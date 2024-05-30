# from torch.autograd
import abc
import collections
import contextlib
import functools
import logging
import threading
import weakref
from collections import defaultdict, namedtuple
from typing import (
    Any,
    Callable,
    cast,
    Deque,
    Dict,
    List,
    Optional,
    Sequence,
    Set,
    Tuple,
    Union,
)

import torch
from torch.autograd.variable import Variable
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils.hooks import RemovableHandle

log = logging.getLogger(__name__)


__all__ = [
    "saved_tensors_hooks",
    "save_on_cpu",
    "disable_saved_tensors_hooks",
    "register_multi_grad_hook",
    "allow_mutation_on_saved_tensors",
    "Node",
    "GradientEdge",
    "get_gradient_edge",
    "increment_version",
]


class Node(abc.ABC):
    @abc.abstractmethod
    def name(self) -> str:
        r"""Return the name.

        Example::

            >>> import torch
            >>> a = torch.tensor([0., 0., 0.], requires_grad=True)
            >>> b = a.clone()
            >>> assert isinstance(b.grad_fn, torch.autograd.graph.Node)
            >>> print(b.grad_fn.name())
            CloneBackward0
        """
        ...

    @property
    @abc.abstractmethod
    def next_functions(self) -> Tuple[Tuple[Optional["Node"], int], ...]:
        ...

    @abc.abstractmethod
    def metadata(self) -> dict:
        r"""Return the metadata."""
        ...

    @abc.abstractmethod
    def _register_hook_dict(self, tensor: torch.Tensor) -> None:
        ...

    @abc.abstractmethod
    def register_hook(self, fn: Callable[..., Any]) -> RemovableHandle:
        r"""Register a backward hook.

        The hook will be called every time a gradient with respect to the
        Node is computed. The hook should have the following signature::

            hook(grad_inputs: Tuple[Tensor], grad_outputs: Tuple[Tensor]) -> Tuple[Tensor] or None


        The hook should not modify its argument, but it can optionally return
        a new gradient which will be used in place of :attr:`grad_inputs`.

        This function returns a handle with a method ``handle.remove()``
        that removes the hook from the module.

        .. note::
            See :ref:`backward-hooks-execution` for more information on how when this hook
            is executed, and how its execution is ordered relative to other hooks.

        Example::

            >>> import torch
            >>> a = torch.tensor([0., 0., 0.], requires_grad=True)
            >>> b = a.clone()
            >>> assert isinstance(b.grad_fn, torch.autograd.graph.Node)
            >>> handle = b.grad_fn.register_hook(lambda gI, gO: (gO[0] * 2,))
            >>> b.sum().backward(retain_graph=True)
            >>> print(a.grad)
            tensor([2., 2., 2.])
            >>> handle.remove() # Removes the hook
            >>> a.grad = None
            >>> b.sum().backward(retain_graph=True)
            >>> print(a.grad)
            tensor([1., 1., 1.])
        """
        ...

    @abc.abstractmethod
    def register_prehook(self, fn: Callable[..., Any]) -> RemovableHandle:
        r"""Register a backward pre-hook.

        The hook will be called every time a gradient with respect to the
        Node is computed. The hook should have the following signature::

            hook(grad_outputs: Tuple[Tensor]) -> Tuple[Tensor] or None

        The hook should not modify its argument, but it can optionally return
        a new gradient which will be used in place of :attr:`grad_outputs`.

        This function returns a handle with a method ``handle.remove()``
        that removes the hook from the module.

        .. note::
            See :ref:`backward-hooks-execution` for more information on how when this hook
            is executed, and how its execution is ordered relative to other hooks.

        Example::

            >>> a = torch.tensor([0., 0., 0.], requires_grad=True)
            >>> b = a.clone()
            >>> assert isinstance(b.grad_fn, torch.autograd.graph.Node)
            >>> handle = b.grad_fn.register_prehook(lambda gI: (gI[0] * 2,))
            >>> b.sum().backward(retain_graph=True)
            >>> print(a.grad)
            tensor([2., 2., 2.])
            >>> handle.remove()
            >>> a.grad = None
            >>> b.sum().backward(retain_graph=True)
            >>> print(a.grad)
            tensor([1., 1., 1.])
        """
        ...

    @classmethod
    def __subclasshook__(cls, C):
        if cls is Node:
            if (
                C is not None and C is getattr(torch._C._functions, C.__name__, None)
            ) or issubclass(C, torch.autograd.function.BackwardCFunction):
                return True
        return NotImplemented


def _get_grad_fn_or_grad_acc(t):
    if t.requires_grad and t.grad_fn is None:
        return t.view_as(t).grad_fn.next_functions[0][0]
    else:
        return t.grad_fn


GradientEdge = namedtuple("GradientEdge", ("node output_nr"))
GradientEdge.__doc__ = """\
Object representing a given gradient edge within the autograd graph.
To get the gradient edge where a given Tensor gradient will be computed,
you can do ``edge = autograd.graph.get_gradient_edge(tensor)``.
"""


def get_gradient_edge(tensor):
    """Get the gradient edge for computing the gradient of the given Tensor.

    In particular, it is equivalent to call
    ``g = autograd.grad(loss, input)`` and ``g = autograd.grad(loss, get_gradient_edge(input))``.
    """
    if not tensor.requires_grad:
        raise RuntimeError(
            "It is not possible to get the gradient edge for a Tensor that does not require gradients"
        )
    grad_fn = _get_grad_fn_or_grad_acc(tensor)

    # Note that output_nr default to 0 which is the right value
    # for the AccumulateGrad node.
    return GradientEdge(grad_fn, tensor.output_nr)


def increment_version(tensor):
    """Update autograd metadata tracking whether the given Tensor was modified in place.

    This is to enable more accurate error checking within the autograd engine.
    It is already done automatically by PyTorch functions and within custom Function
    when mark_dirty() is called appropriately so you only need to call this explicitly
    if you are doing inplace operation on the Tensor data in a way that Pytorch doesn't
    know about. For example a custom kernel that reads the Tensor data_ptr and modifies
    the memory inplace based on this pointer.

    Note that incrementing the version counter multiple times for a single inplace operation
    is not problematic.
    """
    torch._C._increment_version(tensor)


class saved_tensors_hooks:
    """Context-manager that sets a pair of pack / unpack hooks for saved tensors.

    Use this context-manager to define how intermediary results of an operation
    should be packed before saving, and unpacked on retrieval.

    In that context, the ``pack_hook`` function will be called everytime an
    operation saves a tensor for backward (this includes intermediary results
    saved using
    :func:`~torch.autograd.function._ContextMethodMixin.save_for_backward` but
    also those recorded by a PyTorch-defined operation). The output of
    ``pack_hook`` is then stored in the computation graph instead of the
    original tensor.

    The ``unpack_hook`` is called when the saved tensor needs to be accessed,
    namely when executing :func:`torch.Tensor.backward()` or
    :func:`torch.autograd.grad()`. It takes as argument the *packed* object
    returned by ``pack_hook`` and should return a tensor which has the same
    content as the original tensor (passed as input to the corresponding
    ``pack_hook``).

    The hooks should have the following signatures:

        pack_hook(tensor: Tensor) -> Any

        unpack_hook(Any) -> Tensor

    where the return value of ``pack_hook`` is a valid input to ``unpack_hook``.

    In general, you want ``unpack_hook(pack_hook(t))`` to be equal to ``t`` in terms
    of value, size, dtype and device.

    Example::

        >>> # xdoctest: +REQUIRES(env:TORCH_DOCTEST_AUTOGRAD)
        >>> def pack_hook(x):
        ...     print("Packing", x)
        ...     return x
        >>>
        >>> def unpack_hook(x):
        ...     print("Unpacking", x)
        ...     return x
        >>>
        >>> a = torch.ones(5, requires_grad=True)
        >>> b = torch.ones(5, requires_grad=True) * 2
        >>> with torch.autograd.graph.saved_tensors_hooks(pack_hook, unpack_hook):
        ...     y = a * b
        Packing tensor([1., 1., 1., 1., 1.], requires_grad=True)
        Packing tensor([2., 2., 2., 2., 2.], grad_fn=<MulBackward0>)
        >>> y.sum().backward()
        Unpacking tensor([1., 1., 1., 1., 1.], requires_grad=True)
        Unpacking tensor([2., 2., 2., 2., 2.], grad_fn=<MulBackward0>)

    .. warning ::
        Performing an inplace operation on the input to either hooks may lead
        to undefined behavior.

    .. warning ::
        Only one pair of hooks is allowed at a time. When recursively nesting this
        context-manager, only the inner-most pair of hooks will be applied.
    """

    def __init__(
        self,
        pack_hook: Callable[[torch.Tensor], Any],
        unpack_hook: Callable[[Any], torch.Tensor],
    ):
        self.pack_hook = pack_hook
        self.unpack_hook = unpack_hook

    def __enter__(self):
        torch._C._autograd._push_saved_tensors_default_hooks(
            self.pack_hook, self.unpack_hook
        )

    def __exit__(self, *args: object):
        torch._C._autograd._pop_saved_tensors_default_hooks()

class save_on_cpu2(saved_tensors_hooks):
    """Context manager under which tensors saved by the forward pass will be stored on cpu, then retrieved for backward.

    When performing operations within this context manager, intermediary
    results saved in the graph during the forward pass will be moved to CPU,
    then copied back to the original device when needed for the backward pass.
    If the graph was already on CPU, no tensor copy is performed.

    Use this context-manager to trade compute for GPU memory usage (e.g.
    when your model doesn't fit in GPU memory during training).

    Args:
        pin_memory (bool): If ``True`` tensors will be saved to CPU pinned memory
                           during packing and copied to GPU asynchronously during unpacking.
                           Defaults to ``False``.
                           Also see :ref:`cuda-memory-pinning`.


    Example::

        >>> # xdoctest: +REQUIRES(env:TORCH_DOCTEST_CUDA)
        >>> # xdoctest: +REQUIRES(env:TORCH_DOCTEST_AUTOGRAD)
        >>> a = torch.randn(5, requires_grad=True, device="cuda")
        >>> b = torch.randn(5, requires_grad=True, device="cuda")
        >>> c = torch.randn(5, requires_grad=True, device="cuda")
        >>>
        >>> def f(a, b, c):
        ...     prod_1 = a * b           # a and b are saved on GPU
        ...     with torch.autograd.graph.save_on_cpu():
        ...         prod_2 = prod_1 * c  # prod_1 and c are saved on CPU
        ...     y = prod_2 * a           # prod_2 and a are saved on GPU
        ...     return y
        >>>
        >>> y = f(a, b, c)
        >>> del a, b, c  # for illustration only
        >>> # the content of a, b, and prod_2 are still alive on GPU
        >>> # the content of prod_1 and c only live on CPU
        >>> y.sum().backward()  # all CPU tensors are moved back to GPU, for backward
        >>> # all intermediary tensors are released (deleted) after the call to backward

    """

    def __init__(self, pin_memory=False, device_type="cuda"):
        device_module = getattr(torch, device_type, torch.cuda)

        def pack_to_cpu(tensor):
            if not pin_memory:
                return (tensor.device, tensor.cpu())
            packed = torch.empty(
                tensor.size(),
                dtype=tensor.dtype,
                layout=tensor.layout,
                pin_memory=(device_module.is_available() and not tensor.is_sparse),
            )
            packed.copy_(tensor)
            return (tensor.device, packed)

        def unpack_from_cpu(packed):
            device, tensor = packed
            return tensor.to(device, non_blocking=pin_memory)

        super().__init__(pack_to_cpu, unpack_from_cpu)

def perf_timer(func):
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        output = func(*args, **kwargs)
        elapsed_time = time.perf_counter() - start
        print(elapsed_time)
        return output, elapsed_time

    return wrapper

import time
import secrets
import importlib

_gb = 1024 * 1024 * 1024
def get_tensor_id()-> str:
        return secrets.token_urlsafe(nbytes=8)

def _get_num_bytes_tensor(x):
    return x.dtype.itemsize * x.numel()

def _get_num_bytes_shape(s, dtype=torch.float32):
    return _dtype_size[dtype] * math.prod(s)

global uvm_mgr
uvm_mgr = None

_uvmlib = "uvm_pytorch"

def get_uvm_manager():
    """ single instance mgr for unified memory tensors """
    global uvm_mgr
    if uvm_mgr is None:
        uvm_mgr = importlib.import_module(_uvmlib)
    return uvm_mgr


class save_on_cpu(saved_tensors_hooks):
    """Context manager under which tensors saved by the forward pass will be stored on cpu, then retrieved for backward.

        import bitsandbytes as bnb
        from bitsandbytes import functional as F

    """


    def __init__(self, pinned_memory=True, device_type="cuda"):
        print(f" ^^^^^^^^^   Init called! ^^^^^^^^^^^^ ")
        import bitsandbytes as bnb
        from bitsandbytes import functional as F
        device_module = getattr(torch, device_type, torch.cuda)
        total_side_streams = 8
        all_streams = [torch.cuda.Stream() for _ in range(total_side_streams)]
        all_current_stream_idx = 0
        torch_null_stream = torch.cuda.current_stream()
        max_num_inflight_tensors = 0
        min_copy_size = 256
        self.blocksize = 64
        self.quantdict = {}

        is_first_forward = True
        is_first_backward = True

        colossal_tensor_min_bytes = 8192
        cpu_cache = True

        forward_start_time = 0
        backward_start_time = 0
        index_ready = False

        self.ignore_types = [torch.complex64, torch.int64]
        e_bits = 5
        p_bits = 7 - e_bits
        self.code = F.create_fp8_map(True, e_bits, p_bits).cuda()
        #print(f"code = {self.code}")

        current_stream_idx = 0
        fast_lookup = {}  # (tensor_id, prev, location, stream)
        prev = None  # Declare prev in the outer scope
        tensor_cache = defaultdict(list)
        import actnn.cpp_extension.quantization as ext_quantization

        self.threshold = 0.03
        self.quant_index = []
        self.quant = 0

        self.index_ready = False
        self.count =0
        self.uvm_mgr= get_uvm_manager()
        assert self.uvm_mgr is not None, "failed to import uvm_pytorch"



        def get_tensor_size_id(tensor):
            #num_bytes = _get_num_bytes_tensor(tensor)
            #dims = str(tensor.dim())
            size_id = tuple(tensor.size()) # str(dims+str(tuple(tensor.size())))
            #print(f"size_id = {size_id}")
            return size_id

        def aam_pack_hook(input_tensor: torch.Tensor) -> str:
            from bitsandbytes import functional as F

            nonlocal prev
            nonlocal all_current_stream_idx
            nonlocal is_first_forward
            nonlocal is_first_backward
            nonlocal torch_null_stream
            nonlocal forward_start_time
            nonlocal backward_start_time

            if is_first_forward:
                #if backward_start_time != 0:
                #    end_backward_time = time.perf_counter()
                #   print(f"***** backward took {(end_backward_time - backward_start_time):.3f} seconds")

                #forward_start_time = time.perf_counter()
                torch_null_stream = torch.cuda.current_stream()
                #print(f"fast lookup size {len(fast_lookup)=}")
                fast_lookup.clear()

                print(f"***** first forward")
                is_first_forward = False
                is_first_backward = True
                prev = None
                self.count = -1

            #print(f"***** forward {input_tensor.shape=}, {input_tensor.dtype=}")
            #if input_tensor.dtype == torch.float32:
            #    print(f"***** float32 {input_tensor[0:10]=}")
            tensor_id = get_tensor_id()
            self.count +=1

            tensor_dtype = input_tensor.dtype
            num_bytes = _get_num_bytes_tensor(input_tensor)
            sizes = input_tensor.size()
            if input_tensor.numel() < min_copy_size or (input_tensor.dtype in self.ignore_types):
                #print(f"skipping {input_tensor.shape=}, {input_tensor.dtype=}")
                gpu_clone = input_tensor.clone().detach()
                fast_lookup[tensor_id] = (gpu_clone, None, input_tensor.dtype)  #False = not quantized
                #prev = tensor_id
                #if not self.index_ready:
                #    self.quant_index.append(False)

                #self.quant_index
                return tensor_id


            size_id = get_tensor_size_id(input_tensor)
            #if cpu_cache and num_bytes > colossal_tensor_min_bytes and len(tensor_cache[size_id]):
                #print(f"***** re-using cpu memory {num_bytes=} bytes and size_id = {size_id}")
                #print(f"{size_id=}, {tensor_cache[size_id]=}")
            #    cpu_tensor = tensor_cache[size_id].pop()
                #if cpu_tensor.size != input_tensor.size():
                #    print(f"size mismatch {size_id=}, {cpu_tensor.shape=}, {input_tensor.shape=}")
                #    print(f"{size_id=}")
                #    for item in tensor_cache[size_id]:
                #        print(f"item in {item.shape=}, {item.dtype=}")

                #assert cpu_tensor.size() == input_tensor.size(), f"size mismatch {size_id=}, {cpu_tensor.shape=}, {input_tensor.shape=}"
            '''else:
                cpu_tensor = torch.empty(
                    input_tensor.size(),
                    dtype=input_tensor.dtype,
                    layout=input_tensor.layout,
                    pin_memory=True,
                    # and device_module.is_available() and not last_tensor.is_sparse,
                )
            '''


            #current_side_stream = all_streams[all_current_stream_idx]
            # ensure this stream is done before next use
            #current_side_stream.wait_stream(torch_null_stream) # torch.cuda.default_stream('cuda'))
            #current_side_stream.wait_stream(current_side_stream)

            # compress tensor
            #compressed_tensor, stats_compression = F.quantize_4bit(input_tensor, quant_type="nf4")
            #if input_tensor.dtype == torch.float32:


                #print(f"***** quantizing {input_tensor.shape=}, {input_tensor.dtype=}")
            # compress various ways
            '''if not self.index_ready:
                #compressed_tensor, stats_compression = F.quantize_4bit(input_tensor, quant_type="nf4")
                compressed_tensor, stats_compression = F.quantize_blockwise(input_tensor, blocksize=self.blocksize, code=self.code)
                #compressed_tensor, stats_compression = F.vectorwise (input_tensor, quant_type="nf4")

                #compressed_tensor, stats_compression = F.vectorwise_quant(input_tensor, dim=0)
                out2 = F.dequantize_blockwise(compressed_tensor, stats_compression, blocksize=self.blocksize, code=self.code)
                #lookup_tensor = F.dequantize_4bit(maybe_compressed_tensor, compress_stats, quant_type="nf4")
                #lookup_tensor = F.dequantize_4bit(maybe_compressed_tensor, compress_stats, quant_type="nf4")
                #out2 = F.dequantize_4bit(compressed_tensor, stats_compression)
                #out2 = F.vectorwise_dequant(compressed_tensor, stats_compression)
                #print(f"{out2.shape=}, {out2.dtype=}, {input_tensor.shape=}, {input_tensor.dtype=}")
                #if not out2:
                #    print(f"***** quantizing failed {input_tensor.shape=}, {input_tensor.dtype=}")
                #    assert False, "failed to quant"
                diff = torch.abs(out2 - input_tensor).mean().item()
                del out2

                print(f"{diff=}")

                if diff <= self.threshold:
                #print(f"***** quantizing {input_tensor.shape=}, {input_tensor.dtype=}")
                    self.quant+=1
                    self.quant_index.append(True)

                    fast_lookup[tensor_id] = (compressed_tensor, stats_compression, input_tensor.dtype)
                else:
                    #print(f"***** not quantizing {input_tensor.shape=}, {input_tensor.dtype=}")
                    self.quant_index.append(False)
                    gpu_clone = input_tensor.clone().detach()
                    fast_lookup[tensor_id] = (gpu_clone, None, input_tensor.dtype)  #False = not quantized
            else:
                #self.quant+=1
                gpu_clone = input_tensor.clone().detach()
                fast_lookup[tensor_id] = (gpu_clone, None)  #False = not quantized

            if self.index_ready:
                # print(f"{self.quant_index[self.count]=}, {self.count=}")
                if self.quant_index[self.count] == True:
                    self.quant+=1
                    compressed_tensor, stats_compression = F.quantize_blockwise(input_tensor, blocksize=self.blocksize, code=self.code)
                    #compressed_tensor, stats_compression = F.vectorwise (input_tensor, quant_type="nf4")
                    #compressed_tensor, stats_compression = F.vectorwise_quant(input_tensor, dim=0)

                    fast_lookup[tensor_id] = (compressed_tensor, stats_compression, input_tensor.dtype)
                else:
                    gpu_clone = input_tensor.clone().detach()
                    fast_lookup[tensor_id] = (gpu_clone, None, input_tensor.dtype)  #False = not quantized
            '''
            sizes = input_tensor.size()
            uvm_storage_tensor = self.uvm_mgr.getManagedTensor(num_bytes, sizes)
            # params = (uvm_storage_tensor, input_tensor.dtype)
            uvm_storage_tensor.copy_(input_tensor, non_blocking=False)
            fast_lookup[tensor_id] = (uvm_storage_tensor, sizes, input_tensor.dtype)
            return tensor_id


            #print(f"packed {tensor_id=}, {cpu_tensor.shape=}")
            #print(f"{len(fast_lookup)=}")
            #self.quantdict[tensor_id] = stats_compression

            #with torch.cuda.stream(current_side_stream):
            #    cpu_tensor.copy_(compressed_tensor, non_blocking=False)


            #all_current_stream_idx = (all_current_stream_idx + 1) % len(all_streams)
            #prev = tensor_id



        # ============ unpacking ==============
        def aam_unpack_hook(unpack_tensor_id: str) -> torch.Tensor:
            nonlocal all_current_stream_idx
            nonlocal current_stream_idx
            nonlocal is_first_backward
            nonlocal is_first_forward
            nonlocal torch_null_stream
            nonlocal backward_start_time

            from bitsandbytes import functional as F


            if is_first_backward:
                self.index_ready = True # we have recorded all tensors

                #end_forward_time = time.perf_counter()
                #print(f"***** forward took {(end_forward_time - forward_start_time):.3f} seconds")
                print(f"***** first backward, managing {len(fast_lookup)} tensors")
                #print(f"{self.quant} tensors quantized")
                #tensor_pct = round(self.quant/len(fast_lookup),4)*100
                #print(f"{tensor_pct}% tensors quantized")
                #self.quant=0
                #backward_start_time = time.perf_counter()
                is_first_backward = False
                is_first_forward = True

            # lookup_tensor, next_id, location, side_stream = fast_lookup[unpack_tensor_id]
            maybe_uvm_tensor, tensor_stats, input_dtype = fast_lookup[unpack_tensor_id]

            if tensor_stats is None:
                return maybe_uvm_tensor

            # move to gpu
            #gpu_tensor = maybe_compressed_tensor.to(device="cuda", non_blocking=False)
            #torch.cuda.synchronize()
            #print(f"{compress_stats=}")
            #lookup_tensor = F.dequantize_4bit(maybe_compressed_tensor, compress_stats, quant_type="nf4")
            #lookup_tensor = F.dequantize_blockwise(maybe_compressed_tensor, compress_stats, blocksize=self.blocksize, code=self.code)
            #lookup_tensor = F.vectorwise_dequant(maybe_compressed_tensor, compress_stats)
            #if lookup_tensor.dtype != input_dtype:
            #    lookup_tensor = lookup_tensor.to(input_dtype) # .to(torch.bfloat16)
            #print(f"unpacked {maybe_uvm_tensor.shape=}, {input_dtype=}, {tensor_stats=}")
            res_tensor = torch.empty_like(maybe_uvm_tensor, dtype=input_dtype, device="cuda")
            res_tensor.copy_(maybe_uvm_tensor, non_blocking=True)
            torch.cuda.synchronize()
            return res_tensor# .to(torch.bfloat16)

                #print(f"***** unpacking {unpack_tensor_id=}, {maybe_compressed_tensor.shape=}")
            '''# asynch load next tensor
            if next_id is not None:
                for i in range(max_num_inflight_tensors):
                    asynch_tensor, next_asynch_id, next_location, stream_id = fast_lookup[next_id]
                    if next_id is None:
                        break
                    if next_location == -1:
                        next_stream = all_streams[all_current_stream_idx]
                        # ensure this stream is done before next use
                        #next_stream.wait_stream(torch_null_stream) # torch.cuda.default_stream('cuda'))
                        with torch.cuda.stream(next_stream):
                            if cpu_cache:
                                num_bytes = _get_num_bytes_tensor(lookup_tensor)
                                if num_bytes > colossal_tensor_min_bytes:
                                    size_id = get_tensor_size_id(lookup_tensor)
                                    #print(f"***** saving for re-use , size_id = {size_id}")
                                    tensor_cache[size_id].append(lookup_tensor)

                            asynch_tensor = asynch_tensor.to(device="cuda", non_blocking=False)
                        next_location = 0
                        fast_lookup[next_id] = (asynch_tensor, next_asynch_id, next_location, next_stream)
                        all_current_stream_idx = (all_current_stream_idx + 1) % len(all_streams)
                    else:
                        # get next next id
                        next_id = next_asynch_id
                        if next_id is None:
                            break   # no more tensors to load

            if location == 0:
                return lookup_tensor

            #if side_stream != -1:
            #    torch_null_stream.wait_stream(side_stream)

            gpu_tensor = lookup_tensor.to(device="cuda", non_blocking=False)
            '''

            #return gpu_tensor

            #@perf_timer
        '''def pack_to_cpu(tensor):
                # if not pin_memory:
                #     return (tensor.device, tensor.cpu())
                print(f"incoming tensor for packing {tensor.shape=}")
                nonlocal last_tensor
                nonlocal current_pack_stream
                if last_tensor is None:
                    last_tensor = tensor
                else:
                    if tensor.numel() < 100:
                        print(f"skipping {tensor.shape=}, {tensor=}")
                        gpu_tensor_stack.append(tensor)
                        return "empty"
                    with torch.cuda.stream(current_pack_stream):
                        packed = torch.empty(
                            last_tensor.size(),
                            dtype=last_tensor.dtype,
                            layout=last_tensor.layout,
                            pin_memory=pinned_memory, # and device_module.is_available() and not last_tensor.is_sparse,
                        )
                        packed.copy_(last_tensor, non_blocking=True)
                    print(f"Asynchronously moved last_tensor to CPU using {current_pack_stream}. Memory on CUDA: {torch.cuda.memory_allocated()}")
                    cpu_tensor_stack.append(packed)
                    last_tensor = tensor
                    print(f"Replaced last_tensor. Memory on CUDA: {torch.cuda.memory_allocated()/_gb:.2f} GB")
                # Alternate between stream1 and stream2
                current_pack_stream = stream2 if current_pack_stream == stream1 else stream1

            #@perf_timer
            def unpack_from_cpu(packed=None):
                nonlocal last_tensor
                nonlocal current_unpack_stream
                if packed == 'empty':
                    #assert False, "empty"
                    print(f"***** returning tensor from gpu stack")
                    res = gpu_tensor_stack.pop()
                    print(f"***** returning {res.shape=}, {res=}")
                    return res

                res = last_tensor
                if len(cpu_tensor_stack):
                    print(f"migrating (cpu->gpu) next tensor, returning {res.shape=} from cpu, and memory on cuda is: {torch.cuda.memory_allocated()/_gb:.2f}")
                    with torch.cuda.stream(current_unpack_stream):
                        last_tensor = cpu_tensor_stack.pop().to(device="cuda", non_blocking=True)
                    # alternate copy to gpu streams
                    current_unpack_stream = stream2 if current_unpack_stream == stream1 else stream1
                else:
                    last_tensor = None
                return res
        '''

        #super().__init__(pack_to_cpu, unpack_from_cpu)
        super().__init__(aam_pack_hook, aam_unpack_hook)


@contextlib.contextmanager
def disable_saved_tensors_hooks(error_message):
    """Context-manager that disables the saved tensors default hooks feature.

    Useful for if you are creating a feature that does not work with saved
    tensors default hooks.

    Args:
        error_message (str): When saved tensors default hooks are used when they
                             have been are disabled, a RuntimeError with this
                             error message gets raised.

    Example::

        >>> # xdoctest: +SKIP(failing)
        >>> message = "saved tensors default hooks are disabled"
        >>> with torch.autograd.graph.disable_saved_tensors_hooks(message):
        ...     # Raises RuntimeError: saved tensors default hooks are disabled
        ...     with torch.autograd.graph.save_on_cpu():
        ...         pass

    """
    try:
        maybe_prev_message = (
            torch._C._autograd._saved_tensors_hooks_get_disabled_error_message()
        )
        torch._C._autograd._saved_tensors_hooks_disable(error_message)
        yield
    finally:
        # See NOTE: [disabled_error_message invariant]
        if maybe_prev_message is None:
            torch._C._autograd._saved_tensors_hooks_enable()
        else:
            torch._C._autograd._saved_tensors_hooks_disable(maybe_prev_message)


class _MultiHandle(RemovableHandle):
    handles: Tuple[RemovableHandle, ...]

    def __init__(self, handles: Tuple[RemovableHandle, ...]):
        self.handles = handles

    def remove(self):
        for handle in self.handles:
            handle.remove()

    def __getstate__(self):
        return self.handles

    def __setstate__(self, state):
        self.handles = state


def register_multi_grad_hook(
    tensors: Sequence[torch.Tensor],
    fn: Union[
        Callable[[Sequence[Optional[torch.Tensor]]], None],
        Callable[[torch.Tensor], None],
    ],
    *,
    mode: str = "all",
):
    r"""Register a multi-grad backward hook.

    There are two supported modes: ``"all"`` and ``"any"``.

    Under the ``"all"`` mode, the hook will be called after gradients with respect to every tensor in
    :attr:`tensors` have been computed. If a tensor is in :attr:`tensors` but
    is not part of the graph, or if a tensor is not needed to compute the gradients
    for any ``inputs`` specified for the current ``.backward()`` or ``.grad()`` call,
    this tensor will be ignored and the hook will not wait for its gradient to be
    computed.

    After every non-ignored tensor's gradient has been computed, :attr:`fn` will be
    called with those gradients. ``None`` will be passed for tensors that did not
    have their gradients computed.

    Under the ``"any"`` mode, the hook will be called after the first gradient
    with respect to a tensor in :attr:`tensors` has been computed. The hook
    will be called with that gradient as its argument.

    The hook should not modify its arguments.

    This function returns a handle with a method ``handle.remove()`` that removes the hook.

    .. note::
        See :ref:`backward-hooks-execution` for more information on how when this hook
        is executed, and how its execution is ordered relative to other hooks.

    Example::

        >>> import torch
        >>>
        >>> a = torch.rand(2, 3, requires_grad=True)
        >>> b = torch.rand(2, 3, requires_grad=True)
        >>> c = a * b
        >>> d = a * b
        >>>
        >>> def fn(grads):
        ...     print([g is not None for g in grads])
        ...
        >>> torch.autograd.graph.register_multi_grad_hook((a, b, c, d), fn)
        >>>
        >>> c.sum().backward(retain_graph=True)
        [True, True, True, False]
        >>> c.sum().backward(inputs=(a,), retain_graph=True)
        [True, False, True, False]
        >>>
    """
    supported_modes = ("all", "any")
    if mode not in supported_modes:
        raise ValueError(f"Expects mode to be one of {supported_modes} but got {mode}")

    if mode == "all":
        count: Dict[int, int] = dict()
        nb_calls = None
        buffer: Dict[int, List[Optional[torch.Tensor]]] = dict()

        grad_fns = list(map(_get_grad_fn_or_grad_acc, tensors))
        len_tensors = len(tensors)

        def get_inner_hook(idx):
            def inner_hook(grad: torch.Tensor):
                nonlocal count, nb_calls, buffer, fn
                id = torch._C._current_graph_task_id()
                assert (
                    id != -1
                ), "expected this hook to be called inside a backward call"
                count[id] = count.get(id, 0)
                buffer[id] = buffer.get(id, [None] * len_tensors)

                if count[id] == 0:
                    # On the first call, compute the actual nb_calls and buffer
                    nb_calls = sum(torch._C._will_engine_execute_node(g) for g in grad_fns)  # type: ignore[attr-defined]

                buffer[id][idx] = grad
                count[id] += 1

                if count[id] == nb_calls:
                    fn = cast(Callable[[Sequence[Optional[torch.Tensor]]], None], fn)
                    fn(buffer[id])
                    del count[id]
                    del buffer[id]

            return inner_hook

        handles: Tuple[RemovableHandle] = tuple(
            t.register_hook(get_inner_hook(i)) for i, t in enumerate(tensors)
        )
    elif mode == "any":
        fn = cast(Callable[[torch.Tensor], None], fn)
        lock = threading.Lock()
        ran_hook: Dict[int, bool] = defaultdict(bool)

        @functools.wraps(fn)
        def wrapped_fn(grad: torch.Tensor):
            nonlocal ran_hook
            id = torch._C._current_graph_task_id()
            assert id != -1, "expected this hook to be called inside a backward call"
            with lock:
                prev, ran_hook[id] = ran_hook[id], True
            if prev:
                return
            fn(grad)

        handles = tuple(
            tensor.register_hook(wrapped_fn)
            for tensor in tensors
            if tensor.requires_grad
        )

    return _MultiHandle(handles)  # type: ignore[possibly-undefined]


# NOTE [Allow mutation on tensors saved for backward]
#
# 1. Tensor gets saved for backward
#    - remember the python object id and the version of the tensor
#    - remember aliasing information (data_ptr of base + version)
#    - save the original so we control its lifetime
# 2. Any time a tensor gets in-placed
#    - for each tensor aliased to it:
#      - check using its object id and version to see if it has been saved
#      - if it has been saved, clone it
#      - delete the reference to the original
# 3. during backward
#    - if the clone exists, the tensor must've been modified in-place
_allow_mutation_on_saved_tensors_enabled = False


def _get_tid(t) -> Tuple[int, int, int]:
    # FIXME: This is almost definitely a bug.
    if isinstance(
        t,
        (
            torch._subclasses.fake_tensor.FakeTensor,
            torch._subclasses.functional_tensor.FunctionalTensor,
        ),
    ):
        data_ptr = 0
    else:
        data_ptr = t.data_ptr()
    return (id(t), data_ptr, t._version)


def _get_sid(t) -> Tuple[int, int]:
    # FIXME: This is almost definitely a bug.
    if isinstance(
        t,
        (
            torch._subclasses.fake_tensor.FakeTensor,
            torch._subclasses.functional_tensor.FunctionalTensor,
        ),
    ):
        data_ptr = 0
    else:
        data_ptr = t.data_ptr()
    return (data_ptr, t._version)


class _Handle:
    pass


class _swap_with_cloned(saved_tensors_hooks):
    def __init__(self, ctx):
        def pack_hook(t):
            tid = _get_tid(t)
            sid = _get_sid(t)
            # Tensors saved for backward have an entry in _tid_to_weakhandle
            handle: Optional[_Handle] = None

            # Save aliasing information
            ctx.sid_to_tid[sid].add(tid)

            # NB: The same tensor (of the same version) can be saved multiple times
            if tid not in ctx.tid_to_weakhandle:
                handle = _Handle()
                ctx.tid_to_weakhandle[tid] = handle
                ctx.original[handle] = t
            else:
                # Store an additional strong reference to the handle
                handle = ctx.tid_to_weakhandle[tid]
            return handle

        def unpack_hook(tup):
            handle = tup
            error_msg = (
                "Trying to backward outside of the 'allow_mutation_on_saved_tensors' context"
                "in which the graph was originally recorded."
            )
            assert _allow_mutation_on_saved_tensors_enabled, error_msg
            if handle in ctx.cloned:
                res = ctx.cloned[handle]
            else:
                assert handle in ctx.original, error_msg
                res = ctx.original[handle]
            return res

        super().__init__(pack_hook, unpack_hook)


class _CloneArgBeforeMutateMode(TorchDispatchMode):
    def __init__(self, ctx):
        self.ctx = ctx

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}

        for idx, arg in enumerate(func._schema.arguments):
            if arg.alias_info is not None and arg.alias_info.is_write:
                t = kwargs["out"] if arg.is_out else args[idx]
                tid = _get_tid(t)
                sid = _get_sid(t)
                ctx = self.ctx
                if sid in ctx.sid_to_tid:
                    for tid in ctx.sid_to_tid[sid]:
                        if tid not in ctx.tid_to_weakhandle:
                            # We know that if tid is in sid_to_tid, then it must also be in
                            # tid_to_weakhandle. However, it is possible for the tensor to be
                            # saved at one point, but cleared by backward before it is modified
                            # in-place. Consider the following example:
                            #
                            # >>> a = torch.randn(2, 3, requires_grad=True).clone()
                            # >>> out = (a**2).sum()
                            # >>> out.backward()
                            # >>> a.sin_()
                            continue
                        handle = ctx.tid_to_weakhandle[tid]
                        if handle in ctx.cloned:
                            # The same exact tensor has been cloned already
                            continue
                        ctx.cloned[handle] = ctx.original[handle].clone()
                        del ctx.original[handle]

        rs = func(*args, **kwargs)
        return rs


class _AllowMutationOnSavedContext:
    def __init__(self):
        self.cloned: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()
        self.original: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()
        self.tid_to_weakhandle: weakref.WeakValueDictionary = (
            weakref.WeakValueDictionary()
        )
        self.sid_to_tid: Dict[Tuple[int, int], Set[Tuple[int, int, int]]] = defaultdict(
            set
        )

    def clear(self):
        self.cloned.clear()
        self.original.clear()
        self.tid_to_weakhandle.clear()
        self.sid_to_tid.clear()


@contextlib.contextmanager
def allow_mutation_on_saved_tensors():
    """Context manager under which mutating tensors saved for backward is allowed.

    Under this context manager, tensors saved for backward are cloned on mutation,
    so the original version can still be used during backward. Normally, mutating a tensor
    saved for backward will result in an error raised when it's used during backward.

    To ensure the correct behavior, both the forward and backward should be run under
    the same context manager.

    returns:
        An _AllowMutationOnSavedContext object storing the state managed by this
        context manager. This object can be useful for debugging purposes. The state
        managed by the context manager is automatically cleared upon exiting.

    Example::

        >>> import torch
        >>> with torch.autograd.graph.allow_mutation_on_saved_tensors():
        ...     # forward
        ...     a = torch.ones(2, 3, requires_grad=True)
        ...     b = a.clone()
        ...     out = (b**2).sum()
        ...     b.sin_()
        ...     # backward
        ...     out.sum().backward()
        ...
        tensor([[0.8415, 0.8415, 0.8415],
                [0.8415, 0.8415, 0.8415]], grad_fn=<SinBackward0>)
    """
    global _allow_mutation_on_saved_tensors_enabled

    ctx = _AllowMutationOnSavedContext()

    with _swap_with_cloned(ctx), _CloneArgBeforeMutateMode(ctx):
        try:
            if _allow_mutation_on_saved_tensors_enabled:
                raise RuntimeError(
                    "allow_mutation_on_saved_tensors contexts cannot be nested"
                )
            _allow_mutation_on_saved_tensors_enabled = True
            yield ctx
        finally:
            ctx.clear()
            _allow_mutation_on_saved_tensors_enabled = False


def _register_logging_hooks_on_whole_graph(t_outputs: List[torch.Tensor]):
    grad_fns = list(map(_get_grad_fn_or_grad_acc, t_outputs))

    def iter_graph(roots):
        if not roots:
            return
        seen = set()
        q: Deque = collections.deque()
        for node in roots:
            if node is not None:
                seen.add(node)
                q.append(node)

        while q:
            node = q.popleft()
            for fn, _idx in node.next_functions:
                if fn in seen or fn is None:
                    continue
                seen.add(fn)
                q.append(fn)

            yield node

    def fmt(t):
        # Avoid circular import
        from torch.testing._internal.common_utils import dtype_abbrs

        if t is None:
            return "None"
        return f"{dtype_abbrs[t.dtype]}[{', '.join(map(str, t.shape))}]"

    def prehook(grad_outputs):
        node = torch._C._current_autograd_node()
        grad_outputs_str = f"[{','.join(fmt(t) for t in grad_outputs)}]"
        log_str = f"Executing: {node} with grad_outputs: {grad_outputs_str}"
        log.debug(log_str)

    handles = []
    for node in iter_graph(grad_fns):
        handles.append(node.register_prehook(prehook))

    def unregister_hooks():
        for handle in handles:
            handle.remove()

    return unregister_hooks


def _engine_run_backward(t_outputs, *args, **kwargs):
    attach_logging_hooks = log.getEffectiveLevel() <= logging.DEBUG
    if attach_logging_hooks:
        unregister_hooks = _register_logging_hooks_on_whole_graph(t_outputs)
    try:
        return Variable._execution_engine.run_backward(  # Calls into the C++ engine to run the backward pass
            t_outputs, *args, **kwargs
        )  # Calls into the C++ engine to run the backward pass
    finally:
        if attach_logging_hooks:
            unregister_hooks()  # type: ignore[possibly-undefined]
