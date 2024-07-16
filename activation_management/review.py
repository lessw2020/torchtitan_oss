import time
import secrets

def perf_timer(func):
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        output = func(*args, **kwargs)
        elapsed_time = time.perf_counter() - start
        print(elapsed_time)
        return output, elapsed_time

    return wrapper

class manage_activations(saved_tensors_hooks):
    """Context manager under which activation tensors created in the forward pass will be managed
    """

    def __init__(self, pin_memory: bool = False, device_type: str = "cuda") -> None:
        device_module = getattr(torch, device_type, torch.cuda)

        self.caching: bool = False # are we managing cpu cached memory blocks
        self.min_tensor_size_bytes = 1024 # we don't want to bother with small tensors
        self.tracker = {} # tensor_id = (new_tensor, dtype, if_modified)  ---> track what saved/offloaded/compressed tensors, are where
        self.mem_offload_cache = {} # cache of available memory blocks for tensors
        self.gb = 1024 * 1024 * 1024 # bytes in a gigabyte
        self.ignore_types = [torch.complex64, torch.int64] # high precision and thus not good for quantization
        self.is_first_forward = True
        self.is_first_backward = True
        # metrics
        self.timing: bool = True
        self.forward_start_time = 0
        self.backward_start_time = 0

        # platform util functions
        def get_tensor_id()-> str:
            # create a unique id for each tensor we are managing
            return secrets.token_urlsafe(nbytes=8)
        
        def get_tensor_size_id( x: torch.Tensor)-> Tuple[int]:
            # get the tensor shape and total bytes as a tuple for cached memory re-use
            num_bytes = self.get_num_bytes_tensor(x) 
            return tuple(num_bytes, x.size())
        
        def get_num_bytes_tensor( x: torch.Tensor) -> int:
            # get the number of bytes in a tensor, for memory management purposes
            return x.element_size() * x.nelement() #x.element_size() * x._base_storage().nbytes()
        
        def get_bytes_per_dimension(x: torch.Tensor)-> Tuple[int]:
            # this might be too slow but is a way to provide a full byte signature for a tensor
            # and used to match available memory sizes for caching 
            # alternative = (total_bytes, tensor.shape) which does not account for strides
            element_size = x.element_size()
            shape = x.shape
            stride = x.stride()
            
            bytes_per_dim = []
            for dim, (size, stride_val) in enumerate(zip(shape, stride)):
                bytes_in_dim = size * stride_val * element_size
                bytes_per_dim.append(bytes_in_dim)
    
            return tuple(bytes_per_dim)
        
        # -------- core pack / unpack work --------
        def pack_tensor(activation: torch.Tensor) -> str:
            # activations are passed in during forward pass - from here we take over and return a unique id
            if self.is_first_forward:
                if self.timing:
                    if self.backward_start_time:
                        end_backward_time = time.perf_counter()
                        print(f"***** backward pass took {(end_backward_time - self.backward_start_time):.3f} seconds")
                    self.forward_start_time = time.perf_counter()
                
                print(f"total managed activations  {len(self.tracker)=}")
                #if not self.caching:
                self.tracker.clear()
                
                print("***** first forward")
                self.is_first_forward = False
                self.is_first_backward = True
            
            # query for basic tensor info
            activation_dtype = activation.dtype
            num_bytes = get_num_bytes_tensor(activation)
            sizes = activation.size()
            tensor_id = get_tensor_id()

            # skipping complex types, small tensors, and tensors with unsupported dtypes
            if num_bytes < self.min_tensor_size_bytes or (activation_dtype in self.ignore_types):
                print(f"skipping activation of {num_bytes}, size= {sizes}, {activation_dtype=}")
                
                gpu_clone = activation.clone().detach()
                self.tracker[tensor_id] = (gpu_clone, activation.dtype, False)  # False = not modified
                return tensor_id
            else:
                # main activation management code
                print(f"Storing activation {sizes}, {num_bytes=}, {activation.dtype=} as {tensor_id}")
                gpu_clone = activation.clone().detach()
                self.tracker[tensor_id] = (gpu_clone, activation_dtype, True)  # True = (in future) modified
                return tensor_id
     
        def unpack_tensor(unpack_tensor_id: str) -> torch.Tensor:
            # backward pass - we are called with the tensor_id.  
            # We then use the tensor_id to retrieve the saved/offloaded/compressed tensor
            # and return it in original state (or near original for quantized)
            if self.is_first_backward:
                self.is_first_backward = False
                self.is_first_forward = True
                if self.timing:
                    end_forward_time = time.perf_counter()
                    print(f"***** forward took {(end_forward_time - self.forward_start_time):.3f} seconds")
                    print(f"***** first backward, managing {len(self.tracker)} tensors")
                    self.backward_start_time = time.perf_counter()
            
            # retrieve the saved/offloaded/compressed tensor
            assert unpack_tensor_id in self.tracker, f"untracked tensor, {unpack_tensor_id}"
            gpu_tensor, dtype, modified = self.tracker[unpack_tensor_id]
            print(f"Unpacking {unpack_tensor_id}, {gpu_tensor.size()}, {gpu_tensor.dtype=}, {modified=}")
            # clear tensor from tracking
            del self.tracker[unpack_tensor_id]
            return gpu_tensor

        super().__init__(pack_tensor, unpack_tensor)
