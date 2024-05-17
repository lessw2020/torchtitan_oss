# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import math
import importlib
import numbers
import torch
import torch.nn as nn

import triton
import triton.language as tl

from apex._autocast_utils import _cast_if_autocast_enabled

global fused_layer_norm_cuda
fused_layer_norm_cuda = None

def fused_rms_norm_affine(input, weight, normalized_shape, eps=1e-6, memory_efficient=True):
    #args = _cast_if_autocast_enabled(input, weight, normalized_shape, eps, memory_efficient)
    #with torch.cuda.amp.autocast(enabled=False):
    return FusedRMSNormAffineFunction.apply(input, weight, normalized_shape, eps, memory_efficient)


class FusedRMSNormAffineFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight, normalized_shape, eps, memory_efficient=False):
        global fused_layer_norm_cuda
        if fused_layer_norm_cuda is None:
            fused_layer_norm_cuda = importlib.import_module("fused_layer_norm_cuda")
        ctx.normalized_shape = normalized_shape
        ctx.eps = eps
        ctx.memory_efficient = memory_efficient
        input_ = input.contiguous()
        weight_ = weight.contiguous()
        output, invvar = fused_layer_norm_cuda.rms_forward_affine(
            input_, ctx.normalized_shape, weight_, ctx.eps)
        if ctx.memory_efficient:
            ctx.save_for_backward(output, weight_, invvar)
        else:
            ctx.save_for_backward(input_, weight_, invvar)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input_or_output, weight_, invvar = ctx.saved_tensors
        grad_input = grad_weight = None
        grad_input, grad_weight = fused_layer_norm_cuda.rms_backward_affine(
           grad_output.contiguous(), invvar, input_or_output,
           ctx.normalized_shape, weight_, ctx.eps, ctx.memory_efficient
        )
        return grad_input, grad_weight, None, None, None


def create_norm(norm_type: str, dim: int, eps: float = 1e-6):
    """
    Creates the specified normalization layer based on the norm_type.

    Args:
        norm_type (str): The type of normalization layer to create.
            Supported types: 1. rmsnorm 2. fused_rmsnorm 3. layernorm 4. np_layernorm
        dim (int): The dimension of the normalization layer.
        eps (float, optional): The epsilon value for numerical stability. Defaults to 1e-6.

    Returns:
        The created normalization layer.

    Raises:
        NotImplementedError: If an unknown norm_type is provided.
    """
    norm_type = norm_type.lower()  # Normalize to lowercase

    if norm_type == "layernorm":
        return nn.LayerNorm(dim, eps=eps, bias=False)
    elif norm_type == "np_layernorm":
        return nn.LayerNorm(dim, eps=eps, elementwise_affine=False, bias=False)
    elif norm_type == "rmsnorm":
        return RMSNorm(dim, eps=eps)
    elif norm_type == "fused_rmsnorm":
        return FusedRMSNorm(dim, eps=eps)
    elif norm_type == "nvfused_rmsnorm":
        return nvFusedRMSNorm(dim, eps=eps)
    else:
        raise NotImplementedError(f"Unknown norm_type: '{norm_type}'")

class nvFusedRMSNorm(torch.nn.Module):
    r"""Applies RMS Normalization over a mini-batch of inputs

    Currently only runs on cuda() tensors.

    .. math::
        y = \frac{x}{\mathrm{RMS}[x]} * \gamma

    The root-mean-square is calculated separately over the last
    certain number dimensions which have to be of the shape specified by
    :attr:`normalized_shape`.
    :math:`\gamma` is a learnable affine transform parameter of
    :attr:`normalized_shape` if :attr:`elementwise_affine` is ``True``.
    `epsilon` is added to the mean-square, then the root of the sum is taken.

    .. note::
        Unlike Batch Normalization and Instance Normalization, which applies
        scalar scale and bias for each entire channel/plane with the
        :attr:`affine` option, RMS Normalization applies per-element scale
        with :attr:`elementwise_affine`.

    This layer uses statistics computed from input data in both training and
    evaluation modes.

    Args:
        normalized_shape (int or list or torch.Size): input shape from an expected input
            of size

            .. math::
                [* \times \text{normalized}\_\text{shape}[0] \times \text{normalized}\_\text{shape}[1]
                    \times \ldots \times \text{normalized}\_\text{shape}[-1]]

            If a single integer is used, it is treated as a singleton list, and this module will
            normalize over the last dimension which is expected to be of that specific size.
        eps: a value added to the denominator for numerical stability. Default: 1e-5
        elementwise_affine: a boolean value that when set to ``True``, this module
            has learnable per-element affine parameters initialized to ones (for weights)
            and zeros (for biases). Default: ``True``.

    Shape:
        - Input: :math:`(N, *)`
        - Output: :math:`(N, *)` (same shape as input)

    Examples::

        >>> input = torch.randn(20, 5, 10, 10)
        >>> # With Learnable Parameters
        >>> m = apex.normalization.FusedRMSNorm(input.size()[1:])
        >>> # Without Learnable Parameters
        >>> m = apex.normalization.FusedRMSNorm(input.size()[1:], elementwise_affine=False)
        >>> # Normalize over last two dimensions
        >>> m = apex.normalization.FusedRMSNorm([10, 10])
        >>> # Normalize over last dimension of size 10
        >>> m = apex.normalization.FusedRMSNorm(10)
        >>> # Activating the module
        >>> output = m(input)

    .. _`Root Mean Square Layer Normalization`: https://arxiv.org/pdf/1910.07467.pdf
    """

    def __init__(self, normalized_shape, eps=1e-8, elementwise_affine=True, memory_efficient=False):
        super().__init__()

        global fused_layer_norm_cuda
        fused_layer_norm_cuda = importlib.import_module("fused_layer_norm_cuda")

        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = torch.Size(normalized_shape)
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        self.memory_efficient = memory_efficient
        if self.elementwise_affine:
            self.weight = nn.Parameter(torch.empty(*normalized_shape))
        else:
            self.register_parameter("weight", None)
        self.reset_parameters()

    def reset_parameters(self):
        if self.elementwise_affine:
            nn.init.ones_(self.weight)

    def forward(self, input):
        #if torch.jit.is_tracing() or torch.jit.is_scripting() or not input.is_cuda:
        #    return manual_rms_norm(input, self.normalized_shape, self.weight, self.eps)

        if self.elementwise_affine:
            return fused_rms_norm_affine(
                input, self.weight, self.normalized_shape, self.eps, self.memory_efficient
            )
        else:
            return fused_rms_norm(input, self.normalized_shape, self.eps, self.memory_efficient)

    def extra_repr(self):
        return "{normalized_shape}, eps={eps}, " "elementwise_affine={elementwise_affine}".format(**self.__dict__)


class FusedRMSNorm(nn.Module):
    """Fused RMS Norm, wraps a fused Triton Kernel"""

    def __init__(
        self,
        dim: int,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
        self.fused_rms_norm_fn = fused_rms_norm_fn

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """leverages Triton Fused RMS Norm kernel"""
        return self.fused_rms_norm_fn(
            x,
            self.weight,
            eps=self.eps,
        )

    def reset_parameters(self):
        torch.nn.init.ones_(self.weight)  # type: ignore


class RMSNorm(nn.Module):
    """
    Initialize the RMSNorm normalization layer.

    Args:
        dim (int): The dimension of the input tensor.
        eps (float, optional): A small value added to the denominator for numerical stability. Default is 1e-6.

    Attributes:
        eps (float): A small value added to the denominator for numerical stability.
        weight (nn.Parameter): Learnable scaling parameter.

    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x: torch.Tensor):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight

    def reset_parameters(self):
        torch.nn.init.ones_(self.weight)  # type: ignore


# FusedRMSNorm in Triton

# Credit
# Tri Dao's Triton LayerNorm: https://github.com/Dao-AILab/flash-attention/blob/main/flash_attn/ops/triton/layer_norm.py
# Triton LayerNorm tutorial: https://triton-lang.org/main/getting-started/tutorials/05-layer-norm.html


@triton.autotune(
    configs=[
        triton.Config({}, num_warps=1),
        triton.Config({}, num_warps=2),
        triton.Config({}, num_warps=4),
        triton.Config({}, num_warps=8),
        triton.Config({}, num_warps=16),
        triton.Config({}, num_warps=32),
    ],
    key=["N"],
)
@triton.jit
def _rms_norm_fwd_kernel(
    X,
    stride_x,
    Y,
    stride_y,
    W,
    Rstd,
    eps,
    M,  # num rows
    N,  # num cols
    block_N: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, block_N)

    # Load input data and weights
    mask = cols < N
    x = tl.load(X + row * stride_x + cols, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(W + cols, mask=mask, other=0.0).to(tl.float32)

    # Compute mean and variance
    xbar = tl.where(cols < N, x, 0.0)
    var = tl.sum(xbar * xbar, axis=0) / N
    rstd = 1 / tl.sqrt(var + eps)

    # Store the reciprocal standard deviation
    tl.store(Rstd + row, rstd)

    # Normalize and apply linear transformation
    x_hat = x * rstd
    y = x_hat * w

    # Write output
    tl.store(Y + row * stride_y + cols, y, mask=mask)


@triton.autotune(
    configs=[
        triton.Config({}, num_warps=1),
        triton.Config({}, num_warps=2),
        triton.Config({}, num_warps=4),
        triton.Config({}, num_warps=8),
        triton.Config({}, num_warps=16),
        triton.Config({}, num_warps=32),
    ],
    key=["N"],
)
@triton.jit
def _rms_norm_bwd_kernel_sm(
    X,
    stride_x,
    W,
    DY,
    stride_dy,
    DX,
    stride_dx,
    Rstd,
    DW,
    eps,
    M,  # num rows
    N,  # num cols
    rows_per_program,
    block_N: tl.constexpr,
):
    row_block_id = tl.program_id(0)
    row_start = row_block_id * rows_per_program
    cols = tl.arange(0, block_N)
    mask = cols < N

    # Load weights
    w = tl.load(W + cols, mask=mask, other=0.0).to(tl.float32)

    # Accumulate gradients for weights
    dw = tl.zeros((block_N,), dtype=tl.float32)

    row_end = min(row_start + rows_per_program, M)
    for row in range(row_start, row_end):
        # Load input, output gradient, and reciprocal standard deviation
        x = tl.load(X + row * stride_x + cols, mask=mask, other=0.0).to(tl.float32)
        dy = tl.load(DY + row * stride_dy + cols, mask=mask, other=0.0).to(tl.float32)
        rstd = tl.load(Rstd + row)

        # Compute normalized input and gradients
        x_hat = x * rstd
        wdy = w * dy
        dw += dy * x_hat
        c1 = tl.sum(x_hat * wdy, axis=0) / N
        dx = (wdy - x_hat * c1) * rstd

        # Store input gradient
        tl.store(DX + row * stride_dx + cols, dx, mask=mask)

    # Store weight gradients
    tl.store(DW + row_block_id * N + cols, dw, mask=mask)


class TritonFusedRMSNorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, eps):
        x_shape_start = x.shape

        # Flatten input
        x = x.view(-1, x.shape[-1])
        if x.stride(-1) != 1:
            x = x.contiguous()
        if weight.stride(-1) != 1:
            weight = weight.contiguous()

        M, N = x.shape
        y = torch.empty_like(x)
        rstd = torch.empty((M,), dtype=torch.float32, device=x.device)

        max_size = 65536 // x.element_size()
        block_N = min(max_size, triton.next_power_of_2(N))

        if N > block_N:
            raise ValueError(f"N {N} must be <= {block_N=}")

        grid = lambda meta: (M,)
        _rms_norm_fwd_kernel[grid](
            x,
            x.stride(0),
            y,
            y.stride(0),
            weight,
            rstd,
            eps,
            M,
            N,
            block_N,
        )

        ctx.eps = eps
        ctx.save_for_backward(x, weight, rstd)
        ctx.x_shape_start = x_shape_start

        y = y.reshape(x_shape_start)
        return y

    @staticmethod
    def backward(ctx, dy):
        x, weight, rstd = ctx.saved_tensors
        eps = ctx.eps
        x_shape_start = ctx.x_shape_start

        # Flatten input and output gradients
        dy = dy.view(-1, dy.shape[-1])
        if dy.stride(-1) != 1:
            dy = dy.contiguous()

        M, N = dy.shape
        dx = torch.empty_like(x)
        dw = torch.empty_like(weight)

        sm_count = torch.cuda.get_device_properties(x.device).multi_processor_count
        _dw = torch.empty((sm_count, N), dtype=torch.float32, device=weight.device)

        max_size = 65536 // x.element_size()
        block_N = min(max_size, triton.next_power_of_2(N))
        rows_per_sm = math.ceil(M / sm_count)

        if N > block_N:
            raise ValueError(f"N {N} must be <= {block_N=}")

        grid = lambda meta: (sm_count,)
        _rms_norm_bwd_kernel_sm[grid](
            x,
            x.stride(0),
            weight,
            dy,
            dy.stride(0),
            dx,
            dx.stride(0),
            rstd,
            _dw,
            eps,
            M,
            N,
            rows_per_sm,
            block_N,
        )
        dw = _dw.sum(0).to(weight.dtype)
        dx = dx.view(x_shape_start)
        return dx, dw, None


# expose fusedRMSNorm as a function
def fused_rms_norm_fn(
    x,
    weight,
    eps=1e-6,
):
    return TritonFusedRMSNorm.apply(
        x,
        weight,
        eps,
    )
