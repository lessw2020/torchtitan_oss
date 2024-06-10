
"""Expose the interface of MS-AMP tensor package."""

from torchtitan.fp8.tensors.cast import TypeCast
from torchtitan.fp8.tensors.hooks import HookManager
from torchtitan.fp8.tensors.metascaling import ScalingMeta
from torchtitan.fp8.tensors.scaling_tensor import ScalingTensor
#from .tensor_dist import TensorDist

__all__ = ['TypeCast', 'HookManager', 'ScalingMeta', 'ScalingTensor',]# 'TensorDist']
