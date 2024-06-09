
"""Expose the interface of MS-AMP tensor package."""

from .cast import TypeCast
from .hooks import HookManager
from .metascaling import ScalingMeta
from .scaling_tensor import ScalingTensor
#from .tensor_dist import TensorDist

__all__ = ['TypeCast', 'HookManager', 'ScalingMeta', 'ScalingTensor',]# 'TensorDist']
