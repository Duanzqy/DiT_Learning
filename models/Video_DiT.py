import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple, Optional
from einops import rearrange  # use einops for easier tensor manipulation
from .utils import hash_state_dict_keys

# Check for available attention libraries
try:
    import flash_attn_interface
    FLASH_ATTN_3_AVAILABLE = True
except ModuleNotFoundError:
    FLASH_ATTN_3_AVAILABLE = False

try:
    import flash_attn
    FLASH_ATTN_2_AVAILABLE = True
except ModuleNotFoundError:
    FLASH_ATTN_2_AVAILABLE = False

try:
    from sageattention import sageattn
    SAGE_ATTN_AVAILABLE = True
except ModuleNotFoundError:
    SAGE_ATTN_AVAILABLE = False


# precompute RoPE

# cis means Complex In Sine
def precompute_freqs_cis(dim:int, end: int = 1024, theta: float = 10000.0):
    '''
    Parameters:
        dim: dimension of the embeddings
        end: maximum sequence length
        theta: scaling factor for frequency
    '''
    # precompute rope for 1d
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[:(dim//2)].double() / dim))
    freqs = torch.outer(torch.arange(end, device=freqs.device, dtype=freqs.dtype), freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_cis

def precompute_freqs_cis_3d(dim: int, end: int = 1024, theta: float = 10000.0):
    '''
    Parameters:
        dim: dimension of the embeddings
        end: maximum sequence length
        theta: scaling factor for frequency
    '''
    # precompute rope for 3d
    f_freqs_cis = precompute_freqs_cis(dim - 2 * (dim//3), end, theta)
    h_freqs_cis = precompute_freqs_cis(dim // 3, end, theta)
    w_freqs_cis = precompute_freqs_cis(dim // 3, end, theta)
    return f_freqs_cis, h_freqs_cis, w_freqs_cis

def rope_apply(x, freqs, num_heads):
    '''
    b: batch size
    s: sequence length
    n: num_heads
    d: head_dim
    '''
    x = rearrange(x, "b s (n d) -> b s n d", n = num_heads)
    x_out = x.torch_view_as_complex(x.to(torch.float64()).reshape(x.shape[0], x.shape[1], x.shape[2], -1, 2))
    x_out = torch.view_as_real(x_out * freqs).flatten(start_dim=2)
    '''
    flatten(2): 把第2维及以后的所有维度展平成一个维度, 也就是:
    输入 shape: (b, s, n, d, 2)
    输出 shape: (b, s, n * d * 2)
    '''
    return x_out.to(x.dtype)



# Modulation function
def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """
    Modulate the input tensor x with shift and scale tensors.
    
    Parameters:
        x (torch.Tensor): Input tensor to be modulated.
        shift (torch.Tensor): Shift tensor.
        scale (torch.Tensor): Scale tensor.
        
    Returns:
        torch.Tensor: Modulated tensor.
    """
    return (x * (scale + 1.0) + shift)



class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization.
    
    Parameters:
        dim (int): Dimension of the input tensor.
        eps (float): Small value to avoid division by zero.
    """
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def norm(self, x):
        return torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        return x.norm(x.to(float)).to(dtype) * self.weight


