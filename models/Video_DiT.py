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



### Attention

def flash_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, num_heads: int, compatibility_mode=False):
    if compatibility_mode:
        q = rearrange(q, "b s (n d) -> b n s d", n=num_heads)
        k = rearrange(k, "b s (n d) -> b n s d", n=num_heads)
        v = rearrange(v, "b s (n d) -> b n s d", n=num_heads)
        x = F.scaled_dot_product_attention(q, k, v)
        x = rearrange(x, "b n s d -> b s (n d)", n=num_heads)
    elif FLASH_ATTN_3_AVAILABLE:
        q = rearrange(q, "b s (n d) -> b s n d", n=num_heads)
        k = rearrange(k, "b s (n d) -> b s n d", n=num_heads)
        v = rearrange(v, "b s (n d) -> b s n d", n=num_heads)
        x, _ = flash_attn_interface.flash_attn_func(q, k, v)
        x = rearrange(x, "b s n d -> b s (n d)", n=num_heads)
    elif FLASH_ATTN_2_AVAILABLE:
        q = rearrange(q, "b s (n d) -> b s n d", n=num_heads)
        k = rearrange(k, "b s (n d) -> b s n d", n=num_heads)
        v = rearrange(v, "b s (n d) -> b s n d", n=num_heads)
        x = flash_attn.flash_attn_func(q, k, v)
        x = rearrange(x, "b s n d -> b s (n d)", n=num_heads)
    elif SAGE_ATTN_AVAILABLE:
        q = rearrange(q, "b s (n d) -> b n s d", n=num_heads)
        k = rearrange(k, "b s (n d) -> b n s d", n=num_heads)
        v = rearrange(v, "b s (n d) -> b n s d", n=num_heads)
        x = sageattn(q, k, v)
        x = rearrange(x, "b n s d -> b s (n d)", n=num_heads)
    else:
        q = rearrange(q, "b s (n d) -> b n s d", n=num_heads)
        k = rearrange(k, "b s (n d) -> b n s d", n=num_heads)
        v = rearrange(v, "b s (n d) -> b n s d", n=num_heads)
        x = F.scaled_dot_product_attention(q, k, v)
        x = rearrange(x, "b n s d -> b s (n d)", n=num_heads)
    return x


class AttentionModule(nn.Module):
    def __init__(self, num_heads: int):
        super().__init__()
        self.num_heads = num_heads

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        x = flash_attention(q, k, v, num_heads= self.num_heads)
        return x


class SelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, eps: float = 1e-6):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.q_norm = RMSNorm(dim, eps)
        self.k_norm = RMSNorm(dim, eps)

        self.attn = AttentionModule(self.num_heads)

    def forward(self, x, freqs):
        q = self.q_norm(self.q(x))
        k = self.k_norm(self.k(x))
        q = rope_apply(q, freqs, self.num_heads)
        k = rope_apply(k, freqs, self.num_heads)
        v = self.v(x)
        x = self.attn(q, k, v)
        return self.o(x)


class CrossAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, eps: float = 1e-6, has_image_input: bool = False):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.q_norm = RMSNorm(dim, eps)
        self.k_norm = RMSNorm(dim, eps)

        
        self.has_image_input = has_image_input
        # 如果有图像输入，多加两个线性层
        if self.has_image_input:
            self.k_img = nn.Linear(dim, dim)
            self.v_img = nn.Linear(dim, dim)
            self.k_img_norm = RMSNorm(dim, eps)

        self.attn = AttentionModule(self.num_heads)

    def forward(self, x: torch.Tensor, y: torch.tensor):
            if self.has_image_input:
                img = y[:, :257]
                ctx = y[:, 257:]
            else:
                ctx = y
            
            q = self.q_norm(self.q(x))
            k = self.k_norm(self.k(ctx))
            v = self.v(ctx)
            x = self.attn(q, k, v)

            if self.has_image_input:
                k_img = self.k_img_norm(self.k_img(img))
                v_img = self.v_img(img)
                x_img = self.attn(q, k_img, v_img)
                x = x + x_img

            return self.o(x)     
    

##### DiTBlock
class DiTBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, ffn_dim: int, eps: float = 1e-6, has_image_input: bool =False):
        super().__init__()
        self.has_image_input = has_image_input
        self.num_heads = num_heads
        self.ffn_dim = ffn_dim

        self.self_attn = SelfAttention(dim, num_heads, eps)
        self.cross_attn = CrossAttention(dim, num_heads, eps, has_image_input)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(ffn_dim, dim)
        )
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False)
        self.norm3 = nn.LayerNorm(dim)

        self.modulation = nn.Parameter(nn.randn(1, 6, dim) / dim** 0.5)

    def forward(self, x, context, freqs, t_mod):
        # msa means multi-head self attention, mlp means multi-layer perceptron
        msa_shift, msa_scale, msa_gate, mlp_shift, mlp_scale, mlp_gate = (
            self.modulation.to(dtype=t_mod.dtype, device=t_mod.device) + t_mod).chunk(6, dim=1)
        input_x = modulate(self.norm1(x), msa_shift, msa_scale)
        x = input_x + msa_gate * self.self_attn(input_x, freqs)

        x = x + self.cross_attn(self.norm3(x), context)

        input_x = modulate(self.norm2(x), mlp_shift, mlp_scale)
        x = x + mlp_gate * self.ffn(input_x)

        return x
