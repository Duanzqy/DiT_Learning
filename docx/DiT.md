# Video-DiT
---
## RoPE（Rotary Position Embedding）数学原理详解

### 1. 简介

Rotary Position Embedding（RoPE，旋转位置编码）是一种在Transformer模型中编码位置信息的方法。与传统的正弦余弦位置编码不同，RoPE将嵌入空间中的每一对维度视为一个二维平面，并在该平面上以与位置相关的角度进行旋转，从而将位置信息嵌入到模型的表达中。

---

### 2. 数学公式

### 2.1 频率构造

对于嵌入维度 $d$，我们将其分为 $d/2$ 组，每组两个维度。每组对应一个频率参数：

$$
\omega_k = \frac{1}{\theta^{2k/d}}
$$

其中，$k=0,1,...,d/2-1$，$\theta$ 通常取 $10,000$。

### 2.2 角度计算

对于第 $pos$ 个 token 位置，每组旋转的角度为：

$$
\phi_k = pos \cdot \omega_k
$$

### 2.3 嵌入向量旋转

假设原始嵌入向量 $x \in \mathbb{R}^d$，每组为 $(x_{2k}, x_{2k+1})$，旋转后为：

$$
\begin{pmatrix}
x'_{2k} \\
x'_{2k+1}
\end{pmatrix}
=
\begin{pmatrix}
\cos(\phi_k) & -\sin(\phi_k) \\
\sin(\phi_k) & \cos(\phi_k)
\end{pmatrix}
\begin{pmatrix}
x_{2k} \\
x_{2k+1}
\end{pmatrix}
$$

即每组二维向量都旋转了 $\phi_k$ 角度。

### 2.4 复数形式（欧拉公式）

也可以将每组用复数表示：

$$
z_k = x_{2k} + i x_{2k+1}
$$

则旋转过程等价于乘以单位复数：

$$
z'_k = z_k \cdot e^{i\phi_k}
$$

其中，$e^{i\phi_k} = \cos(\phi_k) + i\sin(\phi_k)$。

---

### 3. RoPE应用于注意力机制

在多头自注意力中，Query和Key向量通过上述旋转后参与点积：

$$
\text{Attention}(Q, K) = \text{softmax}\left(\frac{Q' \cdot K'^T}{\sqrt{d}}\right)
$$

这种方式使得注意力机制能够自然地捕捉序列中的相对和绝对位置关系。

---

### 4. PyTorch代码实现对应

```python
def precompute_freqs_cis(dim: int, end: int = 1024, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2) / dim))
    freqs = torch.outer(torch.arange(end), freqs)  # 计算所有位置的角度
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # 生成 e^{i\phi_k}
    return freqs_cis
```

- `freqs` 对应 $\omega_k$，每组维度的频率
- `torch.outer` 计算所有位置和维度的角度 $\phi_k$
- `torch.polar` 生成单位复数 $e^{i\phi_k}$，用于后续嵌入向量的旋转

---
3D 位置编码
```python
def precompute_freqs_cis_3d(dim: int, end: int = 1024, theta: float = 10000.0):
    # precompute rope for 3d
    f_freqs_cis = precompute_freqs_cis(dim - 2 * (dim//3), end, theta)
    h_freqs_cis = precompute_freqs_cis(dim // 3, end, theta)
    w_freqs_cis = precompute_freqs_cis(dim // 3, end, theta)
    return f_freqs_cis, h_freqs_cis, w_freqs_cis
```
##### 为什么这里的h、w的频率编码是一样的？
	- 由于输入参数都是 `dim // 3`，`end`，`theta`，这两次调用会产生**同样长度和频率分布**的编码“组”
	- 但实际编码内容不同，这些频率编码会分别应用于 height（h）和 width（w）这两个空间维度的位置。
- 这里为什么要用 `dim - 2 * (dim // 3)` 作为第一个维度的大小？
	- 输入的 dim 是**输入特征的通道维度**，也就是每个时空位置（如一帧中的一个像素点，或一个 patch）的 embedding 维数。这个 `dim` 一般是 Transformer 或神经网络层输出的特征向量长度。
	- 要将 dim 分配给3个空间维度（比如 frame、height、width，或者 depth、height、width），分别做 3D RoPE 编码，通常尽量平分为三份，但 dim 不一定为3的倍数，故这样划分保证三部分加起来为 dim 

---
### 5. 总结

RoPE通过简单高效的旋转，将位置信息编码进向量空间。其核心优势是支持长序列、易于并行计算，并能直接嵌入到Transformer的注意力机制中。



## RMSNorm 数学原理简述

### 1. 简介

**RMSNorm**（Root Mean Square Normalization）是一种用于深度学习模型中的归一化方法。它在Transformer、LLM等模型中被广泛使用，其优点是计算简单、效率高，并且无需减去均值。RMSNorm常被用作LayerNorm的替代方案。

---

### 2. 数学公式

给定输入向量 $x \in \mathbb{R}^d$，RMSNorm 的归一化公式如下：

$$
\text{RMSNorm}(x) = \frac{x}{\operatorname{RMS}(x) + \epsilon} \cdot \gamma
$$

其中：

- $\operatorname{RMS}(x) = \sqrt{\frac{1}{d} \sum_{i=1}^{d} x_i^2}$，即输入向量的均方根
- $\epsilon$ 是一个极小的常数，防止分母为零
- $\gamma$ 是可学习的缩放参数（一般为每个特征一个参数）

---

### 3. 逐步解释

1. **计算均方根：**
   $$
   \operatorname{RMS}(x) = \sqrt{\frac{1}{d} \sum_{i=1}^{d} x_i^2}
   $$
   这是对输入向量所有分量平方，求均值，再开方。

2. **归一化：**
   $$
   x_{\text{norm}} = \frac{x}{\operatorname{RMS}(x) + \epsilon}
   $$
   对输入向量每个分量除以均方根，$\epsilon$保证数值稳定。

3. **缩放：**
   $$
   y = x_{\text{norm}} \cdot \gamma
   $$
   最后乘以可学习参数 $\gamma$。

---

### 4. 与 LayerNorm 的比较

- **LayerNorm** 先减去均值再除以标准差（整体零均值单位方差）。
- **RMSNorm** 仅除以均方根，不减均值，计算更简便，效率更高。

---

### 5. PyTorch 伪代码对应

```python
def rmsnorm(x, gamma, eps):
    rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps)
    x_norm = x / rms
    return x_norm * gamma
```

---

### 6. 总结

RMSNorm 通过均方根归一化，使输入向量具有统一的尺度，并通过可学习参数适应不同特征分布。它是现代大模型常用的高效归一化方式。

## Attention 部分

### Attention调用
```
def flash_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, num_heads: int, compatibility_mode=False):
	...
```
- 将 attention 逻辑集中封装，可以让高层模型代码无感知地切换不同底层实现，兼容各种库和硬件，提升项目的灵活性、可维护性和扩展性，同时代码更简洁易读，方便调试与实验。
- 方便切换不同的实现，例如做消融实验等
- 调用方式一致，参数形状、返回值形状统一，后续换实现、加新实现时不用改高层代码
```
class AttentionModule(nn.Module):
    def __init__(self, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        
    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        x = flash_attention(q, k, v, num_heads= self.num_heads)
        return x
```
##### 问题：**为什么要写一个 `AttentionModule(nn.Module)`，而不是直接用 `flash_attention` 这个函数？**
- 将 Attention 封装为 nn.Module，可以让其像其他神经网络层一样灵活组合、扩展和管理，方便代码模块化和工程实践。**最主要还是融入Pytorch**
	- 如果 Attention 只是一个函数，就无法方便地嵌入 Sequential、ModuleList、模型参数统计、模型保存/加载等体系。
	- 封装成 `nn.Module` 后，可以像其他层一样灵活组合，并参与模型的 save/load、to(device)、half() 等操作。


### 自注意力机制
1. 输入 x，先做线性映射和 RMS 归一化，得到 query 和 key；value 只做线性映射
2. 对 q、k 应用 rope_apply（即 RoPE 位置编码）
3. 用 AttentionModule 计算注意力
4. 输出经过 output 层变换

### 交叉注意力机制
1. 输入 x（主输入）和 y（上下文）。
2. 如果有图片输入，将 y 拆成 img（前257个）和 ctx（context缩写）（剩下的）。
3. q 来自 x；k, v 来自 ctx。
4. 先对 ctx 做注意力。
5. 如果有图像输入，对 img 做注意力，结果加到 x 上。
6. 最后通过输出线性层 o



## DiTBlock
```python
class DiTBlock(nn.Module):
    def __init__(self, has_image_input: bool, dim: int, num_heads: int, ffn_dim: int, eps: float = 1e-6):
        super().__init__()
        ... # 其他初始化
        self.norm1 = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)
        self.norm2 = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)
        self.norm3 = nn.LayerNorm(dim, eps=eps)
        self.ffn = nn.Sequential(nn.Linear(dim, ffn_dim), nn.GELU(
            approximate='tanh'), nn.Linear(ffn_dim, dim))
        self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5) # 可学习的参数
    def forward(self, x, context, t_mod, freqs):
        # msa: multi-head self-attention  mlp: multi-layer perceptron
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.modulation.to(dtype=t_mod.dtype, device=t_mod.device) + t_mod).chunk(6, dim=1)    # 基础调制 + 时间相关调制
        input_x = modulate(self.norm1(x), shift_msa, scale_msa)
        x = x + gate_msa * self.self_attn(input_x, freqs)
        x = x + self.cross_attn(self.norm3(x), context)
        input_x = modulate(self.norm2(x), shift_mlp, scale_mlp)
        x = x + gate_mlp * self.ffn(input_x)
        return x
```
##### 归一化层
```python
self.norm1 = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)
self.norm2 = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)
self.norm3 = nn.LayerNorm(dim, eps=eps)
```
- 前面两层不可学习，因为前两层的 shift、scale 参数在后续通过时间进行调制，如下所示
```python
shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.modulation.to(dtype=t_mod.dtype, device=t_mod.device) + t_mod).chunk(6, dim=1)    # 基础调制 + 时间相关调制
```
- 这两层的 shift、scale 参数加入了时间相关的信息，为了让模型不同扩散步有不同“策略”，极大提升了扩散模型的表达力与生成质量

##### 门控残差连接
```python
x = x + gate_msa * self.self_attn(input_x, freqs)
x = x + gate_mlp * self.ffn(input_x)
```

- 其中 `gate` 是一个**动态生成的标量或向量**（通常shape与F(x)一致），用于“门控”子模块输出对残差流的贡献度。
- `gate` 的取值可以在[0, 1]之间（如果用sigmoid），也可以不限制范围（直接线性）
- 这里的两个 gate 通过“时间调制”机制动态生成的（可随扩散步t变化）
- gate是由当前扩散步调制的，模型能学会在不同t步“侧重”不同分支（比如早期多用FFN，后期多用Attention）
##### FFN网络设计
```python
self.ffn = nn.Sequential(nn.Linear(dim, ffn_dim), nn.GELU(approximate='tanh'), nn.Linear(ffn_dim, dim))
```
- 升维→激活→降维
- GELU（Gaussian Error Linear Unit）是Transformer和Diffusion模型常见的激活函数，效果比ReLU好
- DiTBlock中的FFN（MLP）采用“升维→GELU→降维”结构，是Transformer中最常见和有效的MLP设计，能显著提升特征处理和模型表达能力，几乎是现代Transformer的标配。GELU激活配合近似算法，是当前最优的效率与效果折中。