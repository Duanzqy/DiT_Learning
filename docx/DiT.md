# Video-DiT


# RoPE（Rotary Position Embedding）数学原理详解

## 1. 简介

Rotary Position Embedding（RoPE，旋转位置编码）是一种在Transformer模型中编码位置信息的方法。与传统的正弦余弦位置编码不同，RoPE将嵌入空间中的每一对维度视为一个二维平面，并在该平面上以与位置相关的角度进行旋转，从而将位置信息嵌入到模型的表达中。

---

## 2. 数学公式

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

## 3. RoPE应用于注意力机制

在多头自注意力中，Query和Key向量通过上述旋转后参与点积：

$$
\text{Attention}(Q, K) = \text{softmax}\left(\frac{Q' \cdot K'^T}{\sqrt{d}}\right)
$$

这种方式使得注意力机制能够自然地捕捉序列中的相对和绝对位置关系。

---

## 4. PyTorch代码实现对应

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

## 5. 总结

RoPE通过简单高效的旋转，将位置信息编码进向量空间。其核心优势是支持长序列、易于并行计算，并能直接嵌入到Transformer的注意力机制中。



# RMSNorm 数学原理简述

## 1. 简介

**RMSNorm**（Root Mean Square Normalization）是一种用于深度学习模型中的归一化方法。它在Transformer、LLM等模型中被广泛使用，其优点是计算简单、效率高，并且无需减去均值。RMSNorm常被用作LayerNorm的替代方案。

---

## 2. 数学公式

给定输入向量 $x \in \mathbb{R}^d$，RMSNorm 的归一化公式如下：

$$
\text{RMSNorm}(x) = \frac{x}{\operatorname{RMS}(x) + \epsilon} \cdot \gamma
$$

其中：

- $\operatorname{RMS}(x) = \sqrt{\frac{1}{d} \sum_{i=1}^{d} x_i^2}$，即输入向量的均方根
- $\epsilon$ 是一个极小的常数，防止分母为零
- $\gamma$ 是可学习的缩放参数（一般为每个特征一个参数）

---

## 3. 逐步解释

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

## 4. 与 LayerNorm 的比较

- **LayerNorm** 先减去均值再除以标准差（整体零均值单位方差）。
- **RMSNorm** 仅除以均方根，不减均值，计算更简便，效率更高。

---

## 5. PyTorch 伪代码对应

```python
def rmsnorm(x, gamma, eps):
    rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps)
    x_norm = x / rms
    return x_norm * gamma
```

---

## 6. 总结

RMSNorm 通过均方根归一化，使输入向量具有统一的尺度，并通过可学习参数适应不同特征分布。它是现代大模型常用的高效归一化方式。



# Attention 部分

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
