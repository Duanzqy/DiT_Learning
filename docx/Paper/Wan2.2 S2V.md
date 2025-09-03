#### [Wan 2.2 S2V Tech Report](https://humanaigc.github.io/wan-s2v-webpage/content/wan-s2v.pdf)
![](../attachment/Pasted%20image%2020250901171131.png)
##### 重点关注其中的 **数据处理部分** :
1. 数据来源：
	- [OpenHumanVid](https://github.com/fudan-generative-vision/OpenHumanVid)
	-  [Koala-36M](https://koala36m.github.io/)
	- 手动筛选高质量人类活动视频(e.g. speaking, singing, dancing)
2. Pose：
	- [VitPose](https://github.com/ViTAE-Transformer/ViTPose) tracking pose + 转化为 [DWPose](https://github.com/ViTAE-Transformer/ViTPose)
	- 以 Pose 作为额外的多模态控制信号
	- 以Pose去掉一些视频：
		- 时空上出现较少人物的视频
		- 保留连续说话人脸序列
		- [Light-ASD](https://github.com/Junhua-Liao/Light-ASD) 处理音视频对齐问题
3. 质量过滤
	1. 清晰度：[Dover](https://github.com/VQAssessment/DOVER)
	2. 运动稳定性：使用 [UniMatch](https://github.com/autonomousvision/unimatch) 预测光流再计算运动强度
	3. 脸部、手部清晰的检测：使用 Laplacian 算子作用在脸部与手部来计算（具体如何没有细说，需要查找相关资料）
	4. 美学评分：[christophschuhmann/improved-aesthetic-predictor: CLIP+MLP Aesthetic Score Predictor](https://github.com/christophschuhmann/improved-aesthetic-predictor)
	5. 字幕遮挡检测：检测是否有视频字幕遮挡到手部与脸部

##### Model Architecture：

![](../attachment/Pasted%20image%2020250901171159.png)
###### 概述：
**输入**： Ref Image (单张) + audio + 描述视频的文字 Prompt
**输出**： 保持 Ref Image 身份的音频对齐视频

- 训练时：
	- 输入视频(RGB frames) $X$ 通过 3D VAE -> Latent presentation $x_0$ 
	- 根据 **Flow Matching** 公式 $x_t = t \epsilon + (1 - t) x_0$ 得到 noisy latent $x_t$ 与噪声 $\epsilon$ 的关系
因此，模型的目标是：根据输入的噪声表示 $x_t$ (由目标 latent 加噪而得)，预测速度 velocity $\frac{dx}{dt} = \epsilon - x_0$ 
推理过程中，模型通过条件(reference frame, motion frames, audio input and prompt) 从噪声 $x_t$ 恢复 $x_0$.

###### 具体做法
- 根据 [EMO](https://github.com/HumanAIGC/EMO) ，将 ref、target、motion frames(Optional) 通过 3D VAE 进行时空下采样，并且 concatenate 成为 visual tokens
- motion frames 能提供额外的历史信息，有助于生成长时的连续视频，但是直接 flatten motion frames 会带来大量计算负担，因此采用 [FramePack](https://github.com/lllyasviel/FramePack) 的模块以更高压缩率来压缩之前的frames

###### Audio注入
![](../attachment/Pasted%20image%2020250901171210.png)
 - 原始 audio 通过 [wav2vec](https://arxiv.org/abs/1904.05862) 编码，并采用 [EMO](https://github.com/HumanAIGC/EMO)中的 weighted average layer 结合不同 layer 中的特征
 - 之后这些特征通过 Causal Conv1D 层进行时间压缩，生成与 video latent frame 对齐的 latent audio 特征 $a_i \in \mathcal{R}^{f \times t \times c}$ ( t 为每个帧对于特征数量) 
 - latent audio feature $a$ 注入到 Audio Block，与带噪声的 video latent $x_t = \sum_{i = 1}^{f'} x_{ti} \in \mathcal{R} ^ {(f' \times h \times w) \times c}$ 在时间维度上分别计算注意力($a_i$ 与 $x_{ti}$ 之间计算，而不是整体做完整三维注意力计算，来降低计算开销)
