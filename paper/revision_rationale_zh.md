# ETBQ 论文修订说明

本文档说明 `paper/ijcai25_revised.tex` 相对用户提供的 Overleaf 草稿所做的主要修订及原因。修订目标是提高论文在顶级计算机视觉/机器学习期刊或会议中的表达质量、逻辑一致性和理论可信度，同时尽量保留原有 LaTeX 结构、公式编号标签、图表引用和文献引用命令。

## 1. 全文叙事主线

**修改内容**

- 将核心问题统一为：现有 PTQ 方法大多是 post-hoc reconstruction，而低比特 PTQ 的根本瓶颈之一是全精度模型所处局部极小值过于尖锐，无法吸收量化扰动。
- 将 ETBQ 定位为 PTQ 之前的 lightweight pre-conditioning stage，而不是替代 PTQ 的新量化器。
- 统一使用 “full-precision loss landscape reshaping / pre-conditioning / quantization-friendly FP model” 等表述。

**修改原因**

原稿中同时出现了“PTQ 后修正”“QAT 速度对比”“lossless UNet”“AP/GFlops”等多个未完全支撑的叙事方向，容易让审稿人无法判断文章的主要贡献。修订后，论文主线更清楚：  
1. 标准 SGD 得到的 FP 模型可能处于 sharp minimum；  
2. 量化误差会把模型推出该盆地；  
3. ETBQ 在量化前注入真实量化误差统计，使 FP 模型迁移到量化友好的 flat basin；  
4. 后续任意 PTQ 方法都能从这个更好的起点获益。

## 2. 摘要

**修改内容**

- 删除了原摘要中的语法错误、占位符和不一致实验描述，例如 “Rather than,”、“traing”、“quantizaiont-unawared”、“On Tiny-ImageNet, xxx”、“AP 88.32% with 16.89 M parameters under 45.36 GFlops”等。
- 摘要按“背景—问题—方法—机制—实验结论”重写。
- 强调 WQN、AQN、SWA 三者作用：WQN 对齐权重量化误差并正则化 Hessian trace；AQN 通过显著性掩码处理高方差激活噪声；SWA 巩固平坦盆地。

**修改原因**

顶会/期刊摘要需要快速建立可信贡献，不能包含占位符、跨任务串稿指标或未在正文表格支撑的数据。修订后的摘要减少夸张表述，避免 “all you need” 式标题口号，突出可验证的技术贡献。

## 3. Introduction

**修改内容**

- 重写前两段，先介绍 PTQ/QAT 背景，再指出 post-hoc PTQ 的隐含假设。
- 将“FP model as infallible oracle”的说法改成更学术的 “implicitly assuming that the FP model is a reliable target under quantization perturbations”。
- 重新解释 Fig.~\ref{fig:comparison_loss} 和 Fig.~\ref{fig:hessian}：前者说明噪声敏感性，后者说明 Hessian 光谱更平坦。
- 删除或替换了大量语法错误表达，如 “connectection”、“comparsion”、“quanzizated”、“nun-uniform”、“absored”等。
- 将贡献列表扩展为四点：几何瓶颈、ETBQ 框架、收敛性分析、实验验证。

**修改原因**

原引言中的关键思想是好的，但表达存在重复和逻辑跳跃：先说激活误差，再说 Hessian，再说极小值，没有形成可审稿的因果链。修订后，引言围绕一个清晰命题展开：低比特 PTQ 失败不仅是校准问题，也是 FP 模型几何问题。

## 4. Related Work

**修改内容**

- 将 PTQ 相关工作整理为从 RTN/AdaRound 到 BRECQ/QDrop/MRECG，再到 FlatQuant/Quantization without Tears 的发展脉络。
- 将 FlatQuant 和 Quantization without Tears 的差异写得更明确：FlatQuant 是 post-hoc re-parameterization，Quantization without Tears 属于 QAT。
- 将 SWA、Dropout、label smoothing、PGD、SGD noise、NIPQ、FQAT 放在 “Robustness by Flatness and Noise Injection” 中讨论。

**修改原因**

原稿的 Related Work 中存在中文批注、重复引用格式、句子残缺和概念混杂。修订后，每个相关工作都服务于本文定位：ETBQ 不是普通 PTQ，也不是 QAT，而是 PTQ 前的量化误差对齐式景观预整形。

## 5. Preliminaries and Objective

**修改内容**

- 保留原有量化公式 \eqref{eqt:quantization}、scale/zero-point 初始化公式和 MSE 误差展开公式。
- 将误差展开后的解释改为更严谨的 AQE/WQE 分解。
- 删除原稿中 “ERQ?”、“xxx”、“Fig.~\ref{}”、“active error is larger than that of xxx”等占位内容。
- 明确指出 activation-induced term 通常大于 weight-induced term，因此需要解耦处理。

**修改原因**

方法章节的第一部分必须建立问题，而不是留下未完成的 bullet。修订后的文字把公式 \eqref{eqt:quantization_delta} 与 ETBQ 的解耦设计直接关联起来，逻辑更紧。

## 6. WQN 方法

**修改内容**

- 明确权重量化误差统计在 per-output-channel 粒度进行，因为权重采用 per-channel quantization。
- 将 “Given a weight tensor XX?” 改成完整定义：$\boldsymbol{W} \in \mathbb{R}^{C_{out}\times C_{in}\times K_H\times K_W}$。
- 解释为什么 $N_i=C_{in}K_HK_W$，为什么得到 $\boldsymbol{\mu}_w \in \mathbb{R}^{C_{out}}$ 和 diagonal covariance。
- 将差分噪声机制写成临时 evaluation weight：$\boldsymbol{W}'_t = \widetilde{\boldsymbol{W}}_t + \lambda_e\boldsymbol{\delta}_t-\boldsymbol{\delta}_{t-1}$。
- 将原稿中 “physical ???? weight tensor”、“constratected”、“ant-isptropy”等错误替换为清晰说明。
- 保留并修正等价潜在权重轨迹 \eqref{eqt:latent_trajectory} 和有效损失 \eqref{eqt:effective_loss_final}。

**修改原因**

WQN 是论文最关键的方法创新之一。原稿中有正确方向，但符号和表述不稳定，容易被审稿人质疑“噪声是否有偏”和“为什么是 per-channel”。修订后强调两个核心点：  
1. 差分机制避免长期均值漂移；  
2. Taylor 展开得到量化中心损失项 + Hessian trace 正则项，说明 WQN 是量化误差对齐的平坦化，而非普通噪声正则。

## 7. AQN 方法

**修改内容**

- 将 AQN 写成：per-tensor Gaussian error modeling + EMA smoothing + salience-aware stochastic mask。
- 保留原公式 \eqref{eqt:aqe_mean_ema}--\eqref{eqt:aqn_stoch_injection}。
- 解释 EMA 的作用：降低 batch-to-batch fluctuation，避免噪声分布突变。
- 解释温度 $\tau$ 的作用：过低会过度集中，过高会退化为 uniform injection。
- 将 implicit Lipschitz regularization 改成更审慎表达：trace penalty discourages amplification of high-variance activation perturbations。

**修改原因**

原稿中 AQN 的理论表达有“guarantees stable convergence”等过强说法，容易被理论审稿人挑战。修订后使用“implicit regularization / discourages / controls along vulnerable directions”等更严谨措辞，同时保留方法直觉。

## 8. Algorithm 和 Training

**修改内容**

- 修复算法中不存在的引用 `\eqref{eqt:wqn_diff_injection}`，改为已有的 `\eqref{eqt:diff_noise_def}`。
- 在算法中加入 noise intensity annealing 和 BatchNorm statistics recomputation。
- 将训练目标简化为交叉熵/标签平滑下的扰动模型训练，并说明 SWA 后需要 BN 统计重校准。

**修改原因**

原算法存在公式标签不匹配，会导致编译或审稿阅读错误。BN 重校准来自毕业论文章节，是 SWA 后非常重要的实现细节，加入后方法更完整。

## 9. Experiments

**修改内容**

- 删除所有与道路异常检测、AUC/AP、road segment、IPP、UIS、WST、WSA、low-rank decomposition 等无关串稿内容。
- 将 CIFAR-100、Tiny-ImageNet、Cityscapes 三类实验分开叙述。
- 修正表格标签不一致：Tiny-ImageNet 表使用 `\label{tbl:tiny-imagenet}`，正文引用也同步。
- 将表格中的 “drop” 改为 “Gain”，因为数值表示相对 baseline 的提升而不是下降。
- 删除 Tiny-ImageNet 中 MobileNetV2/GhostNet 大量 0 占位结果，仅保留毕业论文中明确给出的 ResNet-18 结果。
- 将 Cityscapes 表说明为 mIoU，而不是 accuracy。

**修改原因**

实验部分必须完全避免串稿和占位符，否则会严重损害论文可信度。修订后，每个结论都能被当前表格支持：  
- CIFAR-100 展示跨架构 SOTA 对比；  
- Tiny-ImageNet 展示更高分辨率分类泛化；  
- Cityscapes 展示跨任务到语义分割的泛化。

## 10. Conclusion

**修改内容**

- 删除语法错误和夸张表达。
- 总结 ETBQ 的三个模块及其作用。
- 将未来工作写为非高斯误差建模和 Adam/自适应优化器扩展。

**修改原因**

结论应简洁回扣本文主张，不应引入未展开的新承诺。Adam 扩展来自方法部分对 SGD-style isotropic updates 的限制说明，是合理未来方向。

## 11. Appendix Convergence Proof

**修改内容**

- 将原附录中混乱的证明重写为标准 non-convex stochastic optimization 证明。
- 保留核心定理形式：
  \[
  \frac{1}{T}\sum_{t=1}^{T}\mathbb{E}\|\nabla \mathcal{L}_\sigma(\boldsymbol{W}_t)\|^2
  \leq
  \frac{2(\mathcal{L}_\sigma(\boldsymbol{W}_1)-\mathcal{L}_\sigma^*)}{\eta T}
  +\eta L_\sigma\sigma_{eff}^2.
  \]
- 将“原始 PTQ 目标不可导，因此高斯平滑后可微”改成假设形式，避免过度声称对任意深网严格成立。
- 将差分噪声的 telescope drift 写成 Assumption 3 中的 trajectory-averaged asymptotic unbiasedness。
- 修正原文中 `\begin{proof} xxx \end{proof}` 后又继续证明的结构错误，将证明完整放入 proof 环境。

**修改原因**

原附录最大问题是逻辑不严谨且结构错误：先写 `xxx`，然后 proof 环境结束后继续证明。修订后的证明遵循下降引理、无偏/有界方差、望远镜求和的标准路线，审稿人更容易接受。需要注意的是，严格理论上 ETBQ 的差分噪声是“轨迹平均意义下渐近无偏”，不是每一步严格无偏；修订稿已避免把它说成每步严格无偏。

## 12. 尚需作者确认的内容

以下内容建议在最终 Overleaf 版本中进一步确认：

1. `\url{}` 中的代码链接仍为空，需要填写 GitHub 仓库地址。
2. `ijcai25.bib` 中必须包含所有引用键，例如 `liu-flatquant-iclr-2025`、`fu2025quantizationtears`、`fqat-acmmm-2025`、`ronneberger2015u` 等。
3. 图文件名需要与 Overleaf 中实际文件一致：`fig/comparison_loss.png`、`fig/hessian.png`、`fig/error.png`、`fig/distri_weight.png`、`fig/idea1-1.png`、`fig/resnet18_reduce.png`。
4. 如果要投稿 IJCAI，需要压缩篇幅；当前修订版更接近“完整论文/期刊风格”，可再按页数要求删减 Related Work、实验分析或附录。
5. 若 Tiny-ImageNet 的 MobileNetV2/GhostNet 实验已完成，可将其重新加入表格；否则不要保留 0 或 xxx 占位值。
