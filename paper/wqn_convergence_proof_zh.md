# WQN 权重噪声注入的收敛性证明说明

## 0. 证明对象与边界

本文档只讨论 **WQN（Weight Quantization Noise）权重噪声注入** 的收敛性。完整 ETBQ 框架还包含 AQN 和 SWA，但它们不纳入本证明：

1. AQN 作用于中间激活，噪声依赖输入样本、网络层和显著性掩码，理论分析需要额外处理数据依赖性与层间耦合；
2. SWA 是优化后期的参数平均过程，更接近后处理式的轨迹聚合，不是单步 SGD 更新的一部分；
3. 因此，为保证数学论证干净且可被审稿人接受，本证明只说明：**差分式 WQN 等价于在权重空间中优化一个高斯平滑后的目标函数，并在标准非凸 SGD 假设下收敛到该平滑目标的一阶驻点。**

证明中还采用一个常见的局部分析设定：在被分析的优化区间内，噪声强度 \(\lambda_e\) 和 WQN 的统计量 \((\boldsymbol{\mu}_w,\boldsymbol{\Sigma}_w)\) 视为固定。实际训练中的 annealing 和 EMA 更新可理解为慢变外部调度，不放入当前定理中。

---

## 1. 从权重量化误差出发

设全精度权重为 \(\boldsymbol{W}\)，经过目标比特宽度的模拟量化后得到 \(\hat{\boldsymbol{W}}\)。权重量化误差定义为

\[
\boldsymbol{E}_w=\hat{\boldsymbol{W}}-\boldsymbol{W}.
\]

在逐通道权重量化中，同一输出通道共享同一个量化尺度。因此，我们在输出通道粒度上估计误差均值和方差，并将整体权重量化误差建模为

\[
\boldsymbol{\delta}\sim\mathcal{N}(\boldsymbol{\mu}_w,\boldsymbol{\Sigma}_w),
\]

其中 \(\boldsymbol{\mu}_w\) 表示量化误差的均值，\(\boldsymbol{\Sigma}_w\) 表示量化误差协方差。实际实现中通常采用对角协方差近似：

\[
\boldsymbol{\Sigma}_w=\mathrm{diag}(\boldsymbol{\sigma}_w^2).
\]

这里最重要的一点是：\(\boldsymbol{\mu}_w\) 一般并不严格等于零。如果直接向权重中注入

\[
\boldsymbol{\delta}_t\sim\mathcal{N}(\boldsymbol{\mu}_w,\boldsymbol{\Sigma}_w),
\]

则有

\[
\mathbb{E}[\boldsymbol{\delta}_t]=\boldsymbol{\mu}_w.
\]

这意味着每一步都会把权重系统性地推向某个非零方向，从优化角度看会产生持续的均值漂移。因此，朴素加性噪声并不是一个干净的无偏扰动。

---

## 2. 差分式 WQN：消除均值漂移

WQN 的关键设计是用两个同分布的量化误差样本构造差分扰动。令

\[
\boldsymbol{\delta}_{t}^{+},\boldsymbol{\delta}_{t}^{-}
\overset{i.i.d.}{\sim}
\mathcal{N}(\boldsymbol{\mu}_w,\boldsymbol{\Sigma}_w).
\]

定义差分权重噪声为

\[
\boldsymbol{\zeta}_t
=
\lambda_e(\boldsymbol{\delta}_{t}^{+}-\boldsymbol{\delta}_{t}^{-}),
\]

其中 \(\lambda_e\in[0,1]\) 是当前 epoch 的噪声强度系数。

### 2.1 零均值性质

由期望的线性性可得

\[
\begin{aligned}
\mathbb{E}[\boldsymbol{\zeta}_t]
&=
\lambda_e
\left(
\mathbb{E}[\boldsymbol{\delta}_{t}^{+}]
-
\mathbb{E}[\boldsymbol{\delta}_{t}^{-}]
\right)\\
&=
\lambda_e(\boldsymbol{\mu}_w-\boldsymbol{\mu}_w)\\
&=\mathbf{0}.
\end{aligned}
\]

因此，差分 WQN 严格消除了非零量化误差均值带来的漂移。

### 2.2 协方差性质

由于 \(\boldsymbol{\delta}_{t}^{+}\) 与 \(\boldsymbol{\delta}_{t}^{-}\) 独立同分布，

\[
\begin{aligned}
\mathrm{Cov}(\boldsymbol{\zeta}_t)
&=
\lambda_e^2
\mathrm{Cov}(\boldsymbol{\delta}_{t}^{+}-\boldsymbol{\delta}_{t}^{-})\\
&=
\lambda_e^2
\left(
\mathrm{Cov}(\boldsymbol{\delta}_{t}^{+})
+
\mathrm{Cov}(\boldsymbol{\delta}_{t}^{-})
\right)\\
&=
2\lambda_e^2\boldsymbol{\Sigma}_w.
\end{aligned}
\]

所以，WQN 并没有抹掉量化误差的方向结构。它保留了由 \(\boldsymbol{\Sigma}_w\) 描述的方差方向，只是把非零均值去掉。直观地说：WQN 不再让权重被平均误差推走，而是让权重反复经历与真实量化误差方差一致的局部扰动。

---

## 3. WQN 对应的平滑目标函数

令原始训练损失为

\[
\mathcal{L}(\boldsymbol{W}).
\]

WQN 在每一步不是直接在 \(\boldsymbol{W}_t\) 处计算梯度，而是在扰动点

\[
\boldsymbol{W}_t+\boldsymbol{\zeta}_t
\]

处计算梯度。因此，WQN 实际对应的目标不是原始损失，而是下面的权重空间平滑目标：

\[
\mathcal{L}_{\sigma}^{wqn}(\boldsymbol{W})
=
\mathbb{E}_{\boldsymbol{\zeta}}
\left[
\mathcal{L}(\boldsymbol{W}+\boldsymbol{\zeta})
\right],
\]

其中

\[
\boldsymbol{\zeta}\sim
\mathcal{N}(\mathbf{0},2\lambda_e^2\boldsymbol{\Sigma}_w).
\]

这个式子的含义非常重要：WQN 并不是在优化单点损失 \(\mathcal{L}(\boldsymbol{W})\)，而是在优化权重附近一片由量化误差分布决定的邻域平均损失。若某个解只在一个极窄位置损失很低，但稍微受到量化扰动后损失急剧升高，那么它在 \(\mathcal{L}_{\sigma}^{wqn}\) 下不会是一个好解。反之，若某个解附近足够平坦，则它的扰动平均损失仍然较低。

---

## 4. WQN 的隐式 Hessian Trace 正则化

为了说明 WQN 为什么会鼓励平坦极小值，对

\[
\mathcal{L}(\boldsymbol{W}+\boldsymbol{\zeta})
\]

在 \(\boldsymbol{W}\) 附近做二阶 Taylor 展开：

\[
\mathcal{L}(\boldsymbol{W}+\boldsymbol{\zeta})
\approx
\mathcal{L}(\boldsymbol{W})
+
\nabla \mathcal{L}(\boldsymbol{W})^{\top}\boldsymbol{\zeta}
+
\frac{1}{2}
\boldsymbol{\zeta}^{\top}
\boldsymbol{H}_w
\boldsymbol{\zeta},
\]

其中

\[
\boldsymbol{H}_w=\nabla^2\mathcal{L}(\boldsymbol{W})
\]

是损失函数关于权重的 Hessian 矩阵。

对 \(\boldsymbol{\zeta}\) 取期望。由于 \(\mathbb{E}[\boldsymbol{\zeta}]=\mathbf{0}\)，一阶项消失：

\[
\mathbb{E}
\left[
\nabla \mathcal{L}(\boldsymbol{W})^{\top}\boldsymbol{\zeta}
\right]
=
\nabla \mathcal{L}(\boldsymbol{W})^{\top}
\mathbb{E}[\boldsymbol{\zeta}]
=0.
\]

二阶项满足

\[
\mathbb{E}
\left[
\boldsymbol{\zeta}^{\top}\boldsymbol{H}_w\boldsymbol{\zeta}
\right]
=
\mathrm{Tr}
\left(
\boldsymbol{H}_w
\mathbb{E}[\boldsymbol{\zeta}\boldsymbol{\zeta}^{\top}]
\right).
\]

又因为

\[
\mathbb{E}[\boldsymbol{\zeta}\boldsymbol{\zeta}^{\top}]
=
\mathrm{Cov}(\boldsymbol{\zeta})
=
2\lambda_e^2\boldsymbol{\Sigma}_w,
\]

所以

\[
\begin{aligned}
\mathcal{L}_{\sigma}^{wqn}(\boldsymbol{W})
&=
\mathbb{E}_{\boldsymbol{\zeta}}
\left[
\mathcal{L}(\boldsymbol{W}+\boldsymbol{\zeta})
\right]\\
&\approx
\mathcal{L}(\boldsymbol{W})
+
\frac{1}{2}
\mathrm{Tr}
\left(
\boldsymbol{H}_w
\cdot
2\lambda_e^2\boldsymbol{\Sigma}_w
\right)\\
&=
\mathcal{L}(\boldsymbol{W})
+
\lambda_e^2
\mathrm{Tr}
\left(
\boldsymbol{H}_w\boldsymbol{\Sigma}_w
\right).
\end{aligned}
\]

这说明 WQN 的优化目标近似等价于

\[
\boxed{
\mathcal{L}_{\sigma}^{wqn}(\boldsymbol{W})
\approx
\mathcal{L}(\boldsymbol{W})
+
\lambda_e^2
\mathrm{Tr}
(\boldsymbol{H}_w\boldsymbol{\Sigma}_w)
}
\]

这不是普通的均匀平坦化，而是 **量化误差对齐的平坦化**：

- 如果某个权重方向的量化误差方差大，即 \(\boldsymbol{\Sigma}_w\) 在该方向上较大，那么该方向的曲率会受到更强惩罚；
- 如果某个方向量化误差很小，则该方向无需过度平坦化；
- 因此 WQN 学到的是与目标量化网格相匹配的平坦极小值。

---

## 5. 收敛性证明所需假设

下面给出标准非凸 SGD 证明框架中的三个假设。

### 假设 1：平滑目标的梯度 Lipschitz 连续

存在常数 \(L_\sigma>0\)，使得对任意 \(\boldsymbol{W}_1,\boldsymbol{W}_2\)，有

\[
\left\|
\nabla \mathcal{L}_{\sigma}^{wqn}(\boldsymbol{W}_1)
-
\nabla \mathcal{L}_{\sigma}^{wqn}(\boldsymbol{W}_2)
\right\|
\leq
L_\sigma
\left\|
\boldsymbol{W}_1-\boldsymbol{W}_2
\right\|.
\]

这是假设平滑目标足够光滑。它是非凸 SGD 收敛分析的基本条件。

### 假设 2：平滑目标存在下界

存在有限常数 \(\mathcal{L}_{\sigma}^{wqn,*}\)，使得

\[
\mathcal{L}_{\sigma}^{wqn}(\boldsymbol{W})
\geq
\mathcal{L}_{\sigma}^{wqn,*}.
\]

对于交叉熵损失，损失非负，因此该假设自然成立。

### 假设 3：随机梯度无偏且方差有界

定义

\[
\boldsymbol{g}_t
=
\nabla
\mathcal{L}_{\sigma}^{wqn}
(\boldsymbol{W}_t).
\]

WQN 使用的随机梯度为

\[
\tilde{\boldsymbol{g}}_t
=
\nabla
\mathcal{L}
(\boldsymbol{W}_t+\boldsymbol{\zeta}_t;\xi_t),
\]

其中 \(\xi_t\) 表示 mini-batch 随机性。假设

\[
\mathbb{E}
\left[
\tilde{\boldsymbol{g}}_t
\mid
\boldsymbol{W}_t
\right]
=
\boldsymbol{g}_t,
\]

并且存在常数 \(\sigma_w^2\)，使得

\[
\mathbb{E}
\left[
\left\|
\tilde{\boldsymbol{g}}_t-\boldsymbol{g}_t
\right\|^2
\mid
\boldsymbol{W}_t
\right]
\leq
\sigma_w^2.
\]

这里的方差包含两部分：mini-batch SGD 的采样噪声，以及 WQN 差分噪声本身带来的采样噪声。

---

## 6. 定理：WQN 的非凸收敛性

若假设 1--3 成立，并且 WQN 更新为

\[
\boldsymbol{W}_{t+1}
=
\boldsymbol{W}_t
-
\eta
\tilde{\boldsymbol{g}}_t,
\]

学习率满足

\[
\eta\leq\frac{1}{L_\sigma},
\]

则有

\[
\boxed{
\frac{1}{T}
\sum_{t=1}^{T}
\mathbb{E}
\left[
\left\|
\nabla
\mathcal{L}_{\sigma}^{wqn}
(\boldsymbol{W}_t)
\right\|^2
\right]
\leq
\frac{
2
\left(
\mathcal{L}_{\sigma}^{wqn}(\boldsymbol{W}_1)
-
\mathcal{L}_{\sigma}^{wqn,*}
\right)
}{
\eta T
}
+
\eta L_\sigma\sigma_w^2
}
\]

这个结论说明：WQN 会收敛到平滑目标 \(\mathcal{L}_{\sigma}^{wqn}\) 的一阶驻点附近。若采用递减学习率，例如 \(\eta=\mathcal{O}(1/\sqrt{T})\)，右侧会随 \(T\) 增大而下降，从而得到标准意义上的非凸 SGD 收敛。

---

## 7. 逐步证明

为简化记号，令

\[
\mathcal{F}(\boldsymbol{W})
=
\mathcal{L}_{\sigma}^{wqn}(\boldsymbol{W}),
\qquad
\boldsymbol{g}_t
=
\nabla
\mathcal{F}
(\boldsymbol{W}_t).
\]

### 第一步：使用下降引理

由于 \(\mathcal{F}\) 是 \(L_\sigma\)-smooth 的，对于更新

\[
\boldsymbol{W}_{t+1}
=
\boldsymbol{W}_t
-
\eta\tilde{\boldsymbol{g}}_t,
\]

下降引理给出

\[
\mathcal{F}(\boldsymbol{W}_{t+1})
\leq
\mathcal{F}(\boldsymbol{W}_t)
+
\left\langle
\boldsymbol{g}_t,
\boldsymbol{W}_{t+1}-\boldsymbol{W}_t
\right\rangle
+
\frac{L_\sigma}{2}
\left\|
\boldsymbol{W}_{t+1}-\boldsymbol{W}_t
\right\|^2.
\]

代入

\[
\boldsymbol{W}_{t+1}-\boldsymbol{W}_t
=
-
\eta\tilde{\boldsymbol{g}}_t,
\]

得到

\[
\mathcal{F}(\boldsymbol{W}_{t+1})
\leq
\mathcal{F}(\boldsymbol{W}_t)
-
\eta
\left\langle
\boldsymbol{g}_t,
\tilde{\boldsymbol{g}}_t
\right\rangle
+
\frac{L_\sigma\eta^2}{2}
\left\|
\tilde{\boldsymbol{g}}_t
\right\|^2.
\]

### 第二步：对随机性取条件期望

对 \(\boldsymbol{W}_t\) 条件下取期望，并使用无偏性

\[
\mathbb{E}
\left[
\tilde{\boldsymbol{g}}_t
\mid
\boldsymbol{W}_t
\right]
=
\boldsymbol{g}_t,
\]

可得

\[
\mathbb{E}
\left[
\left\langle
\boldsymbol{g}_t,
\tilde{\boldsymbol{g}}_t
\right\rangle
\mid
\boldsymbol{W}_t
\right]
=
\left\|
\boldsymbol{g}_t
\right\|^2.
\]

同时，由方差有界假设，

\[
\begin{aligned}
\mathbb{E}
\left[
\left\|
\tilde{\boldsymbol{g}}_t
\right\|^2
\mid
\boldsymbol{W}_t
\right]
&=
\left\|
\boldsymbol{g}_t
\right\|^2
+
\mathbb{E}
\left[
\left\|
\tilde{\boldsymbol{g}}_t-\boldsymbol{g}_t
\right\|^2
\mid
\boldsymbol{W}_t
\right]\\
&\leq
\left\|
\boldsymbol{g}_t
\right\|^2
+
\sigma_w^2.
\end{aligned}
\]

因此

\[
\begin{aligned}
\mathbb{E}
\left[
\mathcal{F}(\boldsymbol{W}_{t+1})
\right]
&\leq
\mathbb{E}
\left[
\mathcal{F}(\boldsymbol{W}_t)
\right]
-
\eta
\mathbb{E}
\left[
\left\|
\boldsymbol{g}_t
\right\|^2
\right]\\
&\quad
+
\frac{L_\sigma\eta^2}{2}
\mathbb{E}
\left[
\left\|
\boldsymbol{g}_t
\right\|^2
+
\sigma_w^2
\right]\\
&=
\mathbb{E}
\left[
\mathcal{F}(\boldsymbol{W}_t)
\right]
-
\eta
\left(
1-\frac{L_\sigma\eta}{2}
\right)
\mathbb{E}
\left[
\left\|
\boldsymbol{g}_t
\right\|^2
\right]
+
\frac{L_\sigma\eta^2}{2}
\sigma_w^2.
\end{aligned}
\]

### 第三步：使用学习率条件

因为

\[
\eta\leq\frac{1}{L_\sigma},
\]

所以

\[
1-\frac{L_\sigma\eta}{2}
\geq
\frac{1}{2}.
\]

因此

\[
\mathbb{E}
\left[
\mathcal{F}(\boldsymbol{W}_{t+1})
\right]
\leq
\mathbb{E}
\left[
\mathcal{F}(\boldsymbol{W}_t)
\right]
-
\frac{\eta}{2}
\mathbb{E}
\left[
\left\|
\boldsymbol{g}_t
\right\|^2
\right]
+
\frac{L_\sigma\eta^2}{2}
\sigma_w^2.
\]

移项得到

\[
\frac{\eta}{2}
\mathbb{E}
\left[
\left\|
\boldsymbol{g}_t
\right\|^2
\right]
\leq
\mathbb{E}
\left[
\mathcal{F}(\boldsymbol{W}_t)
\right]
-
\mathbb{E}
\left[
\mathcal{F}(\boldsymbol{W}_{t+1})
\right]
+
\frac{L_\sigma\eta^2}{2}
\sigma_w^2.
\]

### 第四步：对整个训练过程求和

对 \(t=1,\ldots,T\) 求和：

\[
\begin{aligned}
\frac{\eta}{2}
\sum_{t=1}^{T}
\mathbb{E}
\left[
\left\|
\boldsymbol{g}_t
\right\|^2
\right]
&\leq
\sum_{t=1}^{T}
\left(
\mathbb{E}[\mathcal{F}(\boldsymbol{W}_t)]
-
\mathbb{E}[\mathcal{F}(\boldsymbol{W}_{t+1})]
\right)\\
&\quad
+
\frac{L_\sigma\eta^2T}{2}\sigma_w^2.
\end{aligned}
\]

中间的损失项发生望远镜相消：

\[
\sum_{t=1}^{T}
\left(
\mathbb{E}[\mathcal{F}(\boldsymbol{W}_t)]
-
\mathbb{E}[\mathcal{F}(\boldsymbol{W}_{t+1})]
\right)
=
\mathcal{F}(\boldsymbol{W}_1)
-
\mathbb{E}
\left[
\mathcal{F}(\boldsymbol{W}_{T+1})
\right].
\]

由下界假设，

\[
\mathbb{E}
\left[
\mathcal{F}(\boldsymbol{W}_{T+1})
\right]
\geq
\mathcal{L}_{\sigma}^{wqn,*}.
\]

所以

\[
\frac{\eta}{2}
\sum_{t=1}^{T}
\mathbb{E}
\left[
\left\|
\boldsymbol{g}_t
\right\|^2
\right]
\leq
\mathcal{L}_{\sigma}^{wqn}(\boldsymbol{W}_1)
-
\mathcal{L}_{\sigma}^{wqn,*}
+
\frac{L_\sigma\eta^2T}{2}
\sigma_w^2.
\]

两边同除以 \(\eta T/2\)，得到

\[
\frac{1}{T}
\sum_{t=1}^{T}
\mathbb{E}
\left[
\left\|
\boldsymbol{g}_t
\right\|^2
\right]
\leq
\frac{
2
\left(
\mathcal{L}_{\sigma}^{wqn}(\boldsymbol{W}_1)
-
\mathcal{L}_{\sigma}^{wqn,*}
\right)
}{
\eta T
}
+
\eta L_\sigma\sigma_w^2.
\]

由于

\[
\boldsymbol{g}_t
=
\nabla
\mathcal{L}_{\sigma}^{wqn}
(\boldsymbol{W}_t),
\]

最终得到

\[
\boxed{
\frac{1}{T}
\sum_{t=1}^{T}
\mathbb{E}
\left[
\left\|
\nabla
\mathcal{L}_{\sigma}^{wqn}
(\boldsymbol{W}_t)
\right\|^2
\right]
\leq
\frac{
2
\left(
\mathcal{L}_{\sigma}^{wqn}(\boldsymbol{W}_1)
-
\mathcal{L}_{\sigma}^{wqn,*}
\right)
}{
\eta T
}
+
\eta L_\sigma\sigma_w^2
}
\]

证毕。

---

## 8. 这个定理真正说明了什么

该定理的含义可以概括为三点。

第一，WQN 的差分噪声是严格零均值的，因此它不会像朴素加性量化噪声那样把优化过程持续推向某个偏移方向。

第二，WQN 优化的并不是原始尖锐损失 \(\mathcal{L}(\boldsymbol{W})\)，而是量化误差分布平滑后的目标 \(\mathcal{L}_{\sigma}^{wqn}(\boldsymbol{W})\)。因此，WQN 自然偏好在量化扰动下仍然稳定的权重区域。

第三，二阶展开说明 WQN 等价于引入

\[
\lambda_e^2
\mathrm{Tr}
(\boldsymbol{H}_w\boldsymbol{\Sigma}_w)
\]

这一 Hessian trace 正则项。它惩罚的不是所有方向的曲率，而是与实际权重量化误差方差 \(\boldsymbol{\Sigma}_w\) 对齐的方向曲率。因此，WQN 所寻找的是 **量化误差对齐的平坦极小值**。

这正是 ETBQ 的核心思想：在 PTQ 之前，让全精度模型先迁移到一个能够吸收低比特权重量化扰动的宽广盆地中，从而为后续 PTQ 提供更稳定、更鲁棒的起点。
