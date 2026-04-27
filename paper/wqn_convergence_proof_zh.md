# WQN 连续步差分噪声的收敛性证明说明

## 0. 证明对象与边界

本文档只讨论 ETBQ 中的权重空间组件，即 **WQN（Weight Quantization Noise）权重噪声注入更新**。完整 ETBQ 还包含 AQN 和 SWA，但它们不纳入本证明：

1. AQN 作用于中间激活，扰动依赖输入样本、网络层和显著性掩码；
2. SWA 是优化后期的参数平均过程，不属于单步 SGD 更新；
3. 因此，本证明只说明：在标准非凸随机优化假设下，WQN 的权重更新收敛到一个权重空间高斯平滑目标的一阶驻点。

需要特别强调的是：这里不使用 fresh-pair 独立同分布差分噪声。本文采用与实际实现一致的连续步差分形式，即当前注入噪声减去上一时刻保留噪声。

---

## 1. 权重量化误差的统计建模

设全精度权重为 \(\boldsymbol{W}\)，模拟量化后的权重为 \(\hat{\boldsymbol{W}}\)。权重量化误差为

\[
\boldsymbol{E}_w=\hat{\boldsymbol{W}}-\boldsymbol{W}.
\]

对于逐通道权重量化，每个输出通道共享一个量化尺度，因此在输出通道粒度上统计误差均值和方差。我们将第 \(t\) 步的权重量化误差样本记为

\[
\boldsymbol{\epsilon}_{t}
\sim
\mathcal{N}(\boldsymbol{\mu}_{w,t},\boldsymbol{\Sigma}_{w,t}),
\]

其中 \(\boldsymbol{\mu}_{w,t}\) 和 \(\boldsymbol{\Sigma}_{w,t}\) 可以随训练进程、权重状态和 EMA 统计缓慢变化。若显式考虑噪声强度退火系数 \(\lambda_e\)，实际注入噪声写作

\[
\boldsymbol{\delta}_{t}
=
\lambda_e \boldsymbol{\epsilon}_{t}.
\]

因此，\(\boldsymbol{\delta}_{t}\) 与 \(\boldsymbol{\delta}_{t-1}\) 一般不应被视为独立同分布样本；它们的均值和方差都可能不同。

---

## 2. 连续步差分式 WQN

WQN 的实际机制不是每一步直接把 \(\boldsymbol{\delta}_{t}\) 加到物理权重上，而是将当前噪声替换上一时刻噪声。令 \(\widetilde{\boldsymbol{W}}_t\) 表示第 \(t\) 步前的物理可训练权重，则临时评估权重为

\[
\boldsymbol{W}'_t
=
\widetilde{\boldsymbol{W}}_t+\boldsymbol{P}_t,
\]

其中差分扰动为

\[
\boldsymbol{P}_t
=
\boldsymbol{\delta}_{t}-\boldsymbol{\delta}_{t-1}.
\]

这与实现一致：模型保存上一时刻已注入的权重噪声，在下一次前向传播时只添加“新噪声 - 旧噪声”的差值。

### 2.1 轨迹平均漂移消失

由于 \(\boldsymbol{\delta}_{t}\) 和 \(\boldsymbol{\delta}_{t-1}\) 的统计量可能不同，一般不能断言

\[
\mathbb{E}[\boldsymbol{P}_t]=\mathbf{0}.
\]

事实上，

\[
\mathbb{E}[\boldsymbol{P}_t]
=
\mathbb{E}[\boldsymbol{\delta}_{t}]
-
\mathbb{E}[\boldsymbol{\delta}_{t-1}].
\]

但沿整个训练轨迹求平均时，差分项形成望远镜求和：

\[
\begin{aligned}
\frac{1}{T}\sum_{t=1}^{T}
\mathbb{E}[\boldsymbol{P}_t]
&=
\frac{1}{T}\sum_{t=1}^{T}
\mathbb{E}[
\boldsymbol{\delta}_{t}
-
\boldsymbol{\delta}_{t-1}
]\\
&=
\frac{
\mathbb{E}[\boldsymbol{\delta}_{T}]
-
\mathbb{E}[\boldsymbol{\delta}_{0}]
}{T}.
\end{aligned}
\]

如果量化噪声均值有界，即存在常数 \(M_\delta\)，使得

\[
\|\mathbb{E}[\boldsymbol{\delta}_{t}]\|\leq M_\delta,
\]

则

\[
\left\|
\frac{1}{T}\sum_{t=1}^{T}
\mathbb{E}[\boldsymbol{P}_t]
\right\|
\leq
\frac{2M_\delta}{T}
\rightarrow 0.
\]

这说明 WQN 的差分扰动并非逐步严格无偏，而是在 **轨迹平均意义下漂移消失**。这比朴素加性噪声更稳定，因为朴素加性噪声会持续把权重推向量化误差均值方向。

---

## 3. WQN 对应的平滑目标

令权重空间中的高斯平滑目标为

\[
\mathcal{L}_{\sigma}(\boldsymbol{W})
=
\mathbb{E}_{\boldsymbol{\delta}}
\left[
\mathcal{L}(\boldsymbol{W}+\boldsymbol{\delta})
\right],
\]

其中 \(\boldsymbol{\delta}\) 表示有效的权重量化扰动。

由于实际的连续步差分噪声具有轻微的局部偏差，我们不声称每一步都严格满足无偏梯度。相反，我们采用一个标准的、可处理的分析假设：由 WQN 诱导的随机梯度可以理想化为平滑目标梯度的条件无偏估计，并具有有界方差。

记

\[
\boldsymbol{g}_t
=
\nabla \mathcal{L}_{\sigma}(\boldsymbol{W}_t),
\]

\[
\tilde{\boldsymbol{g}}_t
\]

为第 \(t\) 步使用的随机梯度。我们采用如下分析假设：

\[
\mathbb{E}_t[\tilde{\boldsymbol{g}}_t]
=
\boldsymbol{g}_t,
\]

并且存在常数 \(\sigma_{\mathrm{eff}}^2\)，使得

\[
\mathbb{E}_t
\left[
\|
\tilde{\boldsymbol{g}}_t-\boldsymbol{g}_t
\|^2
\right]
\leq
\sigma_{\mathrm{eff}}^2.
\]

这里 \(\mathbb{E}_t[\cdot]\) 表示给定历史信息 \(\mathcal{F}_t\) 后的条件期望。这个假设由前述轨迹平均漂移消失所动机化，但不是由望远镜求和严格推出。

---

## 4. 假设条件

### 假设 1：梯度 Lipschitz 连续

平滑目标 \(\mathcal{L}_{\sigma}\) 满足 \(L_{\sigma}\)-smooth：

\[
\|
\nabla \mathcal{L}_{\sigma}(\boldsymbol{W}_1)
-
\nabla \mathcal{L}_{\sigma}(\boldsymbol{W}_2)
\|
\leq
L_{\sigma}
\|
\boldsymbol{W}_1-\boldsymbol{W}_2
\|.
\]

这是非凸随机优化中的标准假设。高斯平滑可以缓解量化扰动带来的局部非光滑性，从而为该假设提供动机。

### 假设 2：目标函数有下界

存在常数 \(\mathcal{L}_{\sigma}^{*}>-\infty\)，使得

\[
\mathcal{L}_{\sigma}(\boldsymbol{W})
\geq
\mathcal{L}_{\sigma}^{*}.
\]

对于交叉熵训练，损失非负，因此该假设自然成立。

### 假设 3：理想化条件无偏与有界方差

如上所述，我们假设

\[
\mathbb{E}_t[\tilde{\boldsymbol{g}}_t]
=
\boldsymbol{g}_t,
\qquad
\mathbb{E}_t
\left[
\|
\tilde{\boldsymbol{g}}_t-\boldsymbol{g}_t
\|^2
\right]
\leq
\sigma_{\mathrm{eff}}^2.
\]

---

## 5. 收敛定理

**定理。** 在假设 1--3 下，若 WQN 权重更新由 SGD 产生：

\[
\boldsymbol{W}_{t+1}
=
\boldsymbol{W}_t-\eta\tilde{\boldsymbol{g}}_t,
\]

且学习率满足

\[
\eta\leq \frac{1}{L_{\sigma}},
\]

则有

\[
\frac{1}{T}
\sum_{t=1}^{T}
\mathbb{E}
\left[
\|
\nabla \mathcal{L}_{\sigma}(\boldsymbol{W}_t)
\|^2
\right]
\leq
\frac{
2(\mathcal{L}_{\sigma}(\boldsymbol{W}_1)-\mathcal{L}_{\sigma}^{*})
}{
\eta T
}
+
\eta L_{\sigma}\sigma_{\mathrm{eff}}^2.
\]

进一步，若取 horizon-dependent constant stepsize：

\[
\eta
=
\min
\left\{
\frac{1}{L_{\sigma}},
\frac{c}{\sqrt{T}}
\right\},
\]

则右端为

\[
\mathcal{O}\left(\frac{1}{\sqrt{T}}\right),
\]

因此

\[
\lim_{T\rightarrow\infty}
\frac{1}{T}
\sum_{t=1}^{T}
\mathbb{E}
\left[
\|
\nabla \mathcal{L}_{\sigma}(\boldsymbol{W}_t)
\|^2
\right]
=0.
\]

---

## 6. 证明

由 \(L_{\sigma}\)-smoothness，对任意一步更新有下降引理：

\[
\mathcal{L}_{\sigma}(\boldsymbol{W}_{t+1})
\leq
\mathcal{L}_{\sigma}(\boldsymbol{W}_t)
+
\langle
\nabla \mathcal{L}_{\sigma}(\boldsymbol{W}_t),
\boldsymbol{W}_{t+1}-\boldsymbol{W}_t
\rangle
+
\frac{L_{\sigma}}{2}
\|
\boldsymbol{W}_{t+1}-\boldsymbol{W}_t
\|^2.
\]

代入 SGD 更新

\[
\boldsymbol{W}_{t+1}-\boldsymbol{W}_t
=
-\eta\tilde{\boldsymbol{g}}_t,
\]

得到

\[
\mathcal{L}_{\sigma}(\boldsymbol{W}_{t+1})
\leq
\mathcal{L}_{\sigma}(\boldsymbol{W}_t)
-
\eta
\langle
\nabla \mathcal{L}_{\sigma}(\boldsymbol{W}_t),
\tilde{\boldsymbol{g}}_t
\rangle
+
\frac{L_{\sigma}\eta^2}{2}
\|
\tilde{\boldsymbol{g}}_t
\|^2.
\]

对历史 \(\mathcal{F}_t\) 条件取期望。由于

\[
\mathbb{E}_t[\tilde{\boldsymbol{g}}_t]
=
\boldsymbol{g}_t
=
\nabla \mathcal{L}_{\sigma}(\boldsymbol{W}_t),
\]

所以

\[
\mathbb{E}_t
\left[
\langle
\nabla \mathcal{L}_{\sigma}(\boldsymbol{W}_t),
\tilde{\boldsymbol{g}}_t
\rangle
\right]
=
\|
\nabla \mathcal{L}_{\sigma}(\boldsymbol{W}_t)
\|^2.
\]

同时，由二阶矩分解：

\[
\begin{aligned}
\mathbb{E}_t
\left[
\|
\tilde{\boldsymbol{g}}_t
\|^2
\right]
&=
\mathbb{E}_t
\left[
\|
\tilde{\boldsymbol{g}}_t-\boldsymbol{g}_t+\boldsymbol{g}_t
\|^2
\right]\\
&=
\mathbb{E}_t
\left[
\|
\tilde{\boldsymbol{g}}_t-\boldsymbol{g}_t
\|^2
\right]
+
\|
\boldsymbol{g}_t
\|^2\\
&\leq
\sigma_{\mathrm{eff}}^2
+
\|
\nabla \mathcal{L}_{\sigma}(\boldsymbol{W}_t)
\|^2.
\end{aligned}
\]

代回下降不等式：

\[
\begin{aligned}
\mathbb{E}_t[
\mathcal{L}_{\sigma}(\boldsymbol{W}_{t+1})
]
&\leq
\mathcal{L}_{\sigma}(\boldsymbol{W}_t)
-
\eta
\|
\nabla \mathcal{L}_{\sigma}(\boldsymbol{W}_t)
\|^2\\
&\quad+
\frac{L_{\sigma}\eta^2}{2}
\left(
\|
\nabla \mathcal{L}_{\sigma}(\boldsymbol{W}_t)
\|^2
+
\sigma_{\mathrm{eff}}^2
\right)\\
&=
\mathcal{L}_{\sigma}(\boldsymbol{W}_t)
-
\eta
\left(
1-\frac{L_{\sigma}\eta}{2}
\right)
\|
\nabla \mathcal{L}_{\sigma}(\boldsymbol{W}_t)
\|^2
+
\frac{L_{\sigma}\eta^2}{2}
\sigma_{\mathrm{eff}}^2.
\end{aligned}
\]

当 \(\eta\leq 1/L_{\sigma}\) 时，

\[
1-\frac{L_{\sigma}\eta}{2}
\geq
\frac{1}{2}.
\]

因此

\[
\frac{\eta}{2}
\|
\nabla \mathcal{L}_{\sigma}(\boldsymbol{W}_t)
\|^2
\leq
\mathcal{L}_{\sigma}(\boldsymbol{W}_t)
-
\mathbb{E}_t[
\mathcal{L}_{\sigma}(\boldsymbol{W}_{t+1})
]
+
\frac{L_{\sigma}\eta^2}{2}
\sigma_{\mathrm{eff}}^2.
\]

对全随机性取期望，并从 \(t=1\) 到 \(T\) 求和：

\[
\begin{aligned}
\frac{\eta}{2}
\sum_{t=1}^{T}
\mathbb{E}
\left[
\|
\nabla \mathcal{L}_{\sigma}(\boldsymbol{W}_t)
\|^2
\right]
&\leq
\sum_{t=1}^{T}
\left(
\mathbb{E}[
\mathcal{L}_{\sigma}(\boldsymbol{W}_t)
]
-
\mathbb{E}[
\mathcal{L}_{\sigma}(\boldsymbol{W}_{t+1})
]
\right)\\
&\quad+
\frac{L_{\sigma}\eta^2T}{2}
\sigma_{\mathrm{eff}}^2\\
&=
\mathcal{L}_{\sigma}(\boldsymbol{W}_1)
-
\mathbb{E}[
\mathcal{L}_{\sigma}(\boldsymbol{W}_{T+1})
]
+
\frac{L_{\sigma}\eta^2T}{2}
\sigma_{\mathrm{eff}}^2\\
&\leq
\mathcal{L}_{\sigma}(\boldsymbol{W}_1)
-
\mathcal{L}_{\sigma}^{*}
+
\frac{L_{\sigma}\eta^2T}{2}
\sigma_{\mathrm{eff}}^2.
\end{aligned}
\]

两边同时除以 \(\eta T/2\)，得到

\[
\frac{1}{T}
\sum_{t=1}^{T}
\mathbb{E}
\left[
\|
\nabla \mathcal{L}_{\sigma}(\boldsymbol{W}_t)
\|^2
\right]
\leq
\frac{
2(\mathcal{L}_{\sigma}(\boldsymbol{W}_1)-\mathcal{L}_{\sigma}^{*})
}{
\eta T
}
+
\eta L_{\sigma}\sigma_{\mathrm{eff}}^2.
\]

证明完毕。

---

## 7. 直观解释

该证明说明，在 WQN 的连续步差分机制下，只要差分扰动的轨迹平均漂移被控制，并且由此诱导的随机梯度可以在分析中视为平滑目标的条件无偏估计，那么 WQN 的权重更新满足标准非凸 SGD 的一阶收敛界。

直观上，WQN 的作用不是让每一步噪声严格零均值，而是防止量化噪声在训练轨迹上持续累积成系统性漂移。望远镜求和保证了长期平均扰动不会把优化器推离稳定区域；而高斯平滑目标则鼓励模型寻找对权重量化扰动不敏感的平坦区域。
