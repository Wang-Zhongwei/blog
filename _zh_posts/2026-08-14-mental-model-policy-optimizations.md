---
layout: post
title: "用一个心智模型统一 RL 损失函数"
subtitle: "用一张图对比 PPO/GRPO、DAPO、GSPO、SAO、SAPO……"
date: 2026-08-14
tags: [rl, ppo, grpo, gspo, dapo, sao, sapo]
image: /assets/figures/mental-model-policy-optimizations/ppo-clip-schematic.png
lang: zh
---

受 [Feynman 学习法](https://en.wikipedia.org/wiki/Learning_by_teaching)的启发，我会尽量用最简单的方式讲清楚几个著名的 LLM RL 损失函数——先从 PPO[^ppo] 和 GRPO[^grpo] 讲起，再到近期的几个变体 DAPO[^dapo]、GSPO[^gspo]、SAO[^sao] 和 SAPO[^sapo]。把它们写下来能帮我理解得更透彻，希望对你也有帮助。

> 💡 **刚接触 PPO 或 GRPO？** 在读这篇博客之前，可以先看看这两篇入门介绍：[PPO](https://huggingface.co/blog/deep-rl-ppo) 和 [GRPO](https://huggingface.co/blog/garg-aayush/derive-grpo-loss)。

LLM RL 中的策略损失（policy loss），或者说目标函数（objective），通常都会用到重要性比率（importance ratio）。将 token 级别的比率定义为

$$
\rho_t = \frac{\pi_{\theta}(y_t \mid x, y_{<t})}{\pi_{\text{old}}(y_t \mid x, y_{<t})},
\tag{1}
$$

其中 $$\pi_\theta$$ 是当前策略，$$\pi_{\text{old}}$$ 是旧策略。旧策略可以是本轮梯度更新开始时模型权重的快照（如 PPO），也可以是异步训练（如 SAO[^sao]）中的 rollout 策略，后者可能滞后不止一次梯度更新。

我们以 PPO 为例，看看 $$\rho_t$$ 在 RL 中是怎么用的。完整的 PPO 目标函数[^ppo]为

$$
J(\theta)
=
\mathbb{E}_{x \sim D,\; \{y_t\}_{t=1}^{T} \sim \pi_{\text{old}}(\cdot \mid x)}
\left[
\sum_{t=1}^{|y|}
\min\left(
\rho_t A_t,\;
\operatorname{clip}(\rho_t, 1-\epsilon, 1+\epsilon)A_t
\right)
\right].
\tag{2}
$$

为了简化，我们可以只看 token 级别的目标函数，因为完整的目标函数不过是它在轨迹（trajectory）上的期望。

$$
J_t(\theta) = \min\left(
\rho_t A_t,\;
\operatorname{clip}(\rho_t, 1-\epsilon, 1+\epsilon)A_t
\right).
\tag{3}
$$


式 (3) 中的 `min` 和 `clip` 操作，正是大多数文章用一堆符号把读者绕晕的地方。其实只要先按优势（advantage）$$A_t$$ 的正负拆开目标函数，再看 $$\rho_t$$ 落在裁剪（clipping）边界的哪一侧，逻辑就一目了然了。

## PPO 图解

### 情形 $$A_t > 0$$

这个 token 比我们预想的要好，所以应该提高它的概率。事实上，

$$
J_t(\theta)
=
\min\bigl(\rho_t A_t,\,\operatorname{clip}(\rho_t, 1-\epsilon, 1+\epsilon)A_t\bigr)
=
\min\bigl(\rho_t,\,\operatorname{clip}(\rho_t, 1-\epsilon, 1+\epsilon)\bigr)\,A_t.
$$

$$
\min\bigl(\rho_t,\,\operatorname{clip}(\rho_t, 1-\epsilon, 1+\epsilon)\bigr)
=
\begin{cases}
\rho_t, & \rho_t \le 1+\epsilon, \\[4pt]
1+\epsilon, & \rho_t > 1+\epsilon.
\end{cases}
$$

**未裁剪区域（$$\rho_t < 1 + \epsilon$$）**

$$
J_t(\theta) = \rho_t A_t,
\qquad
\frac{\partial J_t}{\partial \rho_t}=A_t > 0
$$

提高这个好 token 的概率 $$\rho_t$$ 会增大目标 $$J_t$$，这*正是*优势 $$A_t$$ 为正时*我们应该做的*。

**裁剪区域（$$\rho_t > 1+\epsilon$$）**

$$
J_t(\theta) = (1 + \epsilon) A_t,
\qquad
\frac{\partial J_t}{\partial \rho_t}= 0
$$

换句话说，如果当前策略 $$\pi_{\theta}$$ 已经*强烈偏好*这个好 token，它就在提醒我们*别太贪心*——不必再继续加大它的权重了。

### 情形 $$A_t < 0$$

这个 token 比我们预想的要差，所以应该降低它的概率。事实上，

$$
J_t(\theta)
=
\min\bigl(\rho_t A_t,\,\operatorname{clip}(\rho_t, 1-\epsilon, 1+\epsilon)A_t\bigr)
=
\max\bigl(\rho_t,\,\operatorname{clip}(\rho_t, 1-\epsilon, 1+\epsilon)\bigr)\,A_t.
$$

$$
\max\bigl(\rho_t,\,\operatorname{clip}(\rho_t, 1-\epsilon, 1+\epsilon)\bigr)
=
\begin{cases}
1-\epsilon, & \rho_t < 1-\epsilon, \\[4pt]
\rho_t, & \rho_t \ge 1-\epsilon.
\end{cases}
$$

**未裁剪区域（$$\rho_t \ge 1-\epsilon$$）**

$$
J_t(\theta) = \rho_t A_t,
\qquad
\frac{\partial J_t}{\partial \rho_t}=A_t < 0
$$

减小 $$\rho_t$$ 会增大 $$J_t$$，这正是负优势下应有的行为。

**裁剪区域（$$\rho_t < 1-\epsilon$$）**

$$
J_t(\theta) = (1 - \epsilon) A_t,
\qquad
\frac{\partial J_t}{\partial \rho_t}= 0
$$

一旦策略已经朝压低这个 token 的方向走得足够远，再往下压的部分就会被裁剪掉。

[图 1](#figure-ppo-clip-schematic) 总结了上述行为。在 $$\log\rho_t$$–$$A_t$$ 平面上，每个采样到的 token 都会落入四个区域之一：在红色区域，PPO/GRPO 通过提高 token 的概率来鼓励它；在蓝色区域，则抑制它；在其余区域，$$\rho_t$$ 上的梯度为零，相当于把这个 token 从参数更新中剔除了。

<figure id="figure-ppo-clip-schematic">
  <img src="{{ '/assets/figures/mental-model-policy-optimizations/ppo-clip-schematic.png' | relative_url }}" alt="PPO 裁剪机制在优势–对数比率平面上的示意图">
  <figcaption><strong>图 1.</strong> PPO/GRPO 在 \(\log{\rho_t}-A_t\) 平面上的示意图。红色是增大 \(\rho_t\) 的区域，蓝色是减小它的区域，其余部分都被裁剪掉。纵轴取 \(\log\rho_t\)，这样比率原本不对称的取值范围 \((0,\infty)\) 就变成了关于 \(0\) 对称。图中采用默认值 \(\epsilon = 0.2\)。</figcaption>
</figure>


## 广义 RL 目标函数

SAPO 论文[^sapo]提供了一个比较这些策略优化方法的好视角。在 token 级别，把替代目标（surrogate objective）写成

$$
J_t(\theta)=f(\rho_t(\theta);A_t)\,A_t.
\tag{4}
$$

PPO/GRPO、DAPO、GSPO、SAO、SAPO 等策略优化方法，主要区别就在于**权重函数（weight function）** $$f$$ 的选取。对式 (4) 求梯度，

$$
\begin{aligned}
\nabla_\theta J_t
&=
A_t f'(\rho_t;A_t)\,\nabla_\theta\rho_t \\[4pt]
&=
\underbrace{f'(\rho_t;A_t)}_{\text{方法相关}}
\underbrace{
\rho_t\,
\overbrace{A_t\nabla_\theta
\log\pi_\theta(y_t\mid x,y_{<t})}^{\text{策略梯度}}
}_{\text{共享}}.
\end{aligned}
\tag{5}
$$

所有方法都共享 $$\rho_t A_t\nabla_\theta\log\pi_\theta$$ 这一项，唯一的区别在于**门控函数（gate function）** $$f'(\rho_t;A_t)$$，它决定了有多少学习信号（learning signal）能够通过。

[表 1](#table-1) 汇总了 LLM RL 中几个里程碑式策略优化方法的权重函数 $$f$$ 与门控函数 $$f'$$。

<figure id="table-1" markdown="block">

| 方法 | $$f(\rho_t;A_t)$$ | $$f'(\rho_t;A_t)$$ | 要点 |
| --- | --- | --- | --- |
| [PPO](https://arxiv.org/abs/1707.06347)（2017 年 7 月）/ [GRPO](https://arxiv.org/abs/2402.03300)（2024 年 2 月） | $$A_t>0:\;\min(\rho_t,1+\epsilon)$$<br>$$A_t\leq0:\;\max(\rho_t,1-\epsilon)$$ | $$\mathbf{1}\left[A_t>0,\;\rho_t<1+\epsilon\right]$$<br>$$+\;\mathbf{1}\left[A_t<0,\;\rho_t>1-\epsilon\right]$$ | 按 $$A_t$$ 的正负做硬性、非对称的门控 |
| [DAPO](https://arxiv.org/abs/2503.14476)（2025 年 3 月） | $$A_t>0:\;\min(\rho_t,1+\epsilon_h)$$<br>$$A_t\leq0:\;\max(\rho_t,1-\epsilon_l)$$ | $$\mathbf{1}\left[A_t>0,\;\rho_t<1+\epsilon_h\right]$$<br>$$+\;\mathbf{1}\left[A_t<0,\;\rho_t>1-\epsilon_l\right]$$ | 更高的上界为提升有用的低概率 token 留出更多空间 |
| [GSPO](https://arxiv.org/abs/2507.18071)（2025 年 7 月） | 与 DAPO 相同，但作用于 $$(\rho_s,A_s)$$：<br>$$A_s>0:\;\min(\rho_s,1+\epsilon_h)$$<br>$$A_s\leq0:\;\max(\rho_s,1-\epsilon_l)$$ | $$\mathbf{1}\left[A_s>0,\;\rho_s<1+\epsilon_h\right]$$<br>$$+\;\mathbf{1}\left[A_s<0,\;\rho_s>1-\epsilon_l\right]$$ | 每个回答（response）只做一次裁剪决策，而不是逐 token 决策 |
| [SAPO](https://arxiv.org/abs/2511.20347)（2025 年 11 月） | $$\dfrac{4}{\tau_t}\sigma\left(\tau_t(\rho_t-1)\right)$$，<br>若 $$A_t>0$$ 则 $$\tau_t=\tau_{\text{pos}}$$，否则取 $$\tau_{\text{neg}}$$ | $$4\sigma\left(\tau_t(\rho_t-1)\right)\left(1-\sigma\left(\tau_t(\rho_t-1)\right)\right)$$<br>$$=\mathrm{sech}^2\left(\tfrac{\tau_t(\rho_t-1)}{2}\right)$$ | 门控平滑衰减，而不是突然跳变到零；$$\tau_{\text{neg}}>\tau_{\text{pos}}$$ 使负优势下衰减得更快 |
| [SAO](https://arxiv.org/abs/2607.07508)（2026 年 7 月） | $$\widetilde f(\rho_t)=\operatorname{clip}(\rho_t,1-\epsilon_l,1+\epsilon_h)$$ | $$\mathbf{1}\left[1-\epsilon_l<\rho_t<1+\epsilon_h\right]$$ | 只要比率离开信任域（trust region），无论优势正负一律屏蔽 |

<figcaption><strong>表 1.</strong> LLM RL 中里程碑式策略优化方法的权重函数 \(f\) 与门控函数 \(f'\)。</figcaption>
</figure>

图 2–6 针对每种方法，在 $$(A_t,\log\rho_t)$$ 平面上画出了门控 $$f'(\rho_t;A_t)$$ 与学习信号 $$f'(\rho_t;A_t)\,\rho_t A_t$$（图 6 用的是序列级（sequence-level）的 $$(A_s,\rho_s)$$）：

- PPO[^ppo]（[图 2](#figure-ppo-gradient-weight)）对重要性比率 $$\rho_t$$ 做裁剪：一旦某次更新已经足够地鼓励了好 token、或抑制了坏 token，门控就降为零，该 token 不再产生贡献。这样可以让策略每次只更新一小步，训练更稳定。
- GRPO[^grpo]（[图 2](#figure-ppo-gradient-weight)）在目标函数中加入了组采样（group rollout）维度 $$G$$。在 token 级别，它的门控与 PPO 完全相同。
- DAPO[^dapo]（[图 3](#figure-dapo-gradient-weight)）提高了裁剪上界 $$\epsilon_h$$，于是稀有 token——它们的分母很小，往往会产生较大的 $$\rho_t$$——不那么容易被过早裁剪。PPO 边界之上多出来的那一条带状区域，恰恰是学习信号最大的地方。
- SAO[^sao]（[图 4](#figure-sao-gradient-weight)）面向异步训练，此时当前策略与 rollout 策略之间的偏离可能比 PPO 中大得多。$$\rho_t$$ 因而可能显著偏离 1，所以它的门控表现为一个矩形窗（top-hat window）：只要比率离开信任域，正负两种符号都会被屏蔽。
- SAPO[^sapo]（[图 5](#figure-sapo-gradient-weight)）用平滑门控取代了这种硬边界。SAPO 不是在 $$\rho_t$$ 越过阈值后直接关掉该 token 的贡献，而是让权重随着比率偏离理想区域而逐渐衰减。
- GSPO[^gspo]（[图 6](#figure-gspo-gradient-weight)）则是把同样的思路搬到序列级别：将 $$(A_t,\rho_t)$$ 换成序列级的 $$(A_s,\rho_s)$$，其中 $$\rho_s=(\pi_\theta(y\mid x)/\pi_{\mathrm{old}}(y\mid x))^{1/\lvert y\rvert}$$——即式 (1) 中各 token 比率的几何平均。注意共享因子也随之改变：$$\nabla_\theta\rho_s=\rho_s\cdot\frac{1}{\lvert y\rvert}\sum_t\nabla_\theta\log\pi_\theta(y_t\mid x,y_{<t})$$，也就是说，回答中的每个 token 都获得相同的、按长度平均后的权重。门控形状与 DAPO 一样是非对称的双楔形，但在 GSPO 的裁剪范围（$$\epsilon\sim 3\text{–}4\times10^{-4}$$）下，$$\rho_s$$ 与 1 的偏差始终在 $$10^{-3}$$ 以内，因此学习信号基本上就只剩 $$A_s$$ 本身。

<figure id="figure-ppo-gradient-weight">
  <img src="{{ '/assets/figures/mental-model-policy-optimizations/ppo-gradient-weight-heatmap.png' | relative_url }}" alt="PPO 在优势–对数比率平面上的门控与学习信号热力图">
  <figcaption><strong>图 2.</strong> PPO/GRPO。<em>左：</em>门控 \(f'\)。<em>右：</em>学习信号 \(f'\rho_t A_t\)。激活区域是两个象限楔形；在保留下来的角落里，学习信号随 \(\rho_t\) 增大而增大。</figcaption>
</figure>

<figure id="figure-dapo-gradient-weight">
  <img src="{{ '/assets/figures/mental-model-policy-optimizations/dapo-gradient-weight-heatmap.png' | relative_url }}" alt="DAPO 在优势–对数比率平面上的门控与学习信号热力图">
  <figcaption><strong>图 3.</strong> DAPO。与图 2 相同的二值门控，但上方的楔形更宽。</figcaption>
</figure>

<figure id="figure-sao-gradient-weight">
  <img src="{{ '/assets/figures/mental-model-policy-optimizations/sao-gradient-weight-heatmap.png' | relative_url }}" alt="SAO 在优势–对数比率平面上的门控与学习信号热力图">
  <figcaption><strong>图 4.</strong> SAO。矩形窗（top-hat）门控。</figcaption>
</figure>

<figure id="figure-sapo-gradient-weight">
  <img src="{{ '/assets/figures/mental-model-policy-optimizations/sapo-gradient-weight-heatmap.png' | relative_url }}" alt="SAPO 在优势–对数比率平面上的门控与学习信号热力图">
  <figcaption><strong>图 5.</strong> SAPO。平滑门控（虚线为 \(f'\) 的等值线）。<em>右：</em>在图示范围内，\(\rho_t\) 的增长与门控的衰减几乎相互抵消。</figcaption>
</figure>

<figure id="figure-gspo-gradient-weight">
  <img src="{{ '/assets/figures/mental-model-policy-optimizations/gspo-gradient-weight-heatmap.png' | relative_url }}" alt="GSPO 在序列优势–对数序列比率平面上的梯度权重热力图">
  <figcaption><strong>图 6.</strong> GSPO，画在 \((A_s,\log\rho_s)\) 平面上。楔形形状与图 3 相同，但坐标轴范围只有 \(\pm10^{-3}\)，因为 GSPO 的裁剪范围要小三个数量级。</figcaption>
</figure>

## 小结

借助 SAPO[^sapo] 中带权重函数 $$f(\rho_t; A_t)$$ 的广义目标函数 $$J$$，不同 RL 方法的学习信号都可以画在同一个 $$(A_t,\log\rho_t)$$ 平面上。

- PPO / GRPO：token 级别的硬门控。
- DAPO：为有用的低概率 token 放宽门控。
- SAO：当 rollout 策略变得陈旧（stale）时，只保留落在信任域内的比率。
- SAPO：把硬门控变成平滑门控。
- GSPO：把同样的思路从 token 推广到序列。

这样看来，这些方法其实是在用不同的方式回答同一个问题：哪些策略梯度信号值得信任，又该信任多少？


[^ppo]: Schulman et al., [*Proximal Policy Optimization Algorithms*](https://arxiv.org/abs/1707.06347), arXiv:1707.06347 (2017).

[^grpo]: Shao et al., [*DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models*](https://arxiv.org/abs/2402.03300), arXiv:2402.03300 (2024).

[^dapo]: Yu et al., [*DAPO: An Open-Source LLM Reinforcement Learning System at Scale*](https://arxiv.org/abs/2503.14476), arXiv:2503.14476 (2025).

[^gspo]: Zheng et al., [*Group Sequence Policy Optimization*](https://arxiv.org/abs/2507.18071), arXiv:2507.18071 (2025).

[^sao]: Hou et al., [*Single-Rollout Asynchronous Optimization for Agentic Reinforcement Learning*](https://arxiv.org/abs/2607.07508), arXiv:2607.07508 (2026).

[^sapo]: Gao et al., [*Soft Adaptive Policy Optimization*](https://arxiv.org/abs/2511.20347), arXiv:2511.20347 (2025). 另见 [Qwen 团队关于 SAPO 的博客文章](https://qwen.ai/blog?id=sapo)。
