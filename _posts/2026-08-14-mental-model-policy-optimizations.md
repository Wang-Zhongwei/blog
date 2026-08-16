---
layout: post
title: "A Mental Model to Unify RL Policy Losses"
subtitle: "The diagram that compares PPO/GRPO, DAPO, GSPO, SAO, SAPO..."
date: 2026-08-14
tags: [rl, ppo, grpo, gspo, dapo, sao, sapo]
image: /assets/figures/mental-model-policy-optimizations/ppo-clip-schematic.png
---

Inspired by the [Feynman technique](https://en.wikipedia.org/wiki/Learning_by_teaching), I'll explain some famous LLM RL losses as simply as I can—starting with PPO[^ppo] and GRPO[^grpo], then recent variants DAPO[^dapo], GSPO[^gspo], SAO[^sao], and SAPO[^sapo]. Writing them out helps me understand them better, and hopefully it helps you too.

> 💡 **New to PPO or GRPO?** These introductions are good places to start before reading this blog: [PPO](https://huggingface.co/blog/deep-rl-ppo) and [GRPO](https://huggingface.co/blog/garg-aayush/derive-grpo-loss).

Policy objectives in LLM RL usually involve importance ratio. Define the token-level ratio as

$$
\rho_t = \frac{\pi_{\theta}(y_t \mid x, y_{<t})}{\pi_{\text{old}}(y_t \mid x, y_{<t})},
\tag{1}\label{eq:importance-ratio}
$$

where $\pi_\theta$ is the current policy and $\pi_{\text{old}}$ is the old policy. The old policy could be the snapshot of the model weights at the beginning of the gradient update, as in PPO, or the rollout policy in asynchronous training such as SAO[^sao], lagged by more than one gradient update. The full PPO objective[^ppo] is

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
\tag{2}\label{eq:ppo-full}
$$

To make it simpler, we can consider the token-level objective alone, since the full objective is an expectation of it over trajectories.

$$
J_t(\theta) = \min\left(
\rho_t A_t,\;
\operatorname{clip}(\rho_t, 1-\epsilon, 1+\epsilon)A_t
\right).
\tag{3}\label{eq:ppo-token}
$$


The `min` and `clip` operations in eq. (3) are where most writeups bury readers in notation. Split the objective by the sign of $A_t$, then by where $\rho_t$ sits relative to the clip bounds, and the logic becomes straightforward.

### Case $A_t > 0$

This token is better than we thought, so we should increase its probability. Indeed,

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

**Unclipped region ($\rho_t < 1 + \epsilon$)**

$$
J_t(\theta) = \rho_t A_t,
\qquad
\frac{\partial J_t}{\partial \rho_t}=A_t > 0
$$

Increasing the better token probability $\rho_t$ increases the objective $J_t$, *exactly what we should* do for a positive advantage $A_t$. 

**Clipped region ($\rho_t > 1+\epsilon$).**

$$
J_t(\theta) = (1 + \epsilon) A_t,
\qquad
\frac{\partial J_t}{\partial \rho_t}= 0
$$

In other words, it tells us *don't be too greedy* if the current policy $\pi_{\theta}$ already *strongly prefers* the better token — we should stop assigning more importance to it. 

### Case $A_t < 0$

This token is worse than we thought, so we should decrease its probability. Indeed,

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

**Unclipped region ($\rho_t \ge 1-\epsilon$)**

$$
J_t(\theta) = \rho_t A_t,
\qquad
\frac{\partial J_t}{\partial \rho_t}=A_t < 0
$$

Decreasing $\rho_t$ increases $J_t$, as it should for a negative advantage.

**Clipped region ($\rho_t < 1-\epsilon$).**

$$
J_t(\theta) = (1 - \epsilon) A_t,
\qquad
\frac{\partial J_t}{\partial \rho_t}= 0
$$

Once the policy has already moved far enough against this token, further decreases are clipped out.

[Figure 1](#figure-ppo-clip-schematic) summarizes this behavior. On the $\log\rho_t$ vs. $A_t$ plane, a sampled token falls into one of four regions: in the red region, PPO/GRPO encourages the token by increasing its probability; in the blue region, it discourages the token; in the remaining regions, the gradient on $\rho_t$ is zero, so the token is effectively dropped from the parameter update.

<figure id="figure-ppo-clip-schematic">
  <img src="{{ '/assets/figures/mental-model-policy-optimizations/ppo-clip-schematic.png' | relative_url }}" alt="PPO clip schematic on the advantage–log-ratio plane">
  <figcaption><strong>Figure 1.</strong> Schematic diagram for PPO/GRPO on the $\log{\rho_t}-A_t$ plane. Red is where we increase $\rho_t$, blue is where we decrease it, the rest are clipped out. The y-axis is $\log\rho_t$ so that the ratio's asymmetric range $(0,\infty)$ becomes symmetric about $0$. The default $\epsilon = 0.2$ is used. </figcaption>
</figure>


## Generalized RL objective

The SAPO paper[^sapo] gives us a useful way to compare these policy optimization methods. At the token level, write the surrogate objective as

$$
J_t(\theta)=f(\rho_t(\theta);A_t)\,A_t.
\tag{4}\label{eq:unified-surrogate}
$$

Policy optimization methods, such as PPO/GRPO, DAPO, GSPO, SAO, and SAPO, differ mainly in their choice of the **weight function** $f$. Taking the gradient of eq. (4),

$$
\begin{aligned}
\nabla_\theta J_t
&=
A_t f'(\rho_t;A_t)\,\nabla_\theta\rho_t \\[4pt]
&=
\underbrace{f'(\rho_t;A_t)}_{\text{method-specific}}
\underbrace{
\rho_t\,
\overbrace{A_t\nabla_\theta
\log\pi_\theta(y_t\mid x,y_{<t})}^{\text{policy gradient}}
}_{\text{shared}}.
\end{aligned}
\tag{5}\label{eq:gradient-decomposition}
$$

All methods share the term $\rho_t A_t\nabla_\theta\log\pi_\theta$. They only differ in the **gate function** $f'(\rho_t;A_t)$ that determines how much learning signal gets through.

[Table 1](#table-1) summarizes the weight function $f$ and gate function $f'$ of some milestone policy optimization methods in LLM RL.

<figure id="table-1" markdown="block">

| Method | $f(\rho_t;A_t)$ | $f'(\rho_t;A_t)$ | Insight |
| --- | --- | --- | --- |
| [PPO](https://arxiv.org/abs/1707.06347) (July 2017) / [GRPO](https://arxiv.org/abs/2402.03300) (February 2024) | $A_t>0:\;\min(\rho_t,1+\epsilon)$<br>$A_t\leq0:\;\max(\rho_t,1-\epsilon)$ | $\mathbf{1}\left[A_t>0,\;\rho_t<1+\epsilon\right]$<br>$+\;\mathbf{1}\left[A_t<0,\;\rho_t>1-\epsilon\right]$ | Hard, asymmetric gating based on the sign of $A_t$ |
| [DAPO](https://arxiv.org/abs/2503.14476) (March 2025) | $A_t>0:\;\min(\rho_t,1+\epsilon_h)$<br>$A_t\leq0:\;\max(\rho_t,1-\epsilon_l)$ | $\mathbf{1}\left[A_t>0,\;\rho_t<1+\epsilon_h\right]$<br>$+\;\mathbf{1}\left[A_t<0,\;\rho_t>1-\epsilon_l\right]$ | A higher upper bound leaves more room to increase useful low-probability tokens |
| [GSPO](https://arxiv.org/abs/2507.18071) (July 2025) | Same as DAPO at $(\rho_s,A_s)$:<br>$A_s>0:\;\min(\rho_s,1+\epsilon_h)$<br>$A_s\leq0:\;\max(\rho_s,1-\epsilon_l)$ | $\mathbf{1}\left[A_s>0,\;\rho_s<1+\epsilon_h\right]$<br>$+\;\mathbf{1}\left[A_s<0,\;\rho_s>1-\epsilon_l\right]$ | One clip decision per response instead of per token |
| [SAPO](https://arxiv.org/abs/2511.20347) (November 2025) | $\dfrac{4}{\tau_t}\sigma\left(\tau_t(\rho_t-1)\right)$,<br>$\tau_t=\tau_{\text{pos}}$ if $A_t>0$, else $\tau_{\text{neg}}$ | $4\sigma\left(\tau_t(\rho_t-1)\right)\left(1-\sigma\left(\tau_t(\rho_t-1)\right)\right)$<br>$=\mathrm{sech}^2\left(\tfrac{\tau_t(\rho_t-1)}{2}\right)$ | The gate decays smoothly instead of switching abruptly to zero; $\tau_{\text{neg}}>\tau_{\text{pos}}$ makes it decay faster for negative advantages |
| [SAO](https://arxiv.org/abs/2607.07508) (July 2026) | $\widetilde f(\rho_t)=\operatorname{clip}(\rho_t,1-\epsilon_l,1+\epsilon_h)$ | $\mathbf{1}\left[1-\epsilon_l<\rho_t<1+\epsilon_h\right]$ | Both signs are masked whenever the ratio leaves the trust region |

<figcaption><strong>Table 1.</strong> Weight function $f$ and gate function $f'$ for milestone policy optimization methods in LLM RL.</figcaption>
</figure>

Figures 2–6 plot the gate $f'(\rho_t;A_t)$ and the learning signal $f'(\rho_t;A_t)\,\rho_t A_t$ on the $(A_t,\log\rho_t)$ plane for each method (Figure 6 uses the sequence pair $(A_s,\rho_s)$):

- PPO[^ppo] ([Figure 2](#figure-ppo-gradient-weight)) clips the importance ratio $\rho_t$: once an update has encouraged a good token or discouraged a bad token enough, the gate drops to zero and the token stops contributing. This keeps policy updates small and training stable.
- GRPO[^grpo] ([Figure 2](#figure-ppo-gradient-weight)) adds a group rollout dimension $G$ to the objective. At the token level, its gate is identical to PPO's.
- DAPO[^dapo] ([Figure 3](#figure-dapo-gradient-weight)) raises the upper clipping threshold $\epsilon_h$, so rare tokens—whose small denominator tends to produce large $\rho_t$ values—are less likely to be clipped prematurely. The extra strip above PPO's boundary is exactly where the largest learning signals live.
- SAO[^sao] ([Figure 4](#figure-sao-gradient-weight)) targets asynchronous training, where the current and rollout policies can drift farther apart than in PPO. $\rho_t$ can then deviate substantially from 1, so the gate behaves like a top-hat window, masking both signs whenever the ratio leaves the trust region.
- SAPO[^sapo] ([Figure 5](#figure-sapo-gradient-weight)) replaces this hard boundary with a smooth gate. Instead of switching off a token's contribution once $\rho_t$ crosses a threshold, SAPO lets the weight decay gradually as the ratio moves away from the preferred region.
- GSPO[^gspo] ([Figure 6](#figure-gspo-gradient-weight)) is the same move at sequence level: swap $(A_t,\rho_t)$ for sequence level $(A_s,\rho_s)$ with $\rho_s=(\pi_\theta(y\mid x)/\pi_{\mathrm{old}}(y\mid x))^{1/\lvert y\rvert}$—the geometric mean of the token ratios in eq. (1). Note that the shared factor changes as well: $\nabla_\theta\rho_s=\rho_s\cdot\frac{1}{\lvert y\rvert}\sum_t\nabla_\theta\log\pi_\theta(y_t\mid x,y_{<t})$, i.e. every token in the response receives the same length-averaged weight. The gate is DAPO's asymmetric two-wedge shape, but with GSPO's clip range ($\epsilon\sim 3\text{–}4\times10^{-4}$) $\rho_s$ stays within $10^{-3}$ of 1, so the learning signal is essentially $A_s$ alone.

<figure id="figure-ppo-gradient-weight">
  <img src="{{ '/assets/figures/mental-model-policy-optimizations/ppo-gradient-weight-heatmap.png' | relative_url }}" alt="PPO gate and learning-signal heatmaps on the advantage–log-ratio plane">
  <figcaption><strong>Figure 2.</strong> PPO/GRPO. <em>Left:</em> gate $f'$. <em>Right:</em> learning signal $f'\rho_t A_t$. Active regions are two quadrant wedges; learning signal grows with $\rho_t$ in the surviving corners.</figcaption>
</figure>

<figure id="figure-dapo-gradient-weight">
  <img src="{{ '/assets/figures/mental-model-policy-optimizations/dapo-gradient-weight-heatmap.png' | relative_url }}" alt="DAPO gate and learning-signal heatmaps on the advantage–log-ratio plane">
  <figcaption><strong>Figure 3.</strong> DAPO. Same binary gate as Figure 2, with a wider upper wedge.</figcaption>
</figure>

<figure id="figure-sao-gradient-weight">
  <img src="{{ '/assets/figures/mental-model-policy-optimizations/sao-gradient-weight-heatmap.png' | relative_url }}" alt="SAO gate and learning-signal heatmaps on the advantage–log-ratio plane">
  <figcaption><strong>Figure 4.</strong> SAO. Top-hat gate. </figcaption>
</figure>

<figure id="figure-sapo-gradient-weight">
  <img src="{{ '/assets/figures/mental-model-policy-optimizations/sapo-gradient-weight-heatmap.png' | relative_url }}" alt="SAPO gate and learning-signal heatmaps on the advantage–log-ratio plane">
  <figcaption><strong>Figure 5.</strong> SAPO. Smooth gate (dashed lines: $f'$ iso-levels). <em>Right:</em> over this crop, $\rho_t$ growth and gate decay nearly cancel.</figcaption>
</figure>

<figure id="figure-gspo-gradient-weight">
  <img src="{{ '/assets/figures/mental-model-policy-optimizations/gspo-gradient-weight-heatmap.png' | relative_url }}" alt="GSPO gradient-weight heatmap on the sequence-advantage / log-sequence-ratio plane">
  <figcaption><strong>Figure 6.</strong> GSPO on $(A_s,\log\rho_s)$. Same wedge shape as Figure 3, but the axis spans $\pm10^{-3}$ because GSPO's clip range is three orders of magnitude tighter.</figcaption>
</figure>

## Takeaways

Using SAPO's[^sapo] generalized form of objective $J$ with weight function $f(\rho_t; A_t)$, learning signals of different RL methods can be shown on the same $(A_t,\log\rho_t)$ plane.

- PPO / GRPO: hard token-level gate.
- DAPO: widens the gate for useful low-probability tokens.
- SAO: keeps only ratios inside a trust region when rollout policies become stale.
- SAPO: turns the hard gate into a smooth one.
- GSPO: moves the same idea from tokens to sequences.

Seen this way, these methods are providing different answers to the same question: which policy-gradient signals should we trust, and how much?


[^ppo]: Schulman et al., [*Proximal Policy Optimization Algorithms*](https://arxiv.org/abs/1707.06347), arXiv:1707.06347 (2017).

[^grpo]: Shao et al., [*DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models*](https://arxiv.org/abs/2402.03300), arXiv:2402.03300 (2024).

[^dapo]: Yu et al., [*DAPO: An Open-Source LLM Reinforcement Learning System at Scale*](https://arxiv.org/abs/2503.14476), arXiv:2503.14476 (2025).

[^gspo]: Zheng et al., [*Group Sequence Policy Optimization*](https://arxiv.org/abs/2507.18071), arXiv:2507.18071 (2025).

[^sao]: Hou et al., [*Single-Rollout Asynchronous Optimization for Agentic Reinforcement Learning*](https://arxiv.org/abs/2607.07508), arXiv:2607.07508 (2026).

[^sapo]: Gao et al., [*Soft Adaptive Policy Optimization*](https://arxiv.org/abs/2511.20347), arXiv:2511.20347 (2025). See also the [Qwen Team blog post on SAPO](https://qwen.ai/blog?id=sapo).
