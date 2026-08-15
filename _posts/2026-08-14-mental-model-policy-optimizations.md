---
layout: post
title: "A mental model to understand RL policy optimizations"
subtitle: "The diagram to unify PPO/GRPO, DAPO, GSPO, SAO, SAPO..."
date: 2026-08-14
tags: [rl, ppo, grpo, gspo, dapo, sao, sapo]
---

Inspired by the [Feynman technique](https://en.wikipedia.org/wiki/Learning_by_teaching), I'll try to explain some landmark policy optimization methods used in LLM RL as simply as I can from the very beginning of PPO to GRPO to GSPO and then some modern variations in the last 2 years DAPO SAO and SAPO. Writing them out helps me understand them better, and hopefully it helps you too.

> 💡 **New to PPO or GRPO?** These introductions are good places to start before reading this blog: [PPO](https://huggingface.co/blog/deep-rl-ppo) and [GRPO](https://huggingface.co/blog/garg-aayush/derive-grpo-loss).

Policy objectives in LLM RL usually involve importance ratio. Define the token-level ratio as

$$
\rho_t = \frac{\pi_{\theta}(y_t \mid x, y_{<t})}{\pi_{\text{old}}(y_t \mid x, y_{<t})},
$$

which compares the probability assigned to token $y_t$ by the current policy, $\pi_\theta$, with that assigned by the old policy, $\pi_{\text{old}}$. The full [PPO objective](https://arxiv.org/abs/1707.06347) is

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
$$

To make it simpler, we can just consider the token level objective since the whole objective is just the expectation over . If it's GRPO there is gonna be another average over the group rollout dimensions. 

$$
J_t(\theta) = \min\left(
\rho_t A_t,\;
\operatorname{clip}(\rho_t, 1-\epsilon, 1+\epsilon)A_t
\right)
$$


The `min` and `clip` operations are where most writeups bury readers in notation. Split the objective by the sign of $A_t$, then by where $\rho_t$ sits relative to the clip bounds, and the logic becomes straightforward.

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

**Clipped region ($\rho_t < 1-\epsilon$).**

$$
J_t(\theta) = (1 - \epsilon) A_t,
\qquad
\frac{\partial J_t}{\partial \rho_t}= 0
$$

[Figure 1](#figure-ppo-clip-schematic) summarizes this behavior. On the $\log\rho_t$ vs. $A_t$ plane, there are four regions where a sampled token could fall into: In the red region, PPO/GRPO encourages the token by increasing its probability; in the blue region, it discourages the token; in the remaining regions, the token is effectively dropped. 

<figure id="figure-ppo-clip-schematic">
  <img src="{{ '/assets/figures/mental-model-policy-optimizations/ppo-clip-schematic.png' | relative_url }}" alt="PPO clip schematic on the advantage–log-ratio plane">
  <figcaption><strong>Figure 1.</strong> Schematic diagram for PPO/GRPO on the $\log{\rho_t}-A_t$ plane. Red is where we increase $\rho_t$, blue is where we decrease it, the rest are clipped out. Log ratio is used to make y-axis expand to infinities. The default $\epsilon = 0.2$ is used. </figcaption>
</figure>


## One equation, four gradient gates

The [SAPO paper](https://arxiv.org/abs/2511.20347) gives us a useful way to compare these policy optimization methods. At the token level, write the surrogate objective as

$$
J_t(\theta)=f(\rho_t(\theta);A_t)\,A_t.
$$

PPO/GRPO, DAPO, and SAPO differ mainly in their choice of $f$. Taking the gradient,

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
$$

All methods share the same importance-weighted policy gradient: $\rho_t A_t\nabla_\theta\log\pi_\theta(y_t\mid x,y_{<t})$. Each method differs in the gate $f'(\rho_t;A_t)$, which controls how much learning signal passes through. Figures 2–5 show (left) the gate and (right) the effective per-token weight $f'(\rho_t;A_t)\,\rho_t A_t$, all on a common color scale for easy comparison.



Table 1 summarizes the gates. PPO and GRPO use the same clipping; DAPO and SAO have different ratio gates. Only the ratio-handling logic is shown for each.

| Method | $f(\rho_t;A_t)$ | $f'(\rho_t;A_t)$ | Effect |
| --- | --- | --- | --- |
| [PPO](https://arxiv.org/abs/1707.06347) (July 2017) / [GRPO](https://arxiv.org/abs/2402.03300) (February 2024) | $A_t>0:\min(\rho_t,1+\epsilon)$; $A_t\leq0:\max(\rho_t,1-\epsilon)$ | $1$ before the sign-dependent clip boundary, $0$ after it | Hard, asymmetric gating based on the sign of $A_t$ |
| [DAPO](https://arxiv.org/abs/2503.14476) (March 2025) | $A_t>0:\min(\rho_t,1+\epsilon_{\mathrm{high}})$; $A_t\leq0:\max(\rho_t,1-\epsilon_{\mathrm{low}})$ | The same binary gate, with separate upper and lower boundaries | A higher upper bound leaves more room to increase useful low-probability tokens |
| [SAO](https://arxiv.org/abs/2607.07508) (July 2026) | $\widetilde f(\rho_t)=\operatorname{clip}(\rho_t,1-\epsilon_{\mathrm{low}},1+\epsilon_{\mathrm{high}})$ | $\mathbf{1}[1-\epsilon_{\mathrm{low}}<\rho_t<1+\epsilon_{\mathrm{high}}]$ | Both signs are masked whenever the ratio leaves the trust region |
| [SAPO](https://arxiv.org/abs/2511.20347) (November 2025) | $\dfrac{4}{\tau_t}\sigma\\left(\tau_t(\rho_t-1)\right)$ | $4\sigma(z_t)(1-\sigma(z_t))=\mathrm{sech}^2(z_t/2)$ | The gate decays smoothly instead of switching abruptly to zero |

<!-- For SAPO,

$$
z_t=\tau_t(\rho_t-1),
\qquad
\tau_t=
\begin{cases}
\tau_{\mathrm{pos}}, & A_t>0,\\
\tau_{\mathrm{neg}}, & A_t\leq0.
\end{cases}
$$

The paper uses $\tau_{\mathrm{neg}}>\tau_{\mathrm{pos}}$, so updates from negative-advantage tokens decay faster as they move off-policy. With the reported defaults, $\tau_{\mathrm{pos}}=1.0$ and $\tau_{\mathrm{neg}}=1.05$, the asymmetry is deliberately mild.

The SAO row needs one qualification. SAO states its loss as a ratio-weighted log-policy objective and masks the weight outside the interval. The clipped $\widetilde f$ above is a gradient-equivalent surrogate: its derivative gives the same strict two-sided mask, but it is not the literal objective printed in the SAO paper. SAO also computes its ratio against the rollout policy rather than a separately maintained old policy.

The plots use illustrative parameters to make these differences visible: $\epsilon_{\mathrm{low}}=0.2$ and $\epsilon_{\mathrm{high}}=0.5$ for DAPO and SAO, and $\tau_{\mathrm{pos}}=2$, $\tau_{\mathrm{neg}}=4$ for SAPO. These are not the papers' default training settings. -->

<figure id="figure-ppo-gradient-weight">
  <img src="{{ '/assets/figures/mental-model-policy-optimizations/ppo-gradient-weight-heatmap.png' | relative_url }}" alt="PPO gate and effective-weight heatmaps on the advantage–log-ratio plane">
  <figcaption><strong>Figure 2.</strong> PPO/GRPO. <em>Left:</em> the gate is exactly one inside the two active wedges and zero beyond the sign-dependent clip boundaries, so each dead zone is a quadrant wedge. <em>Right:</em> the same wedges, now carrying $\rho_t A_t$ — the sign flips with $A_t$ and the magnitude grows with $\rho_t$, which is why the surviving corners are the strongest updates in the whole plane.</figcaption>
</figure>

<figure id="figure-dapo-gradient-weight">
  <img src="{{ '/assets/figures/mental-model-policy-optimizations/dapo-gradient-weight-heatmap.png' | relative_url }}" alt="DAPO gate and effective-weight heatmaps on the advantage–log-ratio plane">
  <figcaption><strong>Figure 3.</strong> DAPO. The gate remains binary, but $\epsilon_{\mathrm{high}}>\epsilon_{\mathrm{low}}$ gives positive-advantage tokens more room to increase before clipping. The right panel shows what that room is worth: the extra strip admitted above PPO’s boundary is exactly where $\rho_t$ is largest, so it carries more weight than its area suggests.</figcaption>
</figure>

<figure id="figure-sao-gradient-weight">
  <img src="{{ '/assets/figures/mental-model-policy-optimizations/sao-gradient-weight-heatmap.png' | relative_url }}" alt="SAO gate and effective-weight heatmaps on the advantage–log-ratio plane">
  <figcaption><strong>Figure 4.</strong> SAO. The ratio alone determines whether a token survives, so both signs are masked outside the allowed interval and the dead zones run the full width. The right panel shows the consequence: capping $\rho_t$ from above caps the effective weight too, which neither PPO nor DAPO does for $A_t<0$.</figcaption>
</figure>

<figure id="figure-sapo-gradient-weight">
  <img src="{{ '/assets/figures/mental-model-policy-optimizations/sapo-gradient-weight-heatmap.png' | relative_url }}" alt="SAPO gate and effective-weight heatmaps on the advantage–log-ratio plane">
  <figcaption><strong>Figure 5.</strong> SAPO. There is no finite clipping boundary: the gate equals one at $\rho_t=1$ and decays smoothly as the token moves off-policy, so nothing is switched off outright. Dashed lines are $f'$ iso-levels, repeated on the right panel for reference. On this window the decay barely dents the effective weight — see Figure 6 for why.</figcaption>
</figure>

PPO and DAPO decide whether a token has moved too far in the direction encouraged by its advantage. SAO rejects any token that has moved too far in either direction. SAPO replaces that binary decision with a continuously varying weight.

<!-- 
### Widening the window

The right panels of Figures 2–5 already carry the shared factor, but they are cropped to $|\log\rho_t|\leq0.6$, and that crop hides SAPO's whole story. The $\rho_t$ out front grows with the ratio while SAPO's gate shrinks, and over this window the two nearly cancel: $f'\rho_t$ falls only from $1.00$ at $\rho_t=1$ to $0.99$ at the top edge. That is why SAPO's right panel looks so much like DAPO's despite the gates being completely different rules. Widen the window and the cancellation breaks.

<figure id="figure-effective-coefficient">
  <img src="{{ '/assets/figures/mental-model-policy-optimizations/effective-coefficient-curves.png' | relative_url }}" alt="Effective per-token weight against log importance ratio, for positive and negative advantage">
  <figcaption><strong>Figure 6.</strong> The effective weight $f'(\rho_t;A_t)\,\rho_t$ at $|A_t|=1$, over a ratio window wide enough to show the tails. Shaded band marks the window of Figures 2–5.</figcaption>
</figure>

Two things follow that the cropped panels do not show.

First, SAPO's weight does reach zero. The gate never does — $\mathrm{sech}^2$ is positive everywhere — but $f'\rho_t$ rises to a peak just above $\rho_t=1$ and then collapses, because the exponential decay of the gate beats the linear growth of $\rho_t$. The peak sits at $\rho_t\approx1.38$ for the $\tau=2$ used in the positive-advantage panel and at $\rho_t\approx1.11$ for $\tau=4$; with the paper's much gentler defaults it moves out to $\rho_t\approx2.1$, but the collapse still happens. SAPO does not merely attenuate far-off-policy tokens; it discards them, just without a hard boundary.

Second, and less comfortably: for $A_t<0$, PPO and DAPO impose no upper bound at all. Their curves coincide — DAPO only moves the *upper* $\epsilon$, which for negative advantage is the side that was never clipped — and both grow linearly in $\rho_t$ without limit. A negative-advantage token that the current policy has come to strongly prefer receives an unboundedly large gradient. This is precisely the failure mode SAO's two-sided clip and SAPO's decay are built to remove, and it is easy to miss in Figures 2–5, where the gate for that region is a flat $1$ and the right panels are cropped before the growth becomes alarming.


other references: https://qwen.ai/blog?id=sapo -->
