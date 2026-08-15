---
layout: post
title: "A mental model to understand RL policy optimizations"
subtitle: "The diagram to unify PPO/GRPO, DAPO, GSPO, SAO, SAPO..."
date: 2026-08-14
tags: [rl, ppo, grpo, gspo, dapo, sao, sapo]
---

> 💡 **New to PPO or GRPO?** These introductions are good places to start before reading this blog: [PPO](https://huggingface.co/blog/deep-rl-ppo) and [GRPO](https://huggingface.co/blog/garg-aayush/derive-grpo-loss).

Inspired by the [Feynman technique](https://en.wikipedia.org/wiki/Learning_by_teaching), I'll try to explain some landmark policy optimization methods used in LLM RL as simply as I can from the very beginning of PPO to GRPO to GSPO and then some modern variations in the last 2 years DAPO SAO and SAPO. Writing them out helps me understand them better, and hopefully it helps you too.

Policy optimizations in LLM RL almost always involve importance ratio. Define the token-level ratio as

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

The importance-weighted policy-gradient term, $\rho_t A_t\nabla_\theta\log\pi_\theta(y_t\mid x,y_{<t})$, is shared by every method in this comparison. The method-specific gate, $f'(\rho_t;A_t)$, determines how much of the token's learning signal survives. Figures 2–5 plot $f'(\rho_t;A_t)$ on its own:

$$
f'(\rho_t;A_t).
$$

Three of the four methods switch on $\mathrm{sign}(A_t)$, so the gate is still a function of both coordinates — hence the semicolon — but it carries no factor of $A_t$ and no factor of $\rho_t$. Both of those are common to every method, so dropping them leaves only what actually distinguishes the four: the shape of the trust region and how sharply the gate falls off at its edge.

Table 1 summarizes the four gates. PPO and GRPO use the same clipped policy loss; they differ elsewhere, most notably in how they estimate $A_t$. The table likewise isolates DAPO's and SAO's ratio-handling rules rather than their complete training algorithms. GSPO is deliberately absent from it: it reuses PPO's gate unchanged and swaps the argument instead, so it gets [its own section](#gspo-the-same-gate-a-different-unit) below.

| Method | $f(\rho_t;A_t)$ | $f'(\rho_t;A_t)$ | Effect |
| --- | --- | --- | --- |
| [PPO](https://arxiv.org/abs/1707.06347) (July 2017) / [GRPO](https://arxiv.org/abs/2402.03300) (February 2024) | $A_t>0:\min(\rho_t,1+\epsilon)$; $A_t\leq0:\max(\rho_t,1-\epsilon)$ | $1$ before the sign-dependent clip boundary, $0$ after it | Hard, asymmetric gating based on the sign of $A_t$ |
| [DAPO](https://arxiv.org/abs/2503.14476) (March 2025) | $A_t>0:\min(\rho_t,1+\epsilon_{\mathrm{high}})$; $A_t\leq0:\max(\rho_t,1-\epsilon_{\mathrm{low}})$ | The same binary gate, with separate upper and lower boundaries | A higher upper bound leaves more room to increase useful low-probability tokens |
| [SAO](https://arxiv.org/abs/2607.07508) (July 2026) | $\widetilde f(\rho_t)=\operatorname{clip}(\rho_t,1-\epsilon_{\mathrm{low}},1+\epsilon_{\mathrm{high}})$ | $\mathbf{1}[1-\epsilon_{\mathrm{low}}<\rho_t<1+\epsilon_{\mathrm{high}}]$ | Both signs are masked whenever the ratio leaves the trust region |
| [SAPO](https://arxiv.org/abs/2511.20347) (November 2025) | $\dfrac{4}{\tau_t}\sigma\!\left(\tau_t(\rho_t-1)\right)$ | $4\sigma(z_t)(1-\sigma(z_t))=\mathrm{sech}^2(z_t/2)$ | The gate decays smoothly instead of switching abruptly to zero |

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
  <img src="{{ '/assets/figures/mental-model-policy-optimizations/ppo-gradient-weight-heatmap.png' | relative_url }}" alt="PPO gradient-weight heatmap">
  <figcaption><strong>Figure 2.</strong> PPO/GRPO. The gate is exactly one inside the two active wedges and zero beyond the sign-dependent clip boundaries, so each dead zone is a quadrant wedge.</figcaption>
</figure>

<figure id="figure-dapo-gradient-weight">
  <img src="{{ '/assets/figures/mental-model-policy-optimizations/dapo-gradient-weight-heatmap.png' | relative_url }}" alt="DAPO gradient-weight heatmap">
  <figcaption><strong>Figure 3.</strong> DAPO. The gate remains binary, but $\epsilon_{\mathrm{high}}>\epsilon_{\mathrm{low}}$ gives positive-advantage tokens more room to increase before clipping.</figcaption>
</figure>

<figure id="figure-sao-gradient-weight">
  <img src="{{ '/assets/figures/mental-model-policy-optimizations/sao-gradient-weight-heatmap.png' | relative_url }}" alt="SAO gradient-weight heatmap">
  <figcaption><strong>Figure 4.</strong> SAO. The ratio alone determines whether a token survives. Both positive- and negative-advantage tokens are masked outside the allowed interval, producing full-width dead zones.</figcaption>
</figure>

<figure id="figure-sapo-gradient-weight">
  <img src="{{ '/assets/figures/mental-model-policy-optimizations/sapo-gradient-weight-heatmap.png' | relative_url }}" alt="SAPO gradient-weight heatmap">
  <figcaption><strong>Figure 5.</strong> SAPO. There is no finite clipping boundary. The gate equals one at $\rho_t=1$ and decays smoothly as the token moves off-policy, so finite-ratio updates are attenuated rather than switched off.</figcaption>
</figure>

PPO and DAPO decide whether a token has moved too far in the direction encouraged by its advantage. SAO rejects any token that has moved too far in either direction. SAPO replaces that binary decision with a continuously varying weight.

## GSPO: the same gate, a different unit

[GSPO](https://arxiv.org/abs/2507.18071) (July 2025) fits this framework, but not by adding a fifth $f$. Its objective is PPO's `min`/`clip` copied verbatim, with the token ratio and token advantage swapped for their sequence-level counterparts:

$$
J_{\mathrm{GSPO}}(\theta)
=
\mathbb{E}\left[
\frac{1}{G}\sum_{i=1}^{G}
\min\left(
s_i(\theta)\widehat A_i,\;
\operatorname{clip}(s_i(\theta),1-\epsilon,1+\epsilon)\widehat A_i
\right)
\right],
$$

where the ratio is the *length-normalized* sequence likelihood ratio

$$
s_i(\theta)
=
\left(\frac{\pi_\theta(y_i\mid x)}{\pi_{\mathrm{old}}(y_i\mid x)}\right)^{1/|y_i|}
=
\exp\left(
\frac{1}{|y_i|}\sum_{t=1}^{|y_i|}
\log\frac{\pi_\theta(y_{i,t}\mid x,y_{i,<t})}{\pi_{\mathrm{old}}(y_{i,t}\mid x,y_{i,<t})}
\right).
$$

So the gate is exactly the one already in Table 1, read on new coordinates:

$$
f'(s_i;\widehat A_i)
=
\mathbf{1}\!\left[\widehat A_i>0,\; s_i<1+\epsilon_h\right]
+
\mathbf{1}\!\left[\widehat A_i<0,\; s_i>1-\epsilon_l\right].
$$

[Figure 6](#figure-gspo-gradient-weight) is therefore the same two-wedge picture as Figures 2 and 3. Two things do change, and neither is visible in the algebra.

**The scale changes by three orders of magnitude.** Because $s_i$ is a geometric mean over $|y_i|$ token ratios, it concentrates near 1 far more tightly than any individual $\rho_t$; a window of width $0.2$ around it would essentially never bind. GSPO's reported clipping ranges are $3\times10^{-4}$ (left) and $4\times10^{-4}$ (right), against GRPO's $0.2$-scale. Figure 6 uses those published values rather than the exaggerated ones used elsewhere in this post, because here the magnitude *is* the content. [Figure 7](#figure-gspo-scale-contrast) draws the two windows against each other: GSPO's entire ratio axis, plotted to scale inside PPO's plane, is thinner than the line pointing at it.

**The gating unit changes from token to sequence.** The gradient (Eq. 10 of the paper) is

$$
\nabla_\theta J_{\mathrm{GSPO}}
=
\mathbb{E}\left[
\frac{1}{G}\sum_{i=1}^{G}
s_i(\theta)\,\widehat A_i \cdot
\frac{1}{|y_i|}\sum_{t=1}^{|y_i|}
\nabla_\theta\log\pi_\theta(y_{i,t}\mid x,y_{i,<t})
\right],
$$

so the "shared" factor is no longer $\rho_t A_t\nabla_\theta\log\pi_\theta$ at token $t$; it is $s_i\widehat A_i$ times the *sequence-averaged* score function. One gate decision now covers every token in the response — all $|y_i|$ of them survive together or die together. That is the whole argument of the paper: the clipping unit should match the rewarding unit, and a token-level ratio is a noisy estimate of a quantity the reward never referred to. It is also the one thing the $(A_t,\log\rho_t)$ plane cannot show, because after this substitution it is no longer a per-token plane.

<figure id="figure-gspo-gradient-weight">
  <img src="{{ '/assets/figures/mental-model-policy-optimizations/gspo-gradient-weight-heatmap.png' | relative_url }}" alt="GSPO gradient-weight heatmap on the sequence-advantage / log-sequence-ratio plane">
  <figcaption><strong>Figure 6.</strong> GSPO, on the $(\widehat A_i,\log s_i)$ plane. The gate has the same two-wedge shape as PPO and DAPO — separate left and right ranges make it DAPO's asymmetric version — but the axis spans $\pm10^{-3}$, not $\pm0.6$.</figcaption>
</figure>

<figure id="figure-gspo-scale-contrast">
  <img src="{{ '/assets/figures/mental-model-policy-optimizations/ppo-vs-gspo-scale-contrast.png' | relative_url }}" alt="PPO and GSPO trust regions drawn on their respective ratio scales">
  <figcaption><strong>Figure 7.</strong> The same gate on two scales. Left: PPO/GRPO at $\epsilon=0.2$ on the token ratio. Right: GSPO at its published ranges on the sequence ratio. The red line in the left panel is the right panel's full extent drawn to scale — about $600\times$ shorter than the axis containing it.</figcaption>
</figure>

A token-level variant, GSPO-token, restores per-token advantages while keeping $s_i$ as the clipping quantity; when all $\widehat A_{i,t}$ in a response are equal it is numerically identical to GSPO.

other references: https://qwen.ai/blog?id=sapo
