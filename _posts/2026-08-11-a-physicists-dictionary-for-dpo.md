---
layout: post
title: "A Physicist's Dictionary for DPO"
date: 2026-08-11
tags: [machine-learning, statistical-mechanics, rlhf, dpo]
---

<!-- ═══════════════════════════════════════════════════════════════════
     TEMPLATE — sections follow the agreed outline.
     ✍️  = write your prose here.  Delete these comment blocks as you go.
     Preview locally is NOT needed: GitHub Pages builds Jekyll for you.
     ═══════════════════════════════════════════════════════════════════ -->

<!-- ── 1. HOOK ──────────────────────────────────────────────────────
     One paragraph: "Every RLHF post mentions DPO's closed-form solution
     is a Gibbs distribution. Almost none tell you what the energy,
     entropy, or temperature actually are."
     Then: one short paragraph of DPO setup for the physicists,
     one short paragraph of stat-mech setup for the ML readers. -->

✍️ *Hook goes here.*

## The setup

✍️ *One paragraph each: KL-regularized preference optimization; the Gibbs distribution.*

# From KL-Regularized Preference Optimization to the Gibbs Distribution

Consider the KL-regularized reinforcement-learning objective for a fixed prompt $$x$$:

$$\max_{\pi} \; \mathbb{E}_{y\sim\pi(\cdot\mid x)} \left[ r(x,y) \right] - \beta D_{\mathrm{KL}} \left( \pi(\cdot\mid x) \;\|\; \pi_{\mathrm{ref}}(\cdot\mid x) \right).$$

For readability, we suppress the conditioning on $$x$$ and write $$\pi(y)\equiv\pi(y\mid x)$$, $$\pi_{\mathrm{ref}}(y)\equiv\pi_{\mathrm{ref}}(y\mid x)$$, and $$r(y)\equiv r(x,y)$$. The objective becomes

$$J[\pi] = \mathbb{E}_{y\sim\pi}[r(y)] - \beta D_{\mathrm{KL}} \left( \pi \;\|\; \pi_{\mathrm{ref}} \right).$$

Maximizing $$J[\pi]$$ is equivalent to minimizing

$$\mathcal{F}_{\beta}[\pi] = - \mathbb{E}_{y\sim\pi}[r(y)] + \beta D_{\mathrm{KL}} \left( \pi \;\|\; \pi_{\mathrm{ref}} \right).$$

### Variational derivation

Expanding the expectation and KL divergence, then dividing by $$\beta$$, gives

$$\frac{\mathcal{F}_{\beta}[\pi]}{\beta} = \sum_y \pi(y) \log \frac{ \pi(y) }{ \pi_{\mathrm{ref}}(y) e^{r(y)/\beta} }.$$

The Boltzmann weight $$\pi_{\mathrm{ref}}(y)e^{r(y)/\beta}$$ appears directly from the objective—not as an ansatz. Normalize it by the partition function

$$\boxed{ Z_{\beta} = \sum_y \pi_{\mathrm{ref}}(y) e^{r(y)/\beta} }$$

to obtain the Gibbs policy

$$\boxed{ \pi_{\beta}^{\ast}(y) = \frac{ \pi_{\mathrm{ref}}(y) e^{r(y)/\beta} }{ Z_{\beta} }. }$$

Substituting $$\pi_{\mathrm{ref}}(y)e^{r(y)/\beta}=Z_{\beta}\pi_{\beta}^{\ast}(y)$$ back yields

$$\boxed{ \mathcal{F}_{\beta}[\pi] = -\beta\log Z_{\beta} + \beta D_{\mathrm{KL}} \left( \pi \;\|\; \pi_{\beta}^{\ast} \right). }$$

Since $$D_{\mathrm{KL}}(\pi\|\pi_{\beta}^{\ast})\ge 0$$, with equality iff $$\pi=\pi_{\beta}^{\ast}$$, the unique minimizer is $$\pi_{\beta}^{\ast}$$. The equilibrium free energy is $$\mathcal{F}_{\beta}^{\ast}=-\beta\log Z_{\beta}$$.

## Reading the objective as a free energy

The same functional is a Helmholtz free energy in disguise. Define

$$E(y)=-r(y), \quad U[\pi]=\mathbb{E}_{\pi}[E]=-\mathbb{E}_{\pi}[r], \quad S_{\mathrm{rel}}[\pi]=-D_{\mathrm{KL}}(\pi\|\pi_{\mathrm{ref}}).$$

Then

$$\mathcal{F}_{\beta}[\pi] = U[\pi] - \beta S_{\mathrm{rel}}[\pi],$$

matching $$F=U-TS$$ with

$$\boxed{ E(y)\longleftrightarrow -r(y), \qquad k_B T\longleftrightarrow \beta, \qquad S\longleftrightarrow -D_{\mathrm{KL}}(\pi\|\pi_{\mathrm{ref}}). }$$

The reference policy is the base measure: the optimum tilts it exponentially, $$\pi_{\beta}^{\ast}(y)\propto\pi_{\mathrm{ref}}(y)e^{r(y)/\beta}$$, rather than applying a softmax over rewards alone.

## The same derivation in statistical mechanics

For discrete microstates $$y$$ with energy $$E(y)$$ and base measure $$\pi_{\mathrm{ref}}(y)$$, the identical variational problem uses

$$\mathcal{F}_{\beta}[\pi] = \sum_y \pi(y) E(y) + \beta D_{\mathrm{KL}} \left( \pi \;\|\; \pi_{\mathrm{ref}} \right),$$

with $$\beta\equiv k_B T$$ *not* $$1/k_B T$$—physics texts often use $$\beta_{\mathrm{phys}}=1/(k_B T)$$, which reverses the temperature interpretation when carried over naively.

The algebra is the same as above: expand, divide by $$\beta$$, combine the log, identify weights $$\pi_{\mathrm{ref}}(y)e^{-E(y)/\beta}$$, normalize, and conclude

$$\boxed{ \pi_{\beta}^{\ast}(y) = \frac{ \pi_{\mathrm{ref}}(y) e^{-E(y)/\beta} }{ Z_{\beta} }, \qquad Z_{\beta} = \sum_y \pi_{\mathrm{ref}}(y) e^{-E(y)/\beta}, \qquad \mathcal{F}_{\beta}^{\ast} = -\beta\log Z_{\beta}. }$$

For a uniform $$\pi_{\mathrm{ref}}$$, this reduces to the canonical Gibbs distribution $$e^{-E/\beta}/Z_{\beta}$$.

## The parallel

With $$E=-r$$, the preference-optimization and statistical-mechanics functionals and their minimizers coincide term by term:

$$\mathcal{F}_{\beta}[\pi] = \mathbb{E}_{\pi}[E] + \beta D_{\mathrm{KL}}(\pi\|\pi_{\mathrm{ref}}) = -\beta\log Z_{\beta} + \beta D_{\mathrm{KL}}(\pi\|\pi_{\beta}^{\ast}).$$

The partition function, Gibbs distribution, and free-energy minimum all emerge from the same variational structure.


<!-- ── 2. THE DICTIONARY (centerpiece) ─────────────────────────────── -->

## The dictionary

| Statistical mechanics | Preference optimization |
|---|---|
| Energy $$E$$ | $$-r$$ (negative reward) |
| Base measure / density of states | $$\pi_{\mathrm{ref}}$$ — *not* just a penalty |
| Entropy | **relative** entropy $$-D_{\mathrm{KL}}(\pi \Vert \pi_{\mathrm{ref}})$$, not token entropy |
| Temperature $$k_B T$$ | DPO's $$\beta$$ |
| Partition function $$Z$$ | $$\sum_y q(y)\,e^{r(y)/\beta}$$ |
| Free energy $$F$$ | $$-\beta \log Z_\beta$$ |
| Excess free energy | $$\beta\, D_{\mathrm{KL}}(\pi \Vert \pi_\beta^{\ast})$$ |

The generalized free energy is

$$\mathcal{F}_\beta[\pi] = U[\pi] - \beta\, S_{\mathrm{rel}}[\pi] = -\mathbb{E}_\pi[r] + \beta\, D_{\mathrm{KL}}(\pi \Vert q),$$

minimized uniquely by the Gibbs distribution, with

$$\mathcal{F}_\beta[\pi_\beta^{\ast}] = -\beta \log Z_\beta, \qquad \mathcal{F}_\beta[\pi] - \mathcal{F}_\beta[\pi_\beta^{\ast}] = \beta\, D_{\mathrm{KL}}(\pi \Vert \pi_\beta^{\ast}).$$

<!-- ✍️ Two insights most posts miss — give each its own short passage: -->

### $$\pi_{\mathrm{ref}}$$ is the base measure, not a penalty

✍️ *The reference policy determines the relative density of states before reward
tilting — it is the system's base measure, not merely the location of a penalty.
The entropy is relative entropy w.r.t. that base measure, not ordinary token entropy.*

### The temperature convention trap

✍️ *DPO's $$\beta$$ acts like $$k_B T$$: smaller values → stronger reward tilting.
Physics usually writes inverse temperature $$\beta_{\mathrm{phys}} = 1/(k_B T)$$.
Mixing the two conventions produces the common reversed-temperature reading.*

![Thermodynamic terms vs beta]({{ '/assets/figures/synthetic/thermodynamic_terms_vs_beta.png' | relative_url }})

<!-- Other synthetic figures available:
     synthetic/correct_probability_vs_beta.png
     synthetic/expected_reward_vs_beta.png
     synthetic/kl_vs_beta.png
     synthetic/ess_vs_beta.png
     synthetic/free_energy_partition_identity.png
     synthetic/free_energy_gap_identity.png
     synthetic/reference_probability_vs_reward.png -->

<!-- ── 3. TAKING IT TO A REAL MODEL ────────────────────────────────── -->

## Taking it to a real model

✍️ *GSM8K, candidate generation from the base model, Gibbs reweighting over the
candidate set, the $$\beta$$ sweep. What you measured and how.*

![Reward–KL frontier]({{ '/assets/figures/real/01_reward_kl_frontier.png' | relative_url }})

![Free-energy decomposition]({{ '/assets/figures/real/05_free_energy_landscape.png' | relative_url }})

<!-- Other real-model figures available:
     real/03_reference_normalization.png
     real/04_candidate_trajectories.png
     real/05_free_energy_landscape.png
     rl/pilot_beta_0p2/*.png
     rl/beta_sweep/free_energy_decomposition_eval_beta_sweep.png
     rl/base_pass_rate/pass_at_n_N_32_c_4.png -->

<!-- ── 4. THE HONEST PART ──────────────────────────────────────────
     The section that makes the post memorable. Frame as: "what the
     clean math hides when you go empirical." -->

## What I expected to see, and couldn't

✍️ *You expected to watch $$\mathcal{F}$$ approach $$-\beta \log Z_\beta$$. Explain
why that didn't happen in practice:*

- ✍️ *the finite candidate set truncates the partition sum — $$Z_\beta$$ over 32
  (or 512) samples is not $$Z_\beta$$ over all token sequences;*
- ✍️ *the true KL is a sum over an astronomically large sequence space; the
  sampled estimator has its own bias/variance story;*
- ✍️ *anything else you learned (Gibbs approximation quality, ESS collapse at
  small $$\beta$$, …).*

<!-- ── 5. TAKEAWAYS ────────────────────────────────────────────────── -->

## Takeaways

- ✍️ *takeaway 1*
- ✍️ *takeaway 2*
- ✍️ *takeaway 3*

Code and full experiments: [github.com/YOUR_USERNAME/stat-mech-dpo](https://github.com/YOUR_USERNAME/stat-mech-dpo)
