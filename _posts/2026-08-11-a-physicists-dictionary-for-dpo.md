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

For reference distribution $q$, reward $r$, and temperature $\beta > 0$, the optimal policy is

$$
\pi_\beta^*(y) = \frac{q(y)\,e^{r(y)/\beta}}{Z_\beta},
\qquad
Z_\beta = \sum_y q(y)\,e^{r(y)/\beta}.
$$

<!-- ── 2. THE DICTIONARY (centerpiece) ─────────────────────────────── -->

## The dictionary

| Statistical mechanics | Preference optimization |
|---|---|
| Energy $E$ | $-r$ (negative reward) |
| Base measure / density of states | $\pi_{\mathrm{ref}}$ — *not* just a penalty |
| Entropy | **relative** entropy $-D_{\mathrm{KL}}(\pi \Vert \pi_{\mathrm{ref}})$, not token entropy |
| Temperature $k_B T$ | DPO's $\beta$ |
| Partition function $Z$ | $\sum_y q(y)\,e^{r(y)/\beta}$ |
| Free energy $F$ | $-\beta \log Z_\beta$ |
| Excess free energy | $\beta\, D_{\mathrm{KL}}(\pi \Vert \pi_\beta^*)$ |

The generalized free energy is

$$
\mathcal{F}_\beta[\pi]
= U[\pi] - \beta\, S_{\mathrm{rel}}[\pi]
= -\mathbb{E}_\pi[r] + \beta\, D_{\mathrm{KL}}(\pi \Vert q),
$$

minimized uniquely by the Gibbs distribution, with

$$
\mathcal{F}_\beta[\pi_\beta^*] = -\beta \log Z_\beta,
\qquad
\mathcal{F}_\beta[\pi] - \mathcal{F}_\beta[\pi_\beta^*]
= \beta\, D_{\mathrm{KL}}(\pi \Vert \pi_\beta^*).
$$

<!-- ✍️ Two insights most posts miss — give each its own short passage: -->

### π_ref is the base measure, not a penalty

✍️ *The reference policy determines the relative density of states before reward
tilting — it is the system's base measure, not merely the location of a penalty.
The entropy is relative entropy w.r.t. that base measure, not ordinary token entropy.*

### The temperature convention trap

✍️ *DPO's $\beta$ acts like $k_B T$: smaller values → stronger reward tilting.
Physics usually writes inverse temperature $\beta_{\mathrm{phys}} = 1/(k_B T)$.
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
candidate set, the β sweep. What you measured and how.*

![Reward–KL frontier]({{ '/assets/figures/real/01_reward_kl_frontier.png' | relative_url }})

![Free-energy decomposition]({{ '/assets/figures/real/02_free_energy_decomposition.png' | relative_url }})

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

✍️ *You expected to watch $\mathcal{F}$ approach $-\beta \log Z_\beta$. Explain
why that didn't happen in practice:*

- ✍️ *the finite candidate set truncates the partition sum — $Z_\beta$ over 32
  (or 512) samples is not $Z_\beta$ over all token sequences;*
- ✍️ *the true KL is a sum over an astronomically large sequence space; the
  sampled estimator has its own bias/variance story;*
- ✍️ *anything else you learned (Gibbs approximation quality, ESS collapse at
  small β, …).*

<!-- ── 5. TAKEAWAYS ────────────────────────────────────────────────── -->

## Takeaways

- ✍️ *takeaway 1*
- ✍️ *takeaway 2*
- ✍️ *takeaway 3*

Code and full experiments: [github.com/YOUR_USERNAME/stat-mech-dpo](https://github.com/YOUR_USERNAME/stat-mech-dpo)
