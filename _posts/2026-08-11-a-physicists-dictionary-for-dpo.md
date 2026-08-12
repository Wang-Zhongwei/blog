---
layout: post
title: "A Physicist's Dictionary for DPO"
date: 2026-08-11
tags: [statistical-mechanics, rlhf, dpo]
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

Equation 3) in [Rafailov et al. (2023), *Direct Preference Optimization: Your Language Model Is Secretly a Reward Model*](https://proceedings.neurips.cc/paper_files/paper/2023/file/a85b405ed65c6477a4fe8302b5e06ce7-Paper-Conference.pdf) defined a KL-regularized RLHF objective underlying DPO. For a fixed prompt $$x$$, here is the goal to optimize:

$$\max_{\pi} \; \mathbb{E}_{y\sim\pi(\cdot\mid x)} \left[ r(x,y) \right] - \beta D_{\mathrm{KL}} \left( \pi(\cdot\mid x) \;\|\; \pi_{\mathrm{ref}}(\cdot\mid x) \right).$$

For readability, we suppress the conditioning on $$x$$ and write $$\pi(y)\equiv\pi(y\mid x)$$, $$\pi_{\mathrm{ref}}(y)\equiv\pi_{\mathrm{ref}}(y\mid x)$$, and $$r(y)\equiv r(x,y)$$. The objective becomes

$$J[\pi] = \mathbb{E}_{y\sim\pi}[r(y)] - \beta D_{\mathrm{KL}} \left( \pi \;\|\; \pi_{\mathrm{ref}} \right).$$

Maximizing $$J[\pi]$$ is equivalent to minimizing

$$\boxed{ \mathcal{F}_{\beta}[\pi] = - \mathbb{E}_{y\sim\pi}[r(y)] + \beta D_{\mathrm{KL}} \left( \pi \;\|\; \pi_{\mathrm{ref}} \right). }\tag{1}\label{eq:dpo-free-energy}$$

Here, $$\mathcal{F}_{\beta}$$ is a functional of the policy $$\pi$$. Given a reference policy $$\pi_{\mathrm{ref}}$$ and reward function $$r$$, the goal is to find the policy $$\pi$$ that minimizes $$\mathcal{F}_{\beta}$$.

## The free-energy correspondence

The objective above has the form of a [Helmholtz free-energy](https://en.wikipedia.org/wiki/Helmholtz_free_energy). Responses $$y$$ corresponds to states of a system, with energy

$$E(y)=-r(y).$$

For a policy $$\pi$$, define the mean energy and dimensionless relative entropy as

$$U[\pi]=\mathbb{E}_{\pi}[E]=-\mathbb{E}_{\pi}[r],
\qquad
S_{\mathrm{rel}}[\pi]=-D_{\mathrm{KL}}(\pi\|\pi_{\mathrm{ref}}).$$

The DPO objective is therefore

$$\mathcal{F}_{\beta}[\pi]
=U[\pi]-\beta S_{\mathrm{rel}}[\pi].$$

which matches $$F=U-TS$$ term by term:

$$\boxed{
E(y)\longleftrightarrow-r(y),\qquad
k_B T\longleftrightarrow\beta,\qquad
\frac{S}{k_B}\longleftrightarrow-D_{\mathrm{KL}}(\pi\|\pi_{\mathrm{ref}}).
}\tag{2}\label{eq:dictionary-correspondence}$$

Here DPO's $$\beta$$ corresponds to the thermal energy $$k_B T$$, so its Boltzmann factor $$e^{-E/\beta}$$ is the same as $$e^{-\beta_{\mathrm{phys}}E}$$ under the physics convention $$\beta_{\mathrm{phys}}=1/(k_B T)$$.

But shoudn't [entropy](https://en.wikipedia.org/wiki/Entropy_(statistical_thermodynamics)#Gibbs_entropy_formula) be $$-\sum_{i}\pi_i\log{\pi_i}$$? Why does it involve a reference distribution $$-\sum_{i}\pi_i\log{\pi_i/\pi_{ref,i}}$$? Ordinary entropy acquires exactly this form when states have [degeneracies](https://en.wikipedia.org/wiki/Degenerate_energy_levels). Suppose coarse-grained state $$i$$ contains $$g_i$$ microstates and has total probability $$\pi_i$$. If those microstates are equally likely, each has probability $$p_{i,\alpha}=\pi_i/g_i$$, so

$$
\frac{S}{k_B}
=-\sum_{i,\alpha}p_{i,\alpha}\log p_{i,\alpha}
=-\sum_i\pi_i\log\frac{\pi_i}{g_i}.
$$

Let $$G=\sum_j g_j$$ and normalize the degeneracies as $$q_i=g_i/G$$. Then

$$
\frac{S}{k_B}
=-D_{\mathrm{KL}}(\pi\|q)+\log G.
\tag{3}\label{eq:entropy-kl-decomposition}$$

The constant $$\log G$$ does not affect minimization of the free energy. Thus, up to an additive constant, degeneracy turns ordinary entropy into negative KL divergence relative to the normalized degeneracy measure $$q$$.

Following the steps in [Appendix](#appendix-two-derivations-of-the-optimal-policy), we can then derive 

$$
\pi_i = g_i \exp(-\beta_{\mathrm{phys}} E_i) / Z
\tag{4}\label{eq:degenerate-boltzmann-distribution}$$

$$
Z = \sum_{i} g_i \exp(-\beta_{\mathrm{phys}} E_i)
$$

This is the standard result of Boltzmann distribution for degenerate systems; see [Ellgen, *Thermodynamics and Chemical Equilibrium*, §21.1](https://chem.libretexts.org/Bookshelves/Physical_and_Theoretical_Chemistry_Textbook_Maps/Thermodynamics_and_Chemical_Equilibrium_(Ellgen)/21%3A_The_Boltzmann_Distribution_Function/21.01%3A_Finding_the_Boltzmann_Equation) and [Pathria and Beale, *Statistical Mechanics*, §3.4](https://shop.elsevier.com/books/statistical-mechanics/beale/978-0-12-382188-1).


## Optimal policy and free-energy minimum

In statistical mechanics, for fixed temperature, energy levels, and normalized degeneracy measure $$q$$, the Gibbs distribution uniquely minimizes the free energy. The same variational principle holds here: once $$\beta$$, the reward function, and the reference policy $$\pi_{\mathrm{ref}}$$ are fixed, there is a unique optimal policy that minimizes $$\mathcal{F}_{\beta}$$. While a physical system's $$q$$ is determined by system degeneracies, $$\pi_{\mathrm{ref}}$$ is a modeling choice that may vary across setups; within any one setup, however, it serves as the fixed reference. 

Following steps in [Appendix](#appendix-two-derivations-of-the-optimal-policy) we have free energy 


$$\boxed{ \mathcal{F}_{\beta}[\pi] = -\beta\log Z_{\beta} + \beta D_{\mathrm{KL}} \left( \pi \;\|\; \pi_{\beta}^{\ast} \right). }\tag{5}\label{eq:free-energy-gap}$$

where the partition function is

$$\boxed{ Z_{\beta} = \sum_y \pi_{\mathrm{ref}}(y) e^{r(y)/\beta} }\tag{6}\label{eq:dpo-partition-function}$$

and the corresponding Gibbs policy is

$$\boxed{ \pi_{\beta}^{\ast}(y) = \frac{ \pi_{\mathrm{ref}}(y) e^{r(y)/\beta} }{ Z_{\beta} }. }\tag{7}\label{eq:optimal-gibbs-policy}$$

Since $$D_{\mathrm{KL}}(\pi\|\pi_{\beta}^{\ast})\ge 0$$, with equality iff $$\pi=\pi_{\beta}^{\ast}$$, the unique minimizer is $$\pi_{\beta}^{\ast}$$ and the equilibrium free energy is $$\mathcal{F}_{\beta}^{\ast}=-\beta\log Z_{\beta}$$. This is exactly the expression of free energy in a [canonical ensemble](https://en.wikipedia.org/wiki/Helmholtz_free_energy#Relation_to_the_canonical_partition_function). 

There are two standard ways to derive $$\pi_{\beta}^{\ast}$$:

1. **Partition-function / KL-decomposition.** Rewrite $$\mathcal{F}_{\beta}[\pi]$$ so that an unnormalized Boltzmann weight appears, normalize it by $$Z_{\beta}$$
2. **Lagrange multiplier.** Enforce the normalization constraint $$\sum_y\pi(y)=1$$ while setting the functional derivative of $$\mathcal{F}_{\beta}$$ to zero; the stationary point is again the same policy.

Full algebra for both routes is in the [Appendix](#appendix-two-derivations-of-the-optimal-policy).


<!-- ── 2. THE DICTIONARY (centerpiece) ─────────────────────────────── -->

## The dictionary

| Statistical mechanics | Preference optimization |
|---|---|
| Energy $$E$$ | $$-r$$ (negative reward) |
| Normalized degeneracy $$q$$ | $$\pi_{\mathrm{ref}}$$ — *not* just a penalty |
| Entropy $$S$$ | $$-D_{\mathrm{KL}}(\pi \Vert \pi_{\mathrm{ref}})$$ |
| Temperature $$k_B T$$ | DPO's $$\beta$$ |
| Partition function $$Z$$ | $$\sum_y \pi_{\mathrm{ref}}(y)\,e^{r(y)/\beta}$$ |
| Minimum free energy $$F^{\ast}$$ | $$-\beta \log Z_\beta$$ |
| Free-energy gap | $$\beta\, D_{\mathrm{KL}}(\pi \Vert \pi_\beta^{\ast})$$ |

<!-- The free-energy functional is

$$\mathcal{F}_\beta[\pi] = U[\pi] - \beta\, S_{\mathrm{rel}}[\pi] = -\mathbb{E}_\pi[r] + \beta\, D_{\mathrm{KL}}(\pi \Vert \pi_{\mathrm{ref}}),$$

minimized uniquely at $$\pi_\beta^{\ast}$$, with

$$\mathcal{F}_\beta[\pi_\beta^{\ast}] = -\beta \log Z_\beta, \qquad \mathcal{F}_\beta[\pi] - \mathcal{F}_\beta[\pi_\beta^{\ast}] = \beta\, D_{\mathrm{KL}}(\pi \Vert \pi_\beta^{\ast}).$$ -->


## Taking it to a real model

Everything above is exact. To see what it buys, I ran it on GSM8K with
`Qwen2.5-0.5B-Instruct` under a binary reward: $$r(y)=1$$ when the final answer is
correct, $$0$$ otherwise. Binary reward makes $$\mathbb{E}_{\pi}[r]=P(\text{correct})$$,
so the system has two energy levels and the correct responses are the ground state.

### Offline: reweight a fixed candidate set

Sample 24 candidates for each of 64 test prompts — 1,536 completions at temperature
0.8, top-p 0.95, 512-token cap — score them under $$\pi_{\mathrm{ref}}$$, and treat
those 24 as the entire state space. Then $$Z_{\beta}$$, $$\pi_{\beta}^{\ast}$$, and
$$\mathcal{F}_{\beta}$$ are exact sums over 24 terms, and sweeping $$\beta$$ traces
the frontier with no estimator in the way.

The choice of base measure matters more than I expected. Raw sequence probability is
brutally concentrated — mean effective sample size 1.12 out of 24 candidates — while
length normalization spreads the same mass over 10 to 23. Both reach the same
zero-temperature ceiling of 0.844 expected reward, set by the 54 of 64 prompts holding
at least one correct candidate. They differ in price: sequence probability pays 9.17
nats of $$D_{\mathrm{KL}}(\pi_{\beta}^{\ast}\|\pi_{\mathrm{ref}})$$ to get there,
length normalization 1.17 nats, eight times less. Reward always overcomes reference
improbability at low enough temperature; the base measure sets what that costs.

<figure id="figure-reward-kl-frontier">
  <img src="{{ '/assets/figures/real/01_reward_kl_frontier.png' | relative_url }}" alt="Reward–KL frontier across beta values">
  <figcaption><strong>Figure 1.</strong> Reward–KL frontier across \(\beta\), comparing sequence probability with length-normalized probability.</figcaption>
</figure>

<figure id="figure-free-energy-landscape">
  <img src="{{ '/assets/figures/real/05_free_energy_landscape.png' | relative_url }}" alt="Free-energy landscape across beta values">
  <figcaption><strong>Figure 2.</strong> Free-energy landscape across \(\beta\); color indicates the KL-based gap to the Gibbs optimum.</figcaption>
</figure>



### Online: actually train

The offline sweep says where the floor is. Training asks whether a 0.5B policy can
walk down to it. One RLOO run with a KL penalty per
$$\beta\in\{0.01,0.02,0.05,0.1,0.2\}$$, group size $$G=32$$, 6,500 GSM8K training
prompts, 262,144 rollouts, checkpoints every 32,768, each evaluated on the same 500
held-out prompts. The reference model starts at 0.348 expected reward.

Here $$\mathcal{F}_{\beta}^{\ast}=-\beta\log Z_{\beta}$$ is a fixed horizontal target —
one per $$\beta$$ — and the run should descend toward it.

<figure id="figure-training-free-energy-decomposition">
  <img src="{{ '/assets/figures/rl/beta_sweep/free_energy_decomposition_eval_beta_sweep.png' | relative_url }}" alt="Training-time free-energy decomposition across KL penalty values">
  <figcaption><strong>Figure 3.</strong> Training-time free-energy decomposition across KL penalties \(\beta\): reward, KL, and total free-energy terms on held-out validation.</figcaption>
</figure>

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

For each beta, as training progress, pi should be get closer to pi^{\ast}, ideally we would see the policy would gradually approach the red curve on figure 2. However after 1 epoch of training on GSM8K, that was too optimistic. Runs with different betas didn't go all the way down to their Gibbs energy. I think it is because the search space for \pi is finite while Gibbs equation just show the mathematical optimal but didn't consider the feasibility over a 0.5B model. 



It appears that the reward gains too little with respect to the KL divergence it has gained

✍️ *You expected to watch $$\mathcal{F}$$ approach $$-\beta \log Z_\beta$$. Explain
why that didn't happen in practice:*

- ✍️ *the finite candidate set truncates the partition sum — $$Z_\beta$$ over 32
  (or 512) samples is not $$Z_\beta$$ over all token sequences;*
- ✍️ *the true KL is a sum over an astronomically large sequence space; the
  sampled estimator has its own bias/variance story;*
- ✍️ *anything else you learned (optimal-policy approximation quality, ESS collapse at
  small $$\beta$$, …).*

<!-- ── 5. TAKEAWAYS ────────────────────────────────────────────────── -->

## Takeaways

- ✍️ *takeaway 1*
- ✍️ *takeaway 2*
- ✍️ *takeaway 3*

Code and full experiments: [github.com/YOUR_USERNAME/stat-mech-dpo](https://github.com/YOUR_USERNAME/stat-mech-dpo)

## Appendix: two derivations of the optimal policy

Both routes start from the free-energy functional

$$\boxed{\mathcal{F}_{\beta}[\pi] = -\sum_y \pi(y)r(y) + \beta\sum_y\pi(y)\log\frac{\pi(y)}{\pi_{\mathrm{ref}}(y)}}$$

and the normalization constraint $$\sum_y\pi(y)=1$$. They recover the same partition function, optimal policy, and free-energy minimum.

### A. Partition-function / KL decomposition

Writing the expectation and KL divergence explicitly and dividing by $$\beta$$ gives

$$\frac{\mathcal{F}_{\beta}[\pi]}{\beta} = \sum_y \pi(y)\log\frac{\pi(y)}{\pi_{\mathrm{ref}}(y)e^{r(y)/\beta}}.\tag{A.1}\label{eq:appendix-kl-rewrite}$$

The Boltzmann weight $$\pi_{\mathrm{ref}}(y)e^{r(y)/\beta}$$ appears directly from the objective — not as an ansatz. Normalize it by the partition function

$$Z_{\beta} = \sum_y \pi_{\mathrm{ref}}(y) e^{r(y)/\beta}$$

to obtain the optimal policy

$$\pi_{\beta}^{\ast}(y) = \frac{\pi_{\mathrm{ref}}(y)e^{r(y)/\beta}}{Z_{\beta}}.$$

Substituting $$\pi_{\mathrm{ref}}(y)e^{r(y)/\beta}=Z_{\beta}\pi_{\beta}^{\ast}(y)$$ then yields

$$\frac{\mathcal{F}_{\beta}[\pi]}{\beta}
= \sum_y \pi(y)\log\frac{\pi(y)}{Z_{\beta}\pi_{\beta}^{\ast}(y)}
= D_{\mathrm{KL}}\left(\pi\;\|\;\pi_{\beta}^{\ast}\right)
- \log Z_{\beta}\sum_y\pi(y).\tag{A.2}\label{eq:appendix-kl-expansion}$$

Because $$\pi$$ is a normalized probability distribution, $$\sum_y\pi(y)=1$$. Multiplying by $$\beta$$ therefore gives

$$\mathcal{F}_{\beta}[\pi] = -\beta\log Z_{\beta} + \beta D_{\mathrm{KL}}\left(\pi\;\|\;\pi_{\beta}^{\ast}\right).\tag{A.3}\label{eq:appendix-free-energy-decomposition}$$

Since $$D_{\mathrm{KL}}(\pi\|\pi_{\beta}^{\ast})\ge 0$$, with equality iff $$\pi=\pi_{\beta}^{\ast}$$, the unique minimizer is $$\pi_{\beta}^{\ast}$$ and $$\mathcal{F}_{\beta}^{\ast}=-\beta\log Z_{\beta}$$.

With $$E=-r$$, the same algebra produces the statistical-mechanics form

$$\pi_{\beta}^{\ast}(y)=\frac{\pi_{\mathrm{ref}}(y)e^{-E(y)/\beta}}{Z_{\beta}},\qquad Z_{\beta}=\sum_y\pi_{\mathrm{ref}}(y)e^{-E(y)/\beta}.$$

For a uniform $$\pi_{\mathrm{ref}}$$, this reduces to the canonical Gibbs distribution $$e^{-E/\beta}/Z_{\beta}$$.

### B. Lagrange multiplier

To minimize $$\mathcal{F}_{\beta}[\pi]$$ subject to $$\sum_y\pi(y)=1$$, introduce a Lagrange multiplier $$\lambda$$:

$$\mathcal{L}[\pi,\lambda]
= -\sum_y\pi(y)r(y)
+\beta\sum_y\pi(y)\log\frac{\pi(y)}{\pi_{\mathrm{ref}}(y)}
+\lambda\left(\sum_y\pi(y)-1\right).\tag{B.1}\label{eq:lagrangian}$$

Stationarity with respect to each $$\pi(y)$$ requires

$$\frac{\partial\mathcal{L}}{\partial\pi(y)}
= -r(y)
+\beta\left(\log\frac{\pi(y)}{\pi_{\mathrm{ref}}(y)}+1\right)
+\lambda
=0.\tag{B.2}\label{eq:stationarity}$$

Solving for $$\pi(y)$$ gives

$$\pi(y)=C\,\pi_{\mathrm{ref}}(y)e^{r(y)/\beta},\tag{B.3}\label{eq:unnormalized-policy}$$

where $$C=e^{-1-\lambda/\beta}$$ is independent of $$y$$. Enforcing normalization yields

$$C^{-1}=\sum_y\pi_{\mathrm{ref}}(y)e^{r(y)/\beta}=Z_{\beta},$$

and therefore the same optimal policy

$$\pi_{\beta}^{\ast}(y)=\frac{\pi_{\mathrm{ref}}(y)e^{r(y)/\beta}}{Z_{\beta}}.\tag{B.4}\label{eq:appendix-optimal-policy}$$

Substituting this stationary point back into $$\mathcal{F}_{\beta}$$ recovers the equilibrium free energy $$\mathcal{F}_{\beta}^{\ast}=-\beta\log Z_{\beta}$$.
