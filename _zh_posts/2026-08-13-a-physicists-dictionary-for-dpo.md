---
layout: post
title: "从统计力学看 KL 正则化强化学习"
subtitle: "DPO、PPO 与 GRPO 家族背后的数学——以及训练能否抵达 Gibbs 最小值"
date: 2026-08-13
tags: [statistical-mechanics, rlhf, dpo]
image: /assets/figures/a-physicists-dictionary-for-dpo/free-energy-landscape.png
lang: zh
---

偏好后训练（preference post-training）通常被表述为 **KL 正则化的奖励最大化**（KL-regularized reward maximization）：在提高期望奖励的同时，不让策略偏离参考策略（reference policy）太远。它的最优解具有 Gibbs 形式

$$
\pi_\beta^\ast(y|x) = 
\frac{\pi_{\mathrm{ref}}(y|x)e^{r(x,y)/\beta}}{Z(x)},
\tag{1}
$$

其中 $$Z(x)$$ 是式 (13) 中定义的配分函数（partition function）。DPO 的推导[^dpo]，以及 PPO、GRPO 式训练这类 KL 正则化 RLHF 方法，背后都是同一个结构。[^ppo]

大多数讲解——包括 [Karina Zadorozhny 的后训练算法指南](https://huggingface.co/blog/karina-zadorozhny/guide-to-llm-post-training-algorithms)和 [Ari G 从 RLHF 到 DPO 的推导梳理](https://huggingface.co/blog/ariG23498/rlhf-to-dpo)——基本都止步于此：引入 $$Z(x)$$，提一句它与统计物理的联系，指出它难以计算（intractable），但恰好在 DPO 的成对（pairwise）损失里被消掉，于是推出式 (1) 中的最优策略，然后就翻篇了。

**但在统计力学里，写下配分函数并不是推导的终点，而是起点。** 一旦有了式 (13) 定义的 $$Z$$，自由能（free energy）、内能（internal energy）、熵（entropy）和温度（temperature）都随之而来，平衡态（equilibrium）也有了精确的刻画。KL 正则化 RL 的目标函数几乎可以逐项套进同一个结构。

本文把这本“词典”明确地写出来：奖励对应负能量，$$\beta$$ 对应温度，KL 散度对应相对熵（relative entropy），而 RL 目标函数对应式 (4) 中的自由能原理。接着我会追问这个类比自然引出的问题：**如果理论预言了式 (1) 中唯一的 Gibbs 最优解 $$\pi_\beta^\ast$$，实际训练能离它多近？** 我在 GSM8K 上用一个 GRPO 家族的训练器[^grpo]来检验这一点。


## KL 正则化目标函数 {#the-kl-regularized-objective}

Rafailov et al. (2023)[^dpo] 的式 (3) 定义了 DPO 所依托的 KL 正则化 RLHF 目标。同样的奖励–KL 权衡也出现在 PPO 和 GRPO 中：这些方法优化的是替代目标（surrogate）或基于采样的估计量，再加上一个相对参考策略的 KL 散度项作为正则。一般地，对固定的提示（prompt）$$x$$，目标是优化

$$
\max_{\pi} \; \mathbb{E}_{y\sim\pi(\cdot\mid x)} \left[ r(x,y) \right] - \beta D_{\mathrm{KL}} \left( \pi(\cdot\mid x) \;\|\; \pi_{\mathrm{ref}}(\cdot\mid x) \right).
\tag{2}
$$

为便于阅读，下文省略对 $$x$$ 的条件依赖，记 $$\pi(y)\equiv\pi(y\mid x)$$、$$\pi_{\mathrm{ref}}(y)\equiv\pi_{\mathrm{ref}}(y\mid x)$$ 以及 $$r(y)\equiv r(x,y)$$。目标函数变为

$$
J[\pi] = \mathbb{E}_{y\sim\pi}[r(y)] - \beta D_{\mathrm{KL}} \left( \pi \;\|\; \pi_{\mathrm{ref}} \right).
\tag{3}
$$

最大化 $$J[\pi]$$ 等价于最小化

$$
\boxed{ \mathcal{F}_{\beta}[\pi] = - \mathbb{E}_{y\sim\pi}[r(y)] + \beta D_{\mathrm{KL}} \left( \pi \;\|\; \pi_{\mathrm{ref}} \right). }
\tag{4}
$$

这里 $$\mathcal{F}_{\beta}$$ 是策略 $$\pi$$ 的泛函（functional）。给定参考策略 $$\pi_{\mathrm{ref}}$$ 和奖励函数 $$r$$，问题就是找到使式 (4) 最小的策略 $$\pi$$。

## 映射到统计力学 {#mapping-to-statistical-mechanics}

式 (4) 中的目标函数具有 [Helmholtz 自由能](https://en.wikipedia.org/wiki/Helmholtz_free_energy)（Helmholtz free energy）的形式。每个回答（response）$$y$$ 是系统的一个状态，其能量为

$$
\epsilon(y)=-r(y).
\tag{5}
$$

对策略 $$\pi$$，定义平均能量和无量纲的相对熵为

$$
U[\pi]=\mathbb{E}_{\pi}[\epsilon]=-\mathbb{E}_{\pi}[r],
\qquad
S_{\mathrm{rel}}[\pi]=-D_{\mathrm{KL}}(\pi\|\pi_{\mathrm{ref}}).
\tag{6}
$$

于是目标函数可以写成

$$
\mathcal{F}_{\beta}[\pi]
=U[\pi]-\beta S_{\mathrm{rel}}[\pi],
\tag{7}
$$

与 $$F=U-TS$$ 逐项对应：

$$
\boxed{
\epsilon(y)\longleftrightarrow-r(y),\qquad
k_B T\longleftrightarrow\beta,\qquad
\frac{S}{k_B}\longleftrightarrow-D_{\mathrm{KL}}(\pi\|\pi_{\mathrm{ref}}).
}
\tag{8}
$$

这里 DPO 的 $$\beta$$ 扮演热能 $$k_B T$$ 的角色，因此在物理学约定 $$\beta_{\mathrm{phys}}=1/(k_B T)$$ 下，它的 Boltzmann 因子 $$e^{-\epsilon/\beta}$$ 与 $$e^{-\beta_{\mathrm{phys}}\epsilon}$$ 一致。

可是，[熵](https://en.wikipedia.org/wiki/Entropy_(statistical_thermodynamics)#Gibbs_entropy_formula)不应该是 $$-\sum_{i}\pi_i\log{\pi_i}$$ 吗？为什么会牵扯到一个参考分布，变成 $$-\sum_{i}\pi_i\log{\pi_i/\pi_{\mathrm{ref},i}}$$？其实当状态存在[简并](https://en.wikipedia.org/wiki/Degenerate_energy_levels)（degeneracy）时，普通的熵恰好就是这种形式。设粗粒化（coarse-grained）状态 $$i$$ 包含 $$g_i$$ 个微观态（microstate），总概率为 $$\pi_i$$。若这些微观态等概率，则每个微观态的概率为 $$p_{i,\alpha}=\pi_i/g_i$$，于是

$$
\frac{S}{k_B}
=-\sum_{i,\alpha}p_{i,\alpha}\log p_{i,\alpha}
=-\sum_i\pi_i\log\frac{\pi_i}{g_i}.
$$

令 $$G=\sum_j g_j$$，并把简并度归一化为 $$q_i=g_i/G$$。展开对数，

$$
\frac{S}{k_B}
= -\sum_i \pi_i \log\pi_i + \sum_i \pi_i \log g_i
= -\sum_i \pi_i \log\frac{\pi_i}{q_i} + \log G
= -D_{\mathrm{KL}}(\pi\|q)+\log G.
\tag{9}
$$

常数 $$\log G$$ 不影响式 (4) 中自由能的最小化。因此，在相差一个加性常数的意义下，简并把普通的熵变成了相对于归一化简并测度（normalized degeneracy measure）$$q$$ 的负 KL 散度。

在约束 $$\sum_i\pi_i=1$$ 下最小化 $$F=U-TS/k_B=\sum_i\pi_i\epsilon_i+\beta D_{\mathrm{KL}}(\pi\|q)$$（与[附录 B](#b-lagrange-multiplier) 中的变分问题相同，只是把 $$\pi_{\mathrm{ref}}$$ 换成 $$q$$），得到 Boltzmann 分布

$$
\pi_i = \frac{g_i \exp(-\beta_{\mathrm{phys}} \epsilon_i)}{Z},
\qquad
Z = \sum_{i} g_i \exp(-\beta_{\mathrm{phys}} \epsilon_i).
\tag{10}
$$

这是简并系统的标准形式，参见 [Ellgen, *Thermodynamics and Chemical Equilibrium*, §21.1](https://chem.libretexts.org/Bookshelves/Physical_and_Theoretical_Chemistry_Textbook_Maps/Thermodynamics_and_Chemical_Equilibrium_(Ellgen)/21%3A_The_Boltzmann_Distribution_Function/21.01%3A_Finding_the_Boltzmann_Equation) 和 [Pathria and Beale, *Statistical Mechanics*, §3.4](https://shop.elsevier.com/books/statistical-mechanics/beale/978-0-12-382188-1)。

在 KL 正则化 RL 中，$$\pi_{\mathrm{ref}}(y)$$ 就是归一化简并测度 $$q(y)$$。由式 (5) 的 $$\epsilon(y)=-r(y)$$，再让 DPO 的 $$\beta$$ 对应热能 $$k_B T$$（即 $$\beta_{\mathrm{phys}}=1/\beta$$），有

$$
e^{-\beta_{\mathrm{phys}}\epsilon}=e^{r/\beta},
\tag{11}
$$

这正是式 (1) 中出现的 Boltzmann 权重。


## Gibbs 最优解 {#the-gibbs-optimum}

在统计力学中，给定温度、能级以及归一化简并测度 $$q$$，Gibbs 分布是使自由能最小的唯一分布。同样的变分原理在这里也成立：一旦固定 $$\beta$$、奖励函数和参考策略 $$\pi_{\mathrm{ref}}$$，就存在唯一的最优策略使式 (4) 中的 $$\mathcal{F}_{\beta}$$ 最小。在物理系统中，$$q$$ 由简并度决定；在偏好优化中，$$\pi_{\mathrm{ref}}$$ 是一种建模选择，不同设置下可以不同，但在任一给定设置内，它都充当固定的参考。

按照[附录](#appendix-two-derivations-of-the-optimal-policy)中的步骤，自由能可以分解为

$$
\boxed{ \mathcal{F}_{\beta}[\pi] = -\beta\log Z_{\beta} + \beta D_{\mathrm{KL}} \left( \pi \;\|\; \pi_{\beta}^{\ast} \right). }
\tag{12}
$$

其中配分函数为

$$
\boxed{ Z_{\beta} = \sum_y \pi_{\mathrm{ref}}(y) e^{r(y)/\beta} }
\tag{13}
$$

而对应的 Gibbs 策略就是式 (1)，只需在省略对 $$x$$ 的条件依赖时把 $$Z(x)$$ 换成 $$Z_{\beta}$$。

由于 $$D_{\mathrm{KL}}(\pi\|\pi_{\beta}^{\ast})\ge 0$$，且当且仅当 $$\pi=\pi_{\beta}^{\ast}$$ 时取等号，唯一的最小值点是 $$\pi_{\beta}^{\ast}$$，平衡自由能为 $$\mathcal{F}_{\beta}^{\ast}=-\beta\log Z_{\beta}$$。这恰好就是[正则系综](https://en.wikipedia.org/wiki/Helmholtz_free_energy#Relation_to_the_canonical_partition_function)（canonical ensemble）的自由能。更一般地，一旦知道了式 (13) 中的[配分函数](https://en.wikipedia.org/wiki/Partition_function_(statistical_mechanics)#Relation_to_thermodynamic_variables)，内能、熵、热容（heat capacity）等平衡态物理量都可以由它及其对 $$\beta$$ 的导数得到——这正是下面的词典和式 (8) 所总结的逻辑。

推导 $$\pi_{\beta}^{\ast}$$ 有两种标准方法：

1. **配分函数 / KL 分解。** 改写式 (4)，使一个未归一化的 Boltzmann 权重显现出来，再用式 (13) 中的 $$Z_{\beta}$$ 将其归一化。
2. **Lagrange 乘子（Lagrange multiplier）。** 在施加归一化约束 $$\sum_y\pi(y)=1$$ 的同时令 $$\mathcal{F}_{\beta}$$ 的泛函导数为零；得到的驻点（stationary point）同样是式 (1)。

两种方法的完整推导见[附录](#appendix-two-derivations-of-the-optimal-policy)。

## 词典 {#the-dictionary}

左边是统计力学，右边是 KL 正则化 RL：

| 统计力学                         | KL 正则化 RL                                                                                                                                |
| -------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------ |
| 状态能量 $$\epsilon$$            | $$-r$$（负奖励）                                                                                                                           |
| 归一化简并度 $$q$$               | $$\pi_{\mathrm{ref}}$$                                                                                                                     |
| 内能 $$U$$                       | $$-\mathbb{E}_{\pi}[r]$$                                                                                                                   |
| 熵 $$S$$                         | $$-D_{\mathrm{KL}}(\pi \Vert \pi_{\mathrm{ref}})$$                                                                                         |
| 温度 $$k_B T$$                   | DPO 的 $$\beta$$                                                                                                                           |
| 配分函数 $$Z$$                   | $$\sum_y \pi_{\mathrm{ref}}(y)\,e^{r(y)/\beta}$$                                                                                            |
| 最小自由能 $$F^{\ast}$$          | $$-\beta \log Z_\beta$$                                                                                                                    |
| 自由能差距（free-energy gap）                       | $$\beta\, D_{\mathrm{KL}}(\pi \Vert \pi_\beta^{\ast})$$                                                                                   |
| 热容 $$C_V/k_B$$                 | $$\displaystyle \frac{\operatorname{Var}_{\pi_{\beta}^{\ast}}[r]}{\beta^2} = -\frac{\partial \mathbb{E}_{\pi_{\beta}^{\ast}}[r]}{\partial \beta}$$ |

热容这一项来自对式 (13) 中的配分函数求导。令 $$t=1/\beta$$，则 $$Z_{\beta}=\sum_y\pi_{\mathrm{ref}}(y)e^{tr(y)}$$。对式 (1) 中的 Gibbs 策略，标准恒等式给出 $$\mathbb{E}_{\pi_{\beta}^{\ast}}[r]=\partial_t\log Z_{\beta}$$ 和 $$\operatorname{Var}_{\pi_{\beta}^{\ast}}[r]=\partial_t^2\log Z_{\beta}$$。由于 $$t=1/\beta$$ 意味着 $$\mathrm{d}t/\mathrm{d}\beta=-1/\beta^2$$，

$$-\frac{\partial \mathbb{E}_{\pi_{\beta}^{\ast}}[r]}{\partial \beta}
= -\frac{\mathrm{d}t}{\mathrm{d}\beta}\,\frac{\partial \mathbb{E}_{\pi_{\beta}^{\ast}}[r]}{\partial t}
= \frac{\operatorname{Var}_{\pi_{\beta}^{\ast}}[r]}{\beta^2}.$$

代入 $$\epsilon=-r$$，这就是正则系综中熟悉的关系 $$C_V/k_B=\beta_{\mathrm{phys}}^2\operatorname{Var}(\epsilon)$$。

## GSM8K 上的实验 {#experiments-on-gsm8k}

以上所有结论在概率分布层面都是精确的。为了看看这个理想情形对真实模型意味着什么，我在 GSM8K 上用 `Qwen2.5-0.5B-Instruct` 做了实验，奖励取二值：最终答案正确时 $$r(y)=1$$，否则为 $$0$$。

### Gibbs 下界（Gibbs floor） {#the-gibbs-floor}

对 64 道 GSM8K 测试题中的每个提示 $$x_i$$，我从基座模型（base model）采样 24 个候选回答 $$y_{ij}$$，共得到 1,536 条补全（completion）。生成参数为温度 0.8、top-p 0.95，长度上限 512 个 token。每个候选回答都有一个基座模型给出的对数概率，以及一个二值的正确性奖励 $$r(x_i,y_{ij})\in\{0,1\}$$。把基座模型概率在这 24 个候选上归一化，就在这个有限候选集上定义了 $$\pi_{\mathrm{ref}}(y_{ij}\mid x_i)$$。

把这 24 个候选当作状态空间，所有期望就都变成了有限和。对每个提示，式 (1) 和式 (13) 变为

$$
Z_{\beta}(x_i)
=\sum_{j=1}^{24}\pi_{\mathrm{ref}}(y_{ij}\mid x_i)
e^{r(x_i,y_{ij})/\beta},
\qquad
\pi_{\beta}^{\ast}(y_{ij}\mid x_i)
=\frac{\pi_{\mathrm{ref}}(y_{ij}\mid x_i)e^{r(x_i,y_{ij})/\beta}}
{Z_{\beta}(x_i)}.
\tag{14}
$$

候选集一旦采样完成，Gibbs 策略及其期望奖励、KL 散度以及式 (4) 中的自由能，都可以在这个集合上精确计算。扫描 $$\beta$$，就得到[图 1](#figure-reward-kl-frontier) 中的最优奖励–KL 前沿（frontier），以及[图 2](#figure-free-energy-landscape) 中对应的自由能景观（free-energy landscape）。图 1 还比较了定义参考权重的两种方式：完整序列概率与长度归一化概率。其余实验均使用完整序列概率。

<figure id="figure-reward-kl-frontier">
  <img src="{{ '/assets/figures/a-physicists-dictionary-for-dpo/reward-kl-frontier.png' | relative_url }}" alt="不同 beta 取值下的奖励–KL 前沿">
  <figcaption><strong>图 1.</strong> 不同 \(\beta\) 下的奖励–KL 前沿，比较序列概率与长度归一化概率两种参考权重。</figcaption>
</figure>

<figure id="figure-free-energy-landscape">
  <img src="{{ '/assets/figures/a-physicists-dictionary-for-dpo/free-energy-landscape.png' | relative_url }}" alt="不同 beta 取值下的自由能景观">
  <figcaption><strong>图 2.</strong> 不同 \(\beta\) 下的自由能景观；颜色表示以 KL 度量的、到 Gibbs 最优解的差距。</figcaption>
</figure>



### GRPO 家族训练 vs. Gibbs 下界 {#grpo-family-training-vs-the-floor}

在参考策略处 KL 项为零，因此 $$\mathcal{F}_{\beta}[\pi_{\mathrm{ref}}]=-\mathbb{E}_{\pi_{\mathrm{ref}}}[r]$$，即[图 2](#figure-free-energy-landscape) 中的黑色点线基线。如果训练能到达分布层面的最优解，它的自由能应当从这条基线下降，趋向式 (12) 给出的红色 Gibbs 最小值 $$\mathcal{F}_{\beta}^{\ast}=-\beta\log Z_{\beta}$$。

为了检验这一点，我对 $$\beta\in\{0.01,0.02,0.05,0.1,0.2\}$$ 分别运行了一个 GRPO 家族训练器[^grpo]，组大小 $$G=32$$，使用 6,500 道 GSM8K 训练题，训练约一个 epoch（262,144 次 rollout）。每 32,768 次 rollout 保存一个检查点（checkpoint），并在同一组 500 道留出（held-out）题上评估每个检查点。参考模型的初始期望奖励为 0.348。

<figure id="figure-training-free-energy-decomposition">
  <img src="{{ '/assets/figures/a-physicists-dictionary-for-dpo/training-free-energy-decomposition.png' | relative_url }}" alt="不同 KL 惩罚系数下训练过程中的自由能分解">
  <figcaption><strong>图 3.</strong> 不同 KL 惩罚系数 \(\beta\) 下训练过程中的自由能分解：留出验证集上的奖励项、KL 项与总自由能。</figcaption>
</figure>

### 训练为何达不到下界 {#why-training-falls-short}

我原本预期每个策略都会向式 (1) 中各自的 $$\pi_{\beta}^{\ast}$$ 靠拢，每条自由能曲线都会逼近式 (12) 给出的虚线目标。[图 3](#figure-training-free-energy-decomposition) 表明这过于乐观了：训练一个 epoch 之后，没有一次运行到达其 Gibbs 最小值。奖励确实提高了，但不足以抵消随之而来的、以 $$\beta$$ 加权的 KL 代价，因此实测自由能始终高于目标值。

可能的解释：

1. **表达能力有限。** 优化被限制在一个 0.5B 模型的参数空间内，而这个空间未必包含真正的 Gibbs 策略。
2. **损失非凸。** 即便在该参数空间内，损失也是非凸的，梯度下降并不保证能找到全局最小值。

## 小结 {#takeaways}

- KL 正则化 RL 就是统计物理中的自由能最小化：式 (4) 是 RL 的自由能，式 (8) 是这本词典。
  - 给定 $$\beta$$、参考策略 $$\pi_{\mathrm{ref}}$$ 和奖励函数 $$r$$，存在唯一的策略 $$\pi_{\beta}^{\ast}$$ 使自由能最小，即式 (1)。
  - 从式 (13) 中的 $$Z_{\beta}$$ 可以读出内能、熵、温度、自由能和热容。
- 训练未必能到达这个分布层面的最优解。

代码与完整实验：[github.com/wang-zhongwei/stat-mech-dpo](https://github.com/wang-zhongwei/stat-mech-dpo)

## 附录：最优策略的两种推导 {#appendix-two-derivations-of-the-optimal-policy}

两种方法都从式 (4) 出发，它可以等价地写成

$$
\boxed{\mathcal{F}_{\beta}[\pi] = -\sum_y \pi(y)r(y) + \beta\sum_y\pi(y)\log\frac{\pi(y)}{\pi_{\mathrm{ref}}(y)}}
\tag{A.0}
$$

以及归一化约束 $$\sum_y\pi(y)=1$$。两者都会得到式 (13) 中的配分函数、式 (1) 中的最优策略，以及式 (12) 中的自由能最小值。

### A. 配分函数 / KL 分解 {#a-partition-function--kl-decomposition}

把期望和 KL 散度显式写出，再除以 $$\beta$$，得到

$$
\frac{\mathcal{F}_{\beta}[\pi]}{\beta}
= \sum_y \pi(y)\left[\log\frac{\pi(y)}{\pi_{\mathrm{ref}}(y)}-\frac{r(y)}{\beta}\right]
= \sum_y \pi(y)\log\frac{\pi(y)}{\pi_{\mathrm{ref}}(y)e^{r(y)/\beta}}.
\tag{A.1}
$$

式 (11) 中的 Boltzmann 权重 $$\pi_{\mathrm{ref}}(y)e^{r(y)/\beta}$$ 是直接从目标函数里冒出来的，而不是一个拟设（ansatz）。用式 (13) 中的配分函数将其归一化，就得到式 (1) 中的最优策略。

再代入 $$\pi_{\mathrm{ref}}(y)e^{r(y)/\beta}=Z_{\beta}\pi_{\beta}^{\ast}(y)$$，得到

$$
\frac{\mathcal{F}_{\beta}[\pi]}{\beta}
= \sum_y \pi(y)\log\frac{\pi(y)}{Z_{\beta}\pi_{\beta}^{\ast}(y)}
= D_{\mathrm{KL}}\left(\pi\;\|\;\pi_{\beta}^{\ast}\right)
- \log Z_{\beta}\sum_y\pi(y).
\tag{A.2}
$$

由于 $$\pi$$ 是归一化的概率分布，$$\sum_y\pi(y)=1$$。因此两边乘以 $$\beta$$ 即得式 (12)：

$$
\mathcal{F}_{\beta}[\pi] = -\beta\log Z_{\beta} + \beta D_{\mathrm{KL}}\left(\pi\;\|\;\pi_{\beta}^{\ast}\right).
\tag{A.3}
$$

由于 $$D_{\mathrm{KL}}(\pi\|\pi_{\beta}^{\ast})\ge 0$$，且当且仅当 $$\pi=\pi_{\beta}^{\ast}$$ 时取等号，唯一的最小值点就是式 (1)，且 $$\mathcal{F}_{\beta}^{\ast}=-\beta\log Z_{\beta}$$。

利用式 (5) 的 $$\epsilon=-r$$，同样的代数运算给出统计力学形式

$$
\pi_{\beta}^{\ast}(y)=\frac{\pi_{\mathrm{ref}}(y)e^{-\epsilon(y)/\beta}}{Z_{\beta}},\qquad Z_{\beta}=\sum_y\pi_{\mathrm{ref}}(y)e^{-\epsilon(y)/\beta}.
\tag{A.4}
$$

当 $$\pi_{\mathrm{ref}}$$ 为均匀分布时，它退化为标准的正则 Gibbs 分布 $$e^{-\epsilon/\beta}/Z_{\beta}$$。

### B. Lagrange 乘子 {#b-lagrange-multiplier}

为在约束 $$\sum_y\pi(y)=1$$ 下最小化式 (4)，引入 Lagrange 乘子 $$\lambda$$：

$$
\mathcal{L}[\pi,\lambda]
= -\sum_y\pi(y)r(y)
+\beta\sum_y\pi(y)\log\frac{\pi(y)}{\pi_{\mathrm{ref}}(y)}
+\lambda\left(\sum_y\pi(y)-1\right).
\tag{B.1}
$$

对每个 $$\pi(y)$$ 的驻点条件要求

$$
\frac{\partial\mathcal{L}}{\partial\pi(y)}
= -r(y)
+\beta\left(\log\frac{\pi(y)}{\pi_{\mathrm{ref}}(y)}+1\right)
+\lambda
=0.
\tag{B.2}
$$

解出 $$\pi(y)$$，得到

$$
\pi(y)=C\,\pi_{\mathrm{ref}}(y)e^{r(y)/\beta},
\tag{B.3}
$$

其中 $$C=e^{-1-\lambda/\beta}$$ 与 $$y$$ 无关。施加归一化条件得到 $$C^{-1}=Z_{\beta}$$（即式 (13)），从而得到与式 (1) 相同的最优策略：

$$
\pi_{\beta}^{\ast}(y)=\frac{\pi_{\mathrm{ref}}(y)e^{r(y)/\beta}}{Z_{\beta}}.
\tag{B.4}
$$

把这个驻点代回式 (4)，就得到式 (12) 中的平衡自由能 $$\mathcal{F}_{\beta}^{\ast}=-\beta\log Z_{\beta}$$。

[^dpo]: Rafailov et al., [*Direct Preference Optimization: Your Language Model Is Secretly a Reward Model*](https://proceedings.neurips.cc/paper_files/paper/2023/file/a85b405ed65c6477a4fe8302b5e06ce7-Paper-Conference.pdf), NeurIPS 2023.

[^ppo]: Schulman et al., [*Proximal Policy Optimization Algorithms*](https://arxiv.org/abs/1707.06347), arXiv:1707.06347 (2017)。PPO 并不直接优化式 (2) 中的 KL 正则化目标：它最大化的是关于比值 $$\rho=\pi_\theta/\pi_{\theta_{\mathrm{old}}}$$ 的截断（clipped）替代目标。但在每次更新开始时 $$\pi_\theta=\pi_{\theta_{\mathrm{old}}}$$，截断不起作用，且 $$\nabla_\theta\,\rho A=\nabla_\theta \log\pi_\theta\,A$$——这正是最朴素的策略梯度（vanilla policy gradient）。把相对参考策略的 KL 惩罚并入奖励之后，PPO 在局部沿着的正是同一个 KL 正则化目标的上升方向；截断只是限制了步长。

[^grpo]: Shao et al., [*DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models*](https://arxiv.org/abs/2402.03300), arXiv:2402.03300 (2024).
