# L2T — Learning to Transfer for Evolutionary Multitasking: implementation notes

**Source.** S.-H. Wu, Y. Huang, X. Wu, L. Feng, Z.-H. Zhan, K. C. Tan, "Learning to Transfer for Evolutionary Multitasking," *IEEE Trans. Cybernetics* 55(7):3342–3355, July 2025, DOI 10.1109/TCYB.2025.3561518, plus its supplementary material (Sections S.I–S.III, Algorithms S.1–S.3, Tables S.I–S.XVI).

**Other pointers.**
- Preprint: arXiv:2406.14359 (v1, June 2024). Same method, older wording and supplement table numbering. Everything below follows the published version.
- Supplement URL given in the paper: https://wushenghao8404.github.io/files/L2T-supplement.pdf
- No official code release is named in the paper or supplement. MetaBox (the benchmark platform the paper cites as [19]) documents a third-party module `src.baseline.metabbo.l2t` with `Actor(n_state, n_action, hidden_dim=64)` / `Critic` classes; its fidelity to the paper was not checked here.

**Provenance tags used below**

| Tag | Meaning |
|---|---|
| **[paper]** | Stated in the published paper or supplement |
| **[inferred]** | Not stated; reconstructed from context. Reasoning given |
| **[unspecified]** | Not stated anywhere; a default is suggested |

Section 15 collects every gap and inconsistency in one list. Read it before coding.

---

## 1. What is being built

An RL agent that controls **knowledge transfer (KT)** inside an implicit evolutionary multitasking (EMT) algorithm. Once per generation the agent sees an 18-dim feature vector describing both tasks' populations and outputs 3 numbers per task in [0, 1]:

- `a_k1` — **when** to transfer: fraction of offspring produced by KT (KT share = `0.5·a_k1`, so at most half)
- `a_k2` — **how**: weight of the source task in the *base vector*
- `a_k3` — **how**: weight of the source task in the *differential vector*

Two stages:

1. **Learning stage** — PPO trains an actor–critic MLP by running many short EMT episodes (rollouts of `G_roll` = 100 generations) on task pairs sampled from a training task set. Reward needs each task's optimal value `f*`.
2. **Inference stage** — the frozen actor is plugged into the same EMT loop (MTDE-L2T with a DE base solver, MTGA-L2T with a GA base solver) to solve unseen task pairs for `G_max` generations. No reward, no `f*` needed.

Everything is implemented for **K = 2 tasks**. For more tasks the paper says to randomly pair tasks into 2-task sub-problems and reuse the same agent **[paper]**.

---

## 2. Notation

| Symbol | Meaning |
|---|---|
| `K` | number of tasks in an MTOP instance (2 everywhere) |
| `f_k` | k-th task, minimisation; `f*_k` its optimal value |
| `k`, `j` | target task, source task (`j ≠ k`; with K=2 the other task) |
| `N` | population size per task (50) |
| `D` | dimensionality (value not stated — see §15) |
| `X_k`, `Y_k` | population (N×D) and fitness (N) of task k at generation g |
| `x*_k,g` | best solution found on task k up to generation g |
| `μ_k`, `σ_k` | per-dimension mean / std vectors of `X_k` (D-dim) |
| `F`, `CR` | DE scale factor, crossover rate (0.5, 0.5) |
| `a_k1, a_k2, a_k3` | KT action for task k, each in [0, 1] |
| `G_roll`, `G_max` | generations per training rollout (100), generations at test (250) |
| `O_c`, `O_t,k` | common features (4), task-specific features (7 per task) |
| `b1, b2, b3` | reward weights (1, 10, 100) |
| `ξ` | target accuracy (1e-8) |
| `θ`, `ω` | actor / critic parameters |

---

## 3. Problem formulation

**Task instance [paper, S.III-A].** `g(x; f, M, x_O) = f(M (x − x_O))`, with `f` a function class, `M` a rotation matrix, `x_O` a shift vector (moves the optimum). Tasks are black-box.

**MTOP instance.** A set of K tasks, each drawn i.i.d. from a task instance set `Θ = F × S` (function classes × configurations). Box constraints only.

**Unified search space [paper].** Each task's box `[L_k, U_k]` is mapped to `[0,1]^{D_U}` by `(x − L_k)/(U_k − L_k)`, `D_U = max_k D_k`. All evolution, KT and feature computation happen in the unified space; decode before evaluating.

**MTOP instance set.** `Γ = {T_j}`, 100 instances per test set, sampled from `Θ`.

---

## 4. Base solver: DE/rand/1/bin [paper]

```
Mutation (1):   v_k,i = X_k,r1 + F · (X_k,r2 − X_k,r3)          r1,r2,r3 distinct, random in {1..N}
Crossover (2):  u_k,i,d = v_k,i,d   if rand_d(0,1) ≤ CR  or  d == randint_i(1, D)
                          X_k,i,d   otherwise
```

`F = 0.5`, `CR = 0.5`, `N = 50`.

Selection is only described as "update populations X_k by selection". Standard DE one-to-one replacement (offspring i replaces parent i if not worse) is the natural reading **[inferred]**.

---

## 5. Action space and the KT operator

### 5.1 Existing operators that the action space generalises [paper]

```
(5) arithmetic / SBX-like crossover:  v = λ·X_k,r1 + (1−λ)·X_j,r2
(6) base-vector transfer:             v = X_j,r1 + F·(X_k,r2 − X_k,r3)
(7) differential-vector transfer:     v = X_k,r1 + F·(X_j,r2 − X_j,r3)
(8) direct transfer from source:      v = X_j,r1 + F·(X_j,r2 − X_j,r3)
```

### 5.2 L2T sampling model [paper]

```
(9)  v_k,i ~ (1 − 0.5·a_k1) · p(v | X_k, A)            # self-evolution by base solver, Eq. (1)
           +      0.5·a_k1  · p(v | {X_j}, K)          # KT operator, Eq. (10)

(10) v_k,i = (1 − a_k2)·X_k,r1 + a_k2·X_j,r2
           + F·(1 − a_k3)·(X_k,r3 − X_k,r4)
           + F·     a_k3 ·(X_j,r5 − X_j,r6)
```

- `r1..r6` are random indices in `{1..N}`; `r1, r3, r4` index the target population, `r2, r5, r6` the source population.
- After (10), apply binomial crossover (2) against parent `X_k,i` to get the offspring.
- Self-evolution probability is `α_k = 1 − 0.5·a_k1 ≥ 0.5` by construction (convergence argument from MFEA-II).
- `j` is a randomly chosen source task `≠ k` (the other task when K=2).

### 5.3 Table I — what corner actions mean [paper]

| `a_k1` | `a_k2` | `a_k3` | When | How |
|---|---|---|---|---|
| 0 | – | – | no KT | pure self-evolution, Eq. (1) |
| 1 | 0 | 1 | KT prob. 0.5 | differential-vector transfer, Eq. (7) |
| 1 | 0.5 | 0 | KT prob. 0.5 | target/source crossover, Eq. (5), plus differential vector |
| 1 | 1 | 0 | KT prob. 0.5 | source solution as base vector, Eq. (6) |
| 1 | 1 | 1 | KT prob. 0.5 | sampled solution from source population, Eq. (8) |

Also: `a_k2 = a_k3 = 0` makes the KT branch identical to DE/rand/1 on the target population. MFDE corresponds to the fixed action `(0.5, 0, 1)`; MTDE-B (base-vector transfer) also lies inside the space **[paper]**.

### 5.4 Deterministic KT quota (variance-reduction trick) [paper + inferred]

Instead of flipping a coin per offspring with probability `0.5·a_k1`, the number of KT offspring is fixed per generation:

```
N_k,KT = ceil(0.5 · a_k1 · N)        # 0..25 for N = 50
```

The paper prints `N_k,KT = ⌈0.5 · a_k1⌉` (no `N`) in Algorithm 1, Algorithm S.2 and the text. Taken literally this is always 0 or 1, which contradicts the text ("precise quota of KT solutions", Eq. 11 averaging over `N_k,KT` offspring), so the `· N` factor is **[inferred]**.

Construction of the offspring population: (i) generate all N offspring with the base solver; (ii) pick `N_k,KT` random distinct indices; (iii) replace those offspring with KT-generated ones.

---

## 6. State representation (dimension 4 + 7K = 18)

All eleven features are from Table II of the published paper; every one has range [0, 1].

### 6.1 Common features `O_c` (shared by both tasks)

| Feature | Definition | Source |
|---|---|---|
| `O_c1` | `g / G_max` — ratio of current generation | [paper] |
| `O_c2` | `d(x*_1, x*_2) / sqrt(D)` — distance between best-found solutions up to generation g | [paper] |
| `O_c3` | `d(μ_1, μ_2) / sqrt(D)` — distance between population means (first-moment statistics) | [paper] |
| `O_c4` | `d(σ_1, σ_2) / sqrt(0.5·D)` — distance between per-dimension std vectors (second-moment statistics) | [paper] |

`d` is Euclidean distance. `μ_k`, `σ_k` are the D-dimensional mean and standard-deviation vectors of `X_k`. The normalisers assume solutions in the unified `[0,1]^D` space (max distance `sqrt(D)`; per-dimension std of points in [0,1] is at most 0.5).

### 6.2 Task-specific features `O_t,k` (per task)

| Feature | Definition | Source |
|---|---|---|
| `O_t,k,1` | `n_k,stag / G_max` — number of stagnating generations | [paper] |
| `O_t,k,2` | `flag_k,improved ∈ {0,1}` — best solution improved in previous generation | [paper] |
| `O_t,k,3` | `q_k,KT` — transfer quality of previous KT action, Eq. (11) | [paper] |
| `O_t,k,4` | `mean(σ_k)` — mean over dimensions of per-dimension std of `X_k` | [paper] |
| `O_t,k,5..7` | previous action `a_k1, a_k2, a_k3` | [paper] |

```
(11) q_k,KT = ( Σ_{i=1..N_k,KT} |{ y ∈ Y_k,g : y_k,K,i < y }| ) / (N_k,KT · N)     if N_k,KT > 0
            = 0                                                                     otherwise
```

`Y_k,g` is the **parent** fitness at generation g; `y_k,K,i` is the fitness of the i-th KT-generated offspring. Equivalently, `q_k,KT` is the mean of the score `s(·)` of Eq. (17) over KT offspring.

All features are normalised to [0, 1] **[paper]**. State vector = `concat(O_c, O_t,1, O_t,2)`.

Open points: stagnation counter semantics (consecutive generations without best-fitness improvement is the natural reading **[inferred]**); initial values at g = 0 for the history features (`q`, flag, previous action) — zeros suggested **[unspecified]**; `G_max` in the normalisers during training is presumably `G_roll` **[inferred]**.

---

## 7. Reward (learning stage only) [paper]

```
(12) r   = Σ_k r_k
(13) r_k = b1·r_k,conv + b2·r_k,KT + b3·I( f_k(x*_k,g) − f*_k < ξ )

(14) r_k,conv = −( f_k(x*_k,g) − f*_k ) / ( f_k(x*_k,1) − f*_k )      # in [−1, 0]; 0 at optimum

(16) r_k,KT ≈ (1/N_K)·Σ_j s(y_k,K,j) − (1/N_A)·Σ_j s(y_k,A,j)         # KT offspring vs base-solver offspring
(17) s(y)   = |{ y' ∈ Y_k,g : y < y' }| / N                            # fraction of parents beaten, in [0,1]
```

- `r_k,KT > 0` when KT offspring are on average better (relative to the parent population) than base-solver offspring generated in the same generation; negative under negative transfer. Eq. (15) in the paper is the exact expectation form this approximates.
- `N_K = N_k,KT`, `N_A = N − N_k,KT`. When `N_k,KT = 0`, set `r_k,KT = 0` **[inferred]**.
- Defaults: `b1 = 1`, `b2 = 10`, `b3 = G_roll = 100`, `ξ = 1e-8`.
- Eq. (13) as written adds the `b3` indicator at every step once the accuracy is reached; rollouts always run the full `G_roll` generations (no early termination) **[paper]**.
- If `f*` is unknown for training tasks, approximate it from a set of previously evaluated solutions **[paper]**.
- Sensitivity: `b1`, `b2` matter; `b3` does not. With `b1 = 1`, `b2 = 10` was best at `G_roll` and `b2 = 5` at `G_max` on BBOB. For a new problem family, grid-search `b2` (the HPO experiment used `{0.01, 0.1, 1, 10, 100}`). Ablation: `r_KT` contributes more than `r_conv`.

---

## 8. Agent and PPO

### 8.1 Networks [paper, Table S.IV]

- Actor `φ(s; θ)`: MLP, input 4 + 7K = 18, two hidden layers of 64 units, `tanh`, linear output of size 3K = 6 = `[a_11, a_12, a_13, a_21, a_22, a_23]`.
- Critic `ψ(s; ω)`: same architecture, separate parameters, scalar output.
- Policy: Gaussian with mean `φ(s; θ)` and std `σ`, Eq. (18).

Not stated: how `σ` is parameterised, and how samples are mapped into [0, 1]. Suggested: state-independent learnable log-std and clipping the sampled action to [0, 1] **[unspecified]**.

### 8.2 PPO objective [paper]

```
(19) L_π(θ) = E_t[ min( ρ_t·Â_t , clip(ρ_t, 1−ε, 1+ε)·Â_t ) ],   ρ_t = π(a_t|s_t; θ) / π(a_t|s_t; θ')
     Â_t    = Σ_{l=0..T−t−1} (γλ)^l · δ_{t+l},   δ_t = r_t + γ·ψ(s_{t+1}; ω') − ψ(s_t; ω')
(20) L_ψ(ω) = −Σ_t ( Â_t + ψ(s_t; ω') − ψ(s_t; ω) )²
```

`θ', ω'` are the parameters used to collect the current buffer. Both objectives are maximised by gradient ascent with step size `η`. The paper's printed GAE omits the `γ` in front of `ψ(s_{t+1})`; the standard form above is assumed **[inferred]**.

`γ = 0.99`, `λ = 0.95`, `ε = 0.2`.

### 8.3 Training loop sizes [paper, Table S.IV]

| Item | Value |
|---|---|
| Parallel environments `N_env` | 20 |
| Buffer per PPO update `N_buff` | 2048 × `N_env` = 40,960 steps |
| Total time steps `T` | 5e6 for BBOB_learn; 2e6 for each of VS, S, M, L, VL, C1–C5 |
| Initial population set size `N_P` | 10 (Latin hypercube samples, generated once) |

One time step = one generation of the 2-task EMT (100 function evaluations). So 5e6 steps = 50,000 episodes ≈ 5e8 evaluations and ≈ 122 PPO updates; 2e6 steps ≈ 49 updates.

Learning rate, epochs per update, minibatch size, entropy / value coefficients and gradient clipping are **[unspecified]**. The buffer written as `2048 * N_env` and the 2×64 tanh networks match Stable-Baselines3 PPO defaults (`n_steps=2048`, lr 3e-4, 10 epochs, batch 64), which is a reasonable choice **[inferred]**.

"Record the best-found agent parameter θ*" — the selection criterion is **[unspecified]**; mean episode return on training or held-out instances is the obvious choice.

---

## 9. Algorithms

### 9.1 Algorithm 1 — rollout with DE base solver [paper]

```
Input: task pair {f1, f2}, G_roll, agent π(a|s; θ), initial population set P = {P_1..P_NP}, N
 1  D ← ∅
 2  For each task: X_k ← a population picked at random from P          # pseudo-random init
 3  Evaluate → Y_k
 4  s ← concat(O_c, O_t)
 5  g ← 0
 6  while g < G_roll:
 7      a ~ π(a | s; θ)
 8      for each task k:
 9          (a_k1, a_k2, a_k3) ← a
10          U ← N offspring by DE, Eqs. (1)–(2)
11          N_k,KT ← ceil(0.5 · a_k1 · N)
12          I_KT ← N_k,KT random indices from {1..N}
13          for j in I_KT:
14              v_k,j ← Eq. (10) with a_k2, a_k3
15              u_k,j ← binomial crossover of v_k,j with X_k,j, Eq. (2)
16              U[j] ← u_k,j
18          Y_U ← evaluate U on f_k
19          compute task-specific features O_t,k
20          compute reward r_k, Eq. (13)
21          X_k ← selection(X_k, U)
23      r ← r_1 + r_2
24      compute common features O_c
25      s ← concat(O_c, O_t)
26      D ← D ∪ (s, a, r)
27      g ← g + 1
```

Notes:
- As printed, task 2's KT step uses task 1's already-updated population (selection is inside the per-task loop). A start-of-generation snapshot of both populations is the symmetric alternative.
- As printed, line 26 stores the updated state with the action; store standard transitions `(s_t, a_t, r_t, s_{t+1})`.
- Boundary handling is not stated; clip to [0, 1] in unified space **[unspecified]**.

### 9.2 Algorithm 2 — PPO learning [paper]

```
Input: task instance set Θ, base solver A, G_roll, T, N_env, N_P, N_buff
 1  P ← N_P populations by independent Latin hypercube sampling
 2  init actor θ, critic ω
 3  t ← 0
 4  while t < T:
 5      D ← ∅
 6      while |D| < N_buff:
 7          sample N_env MTOP instances from Θ
 8          run Algorithm 1 on each in parallel workers with current θ
 9          D ← D ∪ (merged worker buffers)
11          t ← t + N_env · G_roll
13      θ ← θ + η ∇_θ L_π
14      ω ← ω + η ∇_ω L_ψ
15      record best-found θ*
```

### 9.3 Algorithm S.2 — MTDE-L2T (inference) [paper]

Same loop as Algorithm 1 for `G_max` generations with these differences: action is `a = π(s | θ*)` ("predict action by actor network"); no reward and no data buffer; track and return `x*_1, x*_2`. Initial populations are still drawn from the pre-generated set `P`. `G_max` may differ from `G_roll` (250 vs 100 in the experiments); the time feature uses `g / G_max`.

Whether inference uses the Gaussian mean or a sample is not explicit; "predict" suggests the mean **[inferred]**.

### 9.4 Algorithms S.1 / S.3 — GA base solver (MTGA-L2T), MFEA-style [paper]

```
per generation:
    a ← agent(s)
    P_m ← X_1 ∪ X_2                                  # merged parent pool
    U_1, U_2 ← ∅
    while each task has fewer than N offspring:
        draw two parents p_a, p_b from P_m without replacement; k_a, k_b their tasks
        if k_a == k_b:
            crossover + mutation on (p_a, p_b) → u_a, u_b for that task
        elif rand < a_{k_a,1}:                        # inter-task pair → KT
            u_a ← Eq. (10) with a_{k_a,2}, a_{k_a,3}
            u_b ← mutation of p_b
        else:
            u_a, u_b ← mutation of p_a, p_b
        add u_a, u_b to the offspring set of their tasks
    evaluate U_1 on f_1, U_2 on f_2
    per task: features, reward (training only), selection
```

- Operators: SBX with `η_c = 2`, polynomial mutation with `η_m = 5`.
- The KT test here is `rand < a_k1` on cross-task pairs only (roughly half of all pairs), which gives about the same overall `0.5·a_k1` KT share **[inferred]**.
- Not stated for GA: the `F` used in Eq. (10), whether crossover/mutation follows the KT vector, and the selection scheme (MFEA-style elitist truncation of parents ∪ offspring per task is the natural reading) **[unspecified]**.
- Algorithm S.1's loop bound is printed as `G_max`; `G_roll` is meant for training.

### 9.5 Complexity of L2T overhead per generation [paper]

Feature extraction `O(K·N² + K·N·D)`, action prediction `O(K)`, KT offspring `O(K·N·D)`; total `O(K·N² + K·N·D)`.

---

## 10. Hyperparameters in one place [paper, Table S.IV]

| Parameter | Value |
|---|---|
| Tasks per MTOP `K` | 2 |
| Population size per task `N` | 50 |
| DE | DE/rand/1 + binomial crossover, `F = 0.5`, `CR = 0.5` |
| GA | SBX (`η_c = 2`) + polynomial mutation (`η_m = 5`) |
| `G_roll` (training rollout length) | 100 |
| `G_max` (test horizon) | 250 |
| Target accuracy `ξ` | 1e-8 |
| Reward weights | `b1 = 1`, `b2 = 10`, `b3 = G_roll = 100` |
| Actor / critic | 2 hidden layers × 64, tanh, linear output |
| PPO | `γ = 0.99`, `λ = 0.95`, `ε = 0.2` |
| `N_P` | 10 |
| `N_env` | 20 |
| `N_buff` | 2048 × 20 = 40,960 |
| `T` | 5e6 (BBOB_learn), 2e6 (each CEC17-based set) |
| Test set size | 100 MTOP instances per set |
| Independent runs per instance | 20 |

---

## 11. Benchmarks

### 11.1 Suite 1 — CEC17MTOP functions with controlled optimum distributions [paper, Table S.II]

- Function set `F = {Ackley, Griewank, Rastrigin, Sphere, Weierstrass}`.
- Each function's box is normalised to `[0,1]^D`; the shift `x_O` is defined in `[0,1]^D`.
- An MTOP instance = 2 tasks, each a function from `F` with a shift drawn from the set's distribution.
- Ten sets; one agent is trained **and** tested per set (train and test instances i.i.d. from the same distribution).

| Set | Optimum distribution `p(x_O)` |
|---|---|
| VS | `U[0.5 − Δ, 0.5 + Δ]^D`, `Δ = 0.025` |
| S | same, `Δ = 0.05` |
| M | same, `Δ = 0.1` |
| L | same, `Δ = 0.2` |
| VL | same, `Δ = 0.4` |
| C1 | `x_c,1 + L·(x_c,2 − x_c,1)`, `L ~ U[0,1]`, `x_c,1, x_c,2 ∈ [0,1]^D` (a line segment) |
| C2 | mixture of `C = 2` uniform boxes `U[x_c,i − Δ_i, x_c,i + Δ_i]^D`, equal weights `1/C` |
| C3 | same, `C = 3` |
| C4 | same, `C = 4` |
| C5 | same, `C = 5` |

Not given: cluster centres `x_c,i` and radii `Δ_i`, whether/how `M` is used for this suite, how the function class is sampled (uniform assumed), and `D` **[unspecified]**.

Standard definitions (shifted/rotated variable `z`, all with `f* = 0` at `z = 0`) and native boxes from the CEC17 MTO benchmark report — not restated in this paper, verify against Da et al., arXiv:1706.03470:

```
Sphere      Σ z_i²                                                     [−100, 100]
Rastrigin   Σ ( z_i² − 10·cos(2π z_i) + 10 )                           [−50, 50]
Ackley      −20·exp(−0.2·sqrt(mean z_i²)) − exp(mean cos(2π z_i)) + 20 + e   [−50, 50]
Griewank    1 + Σ z_i²/4000 − Π cos(z_i / sqrt(i))                     [−100, 100]
Weierstrass Σ_i Σ_{k=0..20} 0.5^k cos(2π 3^k (z_i + 0.5)) − D Σ_{k=0..20} 0.5^k cos(π 3^k)   [−0.5, 0.5]
```

A consistent way to apply the shift **[inferred]**: decode `y = L + x·(U − L)` and `o = L + x_O·(U − L)`, evaluate `f(M·(y − o))`.

### 11.2 Suite 2 — BBOB [paper, Table S.III]

- 24 BBOB functions; search box `[−5, 5]^D`; optima mostly uniform in `[−4, 4]^D`.
- A task instance is `(fid, sid)`: function ID and the seed ID that generates `(M, x_O)`.
- One agent per base solver is trained on BBOB_learn (`T = 5e6`) and tested on BBOB1–BBOB15.
- An MTOP instance = 2 random `(fid, sid)` pairs from the set's `F × S`.

| Purpose | Set | Function IDs `F` | Seeds `S` |
|---|---|---|---|
| Learning | BBOB_learn | {1, 3, 8, 10, 16, 20} | [1, 100] |
| Test, near-i.i.d. | BBOB1 | {1, 3, 8, 10, 16, 20} | [500, 1500] |
| | BBOB2 | {1, 3, 8, 10, 16, 20} | [1000, 1005] |
| | BBOB3 | {1} | [500, 1500] |
| | BBOB4 | {3} | [500, 1500] |
| | BBOB5 | {8} | [500, 1500] |
| | BBOB6 | {10} | [500, 1500] |
| | BBOB7 | {16} | [500, 1500] |
| | BBOB8 | {20} | [500, 1500] |
| Test, o.o.d. | BBOB9 | the other 18 functions: {1..24} \ {1, 3, 8, 10, 16, 20} | [500, 1500] |
| | BBOB10 | {2, 6, 12, 15, 21} | [500, 1500] |
| | BBOB11 | {2} | [500, 1500] |
| | BBOB12 | {6} | [500, 1500] |
| | BBOB13 | {12} | [500, 1500] |
| | BBOB14 | {15} | [500, 1500] |
| | BBOB15 | {21} | [500, 1500] |

BBOB groups: separable f1–f5; low/moderate conditioning f6–f9; high conditioning unimodal f10–f14; multimodal with adequate global structure f15–f19; multimodal with weak global structure f20–f24.

Treating `sid` as the BBOB instance number (which seeds the rotation and shift) in a library such as COCO or IOHexperimenter is the natural mapping **[inferred]**. Both give `f*` per instance, which the reward needs.

### 11.3 Real-world suite — hyperparameter optimisation [paper, §IV-I]

- Tasks from the meta-surrogate HPO benchmark of Klein et al. (NeurIPS 2019, the paper's ref [48]) on OpenML datasets: SVM (`D_H = 2`), XGBoost (`D_H = 8`), FCNet (`D_H = 6`). These dimensions match the "Profet" benchmark families — identification is mine, verify.
- An MTOP instance = 2 random tasks (datasets) from one family; 100 instances per family.
- Agent trained from scratch per family, tested on unseen instances; `G_roll = G_max = 100`.
- `b2` grid `{0.01, 0.1, 1, 10, 100}`; best `b2 = 0.01` for MTDE-L2T and `b2 = 10` for MTGA-L2T.
- Target accuracy `ξ`: SVM 1, XGBoost 0.1, FCNet 0.1.

---

## 12. Experiments and evaluation protocol [paper]

**Experiment map**

| Section | Training | Testing | Results |
|---|---|---|---|
| IV-B effectiveness | from scratch per set: 10 CEC17-based sets + BBOB9, BBOB10 (DE) | unseen instances, same distribution | Table S.V, Fig. 3 |
| IV-C adaptability | once on BBOB_learn (DE and GA agents, `T = 5e6`) | BBOB1–BBOB15 at `G_roll` and `G_max` | Table III, S.VI, S.VII, S.VIII |
| IV-D transferability | BBOB_learn agent fine-tuned on each of the 12 sets | same 12 sets | Table S.IX, Fig. 4 |
| IV-E components | fixed/random agents; ablated variants trained per set | 14 sets | Tables S.X, S.XI |
| IV-G sensitivity | `b2 ∈ {0.1, 0.5, 1, 5, 10, 50, 100}`, `b1 = 1` | BBOB1–15, mean standard score → average rank | Tables S.XII, S.XIII |
| IV-H further | SAC / TD3 instead of PPO; JADE (no archive) as base solver; overhead timing | BBOB / 12 sets | Tables S.XIV, S.XV, Fig. S.1 |
| IV-I real-world | from scratch per HPO family | unseen HPO pairs | Table S.XVI |

**Per-instance comparison (Definition 1).** For each MTOP instance and each task, collect the best-found fitness at generation `g` over `R = 20` runs for the target algorithm `A_t` and the baseline `A_b`; apply a Wilcoxon rank-sum test at `α = 0.05` per task.

- **Win** (`A_t < A_b`): `A_t` is significantly better or statistically equal on **every** task (and not equal on all).
- **Tie**: statistically equal on all tasks.
- **Lose**: anything else — including better on one task and worse on the other.

Report `W/T/L` counts over the 100 instances of a set, at `g = G_roll` and `g = G_max`. The `(+)/(−)/(=)` suffix in the tables is W > L, W < L, W = L **[inferred]** (consistent with every entry).

**Positive transfer rate (Eq. 21).** Fraction of the 100 instances on which an EMT algorithm wins (Definition 1) against single-task DE (STDE).

---

## 13. Baselines, fixed agents and ablations [paper]

**DE-based implicit EMT:** AEMTO, MFDE, MKTDE, MTDE-AD, MTDE-B. **GA-based implicit EMT:** MFEA, MFEA-II (MFEA2), G-MFEA, MFEA-AKT, MTEA-AD. **Explicit EMT:** MTDE-EA (explicit autoencoding), AT-MFEA. **RL-based EMT:** MFEA-RL. **Single-task:** STDE, STJADE. All comparisons keep the base solver identical.

**Trivial agents (cheap to implement, good first sanity check):**
- `MTDE-f(x, y, z)`: constant action `a1 = x, a2 = y, a3 = z` for both tasks. Tested: (.5,0,1), (.5,1,0), (.5,1,1), (1,0,1), (1,1,0), (1,1,1).
- `MTDE-r`: uniform random action in `[0,1]^3` per task per generation.
- `STDE`: no KT.

**Ablations** (each retrained):
- `L2T-w/o-a1`, `-a2`, `-a3`: actor outputs 2K values; the removed action is fixed to `a1 = 0.5`, `a2 = 0`, `a3 = 0` respectively.
- `L2T-w/o-Oc`, `L2T-w/o-Ot`: drop common or task-specific features (input size shrinks).
- `L2T-w/o-FE`: feed raw populations and fitness (`K·N·(D+1)` inputs) instead of features.
- `L2T-w/o-rconv`, `L2T-w/o-rKT`: drop one reward term.

---

## 14. Reference results for sanity-checking a reimplementation

W/T/L of MTDE-L2T (rows) against the named algorithm, 100 instances per set.

**Learned agent vs trivial agents and STDE at `G_max` (Table S.X, selection):**

| Set | f(.5,0,1) | f(1,1,0) | MTDE-r | STDE |
|---|---|---|---|---|
| VS | 76/6/18 | 100/0/0 | 92/6/2 | 95/5/0 |
| M | 67/9/24 | 100/0/0 | 90/8/2 | 72/25/3 |
| VL | 28/38/34 | 81/4/15 | 43/23/34 | 7/91/2 |
| C5 | 26/35/39 | 80/8/12 | 51/32/17 | 5/93/2 |
| BBOB1 | 37/57/6 | 92/7/1 | 76/21/3 | 31/58/11 |
| BBOB9 | 36/49/15 | 72/21/7 | 51/29/20 | 29/59/12 |
| BBOB10 | 35/42/23 | 72/15/13 | 62/18/20 | 36/34/30 |

Reading: gains over STDE are large when optima are close (VS, S, M) and shrink to mostly ties when they are far apart (VL, C5) — the agent learns to transfer little there. Heavy base-vector transfer `f(1,1,0)` is badly beaten everywhere; the differential-vector-only actions `f(.5,0,1)` (MFDE-like) and `f(1,0,1)` are the hardest fixed actions to beat, and win more often than they lose against the learned agent on VL, C2, C3 and C5.

**Agents trained per set vs DE baselines at `G_roll` (Table S.V, selection):**

| Set | AEMTO | MFDE | MKTDE | MTDE-AD | MTDE-B |
|---|---|---|---|---|---|
| VS | 100/0/0 | 100/0/0 | 86/7/7 | 99/1/0 | 97/3/0 |
| M | 87/13/0 | 72/15/13 | 50/17/33 | 84/14/2 | 82/16/2 |
| VL | 50/48/2 | 31/47/22 | 35/34/31 | 45/54/1 | 2/92/6 |
| C2 | 47/48/5 | 51/33/16 | 32/26/42 | 34/59/7 | 16/75/9 |

MKTDE is the strongest baseline; MTDE-B mostly ties on wide distributions.

**BBOB_learn agent on BBOB sets at `G_roll` (Table III of the paper, selection):**

| Set | AEMTO | MFDE | MKTDE | MTDE-AD | MTDE-B |
|---|---|---|---|---|---|
| BBOB1 | 56/43/1 | 63/30/7 | 88/10/2 | 53/46/1 | 36/59/5 |
| BBOB9 | 42/57/1 | 51/35/14 | 67/23/10 | 45/52/3 | 18/66/16 |
| BBOB10 | 51/33/16 | 55/36/9 | 62/29/9 | 54/35/11 | 27/48/25 |

**Other qualitative findings**
- The advantage at `G_max = 250` is smaller than at `G_roll = 100` but still present (generalisation to a longer horizon than trained on).
- Fine-tuning the BBOB_learn agent on a new set beats training from scratch and converges faster.
- RL algorithm: PPO agent beats SAC-trained; TD3-trained beats PPO on most BBOB sets.
- MTJADE-L2T vs STJADE: clear wins on VS/S/M, mostly ties elsewhere — a strong adaptive base solver leaves less room for KT.
- Overhead: MTDE-L2T has the highest per-solution overhead among DE-based EMT; it drops to ≤ 10% of evaluation cost once one evaluation takes more than about 1 ms. The authors recommend L2T when an evaluation costs more than 0.1 ms.
- Training curves (Fig. 4, values read off small plots, approximate): mean episode return starts around −100 to −140 and climbs to roughly −40 to −60 within 2e6 steps when training from scratch on BBOB9, BBOB10, VL, C1, C5; the fine-tuned agent starts near that plateau.
- Learned behaviour on two BBOB1 instances (Fig. 5). MTOP15 (f1 sphere + f20, highly multimodal): one-way easy-to-hard transfer — the multimodal task takes `a1` near 1 while the unimodal task keeps all actions ≈ 0; with distant optima `a3 > a2` (differential-vector transfer, `a2 ≈ 0`). The paper's sentence describing this mixes up the task labels; the description here follows the figure. MTOP23 (f10 + f1, both unimodal): actions vary with the evolution state, and `a2` is high in early generations, when the uniformly initialised populations overlap, then decays as they converge to different optima.

---

## 15. Gaps, inconsistencies and suggested resolutions

| # | Issue | Suggested resolution |
|---|---|---|
| 1 | KT quota printed as `⌈0.5·a_k1⌉` (no `N`) | Use `ceil(0.5·a_k1·N)` |
| 2 | Problem dimension `D` never stated | CEC17 MTO functions are normally 50-D (Weierstrass 25-D in some pairs); pick one `D` for both tasks and keep it fixed between training and testing |
| 3 | Buffer collection: Alg. 2 gathers whole 100-step episodes from 20 workers until at least 40,960 steps (i.e. 21 batches = 42,000 steps); a fixed 2048 steps per environment gives exactly 40,960 but cuts episodes mid-way | Either; treat the end of a rollout at `G_roll` as a time-limit truncation when computing advantages |
| 4 | Gaussian policy std and mapping of samples to [0, 1] | Learnable state-independent log-std; clip actions to [0, 1] |
| 5 | PPO learning rate, epochs, minibatch, entropy/value coefficients | SB3 defaults (3e-4, 10, 64, 0, 0.5) |
| 6 | GAE printed without `γ` on `ψ(s_{t+1})` | Standard GAE |
| 7 | DE selection scheme | One-to-one greedy replacement |
| 8 | Distinctness of `r1..r6` in Eq. (10) | Distinct within each population |
| 9 | Boundary handling | Clip to [0, 1] in unified space |
| 10 | `b3` indicator: every step or once | Follow Eq. (13): every step where the accuracy holds |
| 11 | `r_KT` when `N_k,KT = 0` | 0 |
| 12 | Initial values of history features at g = 0 | Zeros |
| 13 | Features computed before or after selection (Alg. 1 computes `O_t,k` before selection, `O_c` after) | Compute `q_KT` and reward against the parent population, then all distribution features after selection |
| 14 | Task 2 sees task 1's updated population within a generation | Either is defensible; snapshot both populations for symmetry, or follow the printed order |
| 15 | Sample vs mean action at inference | Mean (deterministic) |
| 16 | `G_max` used in feature normalisers during training | `G_roll` |
| 17 | Criterion for "best-found θ*" | Highest mean episode return on a fixed validation batch |
| 18 | C1–C5 cluster centres and radii; rotation in suite 1; function sampling | Fix centres/radii with a seed and document them; identity or seeded random rotation; uniform over `F` |
| 19 | BBOB `sid` → instance mapping | Use `sid` as the BBOB instance number |
| 20 | GA variant: `F` in Eq. (10), post-KT variation, selection | `F = 0.5`; no extra variation on the KT child; per-task elitist truncation |
| 21 | Training budget `T` for from-scratch BBOB9/BBOB10, HPO, and for fine-tuning | 2e6 as for the other from-scratch sets; fine-tune until the return plateaus |
| 22 | Text says "CEC19MTOP" in §IV-D and S.III-B where the ten CEC17-based sets are meant; "M2DE-L2T" for MTDE-L2T; Alg. S.1 loops to `G_max` | Typos; ignore |
| 23 | Alg. 1 line 26 stores the post-update state with the action | Store `(s_t, a_t, r_t, s_{t+1})` |

---

## 16. Suggested structure and skeleton

```
l2t/
  problems/      cec17_shifted.py, bbob.py, mtop.py      # task(x_unified) -> f, with .fopt; pair samplers per set
  solvers/       de.py, ga.py, kt.py                     # Eqs (1), (2), (10); SBX/PM
  env/           features.py, reward.py, l2t_de_env.py, l2t_ga_env.py
  train.py       # PPO, 20 parallel envs
  mtde_l2t.py    # inference loop (Algorithm S.2)
  eval/          wilcoxon_wtl.py, baselines.py           # Definition 1, fixed/random agents, STDE
```

Core pieces (sketch, K = 2, minimisation, unified `[0,1]^D` space):

```python
def kt_trial(Xk, Xj, a2, a3, F, rng):                                  # Eq. (10)
    N = len(Xk)
    r1, r3, r4 = rng.choice(N, 3, replace=False)                        # target population
    r2, r5, r6 = rng.choice(N, 3, replace=False)                        # source population
    return ((1 - a2) * Xk[r1] + a2 * Xj[r2]
            + F * (1 - a3) * (Xk[r3] - Xk[r4])
            + F * a3 * (Xj[r5] - Xj[r6]))

def generation(k, X, Y, a, task, st, rng, N=50, F=0.5, CR=0.5, b=(1, 10, 100), xi=1e-8):
    j = 1 - k
    a1, a2, a3 = a[k]
    U = de_rand_1_bin(X[k], F, CR, rng)                                 # Eqs. (1)-(2)
    n_kt = int(np.ceil(0.5 * a1 * N))                                   # KT quota
    is_kt = np.zeros(N, bool)
    is_kt[rng.choice(N, n_kt, replace=False)] = True
    for i in np.flatnonzero(is_kt):
        v = kt_trial(X[k], X[j], a2, a3, F, rng)
        U[i] = binomial_crossover(X[k][i], v, CR, rng)                  # Eq. (2)
    U = np.clip(U, 0.0, 1.0)
    YU = task(U)

    score = (YU[:, None] < Y[k][None, :]).mean(axis=1)                  # Eq. (17), vs parents
    q_kt = score[is_kt].mean() if n_kt else 0.0                         # Eq. (11)
    r_kt = q_kt - score[~is_kt].mean() if n_kt else 0.0                 # Eq. (16)

    better = YU <= Y[k]                                                 # one-to-one selection
    X[k][better], Y[k][better] = U[better], YU[better]

    improved = Y[k].min() < st.best[k]
    st.best[k] = min(st.best[k], Y[k].min())
    st.stag[k] = 0 if improved else st.stag[k] + 1
    err = st.best[k] - task.fopt
    r_conv = -err / st.err0[k]                                          # Eq. (14); err0 = error of initial best
    r_k = b[0] * r_conv + b[1] * r_kt + b[2] * float(err < xi)          # Eq. (13), training only
    return r_k, q_kt, float(improved)

def observation(X, best_x, st, g, G, D):                                # 4 + 7*2 = 18 dims
    mu = [x.mean(0) for x in X]; sd = [x.std(0) for x in X]
    oc = [g / G,
          np.linalg.norm(best_x[0] - best_x[1]) / np.sqrt(D),
          np.linalg.norm(mu[0] - mu[1]) / np.sqrt(D),
          np.linalg.norm(sd[0] - sd[1]) / np.sqrt(0.5 * D)]
    ot = [[st.stag[k] / G, st.improved[k], st.q[k], sd[k].mean(), *st.prev_a[k]] for k in (0, 1)]
    return np.clip(np.concatenate([oc, *ot]), 0, 1).astype(np.float32)
```

Environment contract: `observation_space = Box(0, 1, (18,))`, `action_space = Box(0, 1, (6,))`; `reset()` samples a new task pair and picks each task's initial population from the 10 pre-generated LHS populations; `step()` runs one generation for both tasks and returns `r_1 + r_2`; the episode is truncated at `G_roll` steps.

PPO setup consistent with Table S.IV (library choice is an assumption, see §8.3):

```python
from stable_baselines3 import PPO
import torch

model = PPO("MlpPolicy", vec_env_with_20_envs,
            n_steps=2048, gamma=0.99, gae_lambda=0.95, clip_range=0.2,
            policy_kwargs=dict(net_arch=dict(pi=[64, 64], vf=[64, 64]),
                               activation_fn=torch.nn.Tanh))
model.learn(total_timesteps=5_000_000)        # 2_000_000 for the CEC17-based sets
```

**Suggested build order**
1. Problems + STDE; confirm STDE converges on single tasks.
2. Eq. (10) KT step with fixed actions; reproduce the qualitative ordering of the `MTDE-f` agents vs STDE on VS and VL (§14).
3. Features and reward; unit-test corner cases (`a1 = 0`, all KT offspring worse/better than all parents, optimum reached).
4. PPO on VS (largest KT gains, `T = 2e6`); check the learned agent beats `MTDE-f(.5,0,1)`, `MTDE-r` and STDE under Definition 1.
5. BBOB_learn training (`T = 5e6`) and BBOB1–15 evaluation at `G_roll` and `G_max`.
6. GA variant, fine-tuning, baselines as needed.

---

## 17. References needed to reproduce baselines and benchmarks

Numbers are the main paper's reference list.

| Item | Reference |
|---|---|
| MFEA | [3] Gupta, Ong, Feng, IEEE TEVC 20(3), 2016 |
| MFEA-II | [23] Bali, Ong, Gupta, Tan, IEEE TEVC 24(1), 2020 |
| MFEA-AKT | [18] Zhou et al., IEEE TCYB 51(5), 2021 |
| G-MFEA | [42] Ding, Yang, Jin, Chai, IEEE TEVC 23(1), 2019 |
| MTEA-AD / MTDE-AD | [29] Wang, Liu, Wu, Wu, IEEE TEVC 26(2), 2022 |
| AEMTO | [26] Xu, Qin, Xia, IEEE TEVC 26(2), 2021 |
| MFDE | [27] Feng et al., IEEE CEC 2017 |
| MTDE-B | [28] Jin, Tsai, Qin, IEEE CEC 2019 |
| MKTDE | [21] Li, Zhan, Tan, Zhang, IEEE TEVC 26(4), 2022 |
| MTDE-EA (explicit autoencoding) | [13] Feng et al., IEEE TCYB 49(9), 2019 |
| AT-MFEA | [43] Xue et al., IEEE TCYB 52(7), 2022 |
| MFEA-RL | [36] Li, Gong, Wang, Gu, IEEE TETCI 8(1), 2024 |
| JADE | [46] Zhang, Sanderson, IEEE TEVC 13(5), 2009 |
| PPO / SAC / TD3 | [20] arXiv:1707.06347 / [44] ICML 2018 / [45] ICML 2018 |
| CEC17 MTO benchmark, mean standard score | [40] Da et al., arXiv:1706.03470 |
| BBOB / COCO | [41] Hansen et al., Optim. Methods Softw. 36(1), 2021 |
| HPO meta-surrogate benchmark | [48] Klein et al., NeurIPS 2019; datasets from OpenML [51] |
| Unified-space normalisation used for suite 1 | Wu, Zhan, Tan, Zhang, "Orthogonal transfer for multitask optimization," IEEE TEVC 27(1), 2023 (supplement ref [16]) |
