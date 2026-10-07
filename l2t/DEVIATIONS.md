# L2T reproduction: decisions and their sources

This package reimplements L2T (Wu et al., "Learning to Transfer for Evolutionary
Multitasking", IEEE TCYB 55(7):3342–3355, 2025, plus supplement) starting from MetaBox's
L2T code. Every choice below is tagged:

- **paper**: stated in the paper or supplement
- **MetaBox**: the paper is silent, so MetaBox's choice is kept
- **chosen**: the paper is silent and MetaBox has no equivalent, or MetaBox's choice
  doesn't fit; the reason is given

## 1. Changes from MetaBox's L2T (`src/environment/optimizer/l2t_optimizer.py`, `src/baseline/metabbo/l2t.py`)

| Item | MetaBox | Here | Source |
|---|---|---|---|
| State | 1 + 7K: no O_c2–O_c4 | 4 + 7K = 18, Table II | paper |
| O_t,k,4 = mean(σ_k) | std over each individual's dimensions (`axis=-1`) | std of each dimension over the population (`axis=0`) | paper |
| Stagnation | counts every non-improving generation | consecutive non-improving generations | paper (Table II: "stagnating generations") |
| Actions | task 0's (a1, a2, a3) applied to every task | each task uses its own three actions | paper (Eq. 9–10) |
| KT quota | ceil(0.5·a1·N), at least 1 | ceil(0.5·a1·N), may be 0 | paper (Eq. 11 has an N_KT = 0 case) |
| q_KT | KT offspring that beat their own parent / N_KT | Eq. 11: mean fraction of the parent population each KT offspring beats | paper |
| r_KT | (non-KT successes − KT successes) / N (sign reversed) | Eq. 16–17: mean score of KT offspring − mean score of DE offspring | paper |
| r_conv | +err / err₀ | −err / err₀ (Eq. 14) | paper |
| b3 | 250 | G_roll = 100 | paper (Table S.IV) |
| Reward returned to the agent | running total over the episode | per-step reward r_t | paper (Eq. 12) |
| DE crossover rate | 0.7 | 0.5 | paper (Table S.IV) |
| DE selection | one-to-one (offspring i replaces parent i if not worse) | truncation: best N of parents ∪ offspring | paper (Section III-B2: α_k ≥ 0.5 guarantees convergence "under truncated selection mechanism"); see §7 |
| KT trial vectors | not clipped | clipped to [0, 1] like MetaBox's DE mutants | chosen |
| Evaluations | parents re-evaluated every generation, `fes` undercounted 3× | each solution evaluated once | chosen |
| Initial populations | fresh uniform random | one of N_P = 10 pre-generated LHS populations per task | paper (Alg. 2, line 1) |
| Horizon | fixed 250 | G_roll = 100 in training, G_max = 250 at test | paper |
| Normaliser of O_c1 = g/G_max and O_t,k,1 = n_stag/G_max | 250 | G_roll = 100 in training and at test (features capped at 1 after generation 100); G_max at test is also evaluated, stored as `<label>-gmaxnorm` | chosen, see §8 |
| Actor mean | (tanh + 1)/2 | linear output layer | paper (Table S.IV) |
| Actor σ | state-dependent head in [0.05, 0.15] | one learnable log σ per action dim, initialised to 0 | paper (Eq. 18: σ doesn't depend on s); init from SB3, see §3 |
| RL algorithm | 10-step returns, 3 epochs, no GAE, lr 1e-4 | PPO with GAE(γ = 0.99, λ = 0.95), ε = 0.2, buffer 2048·N_env, N_env = 20 | paper (Eq. 19–20, Table S.IV) |
| θ* | last checkpoint | the policy whose collected buffer had the highest mean episode return (the quantity of Fig. 4) | chosen (paper: "record the best-found agent", criterion not given), see §8 |
| Benchmarks | 9 fixed CEC17MTO pairs, WCCI2020 | the paper's ten CEC17-based sets and 16 BBOB-based sets | paper |

Kept from MetaBox: the 2×64 tanh actor/critic layout, DE/rand/1 with r1, r2, r3 ≠ i and
mutants clipped to [0, 1], drawing the six indices of
Eq. 10 without replacement, r_conv's denominator using the best of the initial
population, the CEC17 and BBOB base functions and the BBOB instance generator.

## 2. Problem settings

| Item | Choice | Source |
|---|---|---|
| D | 10 for both suites | chosen with the user; the paper gives no D. 50 was tried first for the CEC17 suite, see §9 |
| Unified space | each task encoded to [0, 1]^D with its own bounds | paper (S.III) |
| Suite 1 functions | Ackley, Griewank, Rastrigin, Sphere, Weierstrass (MetaBox's CEC17MTO classes) | paper |
| Suite 1 optimum | x_O in [0, 1]^D, decoded with the function's bounds | paper / Wu et al. 2023 (supplement ref. 16) |
| Suite 1 rotation | none (identity) | chosen; Table S.II only varies x_O |
| Suite 1 function choice | uniform over the five functions, independently per task | chosen |
| C1–C5 centres and radii | centres uniform in [0.1, 0.9]^D, fixed per set by seed 2025 + C; radius 0.05 | chosen; Table S.II leaves them open |
| BBOB instance (fid, sid) | MetaBox's generator seeded with sid: shift U[−4, 4]^D, Householder rotation, bias in {100, …, 2500} | paper (S.III-B) + MetaBox |
| f* for BBOB | MetaBox's `optimum` (= bias; checked numerically for all 24 functions) | MetaBox |
| Test sets | 100 MTOP instances per set, fixed seed; 20 runs each with seeds shared across algorithms | paper (100 × 20); shared seeds chosen |
| Training instances | freshly sampled from the set's distribution for every episode | paper |

## 3. PPO settings the paper doesn't give

γ = 0.99, λ = 0.95, ε = 0.2, the 2048·N_env buffer and 2×64 tanh MLPs are all Stable-Baselines3
PPO defaults, so the rest follow SB3 too: Adam (lr 3e-4, eps 1e-5), 10 epochs, minibatch 64,
value coefficient 0.5, entropy coefficient 0, gradient clipping 0.5, per-minibatch advantage
normalisation, log σ initialised to 0, actions clipped to [0, 1]. Collection follows
Algorithm 2: rounds of 20 whole episodes until the buffer holds ≥ 40,960 steps (21 rounds
= 42,000 steps). The episode end at G_roll is treated as terminal, because the state
contains g / G_max and Eq. 19 is a finite sum.

MetaBox's σ head (σ ∈ [0.05, 0.15], state-dependent) was tried first. Combined with the
paper's linear mean, which starts near 0 and is clipped, it hardly explored: after 2e6
steps on VS the agent kept a1 ≈ 0 and scored a training return of 136, against 394 for
uniformly random actions and 670 for the fixed action (0.5, 0.5, 0.5). That run is kept
in `l2t_runs/DE_VS_metabox-sigma` and the option is available as `--sigma-mode metabox`.

## 4. MTGA-L2T (Algorithm S.1)

| Item | Choice | Source |
|---|---|---|
| Pairing | random pairs from the merged population; each child goes to its own parent's task | paper |
| KT test | `rand < a_k,1` on cross-task pairs only | paper |
| KT child | Eq. 10 vector with F = 0.5, no extra crossover or mutation | chosen (S.1 says "sample u_a by the proposed action formulation") |
| SBX / PM | MToP's operators, η_c = 2, η_m = 5, PM rate 1/D | paper (η) + MToP |
| Selection | per task, elitist truncation of parents ∪ offspring | chosen (MFEA style) |

## 5. MTJADE-L2T (Section IV-H3)

JADE without external archive (MToP's JADE.m operators: current-to-pbest/1 with p = 0.1,
Cauchy F_i and normal CR_i, c = 0.1, midpoint boundary repair, one-to-one selection).
The paper only says the KT actions and the other L2T components stay as designed, so:
KT offspring use their own F_i in Eq. 10 and CR_i in the crossover, and every successful
offspring (KT or not) feeds the F/CR adaptation (**chosen**). STJADE is the same code
with no KT, so the comparison differs only in transfer.

## 6. Baselines

See the docstring of `l2t/baselines.py`. Summary:

- Ported from MTO-Platform: MFEA, MFEA-II, G-MFEA, MFEA-AKT, MTEA-AD, AT-MFEA, MFDE,
  MKTDE, EMEA (as MTDE-EA), JADE without archive (STJADE).
- AEMTO comes from MToP's asynchronous AAEMTO, run synchronously. MTDE-AD is MTEA-AD's
  transfer on a DE solver. MTDE-B (best source solution as base vector, probability 0.3)
  is written from the L2T paper's one-line description and is the least certain.
- MFEA-RL is missing: no description of its method could be found.
- Every baseline uses N = 50, F = 0.5, CR = 0.5, η_c = 2, η_m = 5, as the paper keeps the
  base solver fixed. MToP's own defaults (e.g. CR 0.9 for MFDE, 0.6 for MKTDE) are replaced.
- Boundary handling is clipping to [0, 1] everywhere. MToP's AAEMTO re-samples out-of-bound
  values at random instead; that alone makes its DE far stronger on the 50-D Ackley tasks
  (error ≈ 2.7 vs ≈ 20.8 over 4 seeds), so it was changed to clipping to keep the base
  solver identical across methods. MTDE-B uses truncation selection like the rest.
- G-MFEA's variable shuffling is skipped because both tasks always have the same D.
- MFEA-II's RMP likelihood is computed in log space (MToP multiplies densities, which
  underflows at D = 50).

## 7. Selection and boundary handling: how they were found

A first VS evaluation with MetaBox's one-to-one DE selection gave MTDE-L2T a median final
error of ≈ 7 against ≈ 1.5 for AEMTO, while the paper has MTDE-L2T beating AEMTO 97/2/1.
With transfer switched off, AEMTO's DE alone still reached ≈ 2.8 where one-to-one DE
reached ≈ 21, so the gap came from the base solver, not from transfer. Two differences
explained it:

1. Selection. AEMTO (MToP) uses truncation selection; MetaBox's DE uses one-to-one. The
   paper's convergence remark refers to truncated selection, so the environment now uses
   truncation for DE (one-to-one remains available as `L2TEnv(selection='one_to_one')`).
2. Boundary handling. With truncation selection, clipping still left 50-D Ackley at
   ≈ 20.8 while AEMTO's random re-sampling reached ≈ 2.7. The paper does not specify it;
   clipping (MetaBox, and most MToP baselines) is kept and applied to AEMTO as well.

Results from the one-to-one runs are kept in `l2t_runs/_superseded/`.

## 8. θ* and the time-feature normaliser: how they were chosen

Both are open in the paper. On VS (truncation selection), the first rule tried for θ*
(best deterministic return on 20 fixed validation instances) picked the update-8 agent,
because validation return peaked early. Against AEMTO at G_roll that agent scored
40/10/50, while the final agent scored 100/0/0 (paper: 100/0/0). Return is dominated by
b2·r_KT, which rewards KT offspring relative to DE offspring, so it can peak before the
policy optimises best. θ* is now the policy with the best mean training episode return.
That return keeps rising, so θ* is a late agent.

Beyond G_roll the final agent with the literal g / G_max features (G_max = 250 at test)
lost to AEMTO at G_max (18/12/70 on all 100 instances; 5/10/15 on a 30-instance check),
while normalising by G_roll gave 13/14/3 on the same 30 instances (paper: L2T beats AEMTO
97/2/1 at G_max). With G_max the time features move 2.5× slower than in training, which
puts the agent in states it never saw. With G_roll they read "end of training horizon"
from generation 100 on. The paper's Table II writes G_max, and in training G_max is G_roll.
Normalising by G_roll is the default. The literal reading is evaluated alongside for every
main agent (`L2T-scratch-gmaxnorm`, `L2T-gmaxnorm`), so both can be reported.

## 9. Problem dimension

The paper never states D. The CEC17-based suite was first run at D = 50 (the CEC17 MTO
default). There, DE without transfer stalls within G_max = 250 generations (median VL
error ≈ 21, Ackley stuck on its plateau), so any useful transfer wins outright: L2T
beat STDE 100/0/0 on VS–L and 45/50/5 on VL, against the paper's 95/5/0 … 44/52/4 and
7/91/2, and beat the differential-vector fixed agents and MTDE-B where the paper loses.
The paper's many ties against STDE imply that its base solver often converges on its own.
STDE median VL errors after 250 generations were 1.7e-6 (D = 10), 0.013 (D = 20), 0.32 (D = 30)
and 20.7 (D = 50). With the user's agreement, D = 10 is used for both suites. The D = 50
runs (VS, S, M, L, VL, C1) are kept in `l2t_runs/_superseded/D50/`.
