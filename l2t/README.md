# L2T reproduction

Paper-faithful reimplementation of *Learning to Transfer for Evolutionary Multitasking*
(Wu et al., IEEE TCYB 55(7), 2025), built from MetaBox's L2T code. Every choice the paper
leaves open, and every change from MetaBox, is recorded with its source in
[DEVIATIONS.md](DEVIATIONS.md). Nothing under `src/` is modified.

## Setup (Linux / WSL2 / macOS, Python 3.11)

```bash
git clone <this repo> MetaBox && cd MetaBox
git checkout l2t-reproduction
python3.11 -m venv .venv
.venv/bin/pip install -r l2t/requirements.txt     # exact versions used so far
.venv/bin/python -m l2t.experiments --list        # sanity check: lists the stages
```

Python 3.11 is required (`ray==2.44.1` and `tianshou==1.1.0` with numpy < 2 do not install on 3.13).
The code uses the CPU only (see "Speed" below).

## Run

```bash
l2t/run_pipeline.sh              # everything, in the background; resumable
l2t/run_pipeline.sh 22 core      # 22 workers, first stage only
tail -f l2t_runs/experiments.log
```

Stages (`python -m l2t.experiments --list`): `core` (10 CEC17-based sets + BBOB9/10, Tables
S.V / S.X, Fig. 3), `bbob_de`, `bbob_ga` (Tables III, S.VI), `explicit` (S.VIII), `finetune`
(S.IX, Fig. 4), `jade` (S.XV), `rl_algos` (SAC/TD3, S.XIV), `b2` (S.XII–S.XIII), `ablation` (S.XI).
Not covered: MFEA-RL (Table S.VII) and the HPO suite (Table S.XVI).

Results go to `l2t_runs/` (git-ignored): `DE_<set>/` (agent, log), `eval/<base>/<set>/<algo>.npz`.

## Read results

```bash
.venv/bin/python -m l2t.report --table S.X              # W/T/L next to the paper's numbers
.venv/bin/python -m l2t.report --table S.V --sets VS S M
.venv/bin/python -m l2t.report --target L2T-scratch --vs STDE AEMTO --sets VL --gen 100
.venv/bin/python -m l2t.plots fig3
```

Tables: S.X, S.V, S.VI-DE, S.VI-GA, III-DE, III-GA, S.VIII-DE, S.VIII-GA, S.IX-scratch, S.IX-FT,
S.XI, S.XIV, S.XV. Agent labels: `L2T-scratch` (trained on the test set's own distribution),
`L2T` (BBOB_learn agent), `L2T-FT` (fine-tuned); `…-gmaxnorm` uses the literal g / G_max time features.

## Moving a half-finished run to another machine

Copy `l2t_runs/` (finished trainings carry a `done` file and are skipped; finished evaluations
are the `.npz` files). A training that was interrupted has no `done` file and restarts from scratch.
Results do not depend on the machine: every run is seeded.

## Speed

Measured on an Apple M4 (4 performance + 6 efficiency cores), VS set, one PPO iteration
(20 episodes × 100 generations = 42,000 steps, then one update):

| Part | Time | Scales with cores? |
|---|---|---|
| Simulation (9 workers) | 2.9 s | yes (4 workers: 4.2 s) |
| PPO update, 10 epochs × batch 64 | 3.2 s | no: one core, tiny networks, a GPU would not help |
| Validation episodes | ≈ 0.5 s | yes |

Evaluation (14–15 algorithms × 100 instances × 20 runs) is all simulation and scales with cores.
To compare a machine, run
`python -m l2t.train --set VS --T 126000 --workers N --out /tmp/bench --name bench`
and read the last column of the log (cumulative seconds; the Mac gives 23 s for three updates).

## Layout

| File | Purpose |
|---|---|
| `problems.py` | the 26 MTOP sets (Tables S.II, S.III), task classes, LHS initial populations |
| `operators.py` | DE/rand/1/bin, Eq. (10), SBX, polynomial mutation |
| `env.py` | the environment: features, reward, DE / GA / JADE base solvers, ablations |
| `agent.py` | actor, critic, PPO with GAE |
| `train.py`, `offpolicy.py` | Algorithm 2; SAC / TD3 agents |
| `rollout.py`, `evaluate.py` | episodes on a process pool; test-set evaluation |
| `baselines.py` | 12 peer algorithms (mostly ported from MTO-Platform) |
| `stats.py`, `report.py`, `plots.py` | Definition 1 (W/T/L), tables with the paper's values, Figs. 3–5 |
| `experiments.py` | the stage plan |
