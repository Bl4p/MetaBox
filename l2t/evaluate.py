"""Run algorithms on the test instances of MTOP sets and store best-so-far error curves.

    python -m l2t.evaluate --sets VS M VL --algos L2T=l2t_runs/DE_VS/agent_best.pt STDE "MTDE-f(.5,0,1)" MTDE-r
    python -m l2t.evaluate --sets BBOB1 --base GA --algos L2T=l2t_runs/GA_BBOB_learn/agent_best.pt MFEA

Algorithm names:
    NAME=<checkpoint>        learned agent (mean action), stored under NAME (e.g. L2T, L2T-FT).
                             Its time features are normalised by its training G_roll; append
                             ':gmax' to the path to normalise by the test horizon G_max instead
    STDE                     no knowledge transfer (STGA with --base GA, STJADE with --base JADE)
    MTDE-r                   random action each generation
    MTDE-f(x,y,z)            fixed action, e.g. "MTDE-f(.5,0,1)"
    any name in l2t.baselines.BASELINES (MFDE, MTDE-B, MFEA, ...)

Each algorithm runs on the same 100 instances with the same 20 seeds per instance
(R = 20 independent runs). Results: l2t_runs/eval/<base>/<set>/<name>.npz with
`curves` of shape (100, 20, G_max + 1, 2).
"""
import argparse
import json
import os
import re
import time

import numpy as np
import torch

from .problems import get_set, test_instances
from .rollout import Runner, run_episode

G_MAX = 250
N_RUNS = 20
RUN_SEED = 777


def policy_from_name(name, base, g_max=G_MAX):
    """Returns (label, policy, env overrides)."""
    if '=' in name:
        label, path = name.split('=', 1)
        use_gmax = path.endswith(':gmax')
        path = path[:-len(':gmax')] if use_gmax else path
        ckpt = torch.load(path, weights_only=False)
        params = {k: v.cpu().numpy() for k, v in ckpt['actor'].items()}
        cfg_path = os.path.join(os.path.dirname(path), 'config.json')
        cfg = {}
        if os.path.exists(cfg_path):
            with open(cfg_path) as f:
                cfg = json.load(f)
        # Time features: by default normalised with the G_roll the agent was trained with
        # (g / G_roll, capped at 1 after G_roll); ':gmax' uses the test horizon (DEVIATIONS.md §8)
        fh = g_max if use_gmax else int(cfg.get('g_roll', 100))
        overrides = {'ablation': cfg.get('env_spec', {}).get('ablation'), 'feature_horizon': fh}
        return label, ('agent', params, ckpt.get('squash_mean', False), False), overrides
    if name in ('STDE', 'STGA', 'STJADE'):
        return name, ('stde',), {}
    if name in ('MTDE-r', 'MTGA-r'):
        return name, ('random',), {}
    m = re.fullmatch(r'MT[DG][EA]-f\(([^)]*)\)', name)
    if m:
        return name, ('fixed', tuple(float(x) for x in m.group(1).split(','))), {}
    from .baselines import BASELINES
    if name in BASELINES:
        return name, ('baseline', name), {}
    raise ValueError(f'unknown algorithm {name!r}')


def eval_jobs(set_name, policy, base, overrides, g_max, n_runs, dim=None):
    s = get_set(set_name, dim)
    env_spec = dict(base=base, horizon=g_max, dim=s.dim, compute_reward=False, **overrides)
    jobs = []
    for i, spec in enumerate(test_instances(set_name, dim=dim)):
        for r in range(n_runs):
            jobs.append((env_spec, policy, spec, int(np.random.SeedSequence([RUN_SEED, i, r]).generate_state(1)[0])))
    return jobs


def result_path(out, base, set_name, label):
    return os.path.join(out, 'eval', base, set_name, f'{label}.npz')


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--sets', nargs='+', required=True)
    p.add_argument('--algos', nargs='+', required=True)
    p.add_argument('--base', default='DE', choices=['DE', 'GA', 'JADE'])
    p.add_argument('--g-max', type=int, default=G_MAX)
    p.add_argument('--runs', type=int, default=N_RUNS)
    p.add_argument('--dim', type=int, default=None)
    p.add_argument('--workers', type=int, default=None)
    p.add_argument('--out', default='l2t_runs')
    p.add_argument('--overwrite', action='store_true')
    args = p.parse_args(argv)

    with Runner(args.workers) as runner:
        for set_name in args.sets:
            for algo in args.algos:
                label, policy, overrides = policy_from_name(algo, args.base, args.g_max)
                path = result_path(args.out, args.base, set_name, label)
                if os.path.exists(path) and not args.overwrite:
                    print(f'skip {path} (exists)')
                    continue
                t0 = time.time()
                if policy[0] == 'baseline':
                    from .baselines import baseline_jobs, run_baseline_job
                    jobs = baseline_jobs(set_name, policy[1], args.base, args.g_max, args.runs, args.dim)
                    res = runner.map_fn(run_baseline_job, jobs, chunksize=4)
                    curves = np.stack(res)
                else:
                    jobs = eval_jobs(set_name, policy, args.base, overrides, args.g_max, args.runs, args.dim)
                    res = runner.map(jobs, chunksize=4)
                    curves = np.stack([r['curve'] for r in res])
                n_inst = len(curves) // args.runs
                curves = curves.reshape(n_inst, args.runs, *curves.shape[1:]).astype(np.float32)
                os.makedirs(os.path.dirname(path), exist_ok=True)
                np.savez_compressed(path, curves=curves, algo=algo, set=set_name, base=args.base, g_max=args.g_max)
                print(f'{set_name:10s} {label:18s} median final error {np.median(curves[:, :, -1], axis=(0, 1))}  '
                      f'{time.time() - t0:.0f}s -> {path}', flush=True)


if __name__ == '__main__':
    main()
