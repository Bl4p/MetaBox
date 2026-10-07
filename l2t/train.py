"""Agent learning by PPO (Algorithm 2).

    python -m l2t.train --set VS                      # MTDE-L2T on VS, T = 2e6
    python -m l2t.train --set BBOB_learn --T 5e6      # the BBOB_learn agent
    python -m l2t.train --set BBOB_learn --base GA --T 5e6
    python -m l2t.train --set VL --init l2t_runs/DE_BBOB_learn/agent_best.pt   # fine-tuning (IV-D)
    python -m l2t.train --set VS --ablation wo_a1     # ablations (IV-E)

Writes to l2t_runs/<name>/: config.json, log.csv, agent_best.pt, agent_last.pt.
"""
import argparse
import copy
import csv
import json
import os
import time

import numpy as np
import torch

from .agent import PPO, PPOConfig
from .env import L2TEnv
from .problems import get_set
from .rollout import Runner

N_ENV = 20                  # Table S.IV
BUFFER = 2048 * N_ENV       # N_buff = 40960
G_ROLL = 100
N_VAL = 20                  # fixed validation instances, logged only


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--set', required=True, help='MTOP set to learn on, e.g. VS, C3, BBOB_learn, BBOB9')
    p.add_argument('--base', default='DE', choices=['DE', 'GA', 'JADE'])
    p.add_argument('--T', type=float, default=None, help='total time steps (default 5e6 for BBOB_learn, else 2e6)')
    p.add_argument('--g-roll', type=int, default=G_ROLL)
    p.add_argument('--b1', type=float, default=1.0)
    p.add_argument('--b2', type=float, default=10.0)
    p.add_argument('--b3', type=float, default=None, help='default G_roll')
    p.add_argument('--ablation', default=None, choices=L2TEnv.ABLATIONS)
    p.add_argument('--init', default=None, help='checkpoint to fine-tune from')
    p.add_argument('--dim', type=int, default=None)
    p.add_argument('--n-env', type=int, default=N_ENV)
    p.add_argument('--buffer', type=int, default=BUFFER)
    p.add_argument('--workers', type=int, default=None)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--squash-mean', action='store_true', help="MetaBox's (tanh+1)/2 mean instead of the paper's linear one")
    p.add_argument('--sigma-mode', default='state_independent', choices=['state_independent', 'metabox'])
    p.add_argument('--out', default='l2t_runs')
    p.add_argument('--name', default=None)
    for f, v in vars(PPOConfig()).items():
        p.add_argument(f'--{f.replace("_", "-")}', type=type(v) if not isinstance(v, bool) else int, default=v)
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    T = args.T or (5e6 if args.set == 'BBOB_learn' else 2e6)
    b3 = args.b3 if args.b3 is not None else float(args.g_roll)
    mtop_set = get_set(args.set, args.dim)
    env_spec = dict(base=args.base, horizon=args.g_roll, b=(args.b1, args.b2, b3),
                    ablation=args.ablation, dim=mtop_set.dim)
    probe = L2TEnv(**{**env_spec, 'init_pops': None})
    ppo_cfg = PPOConfig(**{f: (bool(getattr(args, f)) if isinstance(v, bool) else getattr(args, f))
                           for f, v in vars(PPOConfig()).items()})

    name = args.name or '_'.join(filter(None, [args.base, args.set, args.ablation,
                                               'ft' if args.init else None,
                                               None if args.b2 == 10 else f'b2_{args.b2:g}']))
    out = os.path.join(args.out, name)
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, 'config.json'), 'w') as f:
        json.dump({**vars(args), 'T': T, 'b3': b3, 'env_spec': env_spec, 'ppo': vars(ppo_cfg),
                   'state_dim': probe.state_dim, 'action_dim': probe.action_dim}, f, indent=2, default=str)

    torch.manual_seed(args.seed)
    init = torch.load(args.init, weights_only=False) if args.init else None
    if init is not None:                       # fine-tuning keeps the pretrained parameterisation
        args.squash_mean = init.get('squash_mean', False)
        args.sigma_mode = init.get('sigma_mode', 'metabox')
    agent = PPO(probe.state_dim, probe.action_dim, ppo_cfg, squash_mean=args.squash_mean,
                sigma_mode=args.sigma_mode)
    if init is not None:
        agent.load_state_dict(init, load_optimizer=False)
    rng = np.random.default_rng(args.seed)
    val_rng = np.random.default_rng([args.seed, 999])
    val_set = [(mtop_set.sample_mtop(val_rng), int(val_rng.integers(2**31))) for _ in range(N_VAL)]

    log_path = os.path.join(out, 'log.csv')
    fields = ['update', 't', 'train_return', 'val_return', 'best_train_return', 'pi_loss', 'v_loss',
              'approx_kl', 'clip_frac', 'mean_a1', 'mean_a2', 'mean_a3', 'seconds']
    with open(log_path, 'w', newline='') as f:
        csv.writer(f).writerow(fields)

    t, update, best = 0, 0, -np.inf
    t0 = time.time()
    with Runner(args.workers) as runner:
        while t < T:
            # ---- collect: rounds of N_env episodes until the buffer is full (lines 6-12)
            params = agent.actor.numpy_params()
            policy = ('agent', params, args.squash_mean, True)
            collect_ckpt = copy.deepcopy(agent.state_dict())   # the policy that collects this buffer
            episodes = []
            while len(episodes) * args.g_roll < args.buffer:
                jobs = [(env_spec, policy, mtop_set.sample_mtop(rng), int(rng.integers(2**31)), True)
                        for _ in range(args.n_env)]
                episodes += runner.map(jobs)
                t += args.n_env * args.g_roll
            states = np.stack([e['states'] for e in episodes], axis=1)        # (T, E, s)
            actions = np.stack([e['actions'] for e in episodes], axis=1)
            rewards = np.stack([e['rewards'] for e in episodes], axis=1)
            dones = np.zeros_like(rewards)
            dones[-1] = 1.0
            # ---- update (lines 13-14)
            stats = agent.update(states, actions, rewards, dones)
            update += 1
            # ---- record the best-found agent (line 15): the policy that collected the buffer
            # with the highest mean episode return (the quantity plotted in Fig. 4). The
            # deterministic return on fixed validation instances is logged but not used: it
            # peaked early on VS while the later agents optimised much better (DEVIATIONS.md §8).
            train_return = float(np.mean([e['return'] for e in episodes]))
            if train_return > best:
                best = train_return
                torch.save({**collect_ckpt, 'update': update - 1}, os.path.join(out, 'agent_best.pt'))
            ckpt = agent.state_dict()
            torch.save(ckpt, os.path.join(out, 'agent_last.pt'))
            val_policy = ('agent', agent.actor.numpy_params(), args.squash_mean, False)
            val = runner.map([(env_spec, val_policy, spec, seed) for spec, seed in val_set])
            val_return = float(np.mean([v['return'] for v in val]))
            executed = np.clip(actions, 0, 1).reshape(-1, probe.action_dim)
            per_task = executed.reshape(len(executed), 2, -1).mean(axis=(0, 1))
            row = [update, t, train_return, val_return, best,
                   stats['pi_loss'], stats['v_loss'], stats['approx_kl'], stats['clip_frac'],
                   *(list(per_task) + [np.nan] * (3 - len(per_task))), round(time.time() - t0, 1)]
            with open(log_path, 'a', newline='') as f:
                csv.writer(f).writerow(row)
            print(f'[{name}] update {update:4d}  t={t:>9,}  train={row[2]:9.2f}  val={val_return:9.2f} '
                  f'best={best:9.2f}  kl={stats["approx_kl"]:.4f}  a=({", ".join(f"{x:.2f}" for x in per_task)})  '
                  f'{time.time() - t0:7.0f}s', flush=True)
    print(f'done: {out}')


if __name__ == '__main__':
    main()
