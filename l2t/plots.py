"""Figures 3-5 of the paper from stored runs.

    python -m l2t.plots fig3                      # positive transfer rates (Eq. 21)
    python -m l2t.plots fig4                      # training curves, fine-tuned vs from scratch
    python -m l2t.plots fig5 --set BBOB1 --instance 15 --agent l2t_runs/DE_BBOB_learn/agent_best.pt
Figures are written to l2t_runs/figures/.
"""
import argparse
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from .evaluate import G_MAX
from .problems import test_instances, get_set
from .report import load
from .rollout import run_episode
from .stats import positive_transfer_rate

OUT = 'l2t_runs'
FIG = os.path.join(OUT, 'figures')


def fig3(gen=G_MAX):
    algos = ['L2T-scratch', 'AEMTO', 'MFDE', 'MKTDE', 'MTDE-AD', 'MTDE-B']
    groups = [('(a) optimum range', ['VS', 'S', 'M', 'L', 'VL']), ('(b) clusters', ['C1', 'C2', 'C3', 'C4', 'C5'])]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for ax, (title, sets) in zip(axes, groups):
        for a in algos:
            rates = []
            for s in sets:
                emt, st = load(OUT, 'DE', s, a), load(OUT, 'DE', s, 'STDE')
                rates.append(positive_transfer_rate(emt, st, gen) if emt is not None and st is not None else np.nan)
            ax.plot(sets, rates, marker='o', label='MTDE-L2T' if a == 'L2T-scratch' else a)
        ax.set_title(title)
        ax.set_ylim(0, 1)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel(f'positive transfer rate vs STDE (g = {gen})')
    axes[1].legend(fontsize=8)
    return save(fig, 'fig3_positive_transfer.png')


def fig4(sets=('BBOB9', 'BBOB10', 'VL', 'C1', 'C5')):
    fig, axes = plt.subplots(1, len(sets), figsize=(4 * len(sets), 3.2))
    for ax, s in zip(np.atleast_1d(axes), sets):
        for name, label in [(f'DE_{s}_ft', 'L2T-FT'), (f'DE_{s}', 'L2T-w/o-FT')]:
            path = os.path.join(OUT, name, 'log.csv')
            if os.path.exists(path):
                log = pd.read_csv(path)
                ax.plot(log['t'], log['train_return'], label=label)
        ax.set_title(s)
        ax.set_xlabel('time steps')
        ax.grid(alpha=0.3)
    np.atleast_1d(axes)[0].set_ylabel('mean episode return')
    np.atleast_1d(axes)[-1].legend()
    return save(fig, 'fig4_training_curves.png')


def fig5(set_name, instance, agent_path, base='DE', seed=0):
    from .evaluate import policy_from_name
    _, policy, overrides = policy_from_name(f'agent={agent_path}', base)
    spec = test_instances(set_name)[instance]
    env_spec = dict(base=base, horizon=G_MAX, dim=get_set(set_name).dim, compute_reward=False, **overrides)
    r = run_episode(env_spec, policy, spec, seed, record_actions=True)
    a = r['actions']
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.5))
    for k in range(2):
        for j, name in enumerate(['a1 (when)', 'a2 (base vector)', 'a3 (differential vector)']):
            axes[k].plot(a[:, k, j], label=name)
        axes[k].set_title(f'task {k + 1}: {spec[k][1] if spec[k][0] == "cec" else "F%d" % spec[k][1]}')
        axes[k].set_ylim(-0.05, 1.05)
        axes[k].set_xlabel('generation')
        axes[k].legend(fontsize=8)
    axes[2].semilogy(np.maximum(r['curve'], 1e-12))
    axes[2].set_title('best-so-far error')
    axes[2].legend(['task 1', 'task 2'])
    return save(fig, f'fig5_{set_name}_{instance}.png')


def save(fig, name):
    os.makedirs(FIG, exist_ok=True)
    path = os.path.join(FIG, name)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print(path)
    return path


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('figure', choices=['fig3', 'fig4', 'fig5'])
    p.add_argument('--set', default='BBOB1')
    p.add_argument('--instance', type=int, default=15)
    p.add_argument('--agent', default=os.path.join(OUT, 'DE_BBOB_learn', 'agent_best.pt'))
    p.add_argument('--base', default='DE')
    args = p.parse_args(argv)
    if args.figure == 'fig3':
        fig3()
    elif args.figure == 'fig4':
        fig4()
    else:
        fig5(args.set, args.instance, args.agent, args.base)


if __name__ == '__main__':
    main()
