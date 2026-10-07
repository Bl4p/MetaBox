"""Running episodes, serially or on a process pool.

A policy is a small picklable tuple:
    ('agent', numpy_params, squash_mean, stochastic)   learned actor (sampled or mean action)
    ('fixed', (x, y, z))                               MTDE-f(x, y, z): same action for every task
    ('random',)                                        MTDE-r: uniform action in [0, 1]^3 per task
    ('stde',)                                          single-task DE / GA: no KT (a_k1 = 0)
"""
import os
from functools import lru_cache
from multiprocessing import get_context

import numpy as np

from .agent import numpy_policy
from .env import L2TEnv
from .problems import build_mtop, initial_population_set

N_P = 10            # initial population set size (Table S.IV)
INIT_POP_SEED = 0


@lru_cache(maxsize=8)
def _init_pops(n_pops, pop_size, dim, seed):
    return initial_population_set(n_pops, pop_size, dim, seed)


def make_env(env_spec):
    s = dict(env_spec)
    pops = _init_pops(s.pop('n_pops', N_P), s.get('N', 50), s['dim'], s.pop('init_seed', INIT_POP_SEED))
    return L2TEnv(init_pops=pops, **s)


def _act(policy, state, rng, env):
    kind = policy[0]
    if kind == 'agent':
        _, params, squash, stochastic = policy
        raw, mu = numpy_policy(params, state, rng if stochastic else None, squash)
        return raw
    if kind == 'fixed':
        return np.tile(np.asarray(policy[1], dtype=np.float64), env.K)
    if kind == 'random':
        return rng.random(env.action_dim)
    if kind == 'stde':
        return np.zeros(3 * env.K)
    raise ValueError(policy)


def run_episode(env_spec, policy, mtop_spec, seed, record_transitions=False, record_actions=False):
    """Run one episode. Returns a dict with the best-so-far error curve (G+1, K) and,
    optionally, the transitions (for PPO) and the executed (K, 3) actions."""
    env = make_env(env_spec)
    tasks = build_mtop(mtop_spec)
    rng = np.random.default_rng(seed)
    state = env.reset(tasks, rng)
    states, actions, rewards, executed = [], [], [], []
    done = False
    while not done:
        a = _act(policy, state, rng, env)
        if record_transitions:
            states.append(state)
            actions.append(np.asarray(a, dtype=np.float32))
        if record_actions:
            executed.append(env.full_action(a))
        state, r, done, _ = env.step(a)
        rewards.append(r)
    fopt = np.array([t.fopt for t in tasks])
    out = {'curve': np.array(env.curve) - fopt, 'return': float(np.sum(rewards)), 'fes': env.fes}
    if record_transitions:
        out.update(states=np.array(states), actions=np.array(actions), rewards=np.array(rewards, dtype=np.float32))
    if record_actions:
        out['actions'] = np.array(executed)
    return out


def _job(args):
    return run_episode(*args)


class Runner:
    """Maps run_episode over many (policy, instance, seed) jobs with a process pool."""

    def __init__(self, workers=None):
        self.workers = workers or max(1, (os.cpu_count() or 2) - 1)
        # one BLAS thread per worker; spawned workers read these when they import numpy
        for var in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
                    'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
            os.environ[var] = '1'
        self.pool = get_context('spawn').Pool(self.workers, initializer=_worker_init) if self.workers > 1 else None

    def map(self, jobs, chunksize=1):
        if self.pool is None:
            return [_job(j) for j in jobs]
        return self.pool.map(_job, jobs, chunksize=chunksize)

    def map_fn(self, fn, jobs, chunksize=1):
        """Map any picklable top-level function over jobs."""
        if self.pool is None:
            return [fn(j) for j in jobs]
        return self.pool.map(fn, jobs, chunksize=chunksize)

    def close(self):
        if self.pool is not None:
            self.pool.close()
            self.pool.join()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def _worker_init():
    import warnings
    warnings.filterwarnings('ignore')
    try:
        import torch
        torch.set_num_threads(1)
    except ImportError:
        pass
