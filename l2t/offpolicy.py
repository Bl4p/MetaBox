"""L2T agents trained with SAC or TD3 instead of PPO (Section IV-H2, Table S.XIV).

    python -m l2t.offpolicy --algo SAC --set BBOB_learn
    python -m l2t.offpolicy --algo TD3 --set BBOB_learn

Uses tianshou 1.1.0. The paper gives no SAC/TD3 settings; tianshou's defaults are used
(lr 3e-4, tau 0.005, gamma 0.99, batch 256, replay buffer 1e6, 1 gradient step per
environment step, SAC with automatic entropy tuning, TD3 exploration noise 0.1, policy
noise 0.2, noise clip 0.5, delayed updates every 2 steps), with the same 2x64 tanh
networks, 20 parallel environments, environment, reward and T = 5e6 as the PPO agent.

The trained actor is exported in the format of l2t.agent.Actor with squash_mean=True
(tianshou maps tanh outputs in [-1, 1] to the action box [0, 1], i.e. (tanh + 1) / 2), so
evaluation uses the same code path: --algos SAC=l2t_runs/SAC_BBOB_learn/agent_best.pt
"""
import argparse
import json
import os

import gymnasium as gym
import numpy as np
import torch
from torch import nn

from .env import L2TEnv
from .problems import build_mtop, get_set
from .rollout import _init_pops, N_P, INIT_POP_SEED


class L2TGymEnv(gym.Env):
    """Gymnasium wrapper: every reset samples a new MTOP instance from the training set."""

    def __init__(self, set_name, base='DE', horizon=100, b=(1.0, 10.0, 100.0), seed=0, dim=None):
        self.mtop_set = get_set(set_name, dim)
        self.env = L2TEnv(base=base, horizon=horizon, b=b, dim=self.mtop_set.dim,
                          init_pops=_init_pops(N_P, 50, self.mtop_set.dim, INIT_POP_SEED))
        self.observation_space = gym.spaces.Box(0.0, 1.0, (self.env.state_dim,), np.float32)
        self.action_space = gym.spaces.Box(0.0, 1.0, (self.env.action_dim,), np.float32)
        self.rng = np.random.default_rng(seed)

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        tasks = build_mtop(self.mtop_set.sample_mtop(self.rng))
        return self.env.reset(tasks, np.random.default_rng(self.rng.integers(2**31))), {}

    def step(self, action):
        s, r, done, info = self.env.step(action)
        return s, r, done, False, {}


def export_actor(actor, path, algo):
    """Write the actor as l2t.agent.Actor weights (linear1, linear2, mu) with squash_mean=True."""
    linears = [m for m in actor.preprocess.modules() if isinstance(m, nn.Linear)]
    head = actor.mu if hasattr(actor, 'mu') else actor.last
    head = [m for m in head.modules() if isinstance(m, nn.Linear)][-1]
    sd = {'linear1.weight': linears[0].weight, 'linear1.bias': linears[0].bias,
          'linear2.weight': linears[1].weight, 'linear2.bias': linears[1].bias,
          'mu.weight': head.weight, 'mu.bias': head.bias}
    torch.save({'actor': {k: v.detach().cpu().clone() for k, v in sd.items()},
                'squash_mean': True, 'sigma_mode': 'deterministic', 'algo': algo}, path)


def main(argv=None):
    from tianshou.data import Collector, VectorReplayBuffer
    from tianshou.env import SubprocVectorEnv
    from tianshou.exploration import GaussianNoise
    from tianshou.policy import SACPolicy, TD3Policy
    from tianshou.trainer import OffpolicyTrainer
    from tianshou.utils.net.common import Net
    from tianshou.utils.net.continuous import Actor, ActorProb, Critic

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--algo', choices=['SAC', 'TD3'], required=True)
    p.add_argument('--set', default='BBOB_learn')
    p.add_argument('--base', default='DE')
    p.add_argument('--T', type=float, default=5e6)
    p.add_argument('--n-env', type=int, default=20)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--epochs', type=int, default=100, help='checkpoints / validation points over T')
    p.add_argument('--out', default='l2t_runs')
    args = p.parse_args(argv)

    out = os.path.join(args.out, f'{args.algo}_{args.set}')
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, 'config.json'), 'w') as f:
        json.dump(vars(args), f, indent=2)
    torch.manual_seed(args.seed)
    for var in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
        os.environ[var] = '1'

    def make(i):
        return lambda: L2TGymEnv(args.set, args.base, seed=args.seed * 1000 + i)

    train_envs = SubprocVectorEnv([make(i) for i in range(args.n_env)])
    test_envs = SubprocVectorEnv([make(10_000 + i) for i in range(10)])
    probe = make(0)()
    s_dim, a_dim = probe.observation_space.shape[0], probe.action_space.shape[0]

    net_a = Net(s_dim, hidden_sizes=[64, 64], activation=nn.Tanh)
    critic_nets = [Net(s_dim, a_dim, hidden_sizes=[64, 64], activation=nn.Tanh, concat=True) for _ in range(2)]
    critic1, critic2 = (Critic(n) for n in critic_nets)
    c1_opt, c2_opt = torch.optim.Adam(critic1.parameters(), lr=3e-4), torch.optim.Adam(critic2.parameters(), lr=3e-4)
    if args.algo == 'SAC':
        actor = ActorProb(net_a, a_dim, unbounded=False, conditioned_sigma=True)
        a_opt = torch.optim.Adam(actor.parameters(), lr=3e-4)
        log_alpha = torch.zeros(1, requires_grad=True)
        alpha = (-a_dim, log_alpha, torch.optim.Adam([log_alpha], lr=3e-4))
        policy = SACPolicy(actor=actor, actor_optim=a_opt, critic=critic1, critic_optim=c1_opt,
                           critic2=critic2, critic2_optim=c2_opt, tau=0.005, gamma=0.99, alpha=alpha,
                           action_space=probe.action_space, action_scaling=True)
    else:
        actor = Actor(net_a, a_dim, max_action=1.0)
        a_opt = torch.optim.Adam(actor.parameters(), lr=3e-4)
        policy = TD3Policy(actor=actor, actor_optim=a_opt, critic=critic1, critic_optim=c1_opt,
                           critic2=critic2, critic2_optim=c2_opt, tau=0.005, gamma=0.99,
                           exploration_noise=GaussianNoise(sigma=0.1), policy_noise=0.2, update_actor_freq=2,
                           noise_clip=0.5, action_space=probe.action_space, action_scaling=True)

    buffer = VectorReplayBuffer(1_000_000, args.n_env)
    train_collector = Collector(policy, train_envs, buffer, exploration_noise=True)
    test_collector = Collector(policy, test_envs)
    train_collector.reset()
    train_collector.collect(n_step=10_000, random=True)

    def save_best(pol):
        export_actor(pol.actor, os.path.join(out, 'agent_best.pt'), args.algo)

    result = OffpolicyTrainer(
        policy=policy, train_collector=train_collector, test_collector=test_collector,
        max_epoch=args.epochs, step_per_epoch=int(args.T // args.epochs), step_per_collect=args.n_env,
        episode_per_test=10, batch_size=256, update_per_step=1.0, save_best_fn=save_best,
        test_in_train=False).run()
    export_actor(policy.actor, os.path.join(out, 'agent_last.pt'), args.algo)
    print(result)
    open(os.path.join(out, 'done'), 'w').close()


if __name__ == '__main__':
    main()
