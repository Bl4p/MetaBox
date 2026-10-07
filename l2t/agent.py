"""Actor-critic networks and the PPO update (Section III-C, Eqs. (18)-(20), Table S.IV).

Actor and Critic are MetaBox's (src/baseline/metabbo/l2t.py) with two changes:

* the Gaussian mean comes from a linear output layer (Table S.IV: "two hidden layers
  and one linear layer"); MetaBox squashed it with (tanh + 1) / 2
* sigma does not depend on the state (Eq. (18): pi(a|s) = N(phi(s; theta), sigma)); it
  is one learnable log-std per action dimension initialised to 0, as in
  Stable-Baselines3. MetaBox used a state-dependent sigma head bounded to [0.05, 0.15];
  with a linear mean starting near 0 that head barely explores (on VS the agent got
  stuck below a uniformly random policy). It is still available as sigma_mode='metabox'.

MetaBox trained with n-step returns over 10-step windows. The paper uses PPO with GAE.
Its PPO settings (gamma 0.99, lambda 0.95, eps 0.2, buffer 2048 * N_env, 2x64 tanh MLPs)
are exactly Stable-Baselines3's PPO defaults, so the settings it leaves open use SB3's
defaults too (see PPOConfig).
"""
from dataclasses import dataclass, asdict

import numpy as np
import torch
from torch import nn


class Actor(nn.Module):
    def __init__(self, n_state, n_action, hidden_dim=64, squash_mean=False, sigma_mode='state_independent'):
        super().__init__()
        assert sigma_mode in ('state_independent', 'metabox')
        self.linear1 = nn.Linear(n_state, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.mu = nn.Linear(hidden_dim, n_action)
        if sigma_mode == 'metabox':
            self.sigma = nn.Linear(hidden_dim, n_action)
        else:
            self.log_std = nn.Parameter(torch.zeros(n_action))
        self.tanh = nn.Tanh()
        self.max_sigma, self.min_sigma = 0.15, 0.05
        self.squash_mean = squash_mean
        self.sigma_mode = sigma_mode

    def forward(self, state):
        x = self.tanh(self.linear1(state))
        x = self.tanh(self.linear2(x))
        mu = self.mu(x)
        if self.squash_mean:                       # MetaBox's choice; off for the paper
            mu = (torch.tanh(mu) + 1.0) / 2.0
        if self.sigma_mode == 'metabox':
            sigma = (torch.tanh(self.sigma(x)) + 1.0) / 2.0 * (self.max_sigma - self.min_sigma) + self.min_sigma
        else:
            sigma = self.log_std.exp().expand_as(mu)
        return mu, sigma

    def distribution(self, state):
        mu, sigma = self(state)
        return torch.distributions.Normal(mu, sigma)

    def numpy_params(self):
        """Weights as numpy arrays, for cheap action sampling in worker processes."""
        return {k: v.detach().cpu().numpy().copy() for k, v in self.state_dict().items()}


def numpy_policy(params, state, rng=None, squash_mean=False, min_sigma=0.05, max_sigma=0.15):
    """Actor forward pass in numpy. Returns (raw action, mean). Sampled if rng is given."""
    h = np.tanh(state @ params['linear1.weight'].T + params['linear1.bias'])
    h = np.tanh(h @ params['linear2.weight'].T + params['linear2.bias'])
    mu = h @ params['mu.weight'].T + params['mu.bias']
    if squash_mean:
        mu = (np.tanh(mu) + 1.0) / 2.0
    if rng is None:
        return mu, mu
    if 'log_std' in params:
        sigma = np.exp(params['log_std'])
    else:
        sigma = (np.tanh(h @ params['sigma.weight'].T + params['sigma.bias']) + 1.0) / 2.0 \
            * (max_sigma - min_sigma) + min_sigma
    return mu + sigma * rng.standard_normal(mu.shape), mu


class Critic(nn.Module):
    """Same architecture as the actor, separate parameters, scalar output (as MetaBox)."""

    def __init__(self, n_state, hidden_dim=64):
        super().__init__()
        self.linear1 = nn.Linear(n_state, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, 1)
        self.tanh = nn.Tanh()

    def forward(self, x):
        out = self.tanh(self.linear1(x))
        out = self.tanh(self.linear2(out))
        return self.linear3(out).squeeze(-1)


@dataclass
class PPOConfig:
    # Table S.IV
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    # not in the paper: Stable-Baselines3 PPO defaults
    lr: float = 3e-4
    n_epochs: int = 10
    batch_size: int = 64
    vf_coef: float = 0.5
    ent_coef: float = 0.0
    max_grad_norm: float = 0.5
    normalize_advantage: bool = True
    adam_eps: float = 1e-5


def gae(rewards, values, dones, last_values, gamma, lam):
    """Generalised advantage estimation over a (T, n_env) block of transitions.

    `dones[t]` marks that the episode ended after step t. The episode end at the horizon
    is treated as terminal (no bootstrap): the state carries g / G_max, so the end of the
    episode is part of the MDP, matching the finite sum in Eq. (19).
    """
    T = len(rewards)
    adv = np.zeros_like(rewards)
    last = np.zeros_like(last_values)
    for t in reversed(range(T)):
        next_v = last_values if t == T - 1 else values[t + 1]
        nonterminal = 1.0 - dones[t]
        delta = rewards[t] + gamma * next_v * nonterminal - values[t]
        last = delta + gamma * lam * nonterminal * last
        adv[t] = last
    return adv


class PPO:
    def __init__(self, n_state, n_action, cfg: PPOConfig = None, squash_mean=False,
                 sigma_mode='state_independent', device='cpu'):
        self.cfg = cfg or PPOConfig()
        self.device = device
        self.actor = Actor(n_state, n_action, squash_mean=squash_mean, sigma_mode=sigma_mode).to(device)
        self.critic = Critic(n_state).to(device)
        self.optimizer = torch.optim.Adam(list(self.actor.parameters()) + list(self.critic.parameters()),
                                          lr=self.cfg.lr, eps=self.cfg.adam_eps)

    @torch.no_grad()
    def values(self, states):
        return self.critic(torch.as_tensor(states, dtype=torch.float32, device=self.device)).cpu().numpy()

    def update(self, states, actions, rewards, dones):
        """One PPO update (Algorithm 2, lines 13-14) on a buffer of whole episodes.

        Arrays are (T, n_env, ...): T steps of n_env parallel episodes; every column
        holds complete episodes, so the value after the last step is never needed.
        """
        c = self.cfg
        T, E = rewards.shape
        S = torch.as_tensor(states.reshape(T * E, -1), dtype=torch.float32, device=self.device)
        A = torch.as_tensor(actions.reshape(T * E, -1), dtype=torch.float32, device=self.device)
        with torch.no_grad():
            old_logp = self.actor.distribution(S).log_prob(A).sum(-1)
            old_v = self.critic(S)
        values = old_v.cpu().numpy().reshape(T, E)
        adv = gae(rewards, values, dones, np.zeros(E), c.gamma, c.gae_lambda)
        ret = adv + values
        ADV = torch.as_tensor(adv.reshape(-1), dtype=torch.float32, device=self.device)
        RET = torch.as_tensor(ret.reshape(-1), dtype=torch.float32, device=self.device)

        n = T * E
        stats = {'pi_loss': [], 'v_loss': [], 'approx_kl': [], 'clip_frac': []}
        for _ in range(c.n_epochs):
            perm = torch.randperm(n, device=self.device)
            for start in range(0, n, c.batch_size):
                mb = perm[start:start + c.batch_size]
                dist = self.actor.distribution(S[mb])
                logp = dist.log_prob(A[mb]).sum(-1)
                ratio = torch.exp(logp - old_logp[mb])
                a = ADV[mb]
                if c.normalize_advantage and len(mb) > 1:
                    a = (a - a.mean()) / (a.std() + 1e-8)
                pi_loss = -torch.min(ratio * a, torch.clamp(ratio, 1 - c.clip_eps, 1 + c.clip_eps) * a).mean()  # Eq. (19)
                v_loss = ((self.critic(S[mb]) - RET[mb]) ** 2).mean()                                       # Eq. (20)
                loss = pi_loss + c.vf_coef * v_loss - c.ent_coef * dist.entropy().sum(-1).mean()
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(list(self.actor.parameters()) + list(self.critic.parameters()), c.max_grad_norm)
                self.optimizer.step()
                with torch.no_grad():
                    stats['pi_loss'].append(pi_loss.item())
                    stats['v_loss'].append(v_loss.item())
                    stats['approx_kl'].append(((ratio - 1) - torch.log(ratio)).mean().item())
                    stats['clip_frac'].append(((ratio - 1).abs() > c.clip_eps).float().mean().item())
        return {k: float(np.mean(v)) for k, v in stats.items()}

    def state_dict(self):
        return {'actor': self.actor.state_dict(), 'critic': self.critic.state_dict(),
                'optimizer': self.optimizer.state_dict(), 'cfg': asdict(self.cfg),
                'squash_mean': self.actor.squash_mean, 'sigma_mode': self.actor.sigma_mode}

    def load_state_dict(self, d, load_optimizer=True):
        self.actor.load_state_dict(d['actor'])
        self.critic.load_state_dict(d['critic'])
        if load_optimizer and 'optimizer' in d:
            self.optimizer.load_state_dict(d['optimizer'])
