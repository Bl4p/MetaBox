"""The EMT environment of L2T: Algorithm 1 (DE base solver) and Algorithm S.1 (GA base solver).

Structure follows MetaBox's L2T_Optimizer (init_population / self_update / transfer /
selection / get_state), corrected to the paper:

* state has 4 + 7K = 18 features (Table II); MetaBox had 1 + 7K and no O_c2..O_c4
* each task uses its own action (a_k1, a_k2, a_k3); MetaBox applied task 0's to all
* KT quota ceil(0.5 * a_k1 * N), which may be 0; MetaBox forced at least 1
* q_KT (Eq. 11) and r_KT (Eqs. 16-17) are rank scores against the parent population;
  MetaBox used "beat own parent" counts and had r_KT's sign reversed
* r_conv = -(err / err_0) (Eq. 14); MetaBox's sign was positive
* b3 = G_roll = 100; MetaBox used 250
* the per-step reward r_t is returned; MetaBox returned the running total
* sigma_k is the per-dimension std over the population (axis 0); MetaBox used axis -1
* stagnation counts consecutive non-improving generations; MetaBox never reset it
* DE selection is truncation (best N of parents + offspring); MetaBox used one-to-one
* initial populations come from a pre-generated set of N_P LHS populations
* KT trial vectors are clipped to [0, 1]; each solution is evaluated exactly once
"""
import numpy as np

from . import operators as ops


class L2TEnv:
    """Two-task (K=2) implicit EMT driven by a KT action a in [0, 1]^(3K) each generation.

    Parameters
    ----------
    base : 'DE' or 'GA'
    horizon : generations per episode; G_roll in training, G_max at test. Also the
        normaliser G_max of the features O_c1 and O_t,k,1.
    init_pops : (N_P, N, D) array of pre-generated initial populations.
    b : reward weights (b1, b2, b3). Defaults to (1, 10, G_roll) with G_roll = 100.
    compute_reward : rewards need f*; switched off at inference.
    ablation : None or one of the Section IV-E variants:
        'wo_a1' / 'wo_a2' / 'wo_a3'  that action is removed and fixed to 0.5 / 0 / 0
        'wo_Oc' / 'wo_Ot'            common / task-specific features removed
        'wo_FE'                      raw populations and fitness instead of features
        'wo_rconv' / 'wo_rKT'        b1 = 0 / b2 = 0
    """

    K = 2
    ABLATIONS = ('wo_a1', 'wo_a2', 'wo_a3', 'wo_Oc', 'wo_Ot', 'wo_FE', 'wo_rconv', 'wo_rKT')
    _REMOVED_ACTION = {'wo_a1': (0, 0.5), 'wo_a2': (1, 0.0), 'wo_a3': (2, 0.0)}

    def __init__(self, base='DE', horizon=100, init_pops=None, N=50, F=0.5, CR=0.5,
                 eta_c=2.0, eta_m=5.0, b=(1.0, 10.0, 100.0), xi=1e-8, compute_reward=True,
                 ablation=None, dim=None, selection='truncation', feature_horizon=None):
        assert base in ('DE', 'GA', 'JADE')
        assert ablation in (None,) + self.ABLATIONS
        assert selection in ('truncation', 'one_to_one')
        self.selection = selection
        # G_max in O_c1 = g / G_max and O_t,k,1 = n_stag / G_max; defaults to the episode length
        self.feature_horizon = feature_horizon or horizon
        self.base, self.horizon = base, horizon
        self.init_pops = init_pops
        self.N, self.F, self.CR = N, F, CR
        self.eta_c, self.eta_m = eta_c, eta_m
        self.b1, self.b2, self.b3 = b
        if ablation == 'wo_rconv':
            self.b1 = 0.0
        if ablation == 'wo_rKT':
            self.b2 = 0.0
        self.xi = xi
        self.compute_reward = compute_reward
        self.ablation = ablation
        D = dim if dim is not None else (init_pops.shape[2] if init_pops is not None else None)
        n_oc, n_ot = 4, 7 * self.K
        if ablation == 'wo_Oc':
            self.state_dim = n_ot
        elif ablation == 'wo_Ot':
            self.state_dim = n_oc
        elif ablation == 'wo_FE':
            assert D is not None, 'wo_FE needs the dimension'
            self.state_dim = self.K * self.N * (D + 1)
        else:
            self.state_dim = n_oc + n_ot
        self.action_dim = (2 if ablation in self._REMOVED_ACTION else 3) * self.K

    def full_action(self, action):
        """Map the agent's output to the (K, 3) KT action, re-inserting a removed action."""
        a = np.asarray(action, dtype=np.float64).reshape(self.K, -1)
        if self.ablation in self._REMOVED_ACTION:
            i, v = self._REMOVED_ACTION[self.ablation]
            a = np.insert(a, i, v, axis=1)
        return np.clip(a, 0.0, 1.0)

    # ------------------------------------------------------------------ episode API

    def reset(self, tasks, rng):
        """Start an episode on `tasks` (list of K task objects). Returns the initial state."""
        assert len(tasks) == self.K
        self.tasks, self.rng = tasks, rng
        self.D = tasks[0].dim if self.init_pops is None else self.init_pops.shape[2]
        assert all(t.dim <= self.D for t in tasks)
        self.g = 0
        self.fes = 0
        self.X, self.Y = [], []
        for k in range(self.K):
            if self.init_pops is None:
                X = rng.random((self.N, self.D))
            else:                                             # pseudo-random initialisation
                X = self.init_pops[rng.integers(len(self.init_pops))].copy()
            self.X.append(X)
            self.Y.append(self._evaluate(k, X))
        self.best_y = np.array([Y.min() for Y in self.Y])
        self.best_x = [X[Y.argmin()].copy() for X, Y in zip(self.X, self.Y)]
        self.err0 = np.array([max(self.best_y[k] - t.fopt, 1e-300) if self.compute_reward else 1.0
                              for k, t in enumerate(tasks)])
        self.stagnation = np.zeros(self.K)
        self.flag_improved = np.zeros(self.K)
        self.q_kt = np.zeros(self.K)
        self.prev_action = np.zeros((self.K, 3))                # history features start at 0
        self.curve = [self.best_y.copy()]                       # best-so-far per generation
        if self.base == 'JADE':
            self.uF, self.uCR = np.full(self.K, 0.5), np.full(self.K, 0.5)
        return self.get_state()

    def step(self, action):
        """One generation for both tasks. Returns (state, reward, done, info)."""
        a = self.full_action(action)
        if self.base == 'DE':
            U, is_kt = self._offspring_de(a)
        elif self.base == 'JADE':
            U, is_kt = self._offspring_jade(a)
        else:
            U, is_kt = self._offspring_ga(a)

        reward = 0.0
        rewards = np.zeros(self.K)
        for k in range(self.K):
            YU = self._evaluate(k, U[k])
            # s(y) of Eq. (17): fraction of the parent population that offspring y beats
            score = (YU[:, None] < self.Y[k][None, :]).mean(axis=1)
            n_kt = int(is_kt[k].sum())
            if n_kt > 0:
                self.q_kt[k] = score[is_kt[k]].mean()                       # Eq. (11)
                r_kt = self.q_kt[k] - (score[~is_kt[k]].mean() if n_kt < len(YU) else 0.0)  # Eq. (16)
            else:
                self.q_kt[k] = 0.0
                r_kt = 0.0
            self._select(k, U[k], YU)
            improved = self.Y[k].min() < self.best_y[k]
            if improved:
                i = self.Y[k].argmin()
                self.best_y[k], self.best_x[k] = self.Y[k][i], self.X[k][i].copy()
                self.stagnation[k] = 0
            else:
                self.stagnation[k] += 1
            self.flag_improved[k] = float(improved)
            if self.compute_reward:
                err = self.best_y[k] - self.tasks[k].fopt
                r_conv = -err / self.err0[k]                                # Eq. (14)
                rewards[k] = self.b1 * r_conv + self.b2 * r_kt + self.b3 * float(err < self.xi)  # Eq. (13)
        reward = float(rewards.sum())                                       # Eq. (12)
        self.prev_action = a
        self.g += 1
        self.curve.append(self.best_y.copy())
        done = self.g >= self.horizon
        return self.get_state(), reward, done, {'rewards': rewards, 'n_kt': is_kt.sum(axis=1)}

    # ------------------------------------------------------------------ features (Table II)

    def get_state(self):
        if self.ablation == 'wo_FE':
            # Raw populations and fitness (Section III-B3). The paper does not say how
            # fitness is scaled; it is min-max normalised per task here.
            parts = []
            for X, Y in zip(self.X, self.Y):
                span = Y.max() - Y.min()
                parts += [X.ravel(), (Y - Y.min()) / span if span > 0 else np.zeros_like(Y)]
            return np.concatenate(parts).astype(np.float32)
        D, G = self.D, self.feature_horizon
        mu = [X.mean(axis=0) for X in self.X]
        sd = [X.std(axis=0) for X in self.X]
        oc = [self.g / G,
              np.linalg.norm(self.best_x[0] - self.best_x[1]) / np.sqrt(D),
              np.linalg.norm(mu[0] - mu[1]) / np.sqrt(D),
              np.linalg.norm(sd[0] - sd[1]) / np.sqrt(0.5 * D)]
        ot = []
        for k in range(self.K):
            ot += [self.stagnation[k] / G, self.flag_improved[k], self.q_kt[k], sd[k].mean(),
                   *self.prev_action[k]]
        feats = {'wo_Oc': ot, 'wo_Ot': oc}.get(self.ablation, oc + ot)
        return np.clip(np.array(feats, dtype=np.float32), 0.0, 1.0)

    # ------------------------------------------------------------------ base solvers

    def _offspring_de(self, a):
        """Algorithm 1, lines 10-17, for both tasks from the current (parent) populations."""
        U, is_kt = [], np.zeros((self.K, self.N), dtype=bool)
        for k in range(self.K):
            j = self._source(k)
            Uk = ops.de_rand_1_bin(self.X[k], self.F, self.CR, self.rng)
            n_kt = int(np.ceil(0.5 * a[k, 0] * self.N))
            if n_kt > 0:
                idx = self.rng.choice(self.N, n_kt, replace=False)
                Uk[idx] = ops.kt_trial(self.X[k], self.X[j], idx, a[k, 1], a[k, 2], self.F, self.CR, self.rng)
                is_kt[k, idx] = True
            U.append(Uk)
        return U, is_kt

    def _offspring_ga(self, a):
        """Algorithm S.1, lines 8-24: MFEA-style pairing over the merged population."""
        N, rng = self.N, self.rng
        pool = np.concatenate(self.X)
        owner = np.repeat(np.arange(self.K), N)
        order = rng.permutation(len(pool))
        ia, ib = order[0::2], order[1::2]                 # pairs drawn without replacement
        ka, kb = owner[ia], owner[ib]
        pa, pb = pool[ia], pool[ib]
        ua = ops.polynomial_mutation(pa, self.eta_m, rng)  # default: mutate both parents
        ub = ops.polynomial_mutation(pb, self.eta_m, rng)
        kt_a = np.zeros(len(ia), dtype=bool)
        same = ka == kb
        if same.any():                                    # same task: SBX + PM
            c1, c2 = ops.sbx(pa[same], pb[same], self.eta_c, rng)
            ua[same] = ops.polynomial_mutation(c1, self.eta_m, rng)
            ub[same] = ops.polynomial_mutation(c2, self.eta_m, rng)
        cross = np.flatnonzero(~same)
        kt_pairs = cross[rng.random(len(cross)) < a[ka[cross], 0]]   # rand < a_{ka,1}
        for k in range(self.K):                           # KT child for parent a, Eq. (10)
            m = kt_pairs[ka[kt_pairs] == k]
            if len(m):
                j = self._source(k)
                ua[m] = ops.kt_vector(self.X[k], self.X[j], len(m), a[k, 1], a[k, 2], self.F, rng)
                kt_a[m] = True
        U, is_kt = [], np.zeros((self.K, N), dtype=bool)
        for k in range(self.K):
            Uk = np.concatenate([ua[ka == k], ub[kb == k]])
            is_kt[k] = np.concatenate([kt_a[ka == k], np.zeros((kb == k).sum(), dtype=bool)])
            U.append(Uk)
        return U, is_kt

    def _offspring_jade(self, a, p=0.1):
        """JADE without archive as the base solver (Section IV-H3), MToP JADE.m operators.

        Every offspring i uses its own F_i and CR_i, including the KT offspring, whose
        Eq. (10) vector uses F_i and whose crossover uses CR_i.
        """
        rng, N = self.rng, self.N
        U, is_kt = [], np.zeros((self.K, N), dtype=bool)
        self._jade_params = []
        for k in range(self.K):
            X, Y = self.X[k], self.Y[k]
            Fi = np.empty(N)
            for i in range(N):
                f = 0.0
                while f <= 0:
                    f = self.uF[k] + 0.1 * np.tan(np.pi * (rng.random() - 0.5))   # Cauchy(uF, 0.1)
                Fi[i] = min(f, 1.0)
            CRi = np.clip(rng.normal(self.uCR[k], 0.1, N), 0.0, 1.0)
            pool = np.argsort(Y, kind='stable')[:max(round(p * N), 1)]
            pb = pool[rng.integers(len(pool), size=N)]
            r = ops._distinct_indices(rng, N, N, 2, exclude=np.arange(N))
            V = X + Fi[:, None] * (X[pb] - X) + Fi[:, None] * (X[r[:, 0]] - X[r[:, 1]])
            mask = rng.random((N, self.D)) <= CRi[:, None]
            mask[np.arange(N), rng.integers(0, self.D, N)] = True
            Uk = np.where(mask, V, X)
            Uk = np.where(Uk < 0, X / 2, np.where(Uk > 1, (X + 1) / 2, Uk))       # BoundaryMidpoint
            n_kt = int(np.ceil(0.5 * a[k, 0] * N))
            if n_kt > 0:
                j = self._source(k)
                idx = rng.choice(N, n_kt, replace=False)
                for i in idx:
                    v = ops.kt_vector(X, self.X[j], 1, a[k, 1], a[k, 2], Fi[i], rng)
                    Uk[i] = ops.binomial_crossover(v, X[i:i + 1], CRi[i], rng)[0]
                is_kt[k, idx] = True
            U.append(Uk)
            self._jade_params.append((Fi, CRi))
        return U, is_kt

    def _select(self, k, Uk, YU):
        # DE and GA: truncation selection, i.e. the best N of parents + offspring (Section
        # III-B2 states the convergence condition "under truncated selection mechanism").
        # JADE keeps its own one-to-one replacement; MetaBox's DE one-to-one is selection='one_to_one'.
        if self.base == 'JADE' or self.selection == 'one_to_one':
            better = YU <= self.Y[k]
            self.X[k][better], self.Y[k][better] = Uk[better], YU[better]
            if self.base == 'JADE' and better.any():
                Fi, CRi = self._jade_params[k]
                SF, SCR, c = Fi[better], CRi[better], 0.1
                for _ in range(better.sum()):      # MToP JADE.m updates once per success
                    self.uF[k] = (1 - c) * self.uF[k] + c * (SF ** 2).sum() / SF.sum()
                    self.uCR[k] = (1 - c) * self.uCR[k] + c * SCR.mean()
        else:
            X = np.concatenate([self.X[k], Uk])
            Y = np.concatenate([self.Y[k], YU])
            keep = np.argsort(Y, kind='stable')[:self.N]
            self.X[k], self.Y[k] = X[keep], Y[keep]

    def _source(self, k):
        others = [j for j in range(self.K) if j != k]
        return others[self.rng.integers(len(others))]

    def _evaluate(self, k, X):
        self.fes += len(X)
        return self.tasks[k].eval(X[:, :self.tasks[k].dim])
