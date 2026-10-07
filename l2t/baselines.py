"""Peer EMT algorithms of the paper, ported to Python for K = 2 tasks.

Sources
-------
Ported from MTO-Platform (MToP, github.com/intLyc/MTO-Platform, master, MATLAB):
    MFEA, MFEA-II (MFEA2), G-MFEA, MFEA-AKT, MTEA-AD, AT-MFEA (ATMFEA), MFDE, MKTDE,
    EMEA (as MTDE-EA, DE operator for every task)
Written from other descriptions, because MToP has no synchronous version:
    AEMTO   from MToP's stream variant AAEMTO (same adaptive rules, one batch per generation)
    MTDE-AD MTEA-AD's anomaly-detection transfer on a DE/rand/1/bin base solver
    MTDE-B  base-vector transfer (Jin et al., CEC 2019) as summarised in the L2T paper and its
            supplement: "transfer of elite solutions from the source task as the base vector".
            The transfer probability is not given there; 0.3 (MToP's usual RMP) is used.
Not implemented: MFEA-RL (no description available).

Common settings, so that every algorithm has the same base solver as L2T (Section IV-A):
    N = 50 per task, DE F = 0.5 and CR = 0.5, SBX eta_c = 2, PM eta_m = 5, all in [0, 1]^D.
Initial populations come from the same pre-generated LHS set as L2T, chosen by the run
seed, so every algorithm starts run r of instance i from the same populations. MToP's
multifactorial initialisation (evaluating everyone on every task) is replaced by giving
each task its own population, as the L2T environment does.

Every algorithm returns the best-so-far error per generation, shape (G_max + 1, 2).
"""
import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import logsumexp

from . import operators as ops
from .problems import build_mtop, get_set, test_instances
from .rollout import N_P, INIT_POP_SEED, _init_pops

N, F, CR, ETA_C, ETA_M = 50, 0.5, 0.5, 2.0, 5.0
RMP = 0.3


# ============================================================================ harness

class Run:
    """Two tasks, their evaluation, and the best-so-far record per generation."""

    def __init__(self, tasks, rng, g_max, init_pops):
        self.tasks, self.rng, self.g_max = tasks, rng, g_max
        self.K = len(tasks)
        self.fopt = np.array([t.fopt for t in tasks])
        self.best = np.full(self.K, np.inf)
        self.curve = []
        self.X = [init_pops[rng.integers(len(init_pops))].copy() for _ in range(self.K)]
        self.Y = [self.eval(k, X) for k, X in enumerate(self.X)]
        self.end_generation()

    def eval(self, k, X):
        X = np.atleast_2d(X)
        if len(X) == 0:
            return np.zeros(0)
        y = self.tasks[k].eval(X[:, :self.tasks[k].dim])
        self.best[k] = min(self.best[k], y.min())
        return y

    def end_generation(self):
        self.curve.append(self.best - self.fopt)

    @property
    def gen(self):                      # generations completed so far
        return len(self.curve) - 1

    def running(self):
        return self.gen < self.g_max

    def result(self):
        return np.array(self.curve)


def elitist(X, Y, n):
    """Selection_Elit / Selection_MF: keep the n best of the pool, sorted best first."""
    order = np.argsort(Y, kind='stable')[:n]
    return X[order], Y[order], order


def sort_pop(run, k):
    order = np.argsort(run.Y[k], kind='stable')
    run.X[k], run.Y[k] = run.X[k][order], run.Y[k][order]


def de_generation(X, rng, F=F, CR=CR):
    """MToP Generation_DE: DE/rand/1 with x1, x2, x3 distinct and != i, binomial crossover, clip."""
    n = len(X)
    r = ops._distinct_indices(rng, n, n, 3, exclude=np.arange(n))
    V = X[r[:, 0]] + F * (X[r[:, 1]] - X[r[:, 2]])
    return np.clip(ops.binomial_crossover(V, X, CR, rng), 0.0, 1.0)


def ga_generation(X, rng, swap=0.5):
    """MToP MTEA-AD / EMEA single-population GA: random pairs, SBX + PM, variable swap."""
    n = len(X)
    order = rng.permutation(n)
    half = n // 2
    p1, p2 = X[order[:half]], X[order[half:2 * half]]
    c1, c2 = ops.sbx(p1, p2, ETA_C, rng)
    c1, c2 = ops.polynomial_mutation(c1, ETA_M, rng), ops.polynomial_mutation(c2, ETA_M, rng)
    if swap is not None:
        s = rng.random(c1.shape) < swap
        c1, c2 = np.where(s, c2, c1), np.where(s, c1, c2)
    return np.clip(np.concatenate([c1, c2]), 0.0, 1.0)


def merged(run):
    """Multifactorial view: merged decision matrix and skill factors."""
    return np.concatenate(run.X), np.repeat(np.arange(run.K), [len(x) for x in run.X])


def mf_select(run, Off, factor):
    """Evaluate offspring on their skill-factor task, then per-task elitist selection (Selection_MF)."""
    for k in range(run.K):
        Ok = Off[factor == k]
        Yk = run.eval(k, Ok)
        run.X[k], run.Y[k], _ = elitist(np.concatenate([run.X[k], Ok]), np.concatenate([run.Y[k], Yk]), N)


# ============================================================================ GA-based

def mfea(run, rmp=RMP):
    """MFEA (Gupta et al., 2016). MToP MFEA.m."""
    rng = run.rng
    while run.running():
        P, fac = merged(run)
        n = len(P); order = rng.permutation(n); half = n // 2
        Off, ofac = [], []
        for i in range(half):
            a, b = order[i], order[i + half]
            if fac[a] == fac[b] or rng.random() < rmp:
                c1, c2 = ops.sbx(P[a:a + 1], P[b:b + 1], ETA_C, rng)
                Off += [c1[0], c2[0]]
                ofac += [fac[[a, b][rng.integers(2)]], fac[[a, b][rng.integers(2)]]]
            else:
                Off += [ops.polynomial_mutation(P[a], ETA_M, rng)[0], ops.polynomial_mutation(P[b], ETA_M, rng)[0]]
                ofac += [fac[a], fac[b]]
        mf_select(run, np.clip(np.array(Off), 0, 1), np.array(ofac))
        run.end_generation()
    return run.result()


def _learn_rmp(subpops, rng):
    """MToP learnRMP.m for two tasks, with log-space likelihoods (no underflow at D = 50)."""
    models = []
    for data in subpops:
        n_rand = int(np.floor(0.1 * len(data)))
        comb = np.vstack([data, rng.random((n_rand, data.shape[1]))])
        models.append((comb.mean(0), comb.std(0, ddof=1)))

    def loglik_matrix(data):
        cols = []
        for mu, sd in models:
            sd = np.maximum(sd, 1e-12)
            cols.append(np.sum(-np.log(sd * np.sqrt(2 * np.pi)) - 0.5 * ((data - mu) / sd) ** 2, axis=1))
        return np.stack(cols, axis=1)

    L1, L2 = loglik_matrix(subpops[0]), loglik_matrix(subpops[1])

    def nll(r):
        w_diff = 0.5 * r / 2
        w_same = 1 - w_diff
        lw = np.log([w_same, max(w_diff, 1e-300)])
        return -(logsumexp(L1 + lw, axis=1).sum() + logsumexp(L2 + lw[::-1], axis=1).sum())

    r = minimize_scalar(nll, bounds=(0, 1), method='bounded').x
    return float(np.clip(r + rng.normal(0, 0.01), 0, 1))


def mfea_ii(run, swap=0.5):
    """MFEA-II (Bali et al., 2020). MToP MFEA_II.m: online RMP learning every generation."""
    rng = run.rng
    while run.running():
        rmp = _learn_rmp(run.X, rng)
        P, fac = merged(run)
        n = len(P); order = rng.permutation(n); half = n // 2
        Off, ofac = [], []

        def cx_mut_swap(x, y):
            c1, c2 = ops.sbx(x[None], y[None], ETA_C, rng)
            c1, c2 = ops.polynomial_mutation(c1, ETA_M, rng)[0], ops.polynomial_mutation(c2, ETA_M, rng)[0]
            s = rng.random(len(x)) >= swap
            return np.where(s, c2, c1), np.where(s, c1, c2)

        for i in range(half):
            a, b = order[i], order[i + half]
            if fac[a] == fac[b] or rng.random() < rmp:
                c1, c2 = cx_mut_swap(P[a], P[b])
                Off += [c1, c2]
                ofac += [fac[[a, b][rng.integers(2)]], fac[[a, b][rng.integers(2)]]]
            else:
                for p in (a, b):
                    same = np.flatnonzero(fac == fac[p]); same = same[same != p]
                    q = same[rng.integers(len(same))]
                    c1, _ = cx_mut_swap(P[p], P[q])
                    Off.append(c1); ofac.append(fac[p])
        mf_select(run, np.clip(np.array(Off), 0, 1), np.array(ofac))
        run.end_generation()
    return run.result()


def g_mfea(run, rmp=RMP, phi=0.1, theta=0.02, top=0.4):
    """G-MFEA (Ding et al., 2019). MToP G_MFEA.m, decision-variable translation.

    All tasks here have the same dimension, so the decision-variable shuffling strategy
    (meant for tasks of different dimensions) is not applied; MToP's code would permute
    the variables of one task even when the dimensions are equal.
    """
    rng = run.rng
    D = run.X[0].shape[1]
    transfer = {(0, 1): np.zeros(D), (1, 0): np.zeros(D)}
    while run.running():
        P, fac = merged(run)
        n = len(P); order = rng.permutation(n); half = n // 2
        Off, ofac = [], []
        for i in range(half):
            a, b = order[i], order[i + half]
            if fac[a] == fac[b]:
                c1, c2 = ops.sbx(P[a:a + 1], P[b:b + 1], ETA_C, rng)
                Off += [c1[0], c2[0]]
                ofac += [fac[[a, b][rng.integers(2)]], fac[[a, b][rng.integers(2)]]]
            elif rng.random() < rmp:
                ta, tb = transfer[(fac[a], fac[b])], transfer[(fac[b], fac[a])]
                c1, c2 = ops.sbx((P[a] + ta)[None], (P[b] + tb)[None], ETA_C, rng)
                Off += [c1[0] - ta, c2[0] - tb]
                ofac += [fac[[a, b][rng.integers(2)]], fac[[a, b][rng.integers(2)]]]
            else:
                Off += [ops.polynomial_mutation(P[a], ETA_M, rng)[0], ops.polynomial_mutation(P[b], ETA_M, rng)[0]]
                ofac += [fac[a], fac[b]]
        mf_select(run, np.clip(np.array(Off), 0, 1), np.array(ofac))
        run.end_generation()
        g = run.gen
        if g >= phi * run.g_max and g % max(1, round(theta * run.g_max)) == 0:
            alpha = (g / run.g_max) ** 2                       # (FE / maxFE)^2
            mean_t = [run.X[k][:round(top * N)].mean(0) for k in range(run.K)]   # X sorted by elitist()
            transfer[(0, 1)] = alpha * (0.5 - mean_t[0])
            transfer[(1, 0)] = alpha * (0.5 - mean_t[1])
    return run.result()


def _akt_crossover(x, y, alpha, rng):
    """MFEA-AKT's six crossover operators (MToP hyberCX); returns two children."""
    D = len(x)
    def tp(p, q):
        i, j = sorted(rng.integers(0, D, 2))
        c = p.copy(); c[i:j + 1] = q[i:j + 1]; return c
    def uf(p, q):
        return np.where(rng.integers(0, 2, D) == 1, q, p)
    def ari(p, q):
        return 0.25 * p + 0.75 * q
    def geo(p, q):
        return np.abs(p) ** 0.2 * np.abs(q) ** 0.8
    def blx(p, q, a=0.3):
        lo, hi = np.minimum(p, q), np.maximum(p, q); I = hi - lo
        return lo - I * a + (I + 2 * I * a) * rng.random(D)
    if alpha == 6:
        c1, c2 = ops.sbx(x[None], y[None], ETA_C, rng)
        return c1[0], c2[0]
    f = {1: tp, 2: uf, 3: ari, 4: geo, 5: blx}[alpha]
    return f(x, y), f(y, x)


def mfea_akt(run, rmp=RMP, gap=20):
    """MFEA-AKT (Zhou et al., 2021). MToP MFEA_AKT.m: adaptive choice among 6 crossovers."""
    rng = run.rng
    cx = [rng.integers(1, 7, len(x)) for x in run.X]          # CXFactor per individual
    record = []
    while run.running():
        P, fac = merged(run)
        PY = np.concatenate(run.Y)
        Pcx = np.concatenate(cx)
        n = len(P); order = rng.permutation(n); half = n // 2
        Off, ofac, ocx, opar = [], [], [], []
        for i in range(half):
            a, b = order[i], order[i + half]
            if fac[a] == fac[b] or rng.random() < rmp:
                if fac[a] == fac[b]:
                    c1, c2 = ops.sbx(P[a:a + 1], P[b:b + 1], ETA_C, rng)
                    kids, cxs, tran = [c1[0], c2[0]], [Pcx[a], Pcx[b]], False
                else:
                    al = Pcx[[a, b][rng.integers(2)]]
                    c1, c2 = _akt_crossover(P[a], P[b], al, rng)
                    kids, cxs, tran = [c1, c2], [al, al], True
                for kid, c in zip(kids, cxs):
                    p = [a, b][rng.integers(2)]
                    Off.append(kid); ofac.append(fac[p]); ocx.append(c); opar.append(p if tran else -1)
            else:
                for p in (a, b):
                    Off.append(ops.polynomial_mutation(P[p], ETA_M, rng)[0]); ofac.append(fac[p])
                    ocx.append(Pcx[p]); opar.append(-1)
        Off = np.clip(np.array(Off), 0, 1); ofac = np.array(ofac); ocx = np.array(ocx); opar = np.array(opar)
        OY = np.zeros(len(Off))
        for k in range(run.K):
            m = ofac == k
            OY[m] = run.eval(k, Off[m])
        # adapt the crossover factor
        imp = np.zeros(7)
        for i in np.flatnonzero(opar >= 0):
            pf = PY[opar[i]]
            rel = (pf - OY[i]) / pf if pf != 0 else 0.0
            imp[ocx[i]] = max(imp[ocx[i]], rel)
        if imp[1:].any():
            best_cx = int(np.argmax(imp[1:]) + 1)
        else:
            window = record[-gap:] if record else [int(rng.integers(1, 7))]
            best_cx = int(np.bincount(window, minlength=7)[1:].argmax() + 1)
        record.append(best_cx)
        for i in range(len(Off)):
            if opar[i] >= 0:
                pf = PY[opar[i]]
                if pf != 0 and (pf - OY[i]) / pf < 0:
                    ocx[i] = best_cx
            else:
                ocx[i] = [best_cx, int(rng.integers(1, 7))][rng.integers(2)]
        for k in range(run.K):
            m = ofac == k
            X = np.concatenate([run.X[k], Off[m]]); Y = np.concatenate([run.Y[k], OY[m]])
            C = np.concatenate([cx[k], ocx[m]])
            run.X[k], run.Y[k], keep = elitist(X, Y, N)
            cx[k] = C[keep]
        run.end_generation()
    return run.result()


def _anomaly_transfer(curr, source, nl, rng):
    """MTEA-AD learn_anomaly_detection: source solutions most likely under a Gaussian of `curr`."""
    n_rand = int(np.floor(0.01 * len(curr)))
    data = np.vstack([curr, rng.random((n_rand, curr.shape[1]))]) if n_rand else curr
    mu = data.mean(0)
    cov = np.cov(data, rowvar=False) + 1e-5 * np.eye(data.shape[1])
    src = np.unique(source, axis=0)
    diff = src - mu
    sol = np.linalg.solve(cov, diff.T).T
    logp = -0.5 * np.sum(diff * sol, axis=1)                  # mvnpdf up to a constant
    ranked = np.sort(logp)[::-1]
    thr = ranked[0] if nl == 0 else ranked[int(np.ceil(len(ranked) * nl)) - 1]
    return src[logp >= thr]


def _mtea_ad(run, generate, trp=0.1):
    rng = run.rng
    eps = np.zeros(run.K)
    while run.running():
        for t in range(run.K):
            Off = generate(run.X[t], rng)
            if rng.random() < trp:
                nl = 1.0 if run.gen == 0 else eps[t]
                src = np.concatenate([run.X[k] for k in range(run.K) if k != t])
                Tr = np.clip(_anomaly_transfer(Off, src, nl, rng), 0, 1)
                OY, TY = run.eval(t, Off), run.eval(t, Tr)
                X = np.concatenate([run.X[t], Off, Tr]); Y = np.concatenate([run.Y[t], OY, TY])
                run.X[t], run.Y[t], keep = elitist(X, Y, N)
                eps[t] = np.sum(keep >= N + len(Off)) / len(Tr)
            else:
                OY = run.eval(t, Off)
                run.X[t], run.Y[t], _ = elitist(np.concatenate([run.X[t], Off]), np.concatenate([run.Y[t], OY]), N)
        run.end_generation()
    return run.result()


def mtea_ad(run):
    """MTEA-AD (Wang et al., 2022), GA solver. MToP MTEA_AD.m."""
    return _mtea_ad(run, ga_generation)


def mtde_ad(run):
    """MTDE-AD: MTEA-AD's anomaly-detection transfer with DE/rand/1/bin as the solver."""
    return _mtea_ad(run, de_generation)


def at_mfea(run, rmp=RMP, swap=0.5, c_mu=0.5):
    """AT-MFEA (Xue et al., 2022). MToP AT_MFEA.m with diagonal Gaussian affine transfer."""
    rng = run.rng
    mu = [x.mean(0) for x in run.X]
    var = [x.var(0, ddof=1) for x in run.X]

    def transfer(x, s, t):     # AT_Transfer for diagonal covariances: N(mu_s, S_s) -> N(mu_t, S_t)
        return mu[t] + np.sqrt(var[t] / np.maximum(var[s], 1e-12)) * (x - mu[s])

    def cx_mut_swap(x, y, do_swap=True):
        c1, c2 = ops.sbx(x[None], y[None], ETA_C, rng)
        c1, c2 = ops.polynomial_mutation(c1, ETA_M, rng)[0], ops.polynomial_mutation(c2, ETA_M, rng)[0]
        if do_swap:
            s = rng.random(len(x)) >= swap
            c1, c2 = np.where(s, c2, c1), np.where(s, c1, c2)
        return c1, c2

    while run.running():
        P, fac = merged(run)
        n = len(P); order = rng.permutation(n); half = n // 2
        Off, ofac = [], []
        for i in range(half):
            a, b = order[i], order[i + half]
            if fac[a] == fac[b]:
                c1, c2 = cx_mut_swap(P[a], P[b])
                Off += [c1, c2]
                ofac += [fac[[a, b][rng.integers(2)]], fac[[a, b][rng.integers(2)]]]
            elif rng.random() < rmp:
                pa = transfer(P[a], fac[a], fac[b])
                pb = transfer(P[b], fac[b], fac[a])
                c1, _ = ops.sbx(pa[None], P[b][None], ETA_C, rng)
                c2, _ = ops.sbx(P[a][None], pb[None], ETA_C, rng)
                Off += [ops.polynomial_mutation(c1, ETA_M, rng)[0], ops.polynomial_mutation(c2, ETA_M, rng)[0]]
                ofac += [fac[[a, b][rng.integers(2)]], fac[[a, b][rng.integers(2)]]]
            else:
                for p in (a, b):
                    same = np.flatnonzero(fac == fac[p]); same = same[same != p]
                    q = same[rng.integers(len(same))]
                    c1, _ = cx_mut_swap(P[p], P[q])
                    Off.append(c1); ofac.append(fac[p])
        mf_select(run, np.clip(np.array(Off), 0, 1), np.array(ofac))
        for k in range(run.K):
            mu[k] = (1 - c_mu) * mu[k] + c_mu * run.X[k].mean(0)
            var[k] = (1 - c_mu) * var[k] + c_mu * run.X[k].var(0, ddof=1)
        run.end_generation()
    return run.result()


# ============================================================================ DE-based

def mfde(run, rmp=RMP):
    """MFDE (Feng et al., 2017). MToP MFDE.m (CR set to the common 0.5)."""
    rng = run.rng
    while run.running():
        P, fac = merged(run)
        n = len(P)
        Off, ofac = np.empty_like(P), np.empty(n, dtype=int)
        for i in range(n):
            same = np.flatnonzero(fac == fac[i])
            other = np.flatnonzero(fac != fac[i])
            x1 = rng.choice(same[same != i])
            if rng.random() < rmp:
                x2, x3 = rng.choice(other, 2, replace=False)
                ofac[i] = fac[[x1, x2, x3][rng.integers(3)]]
            else:
                x2, x3 = rng.choice(same[(same != i) & (same != x1)], 2, replace=False)
                ofac[i] = fac[i]
            v = P[x1] + F * (P[x2] - P[x3])
            Off[i] = np.clip(ops.binomial_crossover(v[None], P[i][None], CR, rng)[0], 0, 1)
        mf_select(run, Off, ofac)
        run.end_generation()
    return run.result()


def mtde_b(run, rmp=RMP):
    """MTDE-B: DE/rand/1/bin where, with probability rmp, the base vector is the source task's
    best solution; truncation selection like the other DE-based methods."""
    rng = run.rng
    while run.running():
        snapshot = [(x.copy(), y.copy()) for x, y in zip(run.X, run.Y)]
        for t in range(run.K):
            X = snapshot[t][0]
            Xs, Ys = snapshot[1 - t]
            n = len(X)
            r = ops._distinct_indices(rng, n, n, 3, exclude=np.arange(n))
            base = X[r[:, 0]].copy()
            tr = rng.random(n) < rmp
            base[tr] = Xs[np.argmin(Ys)]
            V = base + F * (X[r[:, 1]] - X[r[:, 2]])
            U = np.clip(ops.binomial_crossover(V, X, CR, rng), 0, 1)
            UY = run.eval(t, U)
            run.X[t], run.Y[t], _ = elitist(np.concatenate([run.X[t], U]), np.concatenate([run.Y[t], UY]), N)
        run.end_generation()
    return run.result()


def mktde(run):
    """MKTDE (Li et al., 2022). MToP MKTDE.m: centroid-aligned source differences + elite transfer."""
    rng = run.rng
    for k in range(run.K):
        sort_pop(run, k)
    while run.running():
        cent = [x.mean(0) for x in run.X]
        src = [1 - t for t in range(run.K)]
        for t in range(run.K):
            X, s = run.X[t], src[t]
            popf = np.concatenate([X, run.X[s] - cent[s] + cent[t]])
            n, m = len(X), len(popf)
            Off = np.empty_like(X)
            for i in range(n):
                x1 = rng.choice([j for j in range(n) if j != i])
                x2, x3 = rng.choice([j for j in range(m) if j not in (i, x1)], 2, replace=False)
                v = X[x1] + F * (popf[x2] - popf[x3])
                Off[i] = np.clip(ops.binomial_crossover(v[None], X[i][None], CR, rng)[0], 0, 1)
            OY = run.eval(t, Off)
            run.X[t], run.Y[t], _ = elitist(np.concatenate([run.X[t], Off]), np.concatenate([run.Y[t], OY]), N)
        for t in range(run.K):                            # elite transfer: worst <- source best
            run.X[t][-1] = run.X[src[t]][0]
            run.Y[t][-1] = run.eval(t, run.X[t][-1:])[0]
            sort_pop(run, t)
        run.end_generation()
    return run.result()


def aemto(run, w=0.3, p_lb=0.05, p_ub=0.7):
    """AEMTO (Xu et al., 2021), from MToP's AAEMTO run synchronously, K = 2.

    Each generation a task either transfers (with probability p_transfer) by binomial
    crossover of its parents with roulette-selected source individuals, or evolves
    by DE/rand/1/bin. p_transfer adapts to the survival rates of the two kinds of
    offspring. With two tasks the source-selection probabilities are trivial.
    """
    rng = run.rng
    p_tr = np.full(run.K, (p_lb + p_ub) / 2)
    q_self, q_other = np.zeros(run.K), np.zeros(run.K)
    while run.running():
        for t in range(run.K):
            parent, py = run.X[t], run.Y[t]
            transfer = rng.random() <= p_tr[t]
            if transfer:
                s = 1 - t
                sy = run.Y[s]
                wts = np.cumsum(1.0 / (sy - min(sy.min(), 0) + 1e-6))
                idx = np.searchsorted(wts / wts[-1], rng.random(len(parent)))
                batch = ops.binomial_crossover(run.X[s][idx], parent, CR, rng)
            else:
                # AAEMTO re-samples out-of-bound values at random; clipped here like the other
                # algorithms so that every method shares the same base solver (Section IV-A2)
                batch = de_generation(parent, rng)
            by = run.eval(t, batch)
            X = np.concatenate([batch, parent]); Y = np.concatenate([by, py])
            run.X[t], run.Y[t], keep = elitist(X, Y, N)
            survived = np.sum(keep < len(batch)) / N
            if transfer:
                q_other[t] = w * q_other[t] + (1 - w) * survived
            else:
                q_self[t] = w * q_self[t] + (1 - w) * survived
            p_tr[t] = p_lb + q_other[t] / (q_other[t] + q_self[t] + 0.001) * (p_ub - p_lb)
        run.end_generation()
    return run.result()


def _mda(curr, his):
    """MToP mDA.m: linear map W with W [his; 1] ~ [curr; 1] (least squares, ridge 1e-5)."""
    xx, noise = curr.T, his.T
    d, n = xx.shape
    xxb = np.vstack([xx, np.ones((1, n))])
    nb = np.vstack([noise, np.ones((1, n))])
    Q, Pm = nb @ nb.T, xxb @ nb.T
    reg = 1e-5 * np.eye(d + 1); reg[-1, -1] = 0
    W = np.linalg.solve((Q + reg).T, Pm.T).T
    return W[:-1, :-1]


def mtde_ea(run, s_num=10, t_gap=10):
    """MTDE-EA: EMEA (Feng et al., 2019) with DE for every task. MToP EMEA.m."""
    rng = run.rng
    init_sorted = [run.X[k][np.argsort(run.Y[k], kind='stable')].copy() for k in range(run.K)]
    while run.running():
        for t in range(run.K):
            Off = de_generation(run.X[t], rng)
            if s_num > 0 and (run.gen + 1) % t_gap == 0:
                inj = []
                for k in range(run.K):
                    if k == t:
                        continue
                    best_k = run.X[k][np.argsort(run.Y[k], kind='stable')[:s_num]]
                    W = _mda(init_sorted[t], init_sorted[k])
                    inj.append((W @ best_k.T).T)
                inj = np.clip(np.concatenate(inj), 0, 1)
                Off[rng.choice(len(Off), len(inj), replace=False)] = inj
            OY = run.eval(t, Off)
            run.X[t], run.Y[t], _ = elitist(np.concatenate([run.X[t], Off]), np.concatenate([run.Y[t], OY]), N)
        run.end_generation()
    return run.result()


# ============================================================================ registry & jobs

BASELINES = {
    # DE-based (compared with MTDE-L2T)
    'AEMTO': aemto, 'MFDE': mfde, 'MKTDE': mktde, 'MTDE-AD': mtde_ad, 'MTDE-B': mtde_b,
    'MTDE-EA': mtde_ea,
    # GA-based (compared with MTGA-L2T)
    'MFEA': mfea, 'MFEA2': mfea_ii, 'GMFEA': g_mfea, 'MFEA-AKT': mfea_akt, 'MTEA-AD': mtea_ad,
    'ATMFEA': at_mfea,
}


def run_baseline(name, mtop_spec, seed, g_max, dim, n_pops=N_P, init_seed=INIT_POP_SEED):
    tasks = build_mtop(mtop_spec)
    rng = np.random.default_rng(seed)
    run = Run(tasks, rng, g_max, _init_pops(n_pops, N, dim, init_seed))
    return BASELINES[name](run)


def run_baseline_job(job):
    return run_baseline(*job)


def baseline_jobs(set_name, name, base, g_max, n_runs, dim=None):
    from .evaluate import RUN_SEED
    s = get_set(set_name, dim)
    return [(name, spec, int(np.random.SeedSequence([RUN_SEED, i, r]).generate_state(1)[0]), g_max, s.dim)
            for i, spec in enumerate(test_instances(set_name, dim=dim)) for r in range(n_runs)]
