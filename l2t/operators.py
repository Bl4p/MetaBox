"""Evolution operators in the unified space [0, 1]^D.

DE/rand/1 and binomial crossover follow MetaBox's DE_mutation / DE_crossover
(l2t_optimizer.py), vectorised, with the paper's CR = 0.5 (MetaBox uses 0.7).
`kt_trial` is Eq. (10), from MetaBox's mixed_DE.
"""
import numpy as np


def _distinct_indices(rng, n_rows, pool, k, exclude=None):
    """k distinct indices from range(pool) for each row; optionally never `exclude[row]`."""
    keys = rng.random((n_rows, pool))
    if exclude is not None:
        keys[np.arange(n_rows), exclude] = 2.0      # pushed to the end of the argsort
    return np.argsort(keys, axis=1)[:, :k]


def binomial_crossover(V, X, CR, rng):
    """Eq. (2): take v_d if rand <= CR or d == jrand, else x_d."""
    n, D = V.shape
    mask = rng.random((n, D)) <= CR
    mask[np.arange(n), rng.integers(0, D, n)] = True
    return np.where(mask, V, X)


def de_rand_1_bin(X, F, CR, rng):
    """Eqs. (1)-(2). r1, r2, r3 distinct and different from i; mutant clipped to [0, 1] (as MetaBox)."""
    N = len(X)
    r = _distinct_indices(rng, N, N, 3, exclude=np.arange(N))
    V = np.clip(X[r[:, 0]] + F * (X[r[:, 1]] - X[r[:, 2]]), 0.0, 1.0)
    return binomial_crossover(V, X, CR, rng)


def kt_vector(Xk, Xj, n, a2, a3, F, rng):
    """Eq. (10) for n trial vectors. r1, r3, r4 index the target X_k; r2, r5, r6 the source X_j.

    As in MetaBox's mixed_DE, the six indices of one trial vector are drawn without
    replacement (the paper only says "randomly drawn"). Clipped to [0, 1] like DE mutants.
    """
    r = _distinct_indices(rng, n, min(len(Xk), len(Xj)), 6)
    v = ((1 - a2) * Xk[r[:, 0]] + a2 * Xj[r[:, 1]]
         + F * (1 - a3) * (Xk[r[:, 2]] - Xk[r[:, 3]])
         + F * a3 * (Xj[r[:, 4]] - Xj[r[:, 5]]))
    return np.clip(v, 0.0, 1.0)


def kt_trial(Xk, Xj, idx, a2, a3, F, CR, rng):
    """Eq. (10) followed by binomial crossover with the parents X_k[idx] (Algorithm 1, lines 14-15)."""
    V = kt_vector(Xk, Xj, len(idx), a2, a3, F, rng)
    return binomial_crossover(V, Xk[idx], CR, rng)


# ----------------------------------------------------------------------------- GA (MTGA-L2T)

def sbx(p1, p2, eta, rng):
    """Simulated binary crossover on (n, D) parent arrays in [0, 1]; returns two children.

    MTO-Platform's GA_Crossover (used by its MFEA family): the spread factor gets a
    random sign, and each dimension is left uncrossed with probability 0.5.
    """
    u = rng.random(p1.shape)
    beta = np.where(u <= 0.5, (2 * u) ** (1 / (eta + 1)), (2 * (1 - u)) ** (-1 / (eta + 1)))
    beta = beta * np.where(rng.random(p1.shape) < 0.5, 1.0, -1.0)
    beta[rng.random(p1.shape) < 0.5] = 1.0
    c1 = 0.5 * ((1 + beta) * p1 + (1 - beta) * p2)
    c2 = 0.5 * ((1 + beta) * p2 + (1 - beta) * p1)
    return np.clip(c1, 0.0, 1.0), np.clip(c2, 0.0, 1.0)


def polynomial_mutation(X, eta, rng, pm=None):
    """Polynomial mutation in [0, 1]; each variable mutates with probability pm (default 1/D)."""
    X = np.atleast_2d(X)
    n, D = X.shape
    pm = 1.0 / D if pm is None else pm
    u = rng.random((n, D))
    delta = np.where(u < 0.5,
                     (2 * u + (1 - 2 * u) * (1 - X) ** (eta + 1)) ** (1 / (eta + 1)) - 1,
                     1 - (2 * (1 - u) + 2 * (u - 0.5) * X ** (eta + 1)) ** (1 / (eta + 1)))
    mutate = rng.random((n, D)) < pm
    return np.clip(np.where(mutate, X + delta, X), 0.0, 1.0)
