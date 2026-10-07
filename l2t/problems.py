"""Benchmark suites of the paper (supplement Section S.III, Tables S.II and S.III).

Every task works in the unified space [0, 1]^D: `task.eval(X)` takes an (n, D) array
in [0, 1] and returns n objective values. `task.fopt` is the optimal value f*.

Base functions are MetaBox's: the CEC17MTO basic functions
(environment/problem/MTO/CEC2017MTO/cec2017mto_numpy.py) for suite 1 and the BBOB
functions (environment/problem/SOO/COCO_BBOB/bbob_numpy.py) for suite 2.

MTOP instances are described by small picklable specs, e.g.
    ('cec', 'Rastrigin', xo)  or  ('bbob', fid, sid)
so they can be shipped to worker processes and rebuilt there with `build_task`.
"""
import numpy as np
from scipy.stats import qmc

from metaevobox.environment.problem.MTO.CEC2017MTO import cec2017mto_numpy as cec
from metaevobox.environment.problem.SOO.COCO_BBOB import bbob_numpy as bbob

CEC_FUNCTIONS = ('Ackley', 'Griewank', 'Rastrigin', 'Sphere', 'Weierstrass')
BBOB_LB, BBOB_UB = -5.0, 5.0


# ----------------------------------------------------------------------------- tasks

class CECTask:
    """f(M (y - o)) on the function's native box, encoded to [0, 1]^D.

    The optimum x_O is given in [0, 1]^D and decoded with the function's own bounds
    (Wu et al., "Orthogonal transfer for multitask optimization", supplement ref [16]).
    The paper does not say whether suite 1 is rotated; M = identity here.
    """

    def __init__(self, name, xo):
        self.name = name
        self.dim = len(xo)
        f = getattr(cec, name)(self.dim)          # MetaBox class; sets lb/ub
        f.shift = f.lb + np.asarray(xo) * (f.ub - f.lb)
        f.opt = f.shift
        f.optimum = f.func(f.opt.reshape(1, -1))[0]
        self._f = f
        self.fopt = float(f.optimum)

    def eval(self, X):
        return np.asarray(self._f.eval(X), dtype=np.float64)   # MetaBox eval decodes [0,1] -> box

    def __repr__(self):
        return f'{self.name}'


class BBOBTask:
    """BBOB function `fid` whose rotation, shift and bias are generated from seed `sid`.

    Uses MetaBox's instance generator (bbob_dataset.get_datasets): shift uniform in
    0.8 * [lb, ub] = [-4, 4]^D, Householder rotation, bias in {100, ..., 2500}.
    """

    def __init__(self, fid, sid, dim):
        self.name = f'F{fid}'
        self.fid, self.sid, self.dim = fid, sid, dim
        state = np.random.get_state()
        np.random.seed(sid)                        # MetaBox draws everything from np.random
        try:
            shift = 0.8 * (np.random.random(dim) * (BBOB_UB - BBOB_LB) + BBOB_LB)
            H = bbob.rotate_gen(dim)
            bias = np.random.randint(1, 26) * 100
            self._f = getattr(bbob, f'F{fid}')(dim=dim, shift=shift, rotate=H, bias=bias,
                                               lb=BBOB_LB, ub=BBOB_UB)
        finally:
            np.random.set_state(state)
        self.fopt = float(self._f.optimum)

    def eval(self, X):
        X = np.atleast_2d(X)
        return np.asarray(self._f.func(BBOB_LB + X * (BBOB_UB - BBOB_LB)), dtype=np.float64)

    def __repr__(self):
        return f'F{self.fid}(sid={self.sid})'


def build_task(spec):
    kind = spec[0]
    if kind == 'cec':
        return CECTask(spec[1], spec[2])
    if kind == 'bbob':
        return BBOBTask(spec[1], spec[2], spec[3])
    raise ValueError(spec)


def build_mtop(mtop_spec):
    return [build_task(s) for s in mtop_spec]


# ----------------------------------------------------------------------------- suite 1

CEC_SETS = ('VS', 'S', 'M', 'L', 'VL', 'C1', 'C2', 'C3', 'C4', 'C5')
_RANGE_DELTA = {'VS': 0.025, 'S': 0.05, 'M': 0.1, 'L': 0.2, 'VL': 0.4}
# Not given in the paper (Table S.II leaves x_c,i and Delta_i open): fixed per set by
# this seed, centres uniform in [0.1, 0.9]^D, radius 0.05 for every cluster.
CLUSTER_SEED = 2025
CLUSTER_CENTRE_RANGE = (0.1, 0.9)
CLUSTER_RADIUS = 0.05


class CECSet:
    """One of the ten CEC17-based MTOP sets (Table S.II)."""

    def __init__(self, name, dim=10):
        assert name in CEC_SETS
        self.name, self.dim = name, dim
        if name.startswith('C'):
            c = int(name[1:])
            r = np.random.default_rng(CLUSTER_SEED + c)
            lo, hi = CLUSTER_CENTRE_RANGE
            n_centres = 2 if c == 1 else c
            self.centres = r.uniform(lo, hi, size=(n_centres, dim))

    def sample_xo(self, rng):
        D = self.dim
        if self.name in _RANGE_DELTA:
            d = _RANGE_DELTA[self.name]
            return rng.uniform(0.5 - d, 0.5 + d, D)
        if self.name == 'C1':                      # point on the segment x_c1 -> x_c2
            return self.centres[0] + rng.uniform() * (self.centres[1] - self.centres[0])
        c = self.centres[rng.integers(len(self.centres))]
        return np.clip(rng.uniform(c - CLUSTER_RADIUS, c + CLUSTER_RADIUS), 0.0, 1.0)

    def sample_task(self, rng):
        return ('cec', CEC_FUNCTIONS[rng.integers(len(CEC_FUNCTIONS))], self.sample_xo(rng))

    def sample_mtop(self, rng, K=2):
        return [self.sample_task(rng) for _ in range(K)]


# ----------------------------------------------------------------------------- suite 2

_BBOB_LEARN_F = (1, 3, 8, 10, 16, 20)
BBOB_SETS = {   # Table S.III: name -> (function IDs, inclusive seed range)
    'BBOB_learn': (_BBOB_LEARN_F, (1, 100)),
    'BBOB1': (_BBOB_LEARN_F, (500, 1500)),
    'BBOB2': (_BBOB_LEARN_F, (1000, 1005)),
    'BBOB3': ((1,), (500, 1500)),
    'BBOB4': ((3,), (500, 1500)),
    'BBOB5': ((8,), (500, 1500)),
    'BBOB6': ((10,), (500, 1500)),
    'BBOB7': ((16,), (500, 1500)),
    'BBOB8': ((20,), (500, 1500)),
    'BBOB9': (tuple(f for f in range(1, 25) if f not in _BBOB_LEARN_F), (500, 1500)),
    'BBOB10': ((2, 6, 12, 15, 21), (500, 1500)),
    'BBOB11': ((2,), (500, 1500)),
    'BBOB12': ((6,), (500, 1500)),
    'BBOB13': ((12,), (500, 1500)),
    'BBOB14': ((15,), (500, 1500)),
    'BBOB15': ((21,), (500, 1500)),
}


class BBOBSet:
    """One of the 16 BBOB-based MTOP sets (Table S.III). A task is a (fid, sid) pair."""

    def __init__(self, name, dim=10):
        self.name, self.dim = name, dim
        self.fids, (self.s_lo, self.s_hi) = BBOB_SETS[name]

    def sample_task(self, rng):
        fid = int(self.fids[rng.integers(len(self.fids))])
        sid = int(rng.integers(self.s_lo, self.s_hi + 1))
        return ('bbob', fid, sid, self.dim)

    def sample_mtop(self, rng, K=2):
        return [self.sample_task(rng) for _ in range(K)]


# ----------------------------------------------------------------------------- helpers

DEFAULT_DIM = {'cec': 10, 'bbob': 10}   # not stated in the paper; chosen with the user (DEVIATIONS.md §9)
TEST_SEED = 12345                       # test instances; training uses other seeds
N_TEST_INSTANCES = 100


def get_set(name, dim=None):
    if name in CEC_SETS:
        return CECSet(name, dim or DEFAULT_DIM['cec'])
    if name in BBOB_SETS:
        return BBOBSet(name, dim or DEFAULT_DIM['bbob'])
    raise ValueError(f'unknown MTOP set {name!r}')


def test_instances(name, n=N_TEST_INSTANCES, seed=TEST_SEED, dim=None):
    """The fixed list of test MTOP specs for a set (same list for every algorithm)."""
    s = get_set(name, dim)
    rng = np.random.default_rng([seed, sum(map(ord, name))])
    return [s.sample_mtop(rng) for _ in range(n)]


def initial_population_set(n_pops, pop_size, dim, seed=0):
    """N_P populations by independent Latin hypercube sampling (Algorithm 2, line 1)."""
    return np.stack([qmc.LatinHypercube(d=dim, seed=seed + i).random(pop_size) for i in range(n_pops)])
