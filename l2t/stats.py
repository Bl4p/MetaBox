"""Comparison criterion of the paper (Definition 1) and the positive transfer rate (Eq. 21)."""
import numpy as np
from scipy.stats import ranksums

ALPHA = 0.05


def task_outcome(a, b, alpha=ALPHA):
    """+1 if sample a (minimisation errors) is significantly better than b, -1 if worse, 0 if equal."""
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    if np.array_equal(np.sort(a), np.sort(b)):
        return 0
    stat, p = ranksums(a, b)
    if not np.isfinite(p) or p >= alpha:
        return 0
    return 1 if stat < 0 else -1


def instance_outcome(A, B, alpha=ALPHA):
    """Definition 1 for one MTOP instance. A, B: (runs, K) best-found errors.

    Win  : better or equal on every task, and not equal on all of them
    Tie  : statistically equal on every task
    Lose : anything else (including better on one task and worse on another)
    """
    per_task = [task_outcome(A[:, k], B[:, k], alpha) for k in range(A.shape[1])]
    if all(o == 0 for o in per_task):
        return 0
    if all(o >= 0 for o in per_task):
        return 1
    return -1


def wtl(curves_a, curves_b, g, alpha=ALPHA):
    """W/T/L counts of algorithm a against b at generation g.

    curves_*: (instances, runs, G+1, K) arrays of best-so-far errors, same instance order.
    """
    out = [instance_outcome(curves_a[i, :, g], curves_b[i, :, g], alpha) for i in range(len(curves_a))]
    w, t, l = out.count(1), out.count(0), out.count(-1)
    return w, t, l


def fmt_wtl(w, t, l):
    sign = '+' if w > l else '-' if w < l else '='
    return f'{w}/{t}/{l}({sign})'


def positive_transfer_rate(curves_emt, curves_single, g):
    """Eq. (21): share of instances on which the EMT algorithm wins against single-task DE."""
    w, _, _ = wtl(curves_emt, curves_single, g)
    return w / len(curves_emt)
