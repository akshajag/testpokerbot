"""Vectorised (numpy) Short Deck PLO5 evaluator.

Scores are only compared with each other, never with evaluator.omaha_score,
but the ORDER is identical to the official evaluator (verified in tests).

Key idea: a 5-card rank multiset is encoded as sum(5**rank) (each rank appears
at most 4 times, so base 5 is collision-free).  Two lookup arrays give the
score for non-flush and flush hands.  A 2-card hole subset and a 3-card board
subset combine by simple addition of their keys, so the 100 Omaha combinations
of thousands of simulated hands are evaluated with a handful of numpy ops.
"""
from itertools import combinations, combinations_with_replacement
from math import comb
import numpy as np

POW5 = np.array([5 ** (c >> 2) for c in range(36)], dtype=np.int32)
SUIT = np.array([c & 3 for c in range(36)], dtype=np.int8)

# official category numbers
HIGH, PAIR, TWO, STRAIGHT, TRIPS, FULL, FLUSH, QUADS, SF = range(9)
BASE = 1 << 20


def _pack(cat, ranks):
    s = cat << 20
    for slot, r in enumerate(ranks):
        s |= r << (16 - 4 * slot)
    return s


def _score(ranks_desc, flush):
    cnt = {}
    for r in ranks_desc:
        cnt[r] = cnt.get(r, 0) + 1
    if len(cnt) == 5:
        top = ranks_desc[0] if ranks_desc[0] - ranks_desc[4] == 4 else \
            (3 if tuple(ranks_desc) == (8, 3, 2, 1, 0) else None)
        if top is not None:
            return _pack(SF if flush else STRAIGHT, [top])
        return _pack(FLUSH if flush else HIGH, ranks_desc)
    groups = sorted(cnt.items(), key=lambda kv: (kv[1], kv[0]), reverse=True)
    shape = tuple(g[1] for g in groups)
    order = [g[0] for g in groups]
    cat = {(4, 1): QUADS, (3, 2): FULL, (3, 1, 1): TRIPS, (2, 2, 1): TWO}.get(shape, PAIR)
    return _pack(cat, order)


TN = np.zeros(5 ** 9, dtype=np.int32)
TF = np.zeros(5 ** 9, dtype=np.int32)
for _combo in combinations_with_replacement(range(8, -1, -1), 5):
    if max(_combo.count(r) for r in _combo) > 4:
        continue
    _k = sum(5 ** r for r in _combo)
    TN[_k] = _score(_combo, False)
    if len(set(_combo)) == 5:
        TF[_k] = _score(_combo, True)

PI = np.array([p[0] for p in combinations(range(5), 2)])
PJ = np.array([p[1] for p in combinations(range(5), 2)])
TRI = {n: np.array(list(combinations(range(n), 3))) for n in (3, 4, 5)}


def pair_parts(hole):
    """hole: (N,5) int array -> keys (N,10), suit codes (N,10) (-1 = offsuit)."""
    p5, s = POW5[hole], SUIT[hole]
    a, b = s[:, PI], s[:, PJ]
    return p5[:, PI] + p5[:, PJ], np.where(a == b, a, -1)


def triple_parts(board):
    """board: (N,n) with n in 3..5 -> keys (N,T), suit codes (N,T) (-2 = not monotone)."""
    t = TRI[board.shape[1]]
    p5, s = POW5[board], SUIT[board]
    a, b, c = s[:, t[:, 0]], s[:, t[:, 1]], s[:, t[:, 2]]
    keys = p5[:, t[:, 0]] + p5[:, t[:, 1]] + p5[:, t[:, 2]]
    return keys, np.where((a == b) & (b == c), a, -2)


def best(pk, ps, tk, ts):
    """Best Omaha score per row. pk/ps: (N,10) or (1,10); tk/ts: (N,T)."""
    keys = pk[:, :, None] + tk[:, None, :]
    v = TN[keys]
    fl = ps[:, :, None] == ts[:, None, :]
    if fl.any():
        v = np.where(fl, TF[keys], v)
    return v.reshape(v.shape[0], -1).max(axis=1)


def score(hole, board):
    """Scalar convenience: best score of one hole vs a 3-5 card board."""
    pk, ps = pair_parts(np.asarray([hole]))
    tk, ts = triple_parts(np.asarray([board]))
    return int(best(pk, ps, tk, ts)[0])


# --------------------------------------------------------------- hand indexing
COMB = np.array([[comb(n, k) for k in range(6)] for n in range(37)], dtype=np.int64)
NUM_HANDS = comb(36, 5)


def hand_index(sorted_hands):
    """(N,5) ascending card arrays -> colex index in [0, C(36,5))."""
    h = sorted_hands
    return COMB[h[:, 0], 1] + COMB[h[:, 1], 2] + COMB[h[:, 2], 3] + COMB[h[:, 3], 4] + COMB[h[:, 4], 5]
