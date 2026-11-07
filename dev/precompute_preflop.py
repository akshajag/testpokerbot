"""Offline: equity of every 5-card hand vs a random hand (random board), averaged
over suit-isomorphic hands.  Writes data/preflop_eq.npy (float16, C(36,5) entries).

    python dev/precompute_preflop.py [samples_per_hand]
"""
import os
import sys
from itertools import combinations, permutations
from concurrent.futures import ProcessPoolExecutor
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import fasteval as fe  # noqa: E402

ALL = np.array(list(combinations(range(36), 5)), dtype=np.int64)  # colex order? fix below


def equities(args):
    lo, hi, samples, seed = args
    rng = np.random.default_rng(seed)
    hands = ALL[lo:hi]
    out = np.empty(len(hands))
    B = 400
    for s in range(0, len(hands), B):
        h = hands[s:s + B]
        n = len(h)
        # random permutation of the deck per (hand, sample); drop own cards
        r = rng.random((n, samples, 36))
        np.put_along_axis(r, np.broadcast_to(h[:, None, :], (n, samples, 5)), 2.0, axis=2)
        perm = np.argsort(r, axis=2)[:, :, :10].reshape(n * samples, 10)
        opp, board = perm[:, :5], perm[:, 5:]
        tk, ts = fe.triple_parts(board)
        mk, ms = fe.pair_parts(np.repeat(h, samples, axis=0))
        ok, os_ = fe.pair_parts(opp)
        a, b = fe.best(mk, ms, tk, ts), fe.best(ok, os_, tk, ts)
        res = (a > b) + 0.5 * (a == b)
        out[s:s + n] = res.reshape(n, samples).mean(axis=1)
    return lo, out


def main():
    samples = int(sys.argv[1]) if len(sys.argv) > 1 else 48
    idx = fe.hand_index(ALL)
    n = len(ALL)
    chunks = [(i, min(n, i + 8000), samples, i) for i in range(0, n, 8000)]
    eq = np.empty(n)
    with ProcessPoolExecutor() as ex:
        for lo, out in ex.map(equities, chunks):
            eq[lo:lo + len(out)] = out
            print(f"{lo + len(out)}/{n}", flush=True)
    # canonical suit class: min over 24 suit relabelings of the encoded sorted hand
    ranks, suits = ALL >> 2, ALL & 3
    canon = None
    for p in permutations(range(4)):
        mapped = np.sort(ranks * 4 + np.array(p)[suits], axis=1)
        code = (((mapped[:, 0] * 36 + mapped[:, 1]) * 36 + mapped[:, 2]) * 36 + mapped[:, 3]) * 36 + mapped[:, 4]
        canon = code if canon is None else np.minimum(canon, code)
    _, inv = np.unique(canon, return_inverse=True)
    sums = np.bincount(inv, weights=eq)
    cnts = np.bincount(inv)
    smooth = (sums / cnts)[inv]
    print("classes:", len(cnts), "min/max eq:", smooth.min(), smooth.max())
    table = np.zeros(fe.NUM_HANDS, dtype=np.float16)
    table[idx] = smooth
    os.makedirs(os.path.join(ROOT, "data"), exist_ok=True)
    np.save(os.path.join(ROOT, "data", "preflop_eq.npy"), table)
    print("saved")


if __name__ == "__main__":
    main()
