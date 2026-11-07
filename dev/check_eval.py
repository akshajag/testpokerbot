import os, sys, random, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import fasteval as fe
from evaluator import omaha_score, warmup

warmup()
r = random.Random(3)
bad = 0
for _ in range(20000):
    c = r.sample(range(36), 15)
    b = c[10:]
    a1, a2 = fe.score(c[:5], b), fe.score(c[5:10], b)
    o1, o2 = omaha_score(c[:5], b), omaha_score(c[5:10], b)
    bad += (a1 > a2) != (o1 > o2) or (a1 == a2) != (o1 == o2)
    bad += a1 != o1  # same packing -> identical scores
print("mismatches:", bad)

h = np.array([r.sample(range(36), 10) for _ in range(500)])
t = time.perf_counter()
for _ in range(50):
    tk, ts = fe.triple_parts(h[:, 5:])
    pk, ps = fe.pair_parts(h[:, :5])
    fe.best(pk, ps, tk, ts)
print("ms per 500 evals:", (time.perf_counter() - t) / 50 * 1000)
