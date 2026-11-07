"""In-process duplicate gauntlet using the official game.py (no HTTP), scored
exactly like the judge: mean bb/deal over both halves with a 95% CI.

    python dev/sim.py [deals] [seeds] [opponent,...]
"""
import os
import sys
import time
import math
from concurrent.futures import ProcessPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from game import Game, make_deal, IllegalActionError  # noqa: E402
import brain  # noqa: E402
from opponents import OPPONENTS  # noqa: E402


class Brain:
    def new_half(self, m):
        brain.init_memory(m)

    def choose(self, o, m):
        return brain.choose(o, m)

    def hand_end(self, r, m):
        brain.hand_end(r, m)


def play_half(bots, seed, deals, stats):
    mems = [{}, {}]
    for b, m in zip(bots, mems):
        b.new_half(m)
    nets = []
    errors = [0, 0]
    for i in range(deals):
        g = Game(make_deal(seed, i), button_seat=i % 2, hand_index=i)
        while not g.is_terminal:
            s = g.to_act
            o = g.observation(s)
            t = time.perf_counter()
            a = bots[s].choose(o, mems[s])
            stats[s] += time.perf_counter() - t
            try:
                g.apply(a)
            except IllegalActionError:
                errors[s] += 1
                g.apply(Action.CHECK_CALL if o.to_call == 0 else Action.FOLD)
        for s in (0, 1):
            bots[s].hand_end(g.hand_result(s), mems[s])
        nets.append(g.result().nets)
    return nets, errors


from game import Action  # noqa: E402


def match(args):
    opp_name, seed, deals = args
    me, opp = Brain(), OPPONENTS[opp_name]()
    t1 = [0.0, 0.0]
    t2 = [0.0, 0.0]
    n1, e1 = play_half([me, opp], seed, deals, t1)         # I am seat 0
    n2, e2 = play_half([opp, me], seed, deals, t2)         # I am seat 1, same cards
    per = [(a[0] + b[1]) / 2.0 for a, b in zip(n1, n2)]   # bb per deal
    mean = sum(per) / deals
    var = sum((x - mean) ** 2 for x in per) / (deals - 1)
    se = math.sqrt(var / deals)
    lo, hi = mean - 1.96 * se, mean + 1.96 * se
    verdict = "WIN" if lo > 0 else "LOSS" if hi < 0 else "CHOP"
    return opp_name, seed, mean, lo, hi, verdict, max(t1[0], t2[1]), e1[0] + e2[1]


def main():
    deals = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
    seeds = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    names = sys.argv[3].split(",") if len(sys.argv) > 3 else list(OPPONENTS)
    jobs = [(n, s, deals) for n in names for s in range(1, seeds + 1)]
    wins = 0
    t0 = time.time()
    with ProcessPoolExecutor() as ex:
        for name, seed, mean, lo, hi, v, tmax, err in ex.map(match, jobs):
            wins += v == "WIN"
            print(f"{name:13s} seed {seed}: {mean:+7.3f} bb/deal  CI [{lo:+.3f},{hi:+.3f}]  {v:4s}"
                  f"  my think {tmax:.2f}s/half  errors {err}", flush=True)
    print(f"\nWON {wins}/{len(jobs)} matches = {100 * wins / len(jobs):.1f}%  ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
