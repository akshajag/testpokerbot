"""Sparring bots for local testing: the 4 official baselines plus stronger
archetypes that other teams are likely to submit."""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import fasteval as fe  # noqa: E402
from game import Action  # noqa: E402

FOLD, CALL, RH, RP = Action.FOLD, Action.CHECK_CALL, Action.RAISE_HALF, Action.RAISE_POT
PRE = np.load(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "preflop_eq.npy"))


class Bot:
    def new_half(self, m):
        m["rng"] = np.random.default_rng(len(m) + 7)

    def hand_end(self, res, m):
        pass


class Random(Bot):
    def choose(self, o, m):
        return o.legal_actions[m["rng"].integers(len(o.legal_actions))]


class Station(Bot):
    def choose(self, o, m):
        return CALL


class AlwaysHalf(Bot):
    def choose(self, o, m):
        return RH if RH in o.legal_actions else CALL


class AlwaysPot(Bot):
    def choose(self, o, m):
        return RP if RP in o.legal_actions else CALL


def _pre(o):
    return float(PRE[fe.hand_index(np.array([sorted(o.hole_cards)]))[0]])


def _mc(o, rng, n=300):
    hole = np.array(o.hole_cards)
    board = np.array(o.board, dtype=np.int64)
    rest = np.setdiff1d(np.arange(36), np.concatenate([hole, board]))
    need = 5 - len(board)
    pick = np.argpartition(rng.random((n, len(rest))), 5 + need, axis=1)[:, :5 + need]
    s = rest[pick]
    full = np.concatenate([np.broadcast_to(board, (n, len(board))), s[:, 5:]], axis=1)
    tk, ts = fe.triple_parts(np.ascontiguousarray(full))
    mk, ms = fe.pair_parts(hole[None])
    a = fe.best(np.repeat(mk, n, 0), np.repeat(ms, n, 0), tk, ts)
    ok, os_ = fe.pair_parts(s[:, :5])
    b = fe.best(ok, os_, tk, ts)
    return float(((a > b) + 0.5 * (a == b)).mean())


class Nit(Bot):
    """Plays only strong hands, raises them hard."""
    def choose(self, o, m):
        e = _pre(o) if not o.board else _mc(o, m["rng"], 200)
        if e > (0.6 if not o.board else 0.7) and RP in o.legal_actions:
            return RP
        if e > (0.55 if not o.board else 0.55) or o.to_call == 0:
            return CALL
        return FOLD


class EquityBot(Bot):
    """Typical 'good' team bot: MC equity vs random hand + pot odds."""
    def __init__(self, aggro=0.0, bluff=0.0):
        self.aggro, self.bluff = aggro, bluff

    def choose(self, o, m):
        e = _pre(o) if not o.board else _mc(o, m["rng"])
        tc, pot = o.to_call, o.pot
        can = RP in o.legal_actions
        if can and e > 0.68 - self.aggro:
            return RP
        if can and e > 0.58 - self.aggro:
            return RH
        if tc == 0:
            if can and o.board and m["rng"].random() < self.bluff:
                return RH
            return CALL
        return CALL if e >= tc / (pot + tc) else FOLD


class Maniac(Bot):
    """Raises a lot with anything decent, bluffs often, rarely folds."""
    def choose(self, o, m):
        e = _pre(o) if not o.board else _mc(o, m["rng"], 150)
        can = RP in o.legal_actions
        if can and (e > 0.5 or m["rng"].random() < 0.35):
            return RP if e > 0.6 else RH
        if o.to_call and e < 0.3 and m["rng"].random() < 0.5:
            return FOLD
        return CALL


OPPONENTS = {
    "random": Random,
    "station": Station,
    "always_half": AlwaysHalf,
    "always_pot": AlwaysPot,
    "nit": Nit,
    "equity": EquityBot,
    "equity_aggro": lambda: EquityBot(aggro=0.08, bluff=0.3),
    "maniac": Maniac,
}


class BrainVariant(Bot):
    """Another copy of our own bot, optionally with a different range model."""
    def __init__(self, model):
        self.model = model

    def _with(self, fn, *a):
        import brain
        old = brain.MODEL
        brain.MODEL = self.model
        try:
            return fn(*a)
        finally:
            brain.MODEL = old

    def new_half(self, m):
        import brain
        brain.init_memory(m)
        m["rng"] = np.random.default_rng(99)

    def choose(self, o, m):
        import brain
        return self._with(brain.choose, o, m)

    def hand_end(self, r, m):
        import brain
        self._with(brain.hand_end, r, m)


OPPONENTS["brain_old"] = lambda: BrainVariant("old")
OPPONENTS["mirror"] = lambda: BrainVariant("freq")
