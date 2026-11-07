"""Strategy for heads-up Short Deck 5-Card PLO.

Pieces:
  * Precomputed preflop equity table (every 5-card hand vs a random hand).
  * Vectorised Monte Carlo equity vs the opponent's RANGE: every sampled
    opponent hand is weighted by how likely it was to take the actions it
    actually took this hand (Bayesian range narrowing).
  * Opponent model learned during the half: fold-to-bet, aggression,
    showdown bluff rate -> bluff folders, value-bet stations, call down maniacs.
  * Adaptive time manager: spends the 8 s bank evenly with a big safety margin.
Everything is wrapped so any internal error falls back to a safe legal action.
"""
import os
import time
from collections import namedtuple
import numpy as np

import fasteval as fe
from game import Action, Street

_HERE = os.path.dirname(os.path.abspath(__file__))
PRE = np.load(os.path.join(_HERE, "data", "preflop_eq.npy")).astype(np.float32)
PRE_SORTED = np.sort(PRE)
ALL_CARDS = np.arange(36)

MODEL = os.environ.get("BRAIN_MODEL", "freq")
TIME_BUDGET = 5.0      # seconds of the 8 s bank we plan to use per half
DEFAULT_DPH = 2.6      # prior decisions per hand

FOLD, CALL, RH, RP = Action.FOLD, Action.CHECK_CALL, Action.RAISE_HALF, Action.RAISE_POT
_Rec = namedtuple("_Rec", "street is_me action amount")


def pre_eq(hole):
    h = np.array([sorted(hole)])
    return float(PRE[fe.hand_index(h)[0]])


def pre_pct_vec(hands_sorted):
    e = PRE[fe.hand_index(hands_sorted)]
    return np.searchsorted(PRE_SORTED, e) / len(PRE_SORTED)


# ------------------------------------------------------------------ memory
def init_memory(m, config=None):
    bank = float(getattr(config, "time_bank_seconds", 8.0) or 8.0)
    m.update(
        budget=0.4 * bank, num_hands=int(getattr(config, "num_hands", 1000) or 1000),
        faced=6.0, folded=2.4,          # opp fold-to-our-bet (prior 40%)
        acts=10.0, raises=3.0,           # opp aggression (prior 30%)
        sd_raises=4.0, sd_bluffs=1.2,    # showdown bluff rate (prior 30%)
        pf_faced=5.0, pf_folded=2.0,     # opp preflop fold when facing our raise
        pfa=8.0, pfr=2.4, poa=8.0, por=2.4,  # opp raise frequency preflop / postflop
        t_used=0.0, decisions=0, hands=0,
        us_per_sample=6.0,
        rng=np.random.default_rng(20261107),
    )


def _fold_rate(m):
    return m["folded"] / m["faced"]


def _aggr(m):
    return m["raises"] / m["acts"]


def _bluff(m):
    b = 0.5 * m["sd_bluffs"] / m["sd_raises"] + 0.3 * _aggr(m)
    return min(0.85, max(0.05, b))


# ------------------------------------------------------------------ equity
_REF_CACHE = {}


def _ref_dist(board_part, rng):
    """Sorted scores of 160 random hands on this board (cached per board)."""
    key = tuple(int(c) for c in board_part)
    rv = _REF_CACHE.get(key)
    if rv is None:
        if len(_REF_CACHE) > 64:
            _REF_CACHE.clear()
        rest = np.setdiff1d(ALL_CARDS, board_part)
        ref = rest[np.argpartition(rng.random((160, len(rest))), 5, axis=1)[:, :5]]
        tk, ts = fe.triple_parts(np.asarray(board_part)[None, :])
        rk, rs = fe.pair_parts(ref)
        rv = np.sort(fe.best(rk, rs, np.repeat(tk, 160, 0), np.repeat(ts, 160, 0)))
        _REF_CACHE[key] = (rv, tk, ts)
        return rv, tk, ts
    return rv


def _street_strength(opp, board_part, rng):
    """Opponent hand strength on a partial board: percentile of its made hand vs
    random hands on that board, boosted for flush draws."""
    rv, tk, ts = _ref_dist(board_part, rng)
    n = len(opp)
    pk, ps = fe.pair_parts(opp)
    v = fe.best(pk, ps, np.repeat(tk, n, 0), np.repeat(ts, n, 0))
    x = np.searchsorted(rv, v, side="right") / 160.0
    if len(board_part) < 5:
        bs = fe.SUIT[board_part]
        os_ = fe.SUIT[opp]
        for s in range(4):
            if (bs == s).sum() == 2:
                fd = (os_ == s).sum(axis=1) >= 2
                x = np.where(fd, np.maximum(x, 0.62), x)
    return x


def equity(obs, m, n_samples):
    rng = m["rng"]
    hole = np.array(obs.hole_cards)
    board = np.array(obs.board, dtype=np.int64)
    dead = np.concatenate([hole, board])
    rest = np.setdiff1d(ALL_CARDS, dead)
    need = 5 - len(board)
    k = 5 + need
    n = n_samples
    pick = np.argpartition(rng.random((n, len(rest))), k, axis=1)[:, :k]
    s = rest[pick]
    opp = np.sort(s[:, :5], axis=1)
    full = np.concatenate([np.broadcast_to(board, (n, len(board))), s[:, 5:]], axis=1) if need else \
        np.broadcast_to(board, (n, 5))
    full = np.ascontiguousarray(full)

    # ---- range weights from opponent actions this hand
    w = np.ones(n)
    hist = [a for a in obs.action_history if not a.is_me]
    if hist:
        bluff, fold = _bluff(m), _fold_rate(m)
        aggr = _aggr(m)
        cache = {}
        for a in hist:
            st = int(a.street)
            if st not in cache:
                if st == 0:
                    cache[st] = pre_pct_vec(opp)
                else:
                    nb = 2 + st
                    cache[st] = _street_strength(opp, np.array(obs.board[:nb]), rng)
            x = cache[st]
            if MODEL == "freq":
                fr = (m["pfr"] / m["pfa"]) if st == 0 else (m["por"] / m["poa"])
                eps = min(0.35, max(0.03, 0.6 * m["sd_bluffs"] / m["sd_raises"] - 0.05))
                if a.action in (RH, RP):
                    thr = 1.0 - min(0.95, fr * (1.0 if a.action == RP else 1.15))
                    w *= eps + (1 - eps) / (1.0 + np.exp(-(x - thr) / 0.07))
                elif a.action == CALL and a.amount > 0:
                    thr = min(0.9, fold * 0.9)
                    w *= 0.08 + 0.92 / (1.0 + np.exp(-(x - thr) / 0.1))
                elif a.action == CALL:
                    thr = 1.0 - min(0.95, fr)
                    w *= 1.0 - 0.75 / (1.0 + np.exp(-(x - thr) / 0.08))
            elif a.action in (RH, RP):
                p = 2.0 if a.action == RH else 2.5
                w *= bluff + (1 - bluff) * x ** p
            elif a.action == CALL and a.amount > 0:
                w *= (1 - fold) + fold * np.minimum(1.0, 1.7 * x + 0.05)
            elif a.action == CALL:
                w *= 1.0 - 0.6 * x * min(1.0, 2.0 * aggr)

    tk, ts = fe.triple_parts(full)
    mk, ms = fe.pair_parts(hole[None, :])
    mine = fe.best(np.repeat(mk, n, 0), np.repeat(ms, n, 0), tk, ts)
    ok, os_ = fe.pair_parts(opp)
    theirs = fe.best(ok, os_, tk, ts)
    res = (mine > theirs) + 0.5 * (mine == theirs)
    sw = w.sum()
    ess = sw * sw / max((w * w).sum(), 1e-12)
    eq_w = float((w * res).sum() / sw) if sw > 0 else float(res.mean())
    if ess < 15:  # too few effective samples: shrink toward unweighted
        a = ess / 15.0
        eq_w = a * eq_w + (1 - a) * float(res.mean())
    return eq_w


# ------------------------------------------------------------------ decision
def _samples(obs, m):
    hands_left = max(1, m.get("num_hands", 1000) - obs.hand_index)
    dph = (m["decisions"] / m["hands"]) if m["hands"] > 20 else DEFAULT_DPH
    left = m.get("budget", TIME_BUDGET) - m["t_used"]
    per = left / (hands_left * max(dph, 1.0))
    per -= 0.0004  # fixed overhead per decision
    n = int(per * 1e6 / m["us_per_sample"])
    floor = 40 if left > 1.0 else 12
    if left < 0.3:
        return 0  # bank nearly gone: no simulation
    return max(floor, min(n, 1500 if obs.street else 600))


def decide(obs, m):
    legal = obs.legal_actions
    can_raise = RP in legal
    tc, pot = obs.to_call, obs.pot
    po = tc / (pot + tc) if tc else 0.0
    st = int(obs.street)
    fold = _fold_rate(m)
    station = fold < 0.22
    pf_fold = m["pf_folded"] / m["pf_faced"]
    steal = max(0.0, pf_fold - 0.4)
    r = m["rng"].random()
    opp_acted = any(not a.is_me for a in obs.action_history)

    if st == 0 and not opp_acted:
        e = pre_eq(obs.hole_cards)
    else:
        n = _samples(obs, m)
        t0 = time.perf_counter()
        e = equity(obs, m, n) if n else (pre_eq(obs.hole_cards) if st == 0 else 0.5)
        dt = time.perf_counter() - t0
        if n:
            m["us_per_sample"] = 0.9 * m["us_per_sample"] + 0.1 * max(1.0, dt * 1e6 / n)

    def val_raise(big):
        if not can_raise:
            return CALL
        return RP if big else RH

    # ---------------- preflop
    if st == 0:
        if tc > 0:
            facing_raise = obs.opp_bet > obs.big_blind
            if not facing_raise:  # SB first action vs the big blind
                if can_raise and (e > 0.53 - 0.3 * steal or (pf_fold > 0.62 and e > 0.3)):
                    return RP
                if e > 0.40:
                    return CALL
                return FOLD
            if can_raise and e > 0.60:
                return RP
            # committed: calling leaves little behind
            if obs.my_stack - tc < 0.35 * (pot + tc) and e > po - 0.02:
                return CALL
            if e > po - 0.04:
                return CALL
            return FOLD
        # BB option after a limp
        if can_raise and (e > 0.56 - 0.3 * steal or (pf_fold > 0.62 and e > 0.35)):
            return RP
        return CALL

    if obs.is_button:
        e += 0.015
    value = 0.52 if station else 0.60

    # ---------------- postflop, nothing to call
    if tc == 0:
        if can_raise and e > value:
            return val_raise(e > 0.78 or station)
        if can_raise and not station and e < 0.40:
            bf = min(0.65, max(0.0, (fold - 0.33) * 1.8))
            if r < bf:
                return RH
        return CALL

    # ---------------- postflop, facing a bet
    margin = {1: 0.03, 2: 0.02, 3: 0.0}[st]
    if can_raise and e > 0.74:
        return val_raise(e > 0.82)
    if obs.my_stack - tc < 0.3 * (pot + tc) and e > po - 0.03:
        return CALL
    if e > po + margin:
        return CALL
    return FOLD


def choose(obs, m):
    t0 = time.perf_counter()
    try:
        if "rng" not in m:
            init_memory(m)
        a = decide(obs, m)
        if a not in obs.legal_actions:
            a = CALL
    except Exception:
        if os.environ.get("BRAIN_STRICT"):
            raise
        a = CALL if CALL in obs.legal_actions else obs.legal_actions[0]
    try:
        amt = obs.to_call if a == CALL else (obs.raise_to.get(a, 0) - obs.my_bet if a in (RH, RP) else 0)
        m["last_history"] = tuple(obs.action_history) + (_Rec(obs.street, True, a, amt),)
    except Exception:
        pass
    m["decisions"] = m.get("decisions", 0) + 1
    m["t_used"] = m.get("t_used", 0.0) + time.perf_counter() - t0
    return a


# ------------------------------------------------------------------ learning
def hand_end(res, m):
    t0 = time.perf_counter()
    try:
        _hand_end(res, m)
    except Exception:
        if os.environ.get("BRAIN_STRICT"):
            raise
    finally:
        m["t_used"] = m.get("t_used", 0.0) + time.perf_counter() - t0


def _hand_end(res, m):
    if os.environ.get("BRAIN_LOG") and res.hand_index >= m.get("num_hands", 1000) - 1:
        import sys
        print(f"[brain] internal t_used={m['t_used']:.2f}s decisions={m['decisions']} "
              f"us/sample={m['us_per_sample']:.1f}", file=sys.stderr, flush=True)
    """Update opponent stats. `history` (our view of the action list) is
    remembered from the last decision; the result object itself has none."""
    if "rng" not in m:
        init_memory(m)
    m["hands"] += 1
    hist = m.pop("last_history", None) or ()
    last_raiser = {}
    for a in hist:
        st = int(a.street)
        if not a.is_me:
            m["acts"] += 1
            israise = a.action in (RH, RP)
            if israise:
                m["raises"] += 1
            if st == 0:
                m["pfa"] += 1
                m["pfr"] += israise
            else:
                m["poa"] += 1
                m["por"] += israise
            if last_raiser.get(st) == "me":
                m["faced"] += 1
                if st == 0:
                    m["pf_faced"] += 1
        if a.action in (RH, RP):
            last_raiser[st] = "me" if a.is_me else "opp"
    # A fold by the opponent ends the hand right after our raise, so it never
    # appears in a later observation: infer it from the result.
    if hist and not res.went_to_showdown and res.winner == "me" and hist[-1].is_me \
            and hist[-1].action in (RH, RP):
        m["faced"] += 1
        m["folded"] += 1
        m["acts"] += 1
        if int(hist[-1].street) == 0:
            m["pf_faced"] += 1
            m["pf_folded"] += 1
            m["pfa"] += 1
        else:
            m["poa"] += 1
    if res.went_to_showdown and res.opp_hole and m["t_used"] < m.get("budget", TIME_BUDGET) * 0.8:
        board = np.array(res.board)
        opp = np.array([sorted(res.opp_hole)])
        for a in hist:
            if not a.is_me and a.action in (RH, RP) and int(a.street) > 0:
                m["sd_raises"] += 1
                x = _street_strength(opp, board[:2 + int(a.street)], m["rng"])[0]
                if x < 0.35:
                    m["sd_bluffs"] += 1
