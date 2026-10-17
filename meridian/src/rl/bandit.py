"""Tabular epsilon-greedy contextual bandit with hierarchical back-off. Pure Python, deterministic given a seed.

Myopic by design: it maximises one-step reward given the context (which includes remaining budget and the previous
outcome). It does NOT plan across documents; any benefit from "saving" budget for later is incidental.
"""
from __future__ import annotations

import json
import random

from ..config import p_ai_tier
from .strategies import ALL_STRATEGIES, SAFE_STRATEGIES

MODEL_VERSION = "bandit-1"


def state_key(ctx: dict) -> str:
    return "|".join([ctx["task"], ctx["tier"], ctx["sensitive"], ctx["conf_bucket"], ctx["budget_bucket"], "R" if ctx["required_present"] else "r", ctx["prev"]])


def coarse_key(ctx: dict) -> str:
    return "|".join([ctx["task"], ctx["tier"], ctx["budget_bucket"]])


class ContextualBandit:
    def __init__(self, strategies=None, epsilon=0.2, epsilon_min=0.02, decay=0.9995, min_obs=4, seed=0):
        self.strategies = list(strategies or SAFE_STRATEGIES)
        self.eps0, self.eps_min, self.decay, self.min_obs = epsilon, epsilon_min, decay, min_obs
        self.rng = random.Random(seed)
        self.t = 0
        self.q: dict[str, dict[str, list]] = {"fine": {}, "coarse": {}, "global": {}}      # table -> key -> {strategy: [n, mean]}
        self.seed = seed

    # ------------------------------------------------------------------ selection
    @property
    def epsilon(self):
        return max(self.eps_min, self.eps0 * (self.decay ** self.t))

    def _stats(self, ctx):
        return [("fine", state_key(ctx)), ("coarse", coarse_key(ctx)), ("global", "all")]

    def values(self, ctx: dict, allowed: list[str]) -> dict:
        """Estimated value of each allowed action, from the finest table with enough observations for that action."""
        out = {}
        for s in allowed:
            v, n_used, src = 0.0, 0, "prior"
            for tbl, k in self._stats(ctx):
                n, m = self.q[tbl].get(k, {}).get(s, [0, 0.0])
                if n >= self.min_obs or tbl == "global" and n > 0:
                    v, n_used, src = m, n, tbl
                    break
            out[s] = {"value": v, "n": n_used, "source": src}
        return out

    def select(self, ctx: dict, allowed: list[str], explore: bool = True) -> tuple[str, dict]:
        if not allowed:
            raise ValueError("empty action mask")
        vals = self.values(ctx, allowed)
        if explore and self.rng.random() < self.epsilon:
            return self.rng.choice(allowed), vals
        best = max(vals[s]["value"] for s in allowed)
        ties = [s for s in allowed if vals[s]["value"] >= best - 1e-12]
        pick = "POLICY" if "POLICY" in ties else ties[0]                      # ties resolve to the authoritative policy
        # an action never observed anywhere cannot beat the authoritative policy on a prior
        if vals[pick]["n"] == 0 and "POLICY" in allowed:
            pick = "POLICY"
        return pick, vals

    def update(self, ctx: dict, strategy: str, reward: float):
        self.t += 1
        for tbl, k in self._stats(ctx):
            cell = self.q[tbl].setdefault(k, {}).setdefault(strategy, [0, 0.0])
            cell[0] += 1
            cell[1] += (reward - cell[1]) / cell[0]

    # ------------------------------------------------------------------ persistence
    def to_dict(self, meta=None) -> dict:
        return {"version": MODEL_VERSION, "strategies": self.strategies, "seed": self.seed, "steps": self.t, "min_obs": self.min_obs, "q": self.q, "meta": meta or {}}

    @classmethod
    def from_dict(cls, d: dict) -> "ContextualBandit":
        if d.get("version") != MODEL_VERSION:
            raise ValueError(f"unsupported model version {d.get('version')!r}")
        b = cls(d["strategies"], seed=d.get("seed", 0), min_obs=d.get("min_obs", 4))
        b.q, b.t = d["q"], d.get("steps", 0)
        return b

    def save(self, path, meta=None):
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.to_dict(meta), fh, sort_keys=True)

    @classmethod
    def load(cls, path) -> "ContextualBandit":
        with open(path, encoding="utf-8") as fh:
            return cls.from_dict(json.load(fh))
