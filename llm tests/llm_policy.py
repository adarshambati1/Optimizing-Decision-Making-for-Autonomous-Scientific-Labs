"""An LLM as the lab's decision maker, plugged into Keira's POMDP.

The LLM replaces only the *decision rule*. It is shown the same belief the
other agents use (the GP posterior from keira tests/autolab) for a shortlist
of candidates, plus the economics of the task, and returns one action:

    measure(candidate) | commit(candidate) | abandon

Fairness controls:
  * same simulator, seeds, initial design, GP belief and outlier filter as
    GP-EI and MCTS (Keira's run_episode drives all three);
  * the shortlist is built the way MCTS builds its own action set (top
    candidates by upper confidence bound, plus the best commit candidates);
  * by default feature names and the material are hidden (x1, x2, ...), so the
    LLM cannot use chemistry it may remember from the published datasets.
    Pass names="real" to measure how much that prior knowledge is worth.
"""
from __future__ import annotations

import json
import threading
from typing import Literal, Optional

import numpy as np
from pydantic import BaseModel

from autolab.policies import Policy, terminal_action
from autolab.utility import expected_commit_value, stop_value

DEFAULT_MODEL = "claude-opus-5-5"
# models that take the server-side refusal fallback
_FALLBACK_PREFIXES = ("claude-opus-5", "claude-fable-5-1", "claude-fable-5-2", "claude-sonnet-5-5")

SYSTEM = """You run an autonomous materials lab. Each turn you choose exactly one action.

The lab has a pool of candidate materials. Each has a hidden true score (higher is \
better, standardized so the pool average is 0 and the spread is 1). A candidate \
"meets spec" if its true score is at least the spec threshold.

Actions:
- measure: run one experiment on an untested candidate. It costs money whether or not \
it works. Some experiments fail and tell you nothing; some readings are gross outliers; \
the rest are the true score plus noise.
- commit: end the campaign by sending one candidate to production. You pay the commit \
cost, and receive the payoff only if that candidate truly meets spec.
- abandon: end the campaign with nothing further gained or lost.

Your goal is to maximize expected net utility: payoff received, minus commit cost, \
minus everything spent on experiments. Spending the whole budget is not a goal.

Each turn you get the task's numbers and a table of shortlisted candidates with the \
lab's current statistical belief about each one (a Gaussian-process estimate built \
from the readings so far). You may only measure candidates marked untested, and only \
commit to or measure candidates in the table."""


class Decision(BaseModel):
    action: Literal["measure", "commit", "abandon"]
    candidate_id: Optional[int] = None
    reason: str


class LLMPolicy(Policy):
    """Asks an LLM for every decision. One instance per episode."""
    name = "llm"

    def __init__(self, model: str = DEFAULT_MODEL, effort: str = "medium", names: str = "anonymized",
                 feature_names: list | None = None, dataset: str = "", n_measure: int = 8,
                 n_commit: int = 4, kappa: float = 2.0, client=None, log=None, log_meta: dict | None = None):
        self.model, self.effort, self.names = model, effort, names
        self.feature_names, self.dataset = feature_names, dataset
        self.n_measure, self.n_commit, self.kappa = n_measure, n_commit, kappa
        self.client, self.log, self.log_meta = client, log, log_meta or {}
        self.n_calls = self.n_invalid = self.in_tokens = self.out_tokens = 0

    # ------------------------------------------------------------- prompt --
    def shortlist(self, state) -> np.ndarray:
        """Top untested candidates by UCB (as MCTS does) + best commit candidates."""
        b, meas = state.belief, state.measurable()
        ucb = np.where(meas, b.mu + self.kappa * np.sqrt(b.var), -np.inf)
        top_meas = [i for i in np.argsort(-ucb)[: self.n_measure] if np.isfinite(ucb[i])]
        top_commit = np.argsort(-expected_commit_value(b.mu, b.var, state.model))[: self.n_commit]
        return np.unique(np.r_[top_meas, top_commit]).astype(int)

    def build_prompt(self, state, S: np.ndarray) -> str:
        m, b = state.model, state.belief
        d = state.X.shape[1]
        if self.names == "real" and self.feature_names:
            feats, header = list(self.feature_names), f"Material system: {self.dataset}\n"
        else:
            feats, header = [f"x{i + 1}" for i in range(d)], ""
        eu = expected_commit_value(b.mu, b.var, m)
        v_stop, _ = stop_value(b.mu, b.var, m)
        last = {k: y for k, y in state.obs}   # latest successful reading per candidate
        n_left = int(round(state.remaining / m.cost)) if m.cost > 0 else 0

        rows = []
        for k in S:
            rows.append({
                "id": int(k),
                **{f: round(float(v), 3) for f, v in zip(feats, state.X[k])},
                "status": "tested" if state.tested[k] else "untested",
                "reading": round(float(last[k]), 3) if k in last else None,
                "est_score": round(float(b.mu[k]), 3),
                "est_sd": round(float(np.sqrt(b.var[k])), 3),
                "p_meets_spec": round(float((eu[k] + m.commit_cost) / m.payoff), 3),
                "expected_value_if_committed": round(float(eu[k]), 3),
            })
        return (
            f"{header}"
            f"Economics: each experiment costs {m.cost:g}; committing costs {m.commit_cost:g}; "
            f"payoff for a candidate that meets spec is {m.payoff:g}.\n"
            f"Spec threshold (true score needed): {m.tau:.3f}\n"
            f"Measurement: noise sd {np.sqrt(m.noise_var):.3f}; {m.fail_prob:.0%} of experiments fail; "
            f"{m.outlier_prob:.0%} of readings are outliers.\n"
            f"Budget: spent {state.spent:.3f}, remaining {state.remaining:.3f} "
            f"(about {n_left} more experiments).\n"
            f"So far: {state.n_attempts} experiments attempted, {state.n_fail} failed, "
            f"{len(state.obs)} readings, {int(state.tested.sum())} of {len(state.X)} candidates tested.\n"
            f"Features are scaled to 0-1. Tested candidates cannot be measured again.\n"
            f"Value of stopping now under the current belief (best commit, or abandon): {v_stop:.3f}\n\n"
            f"Shortlisted candidates (JSON, one per line):\n"
            + "\n".join(json.dumps(r) for r in rows)
            + "\n\nChoose one action. For measure or commit, give the candidate id."
        )

    # --------------------------------------------------------------- call --
    def ask(self, prompt: str) -> Decision | None:
        """One API call. Returns None if the model refused or gave no parseable answer."""
        kw = dict(model=self.model, max_tokens=8000, system=SYSTEM,
                  messages=[{"role": "user", "content": prompt}],
                  output_config={"effort": self.effort}, output_format=Decision)
        if self.model.startswith(_FALLBACK_PREFIXES):
            r = self.client.beta.messages.parse(
                betas=["server-side-fallback-2026-07-01"], fallbacks="default", **kw)
        else:
            r = self.client.messages.parse(**kw)
        self.in_tokens += r.usage.input_tokens
        self.out_tokens += r.usage.output_tokens
        if r.stop_reason in ("refusal", "max_tokens"):
            return None
        return r.parsed_output

    # ---------------------------------------------------------------- act --
    def act(self, state):
        S = self.shortlist(state)
        prompt = self.build_prompt(state, S)
        self.n_calls += 1
        dec = self.ask(prompt)
        action = self._validate(dec, state, S)
        if action is None:      # refused / unparseable / illegal move: take the safe terminal action
            self.n_invalid += 1
            action = terminal_action(state)
        if self.log is not None:
            self.log({**self.log_meta, "step": state.n_attempts, "spent": round(state.spent, 4),
                      "action": action[0], "candidate": int(action[1]) if len(action) > 1 else None,
                      "valid": dec is not None and self._validate(dec, state, S) is not None,
                      "reason": dec.reason if dec else None})
        return action

    @staticmethod
    def _validate(dec, state, S):
        if dec is None:
            return None
        if dec.action == "abandon":
            return ("abandon",)
        k = dec.candidate_id
        if k is None or k not in S:
            return None
        if dec.action == "measure":
            return ("measure", int(k)) if state.measurable()[k] else None
        return ("commit", int(k))


class MockClient:
    """Stand-in for the API so the pipeline can be tested with no key and no cost.

    It is NOT an LLM: it measures the untested candidate most likely to meet
    spec five times, then stops. Results from it mean nothing scientifically.
    """

    class _Usage:
        input_tokens = output_tokens = 0

    class _Resp:
        stop_reason = "end_turn"

        def __init__(self, parsed):
            self.parsed_output, self.usage = parsed, MockClient._Usage()

    def __init__(self):
        self.messages = self.beta = self
        self._lock = threading.Lock()

    def parse(self, messages, **_):
        text = messages[0]["content"]
        rows = [json.loads(l) for l in text.splitlines() if l.startswith('{"id"')]
        attempted = int(text.split("So far: ")[1].split(" ")[0])
        untested = [r for r in rows if r["status"] == "untested"]
        if attempted < 8 and untested:
            r = max(untested, key=lambda r: r["p_meets_spec"])
            return self._Resp(Decision(action="measure", candidate_id=r["id"], reason="mock"))
        best = max(rows, key=lambda r: r["expected_value_if_committed"])
        if best["expected_value_if_committed"] > 0:
            return self._Resp(Decision(action="commit", candidate_id=best["id"], reason="mock"))
        return self._Resp(Decision(action="abandon", reason="mock"))
