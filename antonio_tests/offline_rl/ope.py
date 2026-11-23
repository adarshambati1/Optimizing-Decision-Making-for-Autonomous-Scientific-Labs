"""Off-policy evaluation by importance sampling (ADM appendix A.14, used in section 14.2).

Estimate how a NEW stopping policy pi would score, using only campaigns logged
under OTHER (behavior) policies mu. Importance sampling reweights each logged
campaign by how much more (or less) likely pi was to make the same decisions:

    E_{tau ~ pi}[G] = E_{tau ~ mu}[ w(tau) G(tau) ],   w(tau) = prod_t pi(a_t | s_t) / mu(a_t | s_t)

(the environment's own randomness cancels in the ratio, so only the policies'
action probabilities are needed, and the log stores mu(a_t | s_t) as p_action).

Three estimators:
  * ordinary IS          mean_i  w_i G_i                         (unbiased, high variance; A.14)
  * weighted IS          sum_i w_i G_i / sum_i w_i               (self-normalized: small bias, much lower variance)
  * per-decision IS      mean_i sum_t (prod_{t' <= t} rho_t') r_t
                         (a reward only depends on the decisions before it, so it is weighted by those alone)

The log mixes several behavior policies. Each campaign is weighted against the
behavior that actually generated it, which keeps every estimator unbiased.

A target policy here is any function phi -> P(continue | phi) of the 8 logged
belief features, so learned policies (behavioral cloning, batch Q-learning) can
be evaluated directly. Forced stops (budget spent) have mu(stop) = 1 and every
policy must stop there too, so they contribute a ratio of 1.

Returns are reported as net utility: logged return minus the initial design's
cost, which every policy pays.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd

from logger import FEAT_COLS


@dataclass
class OPEResult:
    ordinary: float
    weighted: float
    per_decision: float
    se_ordinary: float
    ess: float           # effective sample size of the campaign weights, (sum w)^2 / sum w^2
    n: int               # campaigns used
    frac_nonzero: float  # share of campaigns whose logged decisions pi could have made


def target_probs(tr: pd.DataFrame, policy: Callable[[np.ndarray], np.ndarray]) -> np.ndarray:
    """pi(logged action | logged features) for every row."""
    p_cont = np.clip(np.asarray(policy(tr[FEAT_COLS].to_numpy()), float), 0.0, 1.0)
    p_cont = np.where(tr.forced.to_numpy() == 1, 0.0, p_cont)       # nothing left: every policy stops
    return np.where(tr.action.to_numpy() == 1, p_cont, 1.0 - p_cont)


def evaluate(tr: pd.DataFrame, ca: pd.DataFrame, policy: Callable[[np.ndarray], np.ndarray]) -> OPEResult:
    """Importance-sampling estimates of a target policy's net utility on the logged campaigns."""
    tr = tr.sort_values(["campaign", "t"], kind="stable")
    rho = target_probs(tr, policy) / tr.p_action.to_numpy()
    df = pd.DataFrame(dict(campaign=tr.campaign.to_numpy(), rho=rho, r=tr.reward.to_numpy()))
    df["cum"] = df.groupby("campaign")["rho"].cumprod()
    g = df.groupby("campaign")
    w = g["rho"].prod()
    G = g["r"].sum()
    pd_ret = (df.cum * df.r).groupby(df.campaign).sum()
    init = ca.set_index("campaign").init_cost.loc[w.index].to_numpy()
    w, G, pd_ret = w.to_numpy(), G.to_numpy(), pd_ret.to_numpy()
    net = G - init
    ordinary = float(np.mean(w * net))
    weighted = float(np.sum(w * net) / np.sum(w)) if w.sum() > 0 else float("nan")
    per_dec = float(np.mean(pd_ret - init))       # the initial cost is paid with weight 1 by every policy
    return OPEResult(ordinary=ordinary, weighted=weighted, per_decision=per_dec,
                     se_ordinary=float(np.std(w * net, ddof=1) / np.sqrt(len(w))),
                     ess=float(w.sum() ** 2 / max(np.sum(w ** 2), 1e-300)), n=len(w),
                     frac_nonzero=float(np.mean(w > 0)))


# ------------------------------------------------- reference target policies --
def always_continue(Phi):
    return np.ones(len(Phi))


def voi_rule(Phi):
    """Adarsh's one-step rule, unrandomized: continue while VOI / cost > 1."""
    return (Phi[:, 7] > 1.0).astype(float)


def gate(theta):
    th = np.asarray(theta, float)

    def pol(Phi):
        return (1.0 / (1.0 + np.exp(-np.clip(Phi[:, : len(th)] @ th, -30, 30))) > 0.5).astype(float)
    return pol
