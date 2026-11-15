"""Expected value of committing, and the value of stopping now.

    E_b[U_k]  = V * P(f_k >= tau) - C_commit = V * Phi((mu_k - tau) / sd_k) - C_commit
    V_stop(b) = max(0, max_k E_b[U_k])     (commit to the best candidate, or abandon for 0)
"""
from __future__ import annotations

import numpy as np
from scipy.special import ndtr


def expected_commit_value(mu, var, m):
    """E_b[U_k] for every candidate (m: AgentModel)."""
    return m.payoff * ndtr((mu - m.tau) / np.sqrt(np.maximum(var, 1e-12))) - m.commit_cost


def stop_value(mu, var, m) -> tuple[float, int]:
    """(value of stopping now, best candidate to commit to)."""
    eu = expected_commit_value(mu, var, m)
    k = int(np.argmax(eu))
    return max(0.0, float(eu[k])), k
