diff --git a//dev/null b/tice.py
index 0000000000000000000000000000000000000000..f585e47c61bb53289ef3269c2cc44217785a892d 100644
--- a//dev/null
+++ b/tice.py
@@ -0,0 +1,110 @@
+"""TICE curvature computation utilities."""
+from __future__ import annotations
+
+from typing import Sequence
+
+import numpy as np
+
+
+def temporal_curvature(
+    delta_psi_sq: Sequence[float],
+    tau: Sequence[float],
+    *,
+    eta: float,
+    gamma: float,
+    eta_dot: float,
+    epsilon: float = 1e-6,
+) -> float:
+    """Compute temporal curvature :math:`\Lambda(t)` for a single agent.
+
+    Parameters
+    ----------
+    delta_psi_sq : Sequence[float]
+        Squared changes in symbolic state :math:`\Delta \psi^2`.
+    tau : Sequence[float]
+        Memory persistence values :math:`\tau` for each state change.
+    eta : float
+        Entropy pressure :math:`\eta`.
+    gamma : float
+        Damping constant :math:`\gamma`.
+    eta_dot : float
+        Rate of change of entropy :math:`\frac{d\eta}{dt}`.
+    epsilon : float, optional
+        Stabiliser to prevent division by zero, by default ``1e-6``.
+
+    Returns
+    -------
+    float
+        Temporal curvature score.
+    """
+
+    delta_psi_sq_arr = np.asarray(delta_psi_sq, dtype=float)
+    tau_arr = np.asarray(tau, dtype=float)
+    numerator = np.sum(delta_psi_sq_arr * tau_arr)
+    denominator = eta + gamma * abs(eta_dot) + epsilon
+    return float(numerator / denominator)
+
+
+def multi_agent_curvature(
+    delta_psi_sq: Sequence[Sequence[float]],
+    tau: Sequence[Sequence[float]],
+    *,
+    eta: Sequence[float],
+    gamma: float,
+    eta_dot: Sequence[float],
+    phi: np.ndarray,
+    coupling: np.ndarray,
+    epsilon: float = 1e-6,
+) -> float:
+    """Compute multi-agent curvature :math:`\Lambda_{multi}`.
+
+    Each agent ``i`` contributes a temporal curvature ``lambda_i``. The
+    pairwise trust matrix ``phi`` and temporal coupling ``coupling`` weight the
+    influence between agents ``i`` and ``j``.
+
+    Parameters
+    ----------
+    delta_psi_sq : Sequence[Sequence[float]]
+        Squared state changes for each agent.
+    tau : Sequence[Sequence[float]]
+        Memory persistence values per agent.
+    eta : Sequence[float]
+        Entropy pressure for each agent.
+    gamma : float
+        Damping constant shared across agents.
+    eta_dot : Sequence[float]
+        Entropy change rate for each agent.
+    phi : np.ndarray
+        Trust correlation matrix ``Phi`` with values in ``[0, 1]``.
+    coupling : np.ndarray
+        Temporal coupling kernel ``K`` with values in ``[0, 1]``.
+    epsilon : float, optional
+        Stabiliser to prevent division by zero, by default ``1e-6``.
+
+    Returns
+    -------
+    float
+        Multi-agent curvature score.
+    """
+
+    n_agents = len(delta_psi_sq)
+    if phi.shape != (n_agents, n_agents) or coupling.shape != (n_agents, n_agents):
+        raise ValueError("phi and coupling must be square matrices matching agent count")
+
+    lambdas = [
+        temporal_curvature(dps, t, eta=e, gamma=gamma, eta_dot=ed, epsilon=epsilon)
+        for dps, t, e, ed in zip(delta_psi_sq, tau, eta, eta_dot)
+    ]
+    lambdas_arr = np.asarray(lambdas)
+
+    pair_sum = 0.0
+    count = 0
+    for i in range(n_agents):
+        for j in range(i + 1, n_agents):
+            weight = phi[i, j] * coupling[i, j]
+            pair_sum += weight * (lambdas_arr[i] + lambdas_arr[j]) / 2
+            count += 1
+    return float(pair_sum / max(count, 1))
+
+
+__all__ = ["temporal_curvature", "multi_agent_curvature"]
