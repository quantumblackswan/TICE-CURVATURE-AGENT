import numpy as np
import pytest

from tice import multi_agent_curvature, temporal_curvature


def test_temporal_curvature_basic():
    delta_psi_sq = [1.0, 2.0]
    tau = [1.0, 0.5]
    result = temporal_curvature(delta_psi_sq, tau, eta=1.0, gamma=0.5, eta_dot=0.1)
    expected = (1.0 * 1.0 + 2.0 * 0.5) / (1.0 + 0.5 * 0.1)
    assert result == pytest.approx(expected)


def test_multi_agent_curvature_pair():
    delta_psi_sq = [[1.0], [4.0]]
    tau = [[1.0], [1.0]]
    eta = [1.0, 1.0]
    eta_dot = [0.1, 0.2]
    phi = np.array([[0.0, 0.5], [0.5, 0.0]])
    coupling = np.array([[0.0, 1.0], [1.0, 0.0]])
    lam = multi_agent_curvature(
        delta_psi_sq,
        tau,
        eta=eta,
        gamma=0.5,
        eta_dot=eta_dot,
        phi=phi,
        coupling=coupling,
    )
    lambda1 = (1.0 * 1.0) / (1.0 + 0.5 * 0.1)
    lambda2 = (4.0 * 1.0) / (1.0 + 0.5 * 0.2)
    expected = 0.5 * 0.5 * (lambda1 + lambda2)  # phi * coupling * mean
    assert lam == pytest.approx(expected)
