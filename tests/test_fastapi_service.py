import math

import pytest
from fastapi.testclient import TestClient

from fastapi_service import app


client = TestClient(app)


def test_simulate_curvature_endpoint():
    resp = client.post("/simulate/curvature", json={"agents": 2, "steps": 2})
    assert resp.status_code == 200
    data = resp.json()
    assert "lambda" in data and "curve_index" in data
    assert isinstance(data["lambda"], float)


def test_compute_xi_chi_endpoint():
    resp = client.post("/compute/xi_chi", json={"probabilities": [0.2, 0.8]})
    assert resp.status_code == 200
    xi_chi = resp.json()["xi_chi"]
    expected = -0.2 * math.log2(0.2) - 0.8 * math.log2(0.8)
    assert xi_chi == pytest.approx(expected)


def test_forecast_scg_endpoint():
    resp = client.post("/forecast/scg", json={"lambdas": [0.1, 0.3, 0.5], "dt": 1.0})
    assert resp.status_code == 200
    scg = resp.json()["scg"]
    expected = (0.5 - 0.1) / 2
    assert scg == pytest.approx(expected)
