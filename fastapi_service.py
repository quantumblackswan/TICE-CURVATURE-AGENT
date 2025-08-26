"""FastAPI microservice exposing TICE curvature metrics."""
from __future__ import annotations

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from tice import multi_agent_curvature
from metrics import curve_index, compute_xi_chi, forecast_scg


app = FastAPI(title="TICE Curvature Service")


class CurvatureRequest(BaseModel):
    agents: int = 3
    steps: int = 3


@app.post("/simulate/curvature")
def simulate_curvature(req: CurvatureRequest):
    agents = req.agents
    steps = req.steps
    delta_psi_sq = np.abs(np.random.normal(0.5, 0.2, size=(agents, steps)))
    tau = np.abs(np.random.normal(1.0, 0.1, size=(agents, steps)))
    eta = np.abs(np.random.normal(0.5, 0.1, size=agents))
    eta_dot = np.random.normal(0.0, 0.05, size=agents)
    phi = np.random.rand(agents, agents)
    phi = (phi + phi.T) / 2
    np.fill_diagonal(phi, 0.0)
    coupling = np.ones((agents, agents)) - np.eye(agents)

    lam = multi_agent_curvature(
        delta_psi_sq,
        tau,
        eta=eta,
        gamma=0.5,
        eta_dot=eta_dot,
        phi=phi,
        coupling=coupling,
    )
    weights = np.ones(agents) / agents
    c_val = curve_index(phi, weights)
    return {"lambda": lam, "curve_index": c_val}


class XiChiRequest(BaseModel):
    probabilities: list[float]


@app.post("/compute/xi_chi")
def api_compute_xi_chi(req: XiChiRequest):
    return {"xi_chi": compute_xi_chi(req.probabilities)}


class ScgRequest(BaseModel):
    lambdas: list[float]
    dt: float = 1.0


@app.post("/forecast/scg")
def api_forecast_scg(req: ScgRequest):
    try:
        scg_val = forecast_scg(req.lambdas, dt=req.dt)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"scg": scg_val}
