"""
Enhanced TICEPlugin
====================

This module extends the original TICEPlugin implementation with
additional analytical and service‑oriented features.  In addition to
computing curvature metrics (Λ, Ξχ, ΩΛ∞) it now tracks the standard
deviation of observer variance over time, forecasts future curvature
values, exposes Prometheus‑compatible metrics, provides a reflective
endpoint for symbolic objectives and secures all API routes with
JWT bearer authentication.  It also offers a cryptographic hash of
the most recent metrics for external verification via smart
contracts.

The primary class exported from this module is ``TICEPlugin``.  The
class inherits from ``torch.nn.Module`` and therefore integrates
seamlessly into PyTorch pipelines.  When run as a script the module
also launches a FastAPI server exposing a REST API.  The API supports
both the original endpoints (/compute, /history, /visualize) and
several new endpoints (/metrics, /agent/reflect, /token) while
maintaining backwards compatibility.

Dependencies
------------

In addition to PyTorch, this module requires the following extra
packages when the API is used:

* ``fastapi`` and ``uvicorn``: for the REST service.
* ``pydantic``: for request and response models.
* ``prometheus_client``: for exporting metrics in Prometheus format.
* ``python‑jose[cryptography]``: for generating and verifying JWTs.
* ``networkx`` and ``matplotlib``: optional features used in the
  Ricci flow graph and visualisation.

These dependencies should be specified in the Dockerfile for
deployment.
"""

from __future__ import annotations

import math
import time
import hashlib
from typing import Iterable, Optional, Sequence, List, Dict, Any

try:
    import torch
except ImportError:
    raise ImportError("PyTorch is required to use TICEPlugin. Please install torch.")

class TICEPlugin(torch.nn.Module):
    """An enhanced PyTorch module implementing symbolic curvature analytics.

    This class extends the original TICEPlugin with additional
    introspection and forecasting capabilities.  It maintains an
    internal ring buffer of recent inputs and computed metrics to
    support moving‑average estimation, variance tracking and future
    predictions.  A cryptographic ``validator_proof`` method is
    provided to emit a hash of the current metrics that can be used
    externally (e.g. by a smart contract) to validate the plugin's
    state.

    Parameters
    ----------
    alpha : float, optional
        Scaling factor applied to the curvature index Λ.
    beta : float, optional
        Offset added to the curvature index Λ.
    phi_gain : float, optional
        Gain controlling the non‑linearity when computing the observer
        variance Ξχ.
    lambda_feedback : float, optional
        Feedback coefficient applied when adjusting Λ by Ξχ.
    omega_gain : float, optional
        Scaling applied when computing the entropy compression metric ΩΛ∞.
    buffer_size : int, optional
        Number of recent samples stored in the ring buffer.
    enable_history : bool, optional
        Whether to record a history of all computed metrics.

    Notes
    -----
    The enhanced plugin remains fully backwards compatible with the
    original implementation.  All existing methods and properties
    retain their semantics.  The new methods are additive.
    """

    def __init__(
        self,
        *,
        alpha: float = 1.0,
        beta: float = 0.0,
        phi_gain: float = 1.0,
        lambda_feedback: float = 1.0,
        omega_gain: float = 1.0,
        buffer_size: int = 100,
        enable_history: bool = True,
    ) -> None:
        super().__init__()
        self.alpha = torch.nn.Parameter(torch.tensor(alpha, dtype=torch.float32), requires_grad=False)
        self.beta = torch.nn.Parameter(torch.tensor(beta, dtype=torch.float32), requires_grad=False)
        self.phi_gain = torch.nn.Parameter(torch.tensor(phi_gain, dtype=torch.float32), requires_grad=False)
        self.lambda_feedback = torch.nn.Parameter(torch.tensor(lambda_feedback, dtype=torch.float32), requires_grad=False)
        self.omega_gain = torch.nn.Parameter(torch.tensor(omega_gain, dtype=torch.float32), requires_grad=False)
        # Internal buffers
        self.buffer_size = buffer_size
        self.delta_history: List[float] = []
        self.tau_history: List[float] = []
        self.eta_history: List[float] = []
        # History of computed metrics
        self.enable_history = enable_history
        self.history: List[Dict[str, float]] = []
        self._last_timestamp: Optional[float] = None

    # ------------------------------------------------------------------
    # Core computations (identical to original implementation)
    # ------------------------------------------------------------------
    def compute_lambda_multi(self, delta_psi_sq: torch.Tensor, tau: torch.Tensor, eta: torch.Tensor) -> torch.Tensor:
        eps = 1e-8
        base = delta_psi_sq / (tau + eps)
        decayed = base * torch.exp(-eta)
        lambda_multi = self.alpha * decayed + self.beta
        return lambda_multi

    def compute_phi(self, lambda_values: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.phi_gain * lambda_values)

    def adjust_lambda(self, lambda_values: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
        return lambda_values * (1.0 + self.lambda_feedback * phi)

    def compute_omega(self, lambda_adj: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
        return self.omega_gain * lambda_adj / (1.0 + torch.abs(phi))

    def forward(
        self,
        delta_psi_sq: Optional[torch.Tensor] = None,
        tau: Optional[torch.Tensor] = None,
        eta: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute adjusted curvature, variance and entropy compression.

        If any of the inputs are ``None``, default values are derived from
        the internal ring buffer via exponential moving averages.  The
        computed metrics are logged to the history if enabled and a
        timestamp is recorded.
        """
        # Helper for exponential moving average
        def ema(values: List[float], alpha: float = 0.1, default: float = 0.0) -> float:
            if not values:
                return default
            avg = values[0]
            for v in values[1:]:
                avg = alpha * v + (1 - alpha) * avg
            return avg

        current_time = time.time()
        # Estimate tau from timestamps when not provided
        if tau is None:
            if self._last_timestamp is None:
                tau = torch.tensor([1.0], dtype=torch.float32)
            else:
                dt = current_time - self._last_timestamp
                tau = torch.tensor([float(max(dt, 1e-3))], dtype=torch.float32)
        # Estimate delta_psi_sq if absent
        if delta_psi_sq is None:
            delta_est = ema(self.delta_history, default=0.0)
            delta_psi_sq = torch.tensor([delta_est], dtype=torch.float32)
        # Estimate eta if absent
        if eta is None:
            eta_est = ema(self.eta_history, default=0.0)
            eta = torch.tensor([eta_est], dtype=torch.float32)

        # Convert scalars into tensors if necessary
        if not isinstance(delta_psi_sq, torch.Tensor):
            delta_psi_sq = torch.tensor(delta_psi_sq, dtype=torch.float32)
        if not isinstance(tau, torch.Tensor):
            tau = torch.tensor(tau, dtype=torch.float32)
        if not isinstance(eta, torch.Tensor):
            eta = torch.tensor(eta, dtype=torch.float32)

        lambda_raw = self.compute_lambda_multi(delta_psi_sq, tau, eta)
        phi = self.compute_phi(lambda_raw)
        lambda_adj = self.adjust_lambda(lambda_raw, phi)
        omega = self.compute_omega(lambda_adj, phi)

        # Update ring buffer and history
        self.update_buffer(delta_psi_sq.detach().cpu(), tau.detach().cpu(), eta.detach().cpu())
        if self.enable_history:
            self.log_metrics(
                delta_psi_sq=float(delta_psi_sq.mean()),
                tau=float(tau.mean()),
                eta=float(eta.mean()),
                lambda_raw=float(lambda_raw.mean()),
                lambda_adj=float(lambda_adj.mean()),
                phi=float(phi.mean()),
                omega=float(omega.mean()),
                timestamp=current_time,
            )
        self._last_timestamp = current_time
        return lambda_adj, phi, omega

    # ------------------------------------------------------------------
    # Ring buffer and history management
    # ------------------------------------------------------------------
    def update_buffer(self, delta_psi_sq: torch.Tensor, tau: torch.Tensor, eta: torch.Tensor) -> None:
        def append_and_trim(lst: List[float], value: float) -> None:
            lst.append(float(value))
            if len(lst) > self.buffer_size:
                del lst[0]

        append_and_trim(self.delta_history, float(delta_psi_sq.mean()))
        append_and_trim(self.tau_history, float(tau.mean()))
        append_and_trim(self.eta_history, float(eta.mean()))

    def log_metrics(
        self,
        *,
        delta_psi_sq: float,
        tau: float,
        eta: float,
        lambda_raw: float,
        lambda_adj: float,
        phi: float,
        omega: float,
        timestamp: float,
    ) -> None:
        self.history.append({
            "timestamp": timestamp,
            "delta_psi_sq": delta_psi_sq,
            "tau": tau,
            "eta": eta,
            "lambda_raw": lambda_raw,
            "lambda_adj": lambda_adj,
            "phi": phi,
            "omega": omega,
        })

    def get_history(self) -> List[Dict[str, float]]:
        return list(self.history)

    def clear_history(self) -> None:
        self.history.clear()

    # ------------------------------------------------------------------
    # Self‑reflection and validation
    # ------------------------------------------------------------------
    def validate(self, lambda_adj: torch.Tensor, phi: torch.Tensor) -> bool:
        past_lambda_mean = 0.0
        if self.history:
            past_lambda_mean = float(sum(h["lambda_adj"] for h in self.history) / len(self.history))
        lambda_excess = lambda_adj.detach().cpu().abs().max() > 10.0 * (past_lambda_mean + 1e-6)
        phi_saturation = phi.detach().cpu().abs().max() > 0.99
        return not (lambda_excess or phi_saturation)

    def criticize(self, lambda_adj: torch.Tensor, phi: torch.Tensor) -> str:
        is_valid = self.validate(lambda_adj, phi)
        if is_valid:
            return "Metrics are within acceptable bounds."
        messages: List[str] = []
        past_lambda_mean = 0.0
        if self.history:
            past_lambda_mean = float(sum(h["lambda_adj"] for h in self.history) / len(self.history))
        if lambda_adj.detach().cpu().abs().max() > 10.0 * (past_lambda_mean + 1e-6):
            messages.append("Λ_adj has diverged beyond 10× the historical mean. Consider reducing Δψ² or τ.")
        if phi.detach().cpu().abs().max() > 0.99:
            messages.append("φ is saturated near ±1, indicating observer variance instability.")
        return " ".join(messages)

    # ------------------------------------------------------------------
    # Advanced analytics
    # ------------------------------------------------------------------
    def curvature_acceleration(self) -> Optional[float]:
        if len(self.history) < 3:
            return None
        l1 = self.history[-3]["lambda_adj"]
        l2 = self.history[-2]["lambda_adj"]
        l3 = self.history[-1]["lambda_adj"]
        t1 = self.history[-3]["timestamp"]
        t2 = self.history[-2]["timestamp"]
        t3 = self.history[-1]["timestamp"]
        v1 = (l2 - l1) / max(t2 - t1, 1e-6)
        v2 = (l3 - l2) / max(t3 - t2, 1e-6)
        acc = (v2 - v1) / max(t3 - t1, 1e-6)
        return acc

    def ricci_flow_graph(self) -> Optional["networkx.Graph"]:
        try:
            import networkx as nx
        except ImportError:
            return None
        if len(self.history) < 2:
            return None
        G = nx.Graph()
        G.add_node(0)
        phi_prev = self.history[-2]["phi"]
        phi_curr = self.history[-1]["phi"]
        weight = abs(phi_curr - phi_prev)
        G.add_edge(0, 0, weight=weight)
        _ = nx.edge_betweenness_centrality(G, weight="weight")
        return G

    # ------------------------------------------------------------------
    # New enhancement methods
    # ------------------------------------------------------------------
    def variance_over_time(self, window: Optional[int] = None) -> Optional[float]:
        """Compute the standard deviation of observer variance (Ξχ) over time.

        Parameters
        ----------
        window : int or None, optional
            Number of recent entries to consider.  If ``None`` all history
            entries are used.  Requires at least two entries to compute
            a variance.  Returns ``None`` if insufficient data.

        Returns
        -------
        float or None
            The standard deviation of φ across the specified window, or
            ``None`` if less than two data points exist.
        """
        if not self.history:
            return None
        subset = self.history[-window:] if window is not None else self.history
        if len(subset) < 2:
            return None
        phi_vals = [entry["phi"] for entry in subset]
        # Use unbiased estimator via torch
        phi_tensor = torch.tensor(phi_vals, dtype=torch.float32)
        return float(phi_tensor.std(unbiased=False).item())

    def forecast_lambda(self, n_steps: int = 5) -> List[float]:
        """Forecast future adjusted curvature indices (Λ_adj) over n steps.

        This method uses a simple exponential smoothing approach.  It
        computes the exponential moving average (EMA) of historical
        Λ_adj values and then projects future values by repeatedly
        smoothing towards that EMA.  If no history exists, a list of
        zeros is returned.

        Parameters
        ----------
        n_steps : int, optional
            How many future time steps to forecast.  Defaults to 5.

        Returns
        -------
        list of float
            Predicted Λ_adj values for the next ``n_steps`` time points.
        """
        if not self.history:
            return [0.0 for _ in range(n_steps)]
        lam_values = [h["lambda_adj"] for h in self.history]
        # Exponential smoothing factor
        alpha = 0.5
        ema_val = lam_values[0]
        for val in lam_values[1:]:
            ema_val = alpha * val + (1 - alpha) * ema_val
        forecasts: List[float] = []
        current = lam_values[-1]
        for _ in range(n_steps):
            current = alpha * current + (1 - alpha) * ema_val
            forecasts.append(current)
        return forecasts

    def validator_proof(self) -> str:
        """Return a SHA256 hash of the most recent curvature metrics.

        If the history buffer is empty, the hash of an empty byte
        string is returned.  Otherwise the latest Λ_adj, φ and ΩΛ∞
        values are concatenated and hashed.  This provides an
        immutable fingerprint of the plugin's state suitable for
        external verification (e.g. by a blockchain smart contract).

        Returns
        -------
        str
            A hexadecimal representation of the SHA256 digest.
        """
        if not self.history:
            return hashlib.sha256(b"").hexdigest()
        latest = self.history[-1]
        data = f"{latest['lambda_adj']},{latest['phi']},{latest['omega']}".encode("utf-8")
        return hashlib.sha256(data).hexdigest()


# Optional REST API with JWT authentication and Prometheus metrics
try:
    from fastapi import FastAPI, HTTPException, Depends, Request, Response, status
    from pydantic import BaseModel
    from jose import JWTError, jwt
    from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
    from prometheus_client import CollectorRegistry, Gauge, generate_latest

    # JWT configuration.  The secret key can be overridden via the
    # ``TICE_SECRET_KEY`` environment variable.  If unset, a
    # deterministic default is used.  This key is used to sign JWT
    # tokens and should be kept confidential in production.
    import os
    SECRET_KEY = os.environ.get("TICE_SECRET_KEY", "ticepluginsecretkey")
    ALGORITHM = "HS256"
    ACCESS_TOKEN_EXPIRE_SECONDS = 3600

    # Security scheme for extracting bearer tokens
    security = HTTPBearer()

    def create_access_token(data: dict, expires_delta: Optional[int] = None) -> str:
        to_encode = data.copy()
        if expires_delta:
            expire = int(time.time()) + expires_delta
        else:
            expire = int(time.time()) + ACCESS_TOKEN_EXPIRE_SECONDS
        to_encode.update({"exp": expire})
        encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
        return encoded_jwt

    async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)) -> Dict[str, Any]:
        token = credentials.credentials
        try:
            payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
            username: str | None = payload.get("sub")
            if username is None:
                raise JWTError()
        except JWTError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Could not validate credentials",
                headers={"WWW-Authenticate": "Bearer"},
            )
        return {"username": username}

    class ComputeRequest(BaseModel):
        delta_psi_sq: Sequence[float] | float
        tau: Sequence[float] | float
        eta: Sequence[float] | float

    class ComputeResponse(BaseModel):
        lambda_adj: List[float]
        phi: List[float]
        omega: List[float]

    class ReflectRequest(BaseModel):
        objectives: List[str] | str

    class ReflectResponse(BaseModel):
        forecast: List[float]
        phi_interpretation: str

    class TokenRequest(BaseModel):
        username: str
        password: str

    class TokenResponse(BaseModel):
        access_token: str
        token_type: str

    app = FastAPI(title="Enhanced TICEPlugin API", description="Compute and analyse curvature metrics via REST with JWT authentication")

    # Instantiate global plugin
    _plugin_instance = TICEPlugin()

    def _ensure_tensor(x: Sequence[float] | float) -> torch.Tensor:
        if isinstance(x, (int, float)):
            return torch.tensor([float(x)], dtype=torch.float32)
        else:
            return torch.tensor(list(x), dtype=torch.float32)

    # Endpoint to obtain a JWT for a user.  In this simple example the
    # credentials are hard‑coded; in production one would validate
    # against a user database.
    @app.post("/token", response_model=TokenResponse)
    async def token_endpoint(req: TokenRequest) -> TokenResponse:
        if not (req.username == "admin" and req.password == "ticepass"):
            raise HTTPException(status_code=401, detail="Invalid username or password")
        token_data = {"sub": req.username}
        token = create_access_token(token_data)
        return TokenResponse(access_token=token, token_type="bearer")

    @app.post("/compute", response_model=ComputeResponse)
    async def compute_endpoint(req: ComputeRequest, user: Dict[str, Any] = Depends(get_current_user)) -> ComputeResponse:
        delta_t = _ensure_tensor(req.delta_psi_sq)
        tau_t = _ensure_tensor(req.tau)
        eta_t = _ensure_tensor(req.eta)
        lambda_adj, phi, omega = _plugin_instance(delta_t, tau_t, eta_t)
        if not _plugin_instance.validate(lambda_adj, phi):
            raise HTTPException(status_code=400, detail=_plugin_instance.criticize(lambda_adj, phi))
        return ComputeResponse(
            lambda_adj=lambda_adj.detach().cpu().numpy().tolist(),
            phi=phi.detach().cpu().numpy().tolist(),
            omega=omega.detach().cpu().numpy().tolist(),
        )

    @app.get("/history")
    async def history_endpoint(user: Dict[str, Any] = Depends(get_current_user)) -> List[Dict[str, float]]:
        return _plugin_instance.get_history()

    @app.get("/visualize")
    async def visualize_endpoint(user: Dict[str, Any] = Depends(get_current_user)) -> Dict[str, str]:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import base64
        from io import BytesIO
        hist = _plugin_instance.get_history()
        if hist:
            timestamps = [h["timestamp"] for h in hist]
            lambda_vals = [h["lambda_adj"] for h in hist]
            phi_vals = [h["phi"] for h in hist]
            omega_vals = [h["omega"] for h in hist]
        else:
            timestamps = [0.0]
            lambda_vals = [0.0]
            phi_vals = [0.0]
            omega_vals = [0.0]
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(timestamps, lambda_vals, label='Λ_adj', color='tab:blue')
        ax.plot(timestamps, phi_vals, label='φ', color='tab:orange')
        ax.plot(timestamps, omega_vals, label='ΩΛ∞', color='tab:green')
        ax.set_xlabel('Timestamp')
        ax.set_ylabel('Metric Value')
        ax.set_title('Curvature Metrics Over Time')
        ax.legend(loc='best')
        fig.tight_layout()
        buf = BytesIO()
        fig.savefig(buf, format='png')
        plt.close(fig)
        buf.seek(0)
        image_base64 = base64.b64encode(buf.read()).decode('ascii')
        return {"image": image_base64}

    @app.get("/metrics")
    async def metrics_endpoint(user: Dict[str, Any] = Depends(get_current_user)) -> Response:
        registry = CollectorRegistry()
        lambda_g = Gauge('tice_lambda_adj', 'Current adjusted curvature index', registry=registry)
        phi_g = Gauge('tice_phi', 'Current observer variance', registry=registry)
        omega_g = Gauge('tice_omega', 'Current entropy compression metric', registry=registry)
        hist = _plugin_instance.get_history()
        if hist:
            latest = hist[-1]
            lambda_g.set(latest['lambda_adj'])
            phi_g.set(latest['phi'])
            omega_g.set(latest['omega'])
        else:
            lambda_g.set(0.0)
            phi_g.set(0.0)
            omega_g.set(0.0)
        data = generate_latest(registry)
        return Response(content=data, media_type="text/plain")

    @app.post("/agent/reflect", response_model=ReflectResponse)
    async def reflect_endpoint(req: ReflectRequest, user: Dict[str, Any] = Depends(get_current_user)) -> ReflectResponse:
        # Determine number of steps to forecast based on number of objectives if a list
        if isinstance(req.objectives, list):
            n_steps = max(1, len(req.objectives))
        else:
            n_steps = 1
        forecast = _plugin_instance.forecast_lambda(n_steps)
        # Interpret variance: classify as stable if std < 0.1 else volatile
        var = _plugin_instance.variance_over_time()
        if var is None:
            interpretation = "insufficient data"
        elif var < 0.1:
            interpretation = "stable"
        elif var < 0.5:
            interpretation = "moderately volatile"
        else:
            interpretation = "highly volatile"
        return ReflectResponse(forecast=forecast, phi_interpretation=interpretation)

    @app.get("/health")
    async def health_endpoint() -> Dict[str, str]:
        """Health check endpoint for deployment monitoring."""
        return {"status": "healthy", "service": "TICE Curvature Agent"}

    def run_api(host: str = "0.0.0.0", port: int = 8000) -> None:
        import uvicorn
        import os
        # Read port from environment variable if available
        port = int(os.environ.get("PORT", port))
        uvicorn.run(app, host=host, port=port)

except ImportError:
    # API dependencies missing; expose a dummy run_api
    app = None  # type: ignore
    def run_api(*args: Any, **kwargs: Any) -> None:
        raise RuntimeError("FastAPI is not installed; install fastapi, uvicorn, python-jose, and prometheus_client to run the API.")

if __name__ == "__main__":
    try:
        print("Starting Enhanced TICEPlugin API…")
        run_api()
    except RuntimeError as e:
        print(e)