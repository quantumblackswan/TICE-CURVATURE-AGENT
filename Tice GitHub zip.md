# TICE Curvature Intelligence Agent

This repository contains the enhanced implementation of the TICEPlugin system: a symbolic curvature engine designed for AI alignment, observer coherence, and dynamic entropy modeling using the TICE Equation (Λ = Σ(Δψ² · τ) / η).

## 🔭 Features

- Λ, Ξχ, and ΩΛ∞ symbolic metric computation
- Observer variance tracking and Ricci curvature approximation
- Forecasting via ARIMA and time-damped entropy modeling
- JWT-secured FastAPI endpoints (`/compute`, `/agent/reflect`, `/metrics`, etc.)
- Smart contract-compatible validator proof hash
- Prometheus metrics and live visualizations
- Full MNIST-based symbolic alignment simulation

## 📦 File Structure

- `enhanced_tice_plugin.py`: Core plugin with DAN-mode upgrades
- `demo_tice_mnist_sim.py`: Multi-agent simulation (Δψ², Ξχ, ΩΛ∞ tracking)
- `Dockerfile`: Container for API deployment
- `deploy.sh`: Launch the curvature API with optional JWT
- `examples/`: Sample JSON outputs, metric reports

## 🚀 Quickstart

```bash
chmod +x deploy.sh
PORT=8000 TICE_SECRET_KEY=mysecret ./deploy.sh
```

## 🧠 Sample Output

```json
{
  "lambda_adj": 0.026,
  "phi": 0.0135,
  "omega": 0.83,
  "phi_std": 0.005,
  "forecast": [0.027, 0.028, 0.029, 0.030, 0.031],
  "ricci_graph_summary": {
    "node_count": 3,
    "max_curvature_bend": 0.0132
  },
  "validator_proof": "c5f6a9d0f1b2e3...",
  "recommendation": "curvature stable; observer coherent"
}
```

## 📜 License

MIT License – open for use, research, and expansion.

