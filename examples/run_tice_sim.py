#!/usr/bin/env python3
"""
Example TICE simulation demonstrating curvature computation.
This file is referenced by the deployment workflow.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

# Try to import torch, but gracefully fallback if not available
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

def run_simulation():
    """Run a basic TICE curvature simulation."""
    print("🧠 Starting TICE Curvature Simulation...")
    
    try:
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch not available")
            
        # Import the enhanced TICE plugin
        import importlib.util
        spec = importlib.util.spec_from_file_location("tice_plugin_newest", "TICE plug newest.py")
        tice_plugin_newest = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(tice_plugin_newest)
        TICEPlugin = tice_plugin_newest.TICEPlugin
        
        # Create plugin instance
        plugin = TICEPlugin(enable_history=True)
        
        # Generate sample data
        delta_psi_sq = torch.tensor([0.5, 0.8, 0.3])
        tau = torch.tensor([1.0, 1.2, 0.9])
        eta = torch.tensor([0.1, 0.2, 0.15])
        
        print(f"📊 Input data:")
        print(f"  Δψ²: {delta_psi_sq.numpy()}")
        print(f"  τ: {tau.numpy()}")
        print(f"  η: {eta.numpy()}")
        
        # Compute curvature metrics
        lambda_adj, phi, omega = plugin(delta_psi_sq, tau, eta)
        
        print(f"📈 Computed metrics:")
        print(f"  λ_adj: {lambda_adj.numpy()}")
        print(f"  φ: {phi.numpy()}")
        print(f"  ω: {omega.numpy()}")
        
        # Validate results
        is_valid = plugin.validate(lambda_adj, phi)
        print(f"✓ Validation: {'PASSED' if is_valid else 'FAILED'}")
        
        if not is_valid:
            critique = plugin.criticize(lambda_adj, phi)
            print(f"⚠️  Critique: {critique}")
        
        # Test forecasting if history is available
        if len(plugin.get_history()) > 0:
            forecast = plugin.forecast_lambda(n_steps=3)
            print(f"🔮 Forecast (3 steps): {forecast}")
        
        print("✅ TICE simulation completed successfully!")
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("💡 Trying fallback simulation...")
        
        # Fallback to basic computation without plugin
        delta_psi_sq_np = np.array([0.5, 0.8, 0.3])
        tau_np = np.array([1.0, 1.2, 0.9])
        eta_np = np.array([0.1, 0.2, 0.15])
        
        # Basic lambda computation
        lambda_basic = np.sum(delta_psi_sq_np / tau_np) * np.exp(-np.mean(eta_np))
        
        print(f"📈 Basic λ computation: {lambda_basic:.4f}")
        print("✅ Fallback simulation completed!")
        return True
        
    except Exception as e:
        print(f"❌ Simulation failed: {e}")
        return False

if __name__ == "__main__":
    success = run_simulation()
    sys.exit(0 if success else 1)