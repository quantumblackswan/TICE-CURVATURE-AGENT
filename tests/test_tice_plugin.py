"""
Unit tests and simulation harness for the TICEPlugin.

This test suite verifies the correctness of the key computations
performed by the ``TICEPlugin`` and exercises the new features
introduced in the upgraded version.  Running this script with
pytest or directly as ``python -m unittest test_tice_plugin.py``
will execute the unit tests.  The module also includes a simple
simulation harness that sweeps across ranges of Δψ², τ and η
values to generate a plot of the curvature metrics.  The plot is
saved to ``simulation_plot.png`` in the current directory.

To run the simulation only, invoke this script directly (e.g.
``python test_tice_plugin.py``).  Note that matplotlib must be
installed for the simulation to produce a plot.
"""

import unittest
import math

import torch

from tice_plugin import TICEPlugin


class TestTICEPlugin(unittest.TestCase):
    """Basic unit tests for the TICEPlugin."""

    def test_compute_lambda_multi(self):
        """Check that the raw λ computation matches expected formula."""
        plugin = TICEPlugin(alpha=1.0, beta=0.0, phi_gain=1.0, lambda_feedback=0.0, omega_gain=1.0, enable_history=False)
        delta = torch.tensor([4.0])
        tau = torch.tensor([2.0])
        eta = torch.tensor([0.0])
        lambda_raw = plugin.compute_lambda_multi(delta, tau, eta)
        expected = (4.0 / 2.0) * math.exp(-0.0)
        self.assertAlmostEqual(lambda_raw.item(), expected, places=5)

    def test_phi_tanh_bounds(self):
        """Ensure φ remains in (-1, 1) for finite λ."""
        plugin = TICEPlugin(phi_gain=5.0, enable_history=False)
        lambda_values = torch.tensor([-10.0, 0.0, 10.0])
        phi = plugin.compute_phi(lambda_values)
        # All outputs must be within (-1, 1)
        self.assertTrue(torch.all(phi < 1.0))
        self.assertTrue(torch.all(phi > -1.0))

    def test_adjust_lambda_feedback(self):
        """Verify that λ is adjusted by the feedback term."""
        plugin = TICEPlugin(lambda_feedback=1.0, enable_history=False)
        lambda_raw = torch.tensor([2.0])
        phi = torch.tensor([0.5])
        lambda_adj = plugin.adjust_lambda(lambda_raw, phi)
        expected = 2.0 * (1 + 0.5)
        self.assertAlmostEqual(lambda_adj.item(), expected, places=5)

    def test_history_logging(self):
        """Confirm that history logging records entries correctly."""
        plugin = TICEPlugin(enable_history=True)
        # Clear history before test
        plugin.clear_history()
        plugin(torch.tensor([1.0]), torch.tensor([1.0]), torch.tensor([0.0]))
        plugin(torch.tensor([2.0]), torch.tensor([1.0]), torch.tensor([0.0]))
        hist = plugin.get_history()
        self.assertEqual(len(hist), 2)
        self.assertIn("lambda_adj", hist[0])
        self.assertIn("phi", hist[1])

    def test_validation_and_critique(self):
        """Check that validation detects large deviations and critique returns messages."""
        plugin = TICEPlugin(enable_history=True)
        plugin.clear_history()
        # Add baseline
        plugin(torch.tensor([0.1]), torch.tensor([1.0]), torch.tensor([0.0]))
        # Produce a large λ to trigger validation failure
        lambda_adj = torch.tensor([100.0])
        phi = torch.tensor([0.0])
        valid = plugin.validate(lambda_adj, phi)
        self.assertFalse(valid)
        critique = plugin.criticize(lambda_adj, phi)
        self.assertIn("Λ_adj has diverged", critique)


def run_simulation() -> None:
    """Run a parameter sweep simulation and save a plot.

    Sweeps Δψ² over [0.0, 2.0], τ over [0.1, 5.0] and η over
    [0.0, 1.0] in a simple loop.  The mean of the resulting
    curvature metrics is plotted as a function of τ.  This
    demonstrates how one might use the TICEPlugin in experimental
    settings.  The plot is written to ``simulation_plot.png``.
    """
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    plugin = TICEPlugin(enable_history=False)

    tau_values = np.linspace(0.1, 5.0, 50)
    lambda_means = []
    phi_means = []
    omega_means = []
    for tau in tau_values:
        # For each τ sample Δψ² and η uniformly from [0, 2] and [0, 1]
        deltas = torch.tensor(np.random.uniform(0.0, 2.0, size=(100,)), dtype=torch.float32)
        etas = torch.tensor(np.random.uniform(0.0, 1.0, size=(100,)), dtype=torch.float32)
        taus = torch.tensor([tau] * 100, dtype=torch.float32)
        lambda_adj, phi, omega = plugin(deltas, taus, etas)
        lambda_means.append(float(lambda_adj.mean()))
        phi_means.append(float(phi.mean()))
        omega_means.append(float(omega.mean()))

    plt.figure(figsize=(8, 4))
    plt.plot(tau_values, lambda_means, label='E[Λ_adj]')
    plt.plot(tau_values, phi_means, label='E[φ]')
    plt.plot(tau_values, omega_means, label='E[ΩΛ∞]')
    plt.xlabel('τ (Time Memory)')
    plt.ylabel('Mean Metric Value')
    plt.title('Simulation Sweep Over τ')
    plt.legend(loc='best')
    plt.tight_layout()
    plt.savefig('simulation_plot.png')
    plt.close()


if __name__ == '__main__':
    # When executed directly, run the simulation harness
    run_simulation()