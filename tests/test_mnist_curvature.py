import numpy as np
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from mnist_curvature import load_mnist, mnist_multi_agent_lambda


def test_mnist_multi_agent_lambda():
    images, labels = load_mnist()
    lam = mnist_multi_agent_lambda(images, labels, agents=3, samples=2)
    assert isinstance(lam, float)
    assert np.isfinite(lam)
