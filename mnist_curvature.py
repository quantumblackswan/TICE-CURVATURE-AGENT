"""MNIST-driven multi-agent curvature utilities."""
from __future__ import annotations

import gzip
import os
import urllib.request
from pathlib import Path
from typing import Tuple

import numpy as np

from tice import multi_agent_curvature

MNIST_URLS = {
    "images": "https://storage.googleapis.com/cvdf-datasets/mnist/train-images-idx3-ubyte.gz",
    "labels": "https://storage.googleapis.com/cvdf-datasets/mnist/train-labels-idx1-ubyte.gz",
}


def load_mnist(data_dir: str = "mnist_data", max_samples: int = 1000) -> Tuple[np.ndarray, np.ndarray]:
    """Download and load MNIST dataset.
    
    Parameters
    ----------
    data_dir : str
        Directory to store MNIST data files.
    max_samples : int
        Maximum number of samples to load (for memory efficiency).
        
    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        Images and labels arrays.
    """
    os.makedirs(data_dir, exist_ok=True)
    
    # Download files if not present
    for name, url in MNIST_URLS.items():
        filepath = Path(data_dir) / f"{name}.gz"
        if not filepath.exists():
            print(f"Downloading {name}...")
            urllib.request.urlretrieve(url, filepath)
    
    # Load images
    with gzip.open(Path(data_dir) / "images.gz", "rb") as f:
        # Skip header (16 bytes)
        f.read(16)
        # Read image data
        buf = f.read(28 * 28 * max_samples)
        images = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
        images = images.reshape(-1, 28 * 28) / 255.0
    
    # Load labels  
    with gzip.open(Path(data_dir) / "labels.gz", "rb") as f:
        # Skip header (8 bytes)
        f.read(8)
        # Read label data
        buf = f.read(max_samples)
        labels = np.frombuffer(buf, dtype=np.uint8)
    
    return images[:max_samples], labels[:max_samples]


def mnist_multi_agent_lambda(
    images: np.ndarray,
    labels: np.ndarray,
    agents: int = 3,
    samples: int = 10,
) -> float:
    """Compute multi-agent lambda using MNIST features.
    
    Parameters
    ----------
    images : np.ndarray
        MNIST image data, shape (N, 784).
    labels : np.ndarray  
        MNIST labels, shape (N,).
    agents : int
        Number of agents in the multi-agent system.
    samples : int
        Number of samples per agent.
        
    Returns
    -------
    float
        Multi-agent curvature value.
    """
    n_samples = min(samples, len(images) // agents)
    
    # Split data among agents
    delta_psi_sq = []
    tau = []
    eta = []
    eta_dot = []
    
    for i in range(agents):
        start_idx = i * n_samples
        end_idx = (i + 1) * n_samples
        
        agent_images = images[start_idx:end_idx]
        agent_labels = labels[start_idx:end_idx]
        
        # Compute delta_psi_sq from image variance
        img_var = np.var(agent_images, axis=1)
        delta_psi_sq.append(img_var.tolist())
        
        # Compute tau from label entropy
        unique_labels, counts = np.unique(agent_labels, return_counts=True)
        label_entropy = -np.sum((counts / n_samples) * np.log(counts / n_samples + 1e-12))
        tau.append([1.0 + label_entropy] * n_samples)
        
        # Agent-specific parameters
        eta.append(0.1 + 0.05 * i)
        eta_dot.append(0.01 * (-1) ** i)
    
    # Create trust and coupling matrices
    phi = np.random.rand(agents, agents) * 0.8 + 0.1
    phi = (phi + phi.T) / 2
    np.fill_diagonal(phi, 0.0)
    
    coupling = np.ones((agents, agents)) - np.eye(agents)
    
    return multi_agent_curvature(
        delta_psi_sq,
        tau,
        eta=eta,
        gamma=0.5,
        eta_dot=eta_dot,
        phi=phi,
        coupling=coupling,
    )