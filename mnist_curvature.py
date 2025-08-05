diff --git a//dev/null b/mnist_curvature.py
index 0000000000000000000000000000000000000000..f2798f2ba5cf799076ee40469f21012b623b7781 100644
--- a//dev/null
+++ b/mnist_curvature.py
@@ -0,0 +1,120 @@
+"""MNIST-driven multi-agent curvature utilities."""
+from __future__ import annotations
+
+import gzip
+import os
+import urllib.request
+from pathlib import Path
+from typing import Tuple
+
+import numpy as np
+
+from tice import multi_agent_curvature
+
+MNIST_URLS = {
+    "images": "https://storage.googleapis.com/cvdf-datasets/mnist/train-images-idx3-ubyte.gz",
+    "labels": "https://storage.googleapis.com/cvdf-datasets/mnist/train-labels-idx1-ubyte.gz",
+}
+
+
+def _download_mnist(data_dir: Path) -> Tuple[Path, Path]:
+    """Download MNIST train set if not already present.
+
+    Parameters
+    ----------
+    data_dir : Path
+        Directory to store dataset files.
+
+    Returns
+    -------
+    Tuple[Path, Path]
+        Paths to the image and label files.
+    """
+    data_dir.mkdir(parents=True, exist_ok=True)
+    image_path = data_dir / "train-images-idx3-ubyte.gz"
+    label_path = data_dir / "train-labels-idx1-ubyte.gz"
+    if not image_path.exists():
+        urllib.request.urlretrieve(MNIST_URLS["images"], image_path)
+    if not label_path.exists():
+        urllib.request.urlretrieve(MNIST_URLS["labels"], label_path)
+    return image_path, label_path
+
+
+def load_mnist(data_dir: str | os.PathLike[str] = "data") -> Tuple[np.ndarray, np.ndarray]:
+    """Load MNIST images and labels as ``float`` arrays in ``[0, 1]``.
+
+    The dataset is downloaded on first use.
+    """
+    img_path, lbl_path = _download_mnist(Path(data_dir))
+    with gzip.open(img_path, "rb") as f:
+        images = np.frombuffer(f.read(), np.uint8, offset=16)
+    images = images.reshape(-1, 28 * 28).astype(float) / 255.0
+    with gzip.open(lbl_path, "rb") as f:
+        labels = np.frombuffer(f.read(), np.uint8, offset=8)
+    return images, labels
+
+
+def mnist_multi_agent_lambda(
+    images: np.ndarray,
+    labels: np.ndarray,
+    *,
+    agents: int = 3,
+    samples: int = 2,
+) -> float:
+    """Compute a multi-agent curvature score from MNIST samples.
+
+    Parameters
+    ----------
+    images : np.ndarray
+        MNIST images flattened to vectors in ``[0, 1]``.
+    labels : np.ndarray
+        Corresponding digit labels.
+    agents : int, optional
+        Number of digit agents to sample, by default ``3``.
+    samples : int, optional
+        Number of successive samples per agent, by default ``2``.
+    """
+    digits = np.random.choice(np.arange(10), size=agents, replace=False)
+    delta_list = []
+    tau_list = []
+    eta_list = []
+    eta_dot_list = []
+    mean_states = []
+
+    for d in digits:
+        idx = np.where(labels == d)[0]
+        chosen = np.random.choice(idx, size=samples + 1, replace=False)
+        imgs = images[chosen]
+        state_changes = np.diff(imgs, axis=0)
+        dpsi = np.sum(state_changes ** 2, axis=1)
+        delta_list.append(dpsi)
+        tau_list.append(np.ones_like(dpsi))
+
+        # Entropy of pixel distribution as eta
+        hist1 = np.histogram(imgs[0], bins=256, range=(0.0, 1.0), density=True)[0] + 1e-8
+        hist2 = np.histogram(imgs[1], bins=256, range=(0.0, 1.0), density=True)[0] + 1e-8
+        entropy1 = -np.sum(hist1 * np.log2(hist1))
+        entropy2 = -np.sum(hist2 * np.log2(hist2))
+        eta_list.append(float(entropy1))
+        eta_dot_list.append(float(entropy2 - entropy1))
+        mean_states.append(imgs.mean(axis=0))
+
+    mean_states_arr = np.asarray(mean_states)
+    norms = np.linalg.norm(mean_states_arr, axis=1, keepdims=True) + 1e-8
+    normed = mean_states_arr / norms
+    phi = normed @ normed.T
+    np.fill_diagonal(phi, 0.0)
+    coupling = np.ones((agents, agents)) - np.eye(agents)
+
+    return multi_agent_curvature(
+        delta_list,
+        tau_list,
+        eta=eta_list,
+        gamma=0.5,
+        eta_dot=eta_dot_list,
+        phi=phi,
+        coupling=coupling,
+    )
+
+
+__all__ = ["load_mnist", "mnist_multi_agent_lambda"]
