diff --git a//dev/null b/tests/test_mnist_curvature.py
index 0000000000000000000000000000000000000000..3afbf1ec59c3fecdfc5abe0ece20f9502a86bfaf 100644
--- a//dev/null
+++ b/tests/test_mnist_curvature.py
@@ -0,0 +1,14 @@
+import numpy as np
+import sys
+from pathlib import Path
+
+sys.path.append(str(Path(__file__).resolve().parents[1]))
+
+from mnist_curvature import load_mnist, mnist_multi_agent_lambda
+
+
+def test_mnist_multi_agent_lambda():
+    images, labels = load_mnist()
+    lam = mnist_multi_agent_lambda(images, labels, agents=3, samples=2)
+    assert isinstance(lam, float)
+    assert np.isfinite(lam)
