diff --git a//dev/null b/qbond_curvature_dashboard.py
index 0000000000000000000000000000000000000000..61d315890962f10bb1ced68e770de27f5c3fcb23 100644
--- a//dev/null
+++ b/qbond_curvature_dashboard.py
@@ -0,0 +1,84 @@
+import numpy as np
+import streamlit as st
+from langchain.llms import HuggingFaceHub
+from langchain.schema import AIMessage, HumanMessage, SystemMessage
+
+try:  # pragma: no cover - optional dependency
+    from langchain_experimental.autonomous_agents import AutoGPT
+except Exception:  # pragma: no cover - optional dependency
+    AutoGPT = None
+
+from tice import multi_agent_curvature
+from mnist_curvature import load_mnist, mnist_multi_agent_lambda
+
+
+st.title("Q-BOND Curvature Dashboard")
+st.subheader("Live Curvature Simulation")
+
+data_source = st.selectbox("Data Source", ["Random", "MNIST"])
+rounds = st.slider("Rounds", 1, 10, 3)
+agents = st.slider("Agents", 2, 10, 2)
+
+if data_source == "MNIST":
+    images, labels = load_mnist()
+
+lambdas = []
+for _ in range(rounds):
+    if data_source == "Random":
+        delta_psi_sq = np.abs(np.random.normal(0.5, 0.2, size=(agents, 3)))
+        tau = np.abs(np.random.normal(1.0, 0.1, size=(agents, 3)))
+        eta = np.abs(np.random.normal(0.5, 0.1, size=agents))
+        eta_dot = np.random.normal(0.0, 0.05, size=agents)
+        phi = np.random.rand(agents, agents)
+        phi = (phi + phi.T) / 2
+        np.fill_diagonal(phi, 0.0)
+        coupling = np.ones((agents, agents)) - np.eye(agents)
+        lam = multi_agent_curvature(
+            delta_psi_sq,
+            tau,
+            eta=eta,
+            gamma=0.5,
+            eta_dot=eta_dot,
+            phi=phi,
+            coupling=coupling,
+        )
+    else:
+        lam = mnist_multi_agent_lambda(images, labels, agents=agents, samples=2)
+    lambdas.append(lam)
+
+st.line_chart(lambdas)
+st.metric("Final Λ_multi", f"{lambdas[-1]:.4f}")
+st.metric("Symbolic Curvature Gain (SCG)", f"{(lambdas[-1] - lambdas[0]):.4f}")
+
+st.header("LangChain Multi-Agent Playground")
+
+if st.button("Chat Between Agents"):
+    llm = HuggingFaceHub(repo_id="gpt2", model_kwargs={"temperature": 0.7})
+    messages = [
+        SystemMessage(content="You are Agent Alpha analyzing curvature."),
+        HumanMessage(content="Discuss the curvature implications on λ."),
+    ]
+    alpha_reply = llm(messages)
+    st.write("Agent Alpha:", alpha_reply.content)
+
+    messages.extend(
+        [
+            AIMessage(content=alpha_reply.content),
+            SystemMessage(content="You are Agent Beta responding to Agent Alpha."),
+        ]
+    )
+    beta_reply = llm(messages)
+    st.write("Agent Beta:", beta_reply.content)
+
+if AutoGPT and st.button("AutoGPT Planning"):
+    llm = HuggingFaceHub(repo_id="gpt2", model_kwargs={"temperature": 0})
+    agent = AutoGPT.from_llm_and_tools(
+        ai_name="CurvaturePlanner",
+        ai_role="Plan curvature experiments",
+        tools=[],
+        llm=llm,
+        human_in_the_loop=False,
+    )
+    plan = agent.run(["Propose a new curvature experiment with λ"])
+    st.write(plan)
+
