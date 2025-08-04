"""
demo_tice_mnist_sim.py

Multi-agent symbolic simulation over MNIST, using TICE metrics:
Λ = Σ(Δψ² · τ) / η
Ξχ = symbolic observer coherence
ΩΛ∞ = entropy compression metric
"""

import torch
import numpy as np
from torch import nn
from torchvision import datasets, transforms
from collections import deque

# Load MNIST or dummy fallback
try:
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    mnist = datasets.MNIST(root=".", train=True, download=True, transform=transform)
    data_loader = torch.utils.data.DataLoader(mnist, batch_size=32, shuffle=True)
except:
    print("MNIST load failed. Using dummy data.")
    data_loader = [(torch.randn(32, 1, 28, 28), torch.randint(0, 10, (32,)))]

class AgentModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(784, 10)
    def forward(self, x):
        return torch.softmax(self.fc(x.view(-1, 784)), dim=1)

def run_sim(num_agents=10, num_rounds=3):
    models = [AgentModel() for _ in range(num_agents)]
    history = [deque(maxlen=5) for _ in range(num_agents)]
    ts = [0] * num_agents
    lambda_values = []
    omega_values = []

    for _ in range(num_rounds):
        embeddings = []
        probs_list = []

        for i, model in enumerate(models):
            inputs, labels = next(iter(data_loader))
            optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
            outputs = model(inputs)
            loss = nn.CrossEntropyLoss()(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            emb = model.fc.weight.mean(dim=0).detach()
            prob = outputs.mean(dim=0).detach()
            history[i].append(emb)
            embeddings.append(emb)
            probs_list.append(prob)
            ts[i] += 1

        delta_psi = [float((e - torch.mean(torch.stack(list(h)), dim=0)).norm()**2) if h else 0.0
                     for e, h in zip(embeddings, history)]
        tau = [float(torch.stack(list(h)).norm()) if h else 1.0 for h in history]
        eta = [float(torch.var(e)) for e in embeddings]
        xi_chi = [-float(torch.sum(p * torch.log(p + 1e-6))) for p in probs_list]

        phi = np.corrcoef(delta_psi) if len(delta_psi) > 1 else np.ones((len(delta_psi), len(delta_psi)))
        temporal_kernel = lambda t1, t2, alpha=0.1: np.exp(-alpha * abs(t1 - t2))
        lam = 0
        for i in range(len(delta_psi)):
            for j in range(len(delta_psi)):
                lam += phi[i][j] * (delta_psi[i] * tau[i] / eta[i]) * temporal_kernel(ts[i], ts[j])

        omega = np.exp(-np.mean(xi_chi))
        lambda_values.append(lam)
        omega_values.append(omega)

    return lambda_values, omega_values

if __name__ == "__main__":
    lambdas, omegas = run_sim()
    print("Λ values:", lambdas)
    print("Ω values:", omegas)
