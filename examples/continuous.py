import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from tqdm import tqdm

from bayesian_flow_torch import BayesianFlow


class Model(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=32):
        super(Model, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(input_dim + 1, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x, t):
        bsz, dim = x.shape
        x = torch.cat((x, t.unsqueeze(-1)), dim=-1)
        output = self.layer(x)
        return output.view(bsz, dim)


def get_two_moons_data(batch_size, noise=0.1, device='cpu'):
    theta = torch.rand(batch_size, device=device) * torch.pi
    moon_idx = torch.randint_like(theta, high=2, dtype=torch.int64)

    x = torch.cos(theta) + moon_idx.float() - 0.5
    y = torch.sin(theta) * (moon_idx.float() * 2 - 1)

    x += torch.randn(batch_size, device=device) * noise
    y += torch.randn(batch_size, device=device) * noise

    coordinates = torch.stack([x / 2, y / 1.5], dim=1)

    return coordinates


def continuous_example():
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'

    model = Model(input_dim=2, output_dim=2)
    model.to(device)

    bayesian_flow = BayesianFlow(sigma=0.01)

    optim = torch.optim.AdamW(model.parameters(), lr=1e-3)

    n = 5000
    losses = []

    model.train()
    for _ in tqdm(range(n)):
        optim.zero_grad()

        x = get_two_moons_data(batch_size=2048, device=device)
        loss = bayesian_flow.continuous_data_continuous_loss(model, x).loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optim.step()

        losses.append(loss.item())

    x = get_two_moons_data(1024, device='cpu').numpy()

    model.eval()
    x_hat = bayesian_flow.continuous_data_sample(model, size=(1024, 2), device=device, num_steps=20).cpu().numpy()

    plt.figure(figsize=(10, 15))

    plt.subplot(3, 1, 1)
    plt.title("Dataset")
    plt.scatter(x[:, 0], x[:, 1])

    plt.subplot(3, 1, 2)
    plt.plot(losses)
    plt.title("Losses over Time")

    plt.subplot(3, 1, 3)
    plt.title("Samples")
    plt.scatter(x_hat[:, 0], x_hat[:, 1])

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    continuous_example()
