import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from tqdm.auto import tqdm

from bayesian_flow_torch import BayesianFlow


class Model(nn.Module):
    def __init__(self, num_vars=2, num_classes=2, hidden_dim=32):
        super(Model, self).__init__()
        num_logits = num_classes if num_classes > 2 else 1
        self.layer = nn.Sequential(
            nn.Linear(num_vars * num_logits + 1, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_vars * num_logits)
        )

    def forward(self, x, t):
        bsz, dim, _ = x.shape
        x = x.view(bsz, -1)
        x = torch.cat((x, t.unsqueeze(-1)), dim=-1)
        output = self.layer(x)
        return output.view(bsz, dim, -1)


def get_xor_data(batch_size, device='cpu'):
    # Generate first bit
    x0 = torch.randint(2, (batch_size,), dtype=torch.bool, device=device)

    # Combine for XOR
    x = torch.stack([x0, ~x0], dim=1).long()

    return x


def discrete_example():
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'

    model = Model(num_vars=2, num_classes=2)
    model.to(device)

    bayesian_flow = BayesianFlow(model, num_classes=3, beta=3.0, reduced_features_binary=True)

    optim = torch.optim.AdamW(model.parameters(), lr=1e-3)

    n = 2000
    losses = []

    model.train()
    for _ in tqdm(range(n)):
        optim.zero_grad()

        x = get_xor_data(batch_size=2048, device=device)
        loss = bayesian_flow.discrete_data_continuous_loss(x)
        loss.backward()

        optim.step()

        losses.append(loss.item())

    x = get_xor_data(128, 'cpu')

    model.eval()
    x_hat_logits = bayesian_flow.discrete_data_sample(size=(128, 2), num_steps=100, device=device)
    x_hat = x_hat_logits.argmax(-1).cpu().numpy()

    plt.figure(figsize=(10, 15))

    plt.subplot(3, 1, 1)
    plt.title("Dataset")
    plt.scatter(x[:, 0].numpy(), x[:, 1].numpy())

    plt.subplot(3, 1, 2)
    plt.plot(losses)
    plt.title("Losses over Time")

    plt.subplot(3, 1, 3)
    plt.title("Samples")
    plt.scatter(x_hat[:, 0], x_hat[:, 1])

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    discrete_example()
