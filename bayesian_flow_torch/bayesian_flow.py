import math
from typing import TypeVar, Any, Union, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

T = TypeVar('T', float, torch.Tensor)


def append_dims(tensor: torch.Tensor, target_dims: int) -> torch.Tensor:
    assert isinstance(target_dims, int), f"Expected 'target_dims' to be an integer, but received {type(target_dims)}."
    tensor_dims = tensor.ndim
    assert tensor_dims <= target_dims, f"Tensor has {tensor_dims} dimensions, but target has {target_dims} dimensions."
    return tensor[(...,) + (None,) * (target_dims - tensor_dims)]


class BayesianFlow:
    def __init__(self, model: nn.Module, num_classes: int = None, beta: float = None, sigma: float = None) -> None:
        super(BayesianFlow, self).__init__()
        self.model = model
        self.num_classes = num_classes
        self.beta = beta
        self.sigma = sigma

    def get_alpha(self, t: T) -> T:
        return self.beta * t

    def get_beta(self, t: T) -> T:
        return self.beta * (t ** 2.0)

    def get_gamma(self, t: T) -> T:
        return 1 - (self.sigma ** (t * 2.0))

    def continuous_output_prediction(
            self,
            mu: torch.Tensor,
            t: torch.Tensor,
            gamma: torch.Tensor,
            t_min: float = 1e-10,
            x_min: float = -1.0,
            x_max: float = 1.0,
            **model_kwargs: Any
    ) -> torch.Tensor:
        output = self.model(mu, t, **model_kwargs)

        gamma = append_dims(gamma, mu.ndim)
        x_hat = (mu / gamma) - (torch.sqrt((1 - gamma) / gamma) * output)
        x_hat = torch.clamp(x_hat, x_min, x_max)

        condition = t < t_min
        return torch.where(append_dims(condition, x_hat.ndim), torch.zeros_like(x_hat), x_hat)

    def continuous_data_continuous_loss(self, x: torch.Tensor, **model_kwargs: Any) -> torch.Tensor:
        assert self.sigma is not None, "Sigma must be set at initialisation for continuous data."

        bsz = x.shape[0]

        t = torch.rand(bsz, device=x.device, dtype=torch.float32)

        gamma = self.get_gamma(t)

        mean = append_dims(gamma, x.ndim) * x
        std = append_dims(gamma * (1 - gamma), x.ndim).sqrt()
        eps = torch.randn_like(x)
        mu = mean + eps * std

        x_hat = self.continuous_output_prediction(mu, t, gamma, **model_kwargs)

        weights = -math.log(self.sigma) * (self.sigma ** (t * -2.0))
        mse = ((x - x_hat) ** 2).mean(-1)
        loss_limit_inf = append_dims(weights, mse.ndim) * mse
        return loss_limit_inf.mean()

    @torch.inference_mode()
    def continuous_data_sample(
            self,
            size: Tuple[int, ...],
            num_steps: int = 100,
            device: Union[str, torch.device] = 'cpu',
            **model_kwargs: Any
    ) -> torch.Tensor:
        assert self.sigma is not None, "Sigma must be set at initialisation for continuous data."

        mu = torch.zeros(size, device=device)
        rho = 1

        for i in range(1, num_steps + 1):
            t = (i - 1) / num_steps
            t = t * torch.ones((mu.shape[0]), device=mu.device, dtype=mu.dtype)

            gamma = self.get_gamma(t)
            x_hat = self.continuous_output_prediction(mu, t, gamma, **model_kwargs)
            alpha = self.sigma ** (-2 * i / num_steps) * (1 - self.sigma ** (2 / num_steps))

            mean = x_hat
            std = torch.full_like(mean, fill_value=alpha).rsqrt()
            eps = torch.randn_like(x_hat)
            y = mean + std * eps

            mu = ((rho * mu) + (alpha * y)) / (rho + alpha)
            rho = rho + alpha

        t = torch.ones((mu.shape[0]), device=mu.device, dtype=mu.dtype)
        x_hat = self.continuous_output_prediction(mu, t, self.get_gamma(t))
        return x_hat

    def discrete_output_distribution(self, theta: torch.Tensor, t: torch.Tensor, **model_kwargs: Any) -> torch.Tensor:
        output = self.model(theta, t, **model_kwargs)

        if output.shape[-1] == 1:
            p_sub_o_true = torch.sigmoid(output)
            p_sub_o = torch.cat((p_sub_o_true, 1 - p_sub_o_true), dim=-1)
        else:
            p_sub_o = torch.nn.functional.softmax(output, dim=-1)
        return p_sub_o

    def discrete_data_continuous_loss(self, ids: torch.Tensor, **model_kwargs: Any) -> torch.Tensor:
        assert self.num_classes is not None, "Number of classes must be set at initialisation for discrete data."
        assert self.beta is not None, "Number of classes must be set at initialisation for discrete data."

        bsz = ids.shape[0]

        t = torch.rand(bsz, device=ids.device, dtype=torch.float32)

        beta = self.get_beta(t)
        one_hot_x = F.one_hot(ids, num_classes=self.num_classes).float()
        mean = append_dims(beta, one_hot_x.ndim) * (self.num_classes * one_hot_x - 1)
        var = append_dims(beta * self.num_classes, one_hot_x.ndim)
        eps = torch.randn_like(mean)
        y = mean + eps * var.sqrt()

        theta = F.softmax(y, dim=-1)

        p_0 = self.discrete_output_distribution(theta, t, **model_kwargs)

        e_x, e_hat = one_hot_x, p_0
        weights = self.num_classes * self.get_alpha(t)
        mse = ((e_x - e_hat) ** 2).mean(-1)
        loss_limit_inf = append_dims(weights, mse.ndim) * mse
        return loss_limit_inf.mean()

    @torch.inference_mode()
    def discrete_data_sample(
            self,
            size: Tuple[int, ...],
            num_steps: int = 100,
            device: Union[str, torch.device] = 'cpu',
            **model_kwargs: Any
    ) -> torch.Tensor:
        assert self.num_classes is not None, "Number of classes must be set at initialisation for discrete data."
        assert self.beta is not None, "Beta must be set at initialisation for discrete data."

        theta = torch.ones((*size, self.num_classes), device=device) / self.num_classes

        for i in range(1, num_steps + 1):
            t = (i - 1) / num_steps
            t = t * torch.ones((theta.shape[0]), device=theta.device, dtype=theta.dtype)

            k_probs = self.discrete_output_distribution(theta, t, **model_kwargs)
            k = torch.distributions.Categorical(probs=k_probs).sample()
            alpha = self.beta * (2 * i - 1) / (num_steps ** 2)

            e_k = F.one_hot(k, num_classes=self.num_classes).float()
            mean = alpha * (self.num_classes * e_k - 1)
            var = (alpha * self.num_classes)
            std = torch.full_like(mean, fill_value=var).sqrt()
            eps = torch.randn_like(e_k)
            y = mean + std * eps

            theta_prime = torch.exp(y) * theta
            theta = theta_prime / theta_prime.sum(-1, keepdim=True)

        k_probs_final = self.discrete_output_distribution(theta, torch.ones_like(t))
        k_final = k_probs_final.argmax(-1)

        return k_final
