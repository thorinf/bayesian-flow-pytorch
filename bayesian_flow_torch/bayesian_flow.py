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
    def __init__(
            self,
            model: nn.Module,
            num_classes: int = None,
            beta: float = None,
            sigma: float = None,
            reduced_features_binary: bool = False
    ) -> None:
        super(BayesianFlow, self).__init__()
        if reduced_features_binary:
            assert (num_classes == 2), f"For `reduced_features_binary` number of classes must be 2, got {num_classes}."
        self.model = model
        self.num_classes = num_classes
        self.beta = beta
        self.sigma = sigma
        self.reduced_features_binary = reduced_features_binary

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

    def continuous_data_continuous_loss(self, target: torch.Tensor, **model_kwargs: Any) -> torch.Tensor:
        assert self.sigma is not None, "Sigma must be set at initialisation for continuous data."

        bsz = target.shape[0]

        t = torch.rand(bsz, device=target.device, dtype=torch.float32)

        gamma = self.get_gamma(t)

        mean = append_dims(gamma, target.ndim) * target
        std = append_dims(gamma * (1 - gamma), target.ndim).sqrt()
        eps = torch.randn_like(target)
        mu = mean + eps * std

        x_hat = self.continuous_output_prediction(mu, t, gamma, **model_kwargs)

        weights = -math.log(self.sigma) * (self.sigma ** (t * -2.0))
        mse = ((target - x_hat) ** 2).mean(-1)
        loss_limit_inf = append_dims(weights, mse.ndim) * mse
        return loss_limit_inf.mean()

    @torch.inference_mode()
    def continuous_data_sample(
            self,
            size: Tuple[int, ...],
            num_steps: int = 100,
            return_all: bool = False,
            device: Union[str, torch.device] = 'cpu',
            **model_kwargs: Any
    ) -> torch.Tensor:
        assert self.sigma is not None, "Sigma must be set at initialisation for continuous data."

        outputs_list = []

        mu = torch.zeros(size, device=device)
        rho = 1

        for i in range(1, num_steps + 1):
            t = (i - 1) / num_steps
            t = t * torch.ones((mu.shape[0]), device=mu.device, dtype=mu.dtype)

            gamma = self.get_gamma(t)
            x_hat = self.continuous_output_prediction(mu, t, gamma, **model_kwargs)

            if return_all:
                outputs_list.append(x_hat)

            alpha = self.sigma ** (-2 * i / num_steps) * (1 - self.sigma ** (2 / num_steps))

            mean = x_hat
            std = torch.full_like(mean, fill_value=alpha).rsqrt()
            eps = torch.randn_like(x_hat)
            y = mean + std * eps

            mu = ((rho * mu) + (alpha * y)) / (rho + alpha)
            rho = rho + alpha

        t = torch.ones((mu.shape[0]), device=mu.device, dtype=mu.dtype)
        x_hat = self.continuous_output_prediction(mu, t, self.get_gamma(t))

        if return_all:
            outputs_list.append(x_hat)
            return outputs_list
        else:
            return x_hat

    def discrete_output_distribution(self, theta: torch.Tensor, t: torch.Tensor, **model_kwargs: Any) -> torch.Tensor:
        if self.num_classes == 2 and self.reduced_features_binary:
            theta = theta[..., :1]

        output = self.model(theta, t, **model_kwargs)

        assert output.shape == theta.shape, f"Model output shape {output.shape} does not match input {theta.shape}."

        if self.num_classes == 2 and self.reduced_features_binary:
            p_sub_o_true = torch.sigmoid(output)
            p_sub_o = torch.cat((p_sub_o_true, 1 - p_sub_o_true), dim=-1)
        else:
            p_sub_o = torch.nn.functional.softmax(output, dim=-1)
        return p_sub_o

    def target_to_distribution(self, target: torch.Tensor) -> torch.Tensor:
        if target.dtype == torch.int64:
            target_dist = F.one_hot(target, num_classes=self.num_classes).float()
        elif target.dtype in (torch.float16, torch.float32, torch.float64):
            final_dim = target.shape[-1]
            if self.num_classes == 2 and self.reduced_features_binary:
                assert final_dim == 1, \
                    f"Target probabilities final dimension must be 1 for `reduced_features_binary`, got {final_dim}."
                target = torch.cat((target, 1 - target), dim=-1)
            else:
                assert final_dim == self.num_classes, \
                    f"Target probabilities last dimension must match {self.num_classes} classes, got {final_dim}."
            target_dist = target
        else:
            assert False, f"Unsupported dtype {target.dtype}. Supported dtypes are int64 and float types."
        return target_dist

    def discrete_data_continuous_loss(self, target: torch.Tensor, **model_kwargs: Any) -> torch.Tensor:
        assert self.num_classes is not None, "Number of classes must be set at initialisation for discrete data."
        assert self.beta is not None, "Number of classes must be set at initialisation for discrete data."

        bsz = target.shape[0]

        t = torch.rand(bsz, device=target.device, dtype=torch.float32)

        target_dist = self.target_to_distribution(target)

        beta = self.get_beta(t)
        mean = append_dims(beta, target_dist.ndim) * (self.num_classes * target_dist - 1)
        var = append_dims(beta * self.num_classes, target_dist.ndim)
        eps = torch.randn_like(mean)
        y = mean + eps * var.sqrt()

        theta = F.softmax(y, dim=-1)

        p_0 = self.discrete_output_distribution(theta, t, **model_kwargs)

        e_x, e_hat = target_dist, p_0
        weights = self.num_classes * self.get_alpha(t)
        mse = ((e_x - e_hat) ** 2).mean(-1)
        loss_limit_inf = append_dims(weights, mse.ndim) * mse
        return loss_limit_inf.mean()

    @torch.inference_mode()
    def discrete_data_sample(
            self,
            size: Tuple[int, ...],
            num_steps: int = 100,
            return_all: bool = False,
            device: Union[str, torch.device] = 'cpu',
            **model_kwargs: Any
    ) -> torch.Tensor:
        assert self.num_classes is not None, "Number of classes must be set at initialisation for discrete data."
        assert self.beta is not None, "Beta must be set at initialisation for discrete data."

        outputs_list = []

        theta = torch.ones((*size, self.num_classes), device=device) / self.num_classes

        for i in range(1, num_steps + 1):
            t = (i - 1) / num_steps
            t = t * torch.ones((theta.shape[0]), device=theta.device, dtype=theta.dtype)

            k_probs = self.discrete_output_distribution(theta, t, **model_kwargs)
            k = torch.distributions.Categorical(probs=k_probs).sample()

            if return_all:
                outputs_list.append(k_probs)

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

        if return_all:
            outputs_list.append(k_probs_final)
            return outputs_list
        else:
            return k_probs_final
