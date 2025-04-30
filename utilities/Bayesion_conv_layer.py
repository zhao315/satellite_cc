import torch
import torch.nn as nn
import torch.nn.functional as F


class BayesianConvLayer(nn.Module):
    """Bayesian Convolutional Layer with weight uncertainties"""
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(BayesianConvLayer, self).__init__()

        # Mean and log variance parameters for the weights
        self.weight_mu = nn.Parameter(torch.Tensor(out_channels, in_channels, kernel_size, kernel_size))
        self.weight_log_var = nn.Parameter(torch.Tensor(out_channels, in_channels, kernel_size, kernel_size))
        # define prior distirbution for weights
        self.register_buffer("prior_weight_mu", torch.zeros_like(self.weight_mu))
        self.register_buffer("prior_weight_sigma", torch.ones_like(self.weight_mu))

        # Bias parameters
        self.bias_mu = nn.Parameter(torch.Tensor(out_channels))
        self.bias_log_var = nn.Parameter(torch.Tensor(out_channels))
        # define prior distribution for bias
        self.register_buffer("prior_bias_mu", torch.zeros_like(self.bias_mu))
        self.register_buffer("prior_bias_sigma", torch.ones_like(self.bias_mu))

        # Layer settings
        self.stride = stride
        self.padding = padding

        # Initialize parameters
        self.reset_parameters()

    def reset_parameters(self):
        # Initialize weights using Kaiming initialization
        nn.init.kaiming_normal_(self.weight_mu)
        nn.init.constant_(self.weight_log_var, -10)  # Start with small variance
        nn.init.constant_(self.bias_mu, 0)
        nn.init.constant_(self.bias_log_var, -10)

    def forward(self, x):
        # Sample weights with reparameterization trick
        weight_sigma = torch.exp(0.5 * self.weight_log_var)
        weight_epsilon = torch.randn_like(self.weight_mu)
        sampled_weight = self.weight_mu + weight_epsilon * weight_sigma

        # Sample bias with reparameterization trick
        bias_sigma = torch.exp(0.5 * self.bias_log_var)
        bias_epsilon = torch.randn_like(self.bias_mu)
        sampled_bias = self.bias_mu + bias_epsilon * bias_sigma

        # Perform convolution with sampled weights and bias
        output = F.conv2d(x, sampled_weight, sampled_bias, stride=self.stride, padding=self.padding)

        return output

    def kl_divergence(self):
        """Calculate KL divergence between posterior and prior for the layer parameters"""
        device = self.weight_mu.device

        # KL for weights
        kl_weights = self._kl_normal(
            self.weight_mu,
            torch.exp(self.weight_log_var),
            self.prior_weight_mu.to(device),
            self.prior_weight_sigma.to(device).pow(2),
        )

        # KL for bias
        kl_bias = self._kl_normal(
            self.bias_mu,
            torch.exp(self.bias_log_var),
            self.prior_bias_mu.to(device),
            self.prior_bias_sigma.to(device).pow(2),
        )

        return kl_weights + kl_bias

    def _kl_normal(self, mu_1, sigma_1_squared, mu_2, sigma_2_squared):
        """KL divergence between two normal distributions"""
        return 0.5 * (
            (sigma_1_squared / sigma_2_squared).sum()
            + ((mu_2 - mu_1).pow(2) / sigma_2_squared).sum()
            + torch.log(sigma_2_squared / sigma_1_squared).sum()
            - mu_1.numel()
        )