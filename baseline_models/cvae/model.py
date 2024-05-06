import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, kl_divergence
import logging

logger = logging.getLogger(__file__)

class Encoder(nn.Module):
    def __init__(self, input_dim, latent_dim, hidden_dim_1, hidden_dim_2):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim_1)
        self.fc2 = nn.Linear(hidden_dim_1, hidden_dim_2)
        # different layers for mean and var
        self.fc31 = nn.Linear(hidden_dim_2, latent_dim)
        self.fc32 = nn.Linear(hidden_dim_2, latent_dim)

    def forward(self, x, eps: float = 1e-8):
        hidden = F.silu(self.fc1(x))
        hidden = F.silu(self.fc2(hidden))
        mean = self.fc31(hidden)
        # soft plus to make var positive
        # eps is supposed to help with numerical stability
        var = F.softplus(self.fc32(hidden)) + eps
        return Normal(mean, var)

class Decoder(nn.Module):
    def __init__(self, input_dim, obs_dim, hidden_dim_1, hidden_dim_2):
        super().__init__()
        # simple MLP
        self.fc1 = nn.Linear(input_dim, hidden_dim_1)
        self.fc2 = nn.Linear(hidden_dim_1, hidden_dim_2)
        self.fc3 = nn.Linear(hidden_dim_2, obs_dim)

    def forward(self, z):
        hidden = F.silu(self.fc1(z))
        hidden = F.silu(self.fc2(hidden))
        y = F.softplus(self.fc3(hidden)) # y's are going to be positive
        return y

class CVAE(nn.Module):
    def __init__(self, obs_dim: int, condition_dim: int, latent_dim: int, hidden_dim: int) -> None:
        super(CVAE, self).__init__()
        self.prior_net = Encoder(condition_dim, latent_dim, hidden_dim, hidden_dim // 2)
        self.recognition_net = Encoder(obs_dim + condition_dim, latent_dim, hidden_dim, hidden_dim // 2)
        self.generation_net = Decoder(latent_dim, obs_dim, hidden_dim // 2, hidden_dim)

    def sample(self, condition):
        # first, get the conditioned latent distribution p(z|x)
        prior = self.prior_net(condition)
        # take some samples
        latents = prior.rsample()
        # and try to reconstruct the observation p(y|x,z)
        obs_hat = self.generation_net(latents)
        return prior, obs_hat

    def forward(self, obs: torch.Tensor, condition: torch.Tensor, compute_loss: bool = True):
        # get prior from condition p(z|x), and try to reconstruct observation
        prior, obs_hat = self.sample(condition)
        # now, we need to get the posterior q(z|x,y)
        x = torch.cat([obs, condition], dim=1)
        posterior = self.recognition_net(x)

        if compute_loss:
            # want reconstructed observation to be close to the observation
            mse = F.mse_loss(obs_hat, obs)
            # want posterior to be close to the prior
            kld = kl_divergence(prior, posterior)
            loss = -kld + mse
            return prior, posterior, obs_hat, mse, kld, loss

        else:
            return prior, posterior, obs_hat
