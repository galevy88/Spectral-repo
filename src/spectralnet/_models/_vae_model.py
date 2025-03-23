import torch
import torch.nn as nn


class VAEModel(nn.Module):
    def __init__(self, architecture: dict, input_dim: int):
        super(VAEModel, self).__init__()
        self.architecture = architecture
        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()

        current_dim = input_dim
        for i, layer in enumerate(self.architecture[:-1]):
            self.encoder.append(nn.Sequential(nn.Linear(current_dim, layer), nn.ReLU()))
            current_dim = layer

        self.mu = nn.Linear(current_dim, self.architecture[-1])
        self.logvar = nn.Linear(current_dim, self.architecture[-1])

        last_dim = input_dim
        current_dim = self.architecture[-1]
        for i, layer in enumerate(reversed(self.architecture[:-1])):
            self.decoder.append(nn.Sequential(nn.Linear(current_dim, layer), nn.ReLU()))
            current_dim = layer
        self.decoder.append(nn.Linear(current_dim, last_dim))

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def encode(self, x: torch.Tensor):
        for layer in self.encoder:
            x = layer(x)
        mu, logvar = self.mu(x), self.logvar(x)
        return mu, logvar

    def decode(self, x: torch.Tensor):
        for layer in self.decoder:
            x = layer(x)
        return x

    def forward(self, x: torch.Tensor):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar