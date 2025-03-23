import os
import torch
import torch.nn as nn
import torch.optim as optim

from tqdm import trange
from torch.utils.data import DataLoader, random_split
from .._models import VAEModel  # Assuming your VAE model is here


class VAETrainer:
    def __init__(self, config: dict, device: torch.device):
        self.device = device
        self.vae_config = config
        self.lr = self.vae_config["lr"]
        self.epochs = self.vae_config["epochs"]
        self.min_lr = self.vae_config["min_lr"]
        self.lr_decay = self.vae_config["lr_decay"]
        self.patience = self.vae_config["patience"]
        self.architecture = self.vae_config["hiddens"]
        self.batch_size = self.vae_config["batch_size"]

        # Save model weights
        self.weights_dir = os.path.join(os.getcwd(), "spectralnet", "_trainers", "vae_weights")
        self.weights_path = os.path.join(self.weights_dir, "vae_weights.pth")
        os.makedirs(self.weights_dir, exist_ok=True)

    def train(self, X: torch.Tensor) -> VAEModel:
        self.X = X.view(X.size(0), -1)  # Flatten input
        self.criterion = nn.MSELoss()

        # Initialize VAE model
        self.vae_net = VAEModel(self.architecture, input_dim=self.X.shape[1]).to(self.device)

        self.optimizer = optim.Adam(self.vae_net.parameters(), lr=self.lr)

        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=self.lr_decay, patience=self.patience
        )

        # Load weights if they exist
        if os.path.exists(self.weights_path):
            self.vae_net.load_state_dict(torch.load(self.weights_path))
            return self.vae_net

        train_loader, valid_loader = self._get_data_loader()

        print("Training Variational Autoencoder:")
        t = trange(self.epochs, leave=True)
        for epoch in t:
            train_loss = 0.0
            for batch_x in train_loader:
                batch_x = batch_x.to(self.device).view(batch_x.size(0), -1)

                self.optimizer.zero_grad()
                x_hat, mu, logvar = self.vae_net(batch_x)

                # Compute VAE loss
                recon_loss = self.criterion(x_hat, batch_x)
                kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / batch_x.shape[0]
                loss = recon_loss + 0.01 * kl_loss  # Weighted KL term

                loss.backward()
                self.optimizer.step()
                train_loss += loss.item()

            train_loss /= len(train_loader)
            valid_loss = self.validate(valid_loader)
            self.scheduler.step(valid_loss)
            current_lr = self.optimizer.param_groups[0]["lr"]

            if current_lr <= self.min_lr:
                break

            t.set_description(
                "Train Loss: {:.7f}, Valid Loss: {:.7f}, LR: {:.6f}".format(
                    train_loss, valid_loss, current_lr
                )
            )
            t.refresh()

        torch.save(self.vae_net.state_dict(), self.weights_path)
        return self.vae_net

    def validate(self, valid_loader: DataLoader) -> float:
        self.vae_net.eval()
        valid_loss = 0.0
        with torch.no_grad():
            for batch_x in valid_loader:
                batch_x = batch_x.to(self.device).view(batch_x.size(0), -1)
                x_hat, mu, logvar = self.vae_net(batch_x)

                # Compute loss
                recon_loss = self.criterion(x_hat, batch_x)
                kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / batch_x.shape[0]
                loss = recon_loss + 0.01 * kl_loss

                valid_loss += loss.item()

        valid_loss /= len(valid_loader)
        return valid_loss

    def embed(self, X: torch.Tensor) -> torch.Tensor:
        print("Embedding data ...")
        self.vae_net.eval()
        with torch.no_grad():
            X = X.view(X.size(0), -1).to(self.device)
            encoded_data, _ = self.vae_net.encode(X)  # Only take the latent space representation
        return encoded_data

    def _get_data_loader(self) -> tuple:
        trainset_len = int(len(self.X) * 0.9)
        validset_len = len(self.X) - trainset_len
        trainset, validset = random_split(self.X, [trainset_len, validset_len])
        train_loader = DataLoader(trainset, batch_size=self.batch_size, shuffle=True)
        valid_loader = DataLoader(validset, batch_size=self.batch_size, shuffle=False)
        return train_loader, valid_loader