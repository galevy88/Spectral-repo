import torch
import numpy as np
import torch.nn as nn

class SpectralNetModel(nn.Module):
    def __init__(self, architecture: dict, input_dim: int):
        super(SpectralNetModel, self).__init__()
        self.architecture = architecture
        self.input_dim = input_dim

        # Define the CNN architecture
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten()
        )

        # Define additional layers based on the architecture
        self.layers = nn.ModuleList()
        current_dim = 128 * 2 * 1  # Adjust based on the output dimension of the CNN
        for i, layer in enumerate(self.architecture):
            next_dim = layer
            if i == len(self.architecture) - 1:
                self.layers.append(
                    nn.Sequential(nn.Linear(current_dim, next_dim), nn.Tanh())
                )
            else:
                self.layers.append(
                    nn.Sequential(nn.Linear(current_dim, next_dim), nn.LeakyReLU())
                )
                current_dim = next_dim

    def _make_orthonorm_weights(self, Y: torch.Tensor) -> torch.Tensor:
        """
        Orthonormalize the output of the network using the Cholesky decomposition.

        Parameters
        ----------
        Y : torch.Tensor
            The output of the network.

        Returns
        -------
        torch.Tensor
            The orthonormalized output.

        Notes
        -----
        This function applies QR decomposition to orthonormalize the output (`Y`) of the network.
        The inverse of the R matrix is returned as the orthonormalization weights.
        """
        m = Y.shape[0]
        _, R = torch.linalg.qr(Y)
        orthonorm_weights = np.sqrt(m) * torch.inverse(R)
        return orthonorm_weights

    def forward(
        self, x: torch.Tensor, should_update_orth_weights: bool = True
    ) -> torch.Tensor:
        """
        Perform the forward pass of the model.

        Parameters
        ----------
        x : torch.Tensor
            The input tensor.
        should_update_orth_weights : bool, optional
            Whether to update the orthonormalization weights using the Cholesky decomposition or not.

        Returns
        -------
        torch.Tensor
            The output tensor.

        Notes
        -----
        This function takes an input tensor `x` and computes the forward pass of the model.
        If `should_update_orth_weights` is set to True, the orthonormalization weights are updated
        using the QR decomposition. The output tensor is returned.
        """
        # Reshape the input to fit the CNN
        x = x.view(x.size(0), 1, 16, 8)  # Example reshape to 1-channel 16x8 images

        # Extract features using the CNN
        x = self.cnn(x)

        for layer in self.layers:
            x = layer(x)

        Y_tilde = x
        if should_update_orth_weights:
            self.orthonorm_weights = self._make_orthonorm_weights(Y_tilde)

        Y = Y_tilde @ self.orthonorm_weights
        return Y