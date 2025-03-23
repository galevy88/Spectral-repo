import torch
import numpy as np
import torch.nn as nn


class SpectralNetModel(nn.Module):
    def __init__(self, architecture: dict, input_dim: int):
        super(SpectralNetModel, self).__init__()
        self.architecture = architecture
        self.layers = nn.ModuleList()
        self.input_dim = input_dim

        current_dim = self.input_dim
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
        Orthonormalize the output of the network using the Gram-Schmidt process.

        Parameters
        ----------
        Y : torch.Tensor
            The output of the network.

        Returns
        -------
        torch.Tensor
            The orthonormalized output.
        """
        m = Y.shape[0]
        n = Y.shape[1]

        # Initialize tensors
        V = Y.clone()  # Input vectors
        U = torch.zeros_like(Y)  # Orthonormal basis

        # Gram-Schmidt process
        for i in range(n):
            # Start with the current vector
            u = V[:, i].clone()  # Create a new tensor instead of assigning inplace

            # Subtract projections onto previous vectors
            for j in range(i):
                proj = torch.dot(U[:, j], V[:, i]) / torch.dot(U[:, j], U[:, j])
                u = u - proj * U[:, j]  # Update temporary vector, not U directly

            # Normalize the vector
            u = u / torch.norm(u)  # Normalize the temporary vector

            # Assign to U without inplace modification of existing U
            U = U.clone()  # Ensure we’re not modifying the original U inplace
            U[:, i] = u  # This is still technically inplace, but u is a new tensor

        # Compute weights using the orthonormal basis
        orthonorm_weights = torch.sqrt(torch.tensor(m, dtype=Y.dtype)) * torch.inverse(U.T @ U) @ U.T

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

        for layer in self.layers:
            x = layer(x)

        Y_tilde = x
        if should_update_orth_weights:
            self.orthonorm_weights = self._make_orthonorm_weights(Y_tilde)

        Y = Y_tilde @ self.orthonorm_weights
        return Y
