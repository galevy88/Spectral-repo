import torch
import torch.nn as nn


class AEModel(nn.Module):
    def __init__(self, architecture: list, input_dim: int):
        super(AEModel, self).__init__()
        self.hidden_dims = architecture
        self.encoder_outputs = []  # Store encoder outputs internally

        # Encoder layers
        self.encoder_layers = nn.ModuleList()
        current_dim = input_dim

        for dim in self.hidden_dims:
            self.encoder_layers.append(
                nn.Sequential(
                    nn.Linear(current_dim, dim),
                    nn.ReLU(),
                    nn.BatchNorm1d(dim)
                )
            )
            current_dim = dim

        # Bottom layer (bottleneck)
        self.bottleneck = nn.Sequential(
            nn.Linear(self.hidden_dims[-1], self.hidden_dims[-1]),
            nn.ReLU(),
            nn.BatchNorm1d(self.hidden_dims[-1])
        )

        # Decoder layers
        self.decoder_layers = nn.ModuleList()
        reversed_dims = self.hidden_dims[::-1]  # Reverse the dimensions

        # Decoder path with skip connection handling
        for i in range(len(reversed_dims) - 1):
            input_dim_decoder = reversed_dims[i] * 2  # Account for skip connection
            output_dim_decoder = reversed_dims[i + 1]
            self.decoder_layers.append(
                nn.Sequential(
                    nn.Linear(input_dim_decoder, output_dim_decoder),
                    nn.ReLU(),
                    nn.BatchNorm1d(output_dim_decoder)
                )
            )

        # Final layer with sigmoid to bound output
        final_input_dim = reversed_dims[-1] * 2  # Last decoder output + first encoder output
        self.final_layer = nn.Sequential(
            nn.Linear(final_input_dim, input_dim),
            nn.Sigmoid()  # Bound output to [0, 1]
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        self.encoder_outputs = []  # Reset encoder outputs for each forward pass
        for layer in self.encoder_layers:
            x = layer(x)
            self.encoder_outputs.append(x)
        return x

    def decode(self, x: torch.Tensor, encoder_outputs: list = None) -> torch.Tensor:
        x = self.bottleneck(x)

        # Decoder with skip connections
        for i, layer in enumerate(self.decoder_layers):
            skip_index = len(self.encoder_outputs) - 1 - i  # Start from last encoder output
            skip_connection = self.encoder_outputs[skip_index]
            x = torch.cat([x, skip_connection], dim=1)
            x = layer(x)

        # Final layer with skip connection from first encoder output
        x = torch.cat([x, self.encoder_outputs[0]], dim=1)
        x = self.final_layer(x)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        encoded = self.encode(x)
        decoded = self.decode(encoded)
        return decoded