# --- START OF FILE utils.py ---

import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    """Injects positional information into the input sequence embeddings.

    This module generates fixed sinusoidal positional encodings based on the
    formula from "Attention Is All You Need". It adds these encodings to the
    input embeddings to give the model information about the position of tokens
    in the sequence.
    """

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        """
        Initializes the PositionalEncoding module.

        Args:
            d_model (int): The embedding dimension of the model (must be even).
            dropout (float): The dropout probability.
            max_len (int): The maximum possible sequence length for which positional
                         encodings will be pre-calculated.
        """
        super().__init__()
        if d_model % 2 != 0:
            raise ValueError("d_model must be an even number for PositionalEncoding.")

        self.dropout = nn.Dropout(p=dropout)

        # Create position encoding matrix
        position = torch.arange(max_len).unsqueeze(1) # Shape: [max_len, 1]
        # Calculate the division term for the sine and cosine functions
        # Shape: [d_model / 2]
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))

        # Initialize positional encoding tensor
        pe = torch.zeros(max_len, 1, d_model) # Shape: [max_len, 1, d_model]

        # Apply sin to even indices (in the embedding dimension)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        # Apply cos to odd indices (in the embedding dimension)
        pe[:, 0, 1::2] = torch.cos(position * div_term)

        # Register pe as a buffer. Buffers are part of the model's state,
        # but are not considered parameters to be optimized during training.
        # They are saved and loaded along with the model state_dict.
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Adds positional encoding to the input tensor.

        Args:
            x (torch.Tensor): The input tensor containing sequence embeddings.
                              Expected shape: [seq_len, batch_size, d_model].

        Returns:
            torch.Tensor: The input tensor with added positional encoding.
                          Output shape: [seq_len, batch_size, d_model].
        """
        # Add the pre-calculated positional encoding to the input embeddings.
        # We only use encodings up to the input sequence length (`x.size(0)`).
        # self.pe has shape [max_len, 1, d_model]. Slicing `self.pe[:x.size(0)]` gives
        # shape [seq_len, 1, d_model], which broadcasts correctly during addition.
        x = x + self.pe[:x.size(0)]
        # Apply dropout
        return self.dropout(x)

# --- END OF FILE utils.py ---