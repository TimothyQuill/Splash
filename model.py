import torch
import torch.nn as nn
import os

from hyperparameters import *


def generate_local_mask(seq_len):
    """Creates a local attention mask for a sequence of length `seq_len`."""
    mask = torch.full((seq_len, seq_len), float("-inf"))  # Initialize mask with -inf
    for i in range(seq_len):
        start = max(0, i - LOCAL_WINDOW_SIZE)
        end = min(seq_len, i + LOCAL_WINDOW_SIZE + 1)
        mask[i, start:end] = 0  # Allow attention within the local window
    return mask


class LocalAttention(nn.Module):
    def __init__(self):
        super(LocalAttention, self).__init__()
        self.transformer = nn.Transformer(
            d_model=D_MODEL,
            nhead=N_HEAD,
            num_encoder_layers=N_LAYERS)

    def forward(self, src, tgt):
        src_len = src.size(0)  # Source sequence length
        tgt_len = tgt.size(0)  # Target sequence length

        # Generate masks for the source and target sequences
        src_mask = generate_local_mask(src_len).to(src.device)
        tgt_mask = generate_local_mask(tgt_len).to(tgt.device)

        return self.transformer(src, tgt, src_mask=src_mask, tgt_mask=tgt_mask)


class SequenceTransformer(nn.Module):
    def __init__(self):
        super(SequenceTransformer, self).__init__()

        # Input embedding
        self.input_embedding = nn.Linear(N_CHANNELS, D_MODEL)

        # Positional encoding
        self.positional_encoding = nn.Embedding(GIVEN_SEQ + TARGET_SEQ, D_MODEL)

        # Localized attention transformer
        self.local_attention = LocalAttention()

        # Output projection
        self.fc_out = nn.Linear(D_MODEL, N_CHANNELS)

    def forward(self, src, tgt):
        """
        Args:
            src: [batch_size, given_seq_len, num_channels]
            tgt: [batch_size, target_seq_len, num_channels]
        Returns:
            output: [batch_size, target_seq_len, num_channels]
        """

        # Embedding and positional encoding
        src_emb = self.input_embedding(src) + self.positional_encoding(torch.arange(GIVEN_SEQ).to(src.device))
        tgt_emb = self.input_embedding(tgt) + self.positional_encoding(torch.arange(TARGET_SEQ).to(tgt.device))

        # Transpose for transformer (seq_len, batch_size, d_model)
        src_emb = src_emb.permute(1, 0, 2)
        tgt_emb = tgt_emb.permute(1, 0, 2)

        # Localized attention
        output = self.local_attention(src_emb, tgt_emb)

        # Output projection
        output = self.fc_out(output.permute(1, 0, 2))  # Shape: [batch_size, target_seq_len, num_channels]
        return output
