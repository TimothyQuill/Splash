"""
Transformer model that forecasts numbers in a sequence,
utilising a local mask for local patterns in the data.
"""
import torch
import torch.nn as nn

from hyperparameters import *


def generate_local_mask(seq_len):
    """Creates a local attention mask for a sequence of length `seq_len`."""
    mask = torch.full((seq_len, seq_len), float("-inf"))  # Initialise mask with -inf
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
            num_encoder_layers=N_LAYERS
        )

    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        return self.transformer(src, tgt, src_mask=src_mask, tgt_mask=tgt_mask)


class SequenceTransformer(nn.Module):
    def __init__(self):
        super(SequenceTransformer, self).__init__()

        self.input_embedding = nn.Linear(N_CHANNELS, D_MODEL)
        self.positional_encoding = nn.Embedding(GIVEN_SEQ + TARGET_SEQ, D_MODEL)
        self.local_attention = LocalAttention()
        self.fc_out = nn.Linear(D_MODEL, N_CHANNELS)

    def forward(self, src, tgt=None, max_len=None):
        """
        Args:
            src: [batch_size, given_seq_len, num_channels]
            tgt: [batch_size, target_seq_len, num_channels] (optional for training)
            max_len: Maximum length for inference (used when tgt is None)
        Returns:
            output: [batch_size, target_seq_len, num_channels]
        """

        batch_size, given_seq_len, _ = src.shape

        src_emb = self.input_embedding(src) + self.positional_encoding(
            torch.arange(given_seq_len, device=src.device)
        ).unsqueeze(0)
        src_emb = src_emb.permute(1, 0, 2)  # Shape: [seq_len, batch_size, d_model]

        if tgt is not None:  # Training mode
            target_seq_len = tgt.shape[1]

            tgt_emb = self.input_embedding(tgt) + self.positional_encoding(
                torch.arange(target_seq_len, device=tgt.device)
            ).unsqueeze(0)
            tgt_emb = tgt_emb.permute(1, 0, 2)  # Shape: [seq_len, batch_size, d_model]

            # Generate masks
            src_mask = generate_local_mask(given_seq_len).to(src.device)
            tgt_mask = generate_local_mask(target_seq_len).to(tgt.device)

            output = self.local_attention(src_emb, tgt_emb, src_mask=src_mask, tgt_mask=tgt_mask)
            output = self.fc_out(output.permute(1, 0, 2))  # Shape: [batch_size, target_seq_len, num_channels]

            return output

        else:  # Inference mode
            assert max_len is not None, "max_len must be specified for inference"
            generated_seq = torch.zeros(
                (batch_size, max_len, N_CHANNELS), device=src.device
            )  # Initialise output sequence

            for i in range(max_len):
                # Generate target embeddings for current step
                tgt_emb = self.input_embedding(generated_seq[:, :i + 1]) + self.positional_encoding(
                    torch.arange(i + 1, device=src.device)
                ).unsqueeze(0)
                tgt_emb = tgt_emb.permute(1, 0, 2)  # Shape: [seq_len, batch_sise, d_model]

                # Generate the target mask for current step
                tgt_mask = generate_local_mask(i + 1).to(src.device)

                # Perform localised attention
                output = self.local_attention(
                    src_emb,
                    tgt_emb,
                    src_mask=generate_local_mask(given_seq_len).to(src.device),
                    tgt_mask=tgt_mask,
                )

                # Decode the last token
                next_token = self.fc_out(output[-1])  # Shape: [batch_size, num_channels]
                generated_seq[:, i] = next_token  # Assign the predicted token

            return generated_seq
