import numpy as np

from hyperparameters import *


def generate_variable_length_data(
    batch_size=50, channels=5, seq_len_range=(10, 30), num_codes=1025):
    """
    Generate variable-length synthetic data for transformer
    training with padding and length information.

    Args:
        batch_size (int): Number of sequences in a batch.
        channels (int): Number of codes to predict at each timestep.
        seq_len_range (tuple): Range of sequence lengths (min_seq_len, max_seq_len).
        num_codes (int): Range of integer codes (default is 0-1024 inclusive).

    Returns:
        tuple:
            - padded_data (numpy.ndarray): Data of shape (batch_size,
            channels, max_seq_len) with zero-padding.
            - seq_lengths (numpy.ndarray): Array of original sequence
            lengths of shape (batch_size,).
    """

    min_seq_len , max_seq_len = seq_len_range
    assert 0 < min_seq_len < max_seq_len, "Invalid sequence length range."

    # Randomly decide sequence lengths for each batch item
    seq_lengths = np.random.randint(min_seq_len , max_seq_len + 1, size=batch_size)
    max_seq_len_in_batch = max(seq_lengths)

    # Initialize padded data array
    padded_data = np.zeros((batch_size, channels, max_seq_len_in_batch), dtype=np.int32)

    for i in range(batch_size):
        seq_len = seq_lengths[i]
        # Generate initial random codes
        initial_codes = np.random.randint(0, num_codes, size=(channels,))
        padded_data[i, :, 0] = initial_codes
        # Define relationships between time steps
        for t in range(1, seq_len):
            previous_timestep = padded_data[i, :, t - 1]

            # Example of complex temporal relationship
            padded_data[i, :, t] = ((previous_timestep * 2 + np.sin(t / seq_len * np.pi) * 50).astype(int)
                                    + np.random.randint(-10, 10, size=channels)) % num_codes

    return padded_data, seq_lengths


def pad_padded_data(padded_data):
    """
    Ensures that each sequence in all batches has a uniform length.

    While generate_variable_length_data enforces this at a per-batch
    level, when dealing with multiple batches, this isn't guaranteed.

    Args:
        padded_data (numpy.ndarray): Data of shape (batch_size,
            channels, max_seq_len) with zero-padding.

    Returns:
        padded_padded_data (numpy.ndarray): Data of shape (batch_size,
            channels, seq_len) with zero-padding.
    """
    padded_padded_data = np.zeros([*padded_data.shape[:2], SEQ_LEN])

    for i in range(padded_data.shape[0]):
        for j in range(padded_data.shape[1]):
            sequence = padded_data[i, j]

            # If the sequence is shorter than SEQ_LEN, pad it
            if len(sequence) < SEQ_LEN:
                padding = np.zeros(SEQ_LEN - len(sequence), dtype=sequence.dtype)
                sequence = np.concatenate([sequence, padding])

            padded_padded_data[i, j] = sequence

    return padded_padded_data


def generate_batch():
    """
    Wrapper function to return a padded batch of data

    Args:
        -

    Returns:
        padded_padded_data (numpy.ndarray): Data of shape (batch_size,
            channels, seq_len) with zero-padding.
    """
    padded_data, _ = generate_variable_length_data(BATCH_SIZE, channels=N_CHANNELS)
    return pad_padded_data(padded_data)


def generate_dataset():
    """
    Create a set of batches of padded data.

    Args:
        -

    Returns:
        result (numpy.ndarray): Data of shape (n_batches,
            batch_size, channels, seq_len) with zero-padding.
    """
    return np.array([generate_batch() for _ in range(N_BATCHES)])
