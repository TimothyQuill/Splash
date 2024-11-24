from model import SequenceTransformer
from helper import *

import numpy as np


def generate_sequence(x):

    # Prepare the data for the model
    x = np.array([x])   # n_batches = 1
    x = torch.Tensor(x)

    # Scale the data to unit range
    x_scaled = minmax_scaling(x)
    x_scaled = x_scaled.view(-1, GIVEN_SEQ, N_CHANNELS)  # Flatten batch

    # Define and load the model
    model = SequenceTransformer()
    model = load_model(model)
    model.eval()

    # Perform inference
    output = model(x_scaled, max_len=26)

    # Reshape and rescale the data
    output = output.permute(0, 2, 1)  # Swap the last two dimensions
    output = reverse_minmax_scaling(output)

    # Add the output to the original input
    output = torch.cat((x[0], output), dim=-1)

    # Convert all values in output to integers
    result = floats_to_ints(output)

    # Convert to json
    return tensor_to_json(result)
