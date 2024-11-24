from model import SequenceTransformer
from helper import *

import numpy as np


def generate_sequence(x):
    x = np.array([x])   # n_batches = 1
    x = normalise(x)
    model = SequenceTransformer()
    model = load_model(model)
    # Pass in the tensor and run
    # Get the output sequence and turn it back into its normal form
    # return sequence
    ...
