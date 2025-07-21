import numpy as np

def get_positional_encoding(max_len, d_model):
    pos = np.arange(max_len)[:, np.newaxis]
    i = np.arange(d_model)[np.newaxis, :]
    angle_rates = 1 / np.power(10000, (2 * (i // 2)) / d_model)
    angle_rads = pos * angle_rates  # shape: (max_len, d_model)
    pe = np.zeros((max_len, d_model))
    pe[:, 0::2] = np.sin(angle_rads[:, 0::2])
    pe[:, 1::2] = np.cos(angle_rads[:, 1::2])

    return pe

# Sample usage
max_len = 10       # Number of positions (sequence length)
d_model = 16       # Embedding dimension
pe_matrix = get_positional_encoding(max_len, d_model)

# Print the positional encoding matrix
print("Positional Encoding Matrix (shape: {}):\n".format(pe_matrix.shape), pe_matrix)