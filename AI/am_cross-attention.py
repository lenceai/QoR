import numpy as np

def cross_attention(query, key, value):
    """
    query: shape (num_queries, d_k)
    key:   shape (num_keys, d_k)
    value: shape (num_keys, d_v)
    
    returns: shape (num_queries, d_v)
    """
    d_k = query.shape[-1]

    # Step 1: Compute raw attention scores (dot product between queries and keys)
    scores = np.dot(query, key.T) / np.sqrt(d_k)  # shape: (num_queries, num_keys)

    # Step 2: Apply softmax to get attention weights
    exp_scores = np.exp(scores - np.max(scores, axis=-1, keepdims=True))  # stability
    attention_weights = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)  # shape: (num_queries, num_keys)

    # Step 3: Compute weighted sum of value vectors
    output = np.dot(attention_weights, value)  # shape: (num_queries, d_v)

    return output
    
# Toy example with 3 queries and 4 key-value pairs
np.random.seed(42)
Q = np.random.rand(3, 8)   # 3 query vectors (decoder tokens)
K = np.random.rand(4, 8)   # 4 key vectors (encoder tokens)
V = np.random.rand(4, 16)  # 4 value vectors with d_v = 16

output = cross_attention(Q, K, V)
print("Output shape:", output.shape)
print("Output:", output)