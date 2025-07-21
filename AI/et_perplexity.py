import math

def calculate_perplexity(log_probs):
    """
    Given a list of log probabilities (base e) for each token in a sequence,
    compute the perplexity of the model for this sequence.
    """
    N = len(log_probs)  # number of tokens
    # Compute the average negative log-likelihood:
    avg_neg_log_likelihood = ...  # hint: use sum(log_probs)
    # Compute perplexity:
    perplexity = ...             # hint: math.exp(...)
    return perplexity

# Example:
# If log_probs = [-1.2, -0.5, -0.3], then
# sum(log_probs) = -2.0, N = 3,
# avg_neg_log_likelihood = -(-2.0)/3 = 0.666...,
# perplexity = exp(0.666...) ≈ 1.95.

avg_neg_log_likelihood = -sum(log_probs) / N
perplexity = math.exp(avg_neg_log_likelihood)