import torch
import torch.nn as nn
import torch.nn.functional as F

class SingleHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim: int):
        super(SingleHeadSelfAttention, self).__init__()
        self.embed_dim = embed_dim

        # Linear layers to create queries, keys, and values
        self.query_linear = nn.Linear(embed_dim, embed_dim)
        self.key_linear = nn.Linear(embed_dim, embed_dim)
        self.value_linear = nn.Linear(embed_dim, embed_dim)

        # Final linear layer after attention
        self.fc_out = nn.Linear(embed_dim, embed_dim)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None):
        batch_size, seq_length, embed_dim = x.shape

        # Compute queries, keys, values
        queries = self.query_linear(x)  # (batch_size, seq_length, embed_dim)
        keys = self.key_linear(x)       # (batch_size, seq_length, embed_dim)
        values = self.value_linear(x)   # (batch_size, seq_length, embed_dim)

        # Compute attention scores
        attention_scores = torch.matmul(queries, keys.transpose(-2, -1)) / (embed_dim ** 0.5)

        # Apply mask if provided (useful for ignoring padding tokens)
        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask.unsqueeze(1) == 0, float('-inf'))

        # Normalize scores into probabilities
        attention_weights = F.softmax(attention_scores, dim=-1)

        # Weighted sum of values
        attention_output = torch.matmul(attention_weights, values)

        # Final linear layer
        output = self.fc_out(attention_output)
        return output