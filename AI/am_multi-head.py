import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int):
        super(MultiHeadSelfAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        # Validate embedding dimension divisibility
        if self.head_dim * num_heads != embed_dim:
            raise ValueError("embed_dim must be divisible by num_heads.")

        # Linear projections for queries, keys, and values
        self.query_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.key_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.value_proj = nn.Linear(embed_dim, embed_dim, bias=False)

        # Final linear layer to combine heads
        self.fc_out = nn.Linear(embed_dim, embed_dim)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None):
        batch_size, seq_length, embed_dim = x.shape

        # Project inputs to queries, keys, and values, and split embeddings into multiple heads
        queries = self.query_proj(x).reshape(batch_size, seq_length, self.num_heads, self.head_dim)
        keys = self.key_proj(x).reshape(batch_size, seq_length, self.num_heads, self.head_dim)
        values = self.value_proj(x).reshape(batch_size, seq_length, self.num_heads, self.head_dim)

        # Transpose for multi-head attention computation (batch_size, num_heads, seq_length, head_dim)
        queries = queries.transpose(1, 2)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)

        # Compute scaled dot-product attention scores per head
        attention_scores = torch.matmul(queries, keys.transpose(-2, -1)) / (self.head_dim ** 0.5)

        # Apply mask if provided
        if mask is not None:
            expanded_mask = mask.unsqueeze(1).unsqueeze(2)  # shape: (batch_size, 1, 1, seq_length)
            attention_scores = attention_scores.masked_fill(expanded_mask == 0, float('-inf'))

        # Normalize scores into attention probabilities
        attention_weights = F.softmax(attention_scores, dim=-1)

        # Compute weighted sum of values
        attention_output = torch.matmul(attention_weights, values)

        # Concatenate heads' outputs back to original embedding dimension
        attention_output = attention_output.transpose(1, 2).contiguous()
        concatenated_output = attention_output.reshape(batch_size, seq_length, embed_dim)

        # Final linear layer to refine concatenated outputs
        output = self.fc_out(concatenated_output)

        return output