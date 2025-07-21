import torch
import torch.nn as nn

class LoRALinear(nn.Module):
    def __init__(self, orig_linear: nn.Linear, r: int = 8, alpha: float = 1.0):
        """
        Wraps a given linear layer with LoRA adaptation.
        orig_linear: an existing nn.Linear layer from a pre-trained model (weights frozen).
        r: rank of the LoRA adapters.
        alpha: scaling factor for LoRA (often set such that alpha/r is 1).
        """
        super().__init__()
        self.in_features = orig_linear.in_features
        self.out_features = orig_linear.out_features
        # Freeze original weight and bias
        self.weight = nn.Parameter(orig_linear.weight.data, requires_grad=False)
        if orig_linear.bias is not None:
            self.bias = nn.Parameter(orig_linear.bias.data, requires_grad=False)
        else:
            self.bias = None
        # LoRA low-rank matrices
        self.r = r
        self.alpha = alpha
        # "Down" projection: reduces dimension from in_features to r
        self.lora_down = nn.Parameter(torch.zeros((r, self.in_features)))
        # "Up" projection: increases dimension from r to out_features
        self.lora_up   = nn.Parameter(torch.zeros((self.out_features, r)))
        # Initialize LoRA weights: usually lora_down random, lora_up zero
        nn.init.kaiming_uniform_(self.lora_down, a=math.sqrt(5))  # He init for down-proj
        nn.init.zeros_(self.lora_up)  # start with no effect
        # Note: starting with lora_up = 0 means initially the LoRA doesn't change the output
        # (since lora_down * 0 = 0), so the model starts exactly like the pre-trained one.
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Compute base linear output (no grad, since weight is frozen)
        # x shape: [batch, in_features]
        # weight shape: [out_features, in_features]
        result = torch.matmul(x, self.weight.T)
        # Compute LoRA adaptation: (x * lora_down^T) * lora_up^T scaled by alpha/r
        # lora_down^T shape: [in_features, r], lora_up^T: [r, out_features]
        lora_out = x @ self.lora_down.T    # shape: [batch, r]
        lora_out = lora_out @ self.lora_up.T  # shape: [batch, out_features]
        # Scale the LoRA output
        result += lora_out * (self.alpha / self.r)
        # Add bias if present
        if self.bias is not None:
            result += self.bias
        return result

# Example usage:
# Suppose we have a GPT-2 model and want to apply LoRA to its first fully-connected layer
from transformers import GPT2Model
model = GPT2Model.from_pretrained('gpt2')
# Pick a linear layer from the model, e.g., the feed-forward layer in the first Transformer block
orig_linear = model.h[0].mlp.c_fc  # (Assume c_fc is a nn.Linear in GPT2 block 0)
# Replace it with a LoRA-wrapped layer
model.h[0].mlp.c_fc = LoRALinear(orig_linear, r=8, alpha=8)
# Now model.h[0].mlp.c_fc will only train the LoRA params. Freeze others as needed.