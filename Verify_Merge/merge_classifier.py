import torch
import torch.nn as nn

def merge_linear_bn(linear, bn):
    """
    Merge a linear layer (nn.Linear) with a BatchNorm1d layer (nn.BatchNorm1d)
    into a single linear layer that is equivalent to the composition of the original layers.

    Args:
        linear (nn.Linear): The linear layer.
        bn (nn.BatchNorm1d): The batch normalization layer applied after the linear layer.
    
    Returns:
        nn.Linear: A new linear layer with merged parameters.
    """
    # Copy original linear parameters.
    W = linear.weight.data.clone()  # shape: (out_features, in_features)
    if linear.bias is not None:
        b = linear.bias.data.clone()  # shape: (out_features,)
    else:
        b = torch.zeros(W.size(0), device=W.device)

    # Extract BN parameters.
    gamma = bn.weight.data         # shape: (out_features,)
    beta = bn.bias.data            # shape: (out_features,)
    running_mean = bn.running_mean # shape: (out_features,)
    running_var = bn.running_var   # shape: (out_features,)
    eps = bn.eps

    # Reshape gamma and running_var for broadcasting.
    gamma = gamma.view(-1, 1)  # shape: (out_features, 1)
    running_std = torch.sqrt(running_var + eps).view(-1, 1)  # shape: (out_features, 1)

    # New weight scaling.
    W_new = W * (gamma / running_std)

    # New bias calculation.
    b_new = (b - running_mean) / running_std.view(-1) * gamma.view(-1) + beta

    # Create new linear layer with merged parameters.
    merged_linear = nn.Linear(linear.in_features, linear.out_features)
    merged_linear.weight.data.copy_(W_new)
    merged_linear.bias.data.copy_(b_new)

    return merged_linear

# Example usage:
# Define a simple sequential model with a linear layer followed by BatchNorm1d.
linear_layer = nn.Linear(128, 64)
bn_layer = nn.BatchNorm1d(64)
model = nn.Sequential(linear_layer, bn_layer)

# Simulate training mode to update running stats.
model.train()
dummy_input = torch.randn(32, 128)
_ = model(dummy_input)

# Merge the layers after training.
merged_linear = merge_linear_bn(linear_layer, bn_layer)

# For inference, we can replace the sequential block with merged_linear.
model_merged = merged_linear
model_merged.eval()
model.eval()
# Test both models to verify equivalence.
with torch.no_grad():
    original_output = model(dummy_input)
    merged_output = model_merged(dummy_input)
    print(original_output)
    print(merged_output)
    print("Difference between outputs:", torch.abs(original_output - merged_output).max().item())