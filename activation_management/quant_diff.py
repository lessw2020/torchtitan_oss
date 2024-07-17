import torch
import numpy as np

# Function to calculate mean squared error
def mse(tensor1, tensor2):
    return torch.mean((tensor1 - tensor2) ** 2).item()

# Function to calculate mean absolute error
def mae(tensor1, tensor2):
    return torch.mean(torch.abs(tensor1 - tensor2)).item()

# Load the tensor from disk (assuming you have a saved tensor)
# If you don't have one, you can create a random tensor and save it first
#original_tensor = torch.randn(1000, 1000)
#torch.save(original_tensor, 'original_tensor.pt')

# Now, let's load it as if it was from disk
loaded_tensor = torch.load('activation_56.pt')

# Make a copy and scale to fp16
scaled_fp16_tensor = loaded_tensor.clone().to(torch.bfloat16)
print(f"{loaded_tensor.shape=}")
# Calculate the scale factor
#scale = torch.max(torch.abs(loaded_tensor)) / torch.max(torch.abs(fp16_tensor))

# Scale the fp16 tensor
#scaled_fp16_tensor = fp16_tensor * scale

# Unscale and convert back to fp32
unscaled_tensor = scaled_fp16_tensor.to(torch.float32) # (scaled_fp16_tensor / scale).to(torch.float32)

# Compare the original and the unscaled tensor
mse_error = mse(loaded_tensor, unscaled_tensor)
mae_error = mae(loaded_tensor, unscaled_tensor)
max_error = torch.max(torch.abs(loaded_tensor - unscaled_tensor)).item()

print(f"Mean Squared Error: {mse_error}")
print(f"Mean Absolute Error: {mae_error}")
print(f"Max Absolute Error: {max_error}")

# Calculate relative errors
relative_mse = mse_error / torch.mean(loaded_tensor ** 2).item()
relative_mae = mae_error / torch.mean(torch.abs(loaded_tensor)).item()

print(f"Relative Mean Squared Error: {relative_mse}")
print(f"Relative Mean Absolute Error: {relative_mae}")

# Check if any values became infinity or NaN
inf_count = torch.isinf(unscaled_tensor).sum().item()
nan_count = torch.isnan(unscaled_tensor).sum().item()

print(f"Number of Infinity values: {inf_count}")
print(f"Number of NaN values: {nan_count}")
