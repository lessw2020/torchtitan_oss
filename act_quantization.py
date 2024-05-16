
import torch
from deepspeed.ops.op_builder import QuantizerBuilder


inference_module = None

def quantize_ds(activations, num_groups, q_bits=8, is_symmetric_quant=True):
    # returns a tuple of (quantized tensor, params)
    global inference_module
    if inference_module is None:
        inference_module = QuantizerBuilder().load()

    return inference_module.quantize(activations, num_groups, q_bits,
                                     inference_module.Symmetric if is_symmetric_quant else inference_module.Asymmetric)


def dequantize_ds(activations, params, num_groups, q_bits=8, is_symmetric_quant=True):
    # returns output tensor
    global inference_module
    if inference_module is None:
        inference_module = QuantizerBuilder().load()
    return inference_module.dequantize(
        activations,
        params,
        num_groups,
        q_bits,
        inference_module.Symmetric if is_symmetric_quant else inference_module.Asymmetric,
    )

activations = torch.randn(100, 512, device="cuda", dtype=torch.bfloat16)
num_groups = activations.shape[0] # (activations.numel() // activations.size(-1))
print(f"{activations.dtype, activations.shape}")
print(f"{num_groups=}")
out_tensor, out_params = quantize_ds(activations, num_groups) # , q_bits, is_symmetric_quant)
dequantized_tensor = dequantize_ds(out_tensor, out_params, num_groups,) #  q_bits, is_symmetric_quant)
print(f"{dequantized_tensor.dtype, dequantized_tensor.shape}")
dequantized_tensor.to(torch.bfloat16)

ds_quantization_error = torch.sum(torch.abs((activations - dequantized_tensor).to(torch.float64)))
print(f"ds quantization error: {ds_quantization_error}")
diff = torch.abs(dequantized_tensor - activations).mean().item()
print(f"diff: {diff}")
