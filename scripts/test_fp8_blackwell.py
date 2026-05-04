import torch
import transformer_engine.pytorch as te
from transformer_engine.common import recipe

# Initialize a standard Transformer layer using TE
# This automatically uses Blackwell's SM 12.0 optimizations
model = te.TransformerLayer(hidden_size=1024, num_attention_heads=16, layernorm_epsilon=1e-5).cuda()

# Create dummy input (Batch Size 16, Seq Length 128, Hidden Size 1024)
inp = torch.randn(16, 128, 1024, device="cuda", dtype=torch.bfloat16)

# Define the FP8 Recipe (E4M3 format is ideal for forward pass)
fp8_recipe = recipe.DelayedScaling(margin=0, fp8_format=recipe.Format.E4M3)

print("Starting FP8 Forward Pass...")
with te.autocast(enabled=True, recipe=fp8_recipe):
    out = model(inp)

print(f"Success! Output shape: {out.shape}")
print(f"VRAM Allocated: {torch.cuda.memory_allocated() / 1e6:.2f} MB")