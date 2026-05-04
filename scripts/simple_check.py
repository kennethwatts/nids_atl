import torch
from transformers import pipeline

# Check if Blackwell is recognized
print(f"Using GPU: {torch.cuda.get_device_name(0)}")
print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

# Run a quick inference task
pipe = pipeline("text-generation", model="gpt2", device=0)
out = pipe("The future of AI on Blackwell is", max_length=30)
print(out[0]['generated_text'])