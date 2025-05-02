import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from sae_kit.sae_kit.sparse_autoencoder import SparseAutoencoder
import random

# Set device and seed
device = "cuda"
torch.manual_seed(42)
random.seed(42)

# Load model and tokenizer
model = AutoModelForCausalLM.from_pretrained(
    "deepseek-ai/DeepSeek-R1-Distill-Llama-8B", 
    torch_dtype=torch.bfloat16,
    device_map=device
)
tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Llama-8B")

# Load dataset
dataset = load_dataset(
    "ServiceNow-AI/R1-Distill-SFT",
    "v1",
    split="train",
    streaming=False,
)
dataset = dataset.shuffle(seed=42)

# Load SAE
n_features = 65536
sae = SparseAutoencoder(
    d_in=4096,
    n_features=n_features,
    k=100,
    device=device,
    use_batch_topk=True
)
sae.load_state_dict(torch.load("r1-sae-layer5_65536_100-27000.pt", map_location=device))
sae.eval()

# Parameters
ctx_len = 512
skip_first_n_tokens = 1
layer_num = 5
# Collect ~25000 activations (much closer to the generator's approach)
num_examples = 1
generator_batch_size = 1

# Collect activations from multiple batches to better match the generator
all_activations = []

offset = 0

example_idx = 4
messages = dataset[example_idx]["messages"]
tokens = tokenizer.apply_chat_template(
    messages, 
    add_generation_prompt=False, 
    return_tensors="pt"
)

print(tokens.shape)

input_ids = tokens[:ctx_len].to(device)

with torch.no_grad():
    outputs = model(input_ids, output_hidden_states=True)
    activations = outputs.hidden_states[layer_num]

last_token_idx = 100

activations = activations[:, skip_first_n_tokens:last_token_idx, :]
flat_activations = activations[0]

print(flat_activations.shape)

# Compute FVU
with torch.no_grad():
    features = sae.encode(flat_activations)
    reconstructions = sae.decode(features)
    
    e = reconstructions - flat_activations
    total_variance = (flat_activations - flat_activations.mean(0)).pow(2).sum()
    squared_error = e.pow(2).sum()
    fvu = squared_error / total_variance

print(f"FVU: {fvu.item():.4f}")
print(f"L0 (avg features active): {features.gt(0).float().mean().item() * n_features:.1f}")
print(f"MSE: {torch.nn.functional.mse_loss(reconstructions, flat_activations).item():.6f}")