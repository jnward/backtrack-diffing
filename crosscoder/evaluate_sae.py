import torch
from transformers import AutoTokenizer
from nnsight import LanguageModel
from datasets import load_dataset
from sae_kit.sae_kit.sparse_autoencoder import SparseAutoencoder
from data_utils import sae_cached_activation_generator

# Set device and seed
device = "cuda"
torch.manual_seed(42)

# Load model and tokenizer
r1_model = LanguageModel("deepseek-ai/DeepSeek-R1-Distill-Llama-8B", device_map=device, torch_dtype=torch.bfloat16)
r1_tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Llama-8B")

with r1_model.trace("test") as tr:
    out = r1_model.output.save()

# Load dataset
dataset = load_dataset(
    "ServiceNow-AI/R1-Distill-SFT",
    "v1",
    split="train",
    streaming=False,
)
dataset = dataset.shuffle(seed=42)

# Configure SAE parameters - same as in training script
n_features = 65536
k = 100
layer_num = 5

# Create SAE model
sae = SparseAutoencoder(
    d_in=r1_model.config.hidden_size,
    n_features=n_features,
    k=k,
    device=device
)

# Load trained SAE weights
sae_path = f"r1-sae-layer{layer_num}_{n_features}_{k}-27000.pt"
sae.load_state_dict(torch.load(sae_path))
sae.to(device)
sae.eval()

# Set up data generator - same as in training script
generator = sae_cached_activation_generator(
    model=r1_model,
    tokenizer=r1_tokenizer,
    dataset=dataset,
    layer_num=layer_num,
    activation_batch_size=256,
    generator_batch_size=16,
    acts_per_run=100_000,
    ctx_len=512,
    skip_first_n_tokens=1,
)

# Generate activations
activations = next(generator).to(device).to(torch.float32)

# Generate reconstructions
with torch.no_grad():
    features = sae.encode(activations)
    reconstructions = sae.decode(features)

# Compute FVU
with torch.no_grad():
    e = reconstructions - activations
    total_variance = (activations - activations.mean(0)).pow(2).sum()
    squared_error = e.pow(2).sum()
    fvu = squared_error / total_variance
    
    # Also compute L0 (feature sparsity)
    l0 = (features > 0).float().sum(1).mean()

print(f"Evaluation results:")
print(f"FVU: {fvu.item():.4f}")
print(f"L0 (avg features active): {l0.item():.1f} out of {n_features}")
print(f"Sparsity: {100 * (1 - l0.item() / n_features):.2f}%") 