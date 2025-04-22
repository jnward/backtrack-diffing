# %%
import torch
import transformers
import os

os.environ["HF_TOKEN"] = "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"
device = "cuda"

# %%
# load llama 3.1 8b
# base = transformers.AutoModelForCausalLM.from_pretrained(
#     "meta-llama/Llama-3.1-8B",
#     torch_dtype=torch.bfloat16,
#     device_map=device,
# )

# %%
# load r1 finetune
finetune_hf = transformers.AutoModelForCausalLM.from_pretrained(
    "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
    # torch_dtype=torch.bfloat16,
    device_map=device,
)

# %%
from transformer_lens import HookedTransformer
import gc

finetune = HookedTransformer.from_pretrained_no_processing(
    "meta-llama/Llama-3.1-8B",
    hf_model=finetune_hf,
    dtype=torch.bfloat16
)
finetune.set_tokenizer(transformers.AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Llama-8B"))
del(finetune_hf)
gc.collect()
torch.cuda.empty_cache()

base = HookedTransformer.from_pretrained_no_processing(
    "meta-llama/Llama-3.1-8B",
    dtype=torch.bfloat16
)
# %%
# count params:
print(f"Base model params: {sum(p.numel() for p in base.parameters())}")
print(f"Finetune model params: {sum(p.numel() for p in finetune.parameters())}")

# %%
import plotly.express as px
import pandas as pd
# examine embeddings

base_embeddings = base.W_E
finetune_embeddings = finetune.W_E

# Calculate the norms of the embeddings along the last dimension
base_norms = torch.norm(base_embeddings, dim=1).float().cpu().detach().numpy()
finetune_norms = torch.norm(finetune_embeddings, dim=1).float().cpu().detach().numpy()

# Create DataFrame for plotting
df = pd.DataFrame({
    'Norm': pd.concat([pd.Series(base_norms), pd.Series(finetune_norms)]),
    'Model': ['Base'] * len(base_norms) + ['Finetune'] * len(finetune_norms)
})

# Create a histogram with specific colors
fig = px.histogram(
    df, 
    x="Norm", 
    color="Model",
    barmode="overlay",
    nbins=64,
    opacity=0.5,
    title="Distribution of Embedding Norms (Llama 3.1 8B vs R1 Finetune)",
    color_discrete_map={"Base": "blue", "Finetune": "red"}
)

fig.show()

# %%
cosine_sims = torch.nn.functional.cosine_similarity(base_embeddings, finetune_embeddings, dim=1)

# plot cosine sims
fig = px.histogram(
    cosine_sims.float().cpu().detach().numpy(),
    title="Cosine Similarity between Base and Finetune Embeddings",
)
fig.show()

# %%
# identify the indices of tokens where cosine sim < 0.95:
divergent_token_idxs = torch.where(cosine_sims < 0.95)[0]
base_tokenizer = base.tokenizer
finetune_tokenizer = finetune.tokenizer
# base_tokenizer = transformers.AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B")
# finetune_tokenizer = transformers.AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Llama-8B")

# Create output file
with open('divergent_tokens.txt', 'w') as f:
    # Print and save tokens with cosine sims
    different_decodings = []
    for idx in divergent_token_idxs:
        base_token = base_tokenizer.decode(idx)
        finetune_token = finetune_tokenizer.decode(idx)
        if base_token != finetune_token:
            different_decodings.append(idx)
            continue
        if "special_token" in base_token:
            continue
        output = f"{repr(base_token)}\n{repr(finetune_token)}\t{cosine_sims[idx].item():.3f}\n"
        print(output)
        f.write(output)

    # Print and save tokens with different decodings
    output = "\n\nTokens with different decodings:"
    print(output)
    f.write(output + "\n")
    for idx in different_decodings:
        output = f"{repr(base_tokenizer.decode(idx))}\n{repr(finetune_tokenizer.decode(idx))}\t{cosine_sims[idx].item():.3f}\n"
        print(output)
        f.write(output)

# %%
# UNEMBED MATRIX ANALYSIS
print("Analyzing unembed matrices (lm_head)...")

# Get unembed matrices (LM head)
# base_unembed = base.lm_head.weight
# finetune_unembed = finetune.lm_head.weight
base_unembed = base.W_U
finetune_unembed = finetune.W_U

# Calculate the norms of the unembed matrices along the last dimension
base_unembed_norms = torch.norm(base_unembed, dim=1).float().cpu().detach().numpy()
finetune_unembed_norms = torch.norm(finetune_unembed, dim=1).float().cpu().detach().numpy()

# Create DataFrame for plotting
df_unembed = pd.DataFrame({
    'Norm': pd.concat([pd.Series(base_unembed_norms), pd.Series(finetune_unembed_norms)]),
    'Model': ['Base'] * len(base_unembed_norms) + ['Finetune'] * len(finetune_unembed_norms)
})

# Create a histogram with specific colors
fig = px.histogram(
    df_unembed, 
    x="Norm", 
    color="Model",
    barmode="overlay",
    nbins=64,
    opacity=0.5,
    title="Distribution of Unembed Matrix Norms (Llama 3.1 8B vs R1 Finetune)",
    color_discrete_map={"Base": "blue", "Finetune": "red"}
)

fig.show()

# %%
# Calculate cosine similarity between unembed matrices
unembed_cosine_sims = torch.nn.functional.cosine_similarity(base_unembed, finetune_unembed, dim=1)

# Plot cosine similarities
fig = px.histogram(
    unembed_cosine_sims.float().cpu().detach().numpy(),
    title="Cosine Similarity between Base and Finetune Unembed Matrices",
)
fig.show()

# %%
# Identify the indices of tokens where cosine sim < 0.95:
unembed_divergent_token_idxs = torch.where(unembed_cosine_sims < 0.95)[0]

# Create output file
with open('divergent_unembed_tokens.txt', 'w') as f:
    f.write("Analysis of divergent tokens in unembed matrices:\n\n")
    
    # Print and save tokens with cosine sims
    different_unembed_decodings = []
    for idx in unembed_divergent_token_idxs:
        base_token = base_tokenizer.decode(idx)
        finetune_token = finetune_tokenizer.decode(idx)
        if base_token != finetune_token:
            different_unembed_decodings.append(idx)
            continue
        if "special_token" in base_token:
            continue
        output = f"{repr(base_token)}\n{repr(finetune_token)}\t{unembed_cosine_sims[idx].item():.3f}\n"
        print(output)
        f.write(output)

    # Print and save tokens with different decodings
    output = "\n\nUnembed tokens with different decodings:"
    print(output)
    f.write(output + "\n")
    for idx in different_unembed_decodings:
        output = f"{repr(base_tokenizer.decode(idx))}\n{repr(finetune_tokenizer.decode(idx))}\t{unembed_cosine_sims[idx].item():.3f}\n"
        print(output)
        f.write(output)
# %%
# for layer_idx in range(len(base.model.layers)):
#     base_layer = base.model.layers[layer_idx]
#     finetune_layer = finetune.model.layers[layer_idx]

#     base_attn = base_layer.self_attn
#     finetune_attn = finetune_layer.self_attn

#     for head_idx in range(base.model.config.num_attention_heads):
#         print(base_attn.q_proj.weight.shape)
#         print(base_attn.k_proj.weight.shape)
#         print(base_attn.v_proj.weight.shape)
#         print(base_attn.o_proj.weight.shape)
#         break
#     break
# %%
prompt = "Ok, so the user is asking me to find the greatest prime factor of 1011."
tokenized = finetune_tokenizer.encode(prompt)
num_prompt_tokens = len(tokenized)

completion = finetune.generate(prompt, max_new_tokens=2048-num_prompt_tokens, stop_at_eos=True)
print(completion)
# %%
base_tokenizer.encode("User: Howdy partner!")
# %%
