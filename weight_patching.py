# %%
import torch
from nnsight import LanguageModel
import einops
import os

device = "cuda"
dtype = torch.float

os.environ["HF_TOKEN"] = "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"

# %%
base_model = LanguageModel("meta-llama/Llama-3.1-8B", device_map=device, torch_dtype=dtype)
finetune_model = LanguageModel("deepseek-ai/DeepSeek-R1-Distill-Llama-8B", device_map=device, torch_dtype=dtype)

# %%
tokens = torch.tensor([0], dtype=torch.int64, device=device)
with base_model.trace(tokens) as tracer:
    pass

with finetune_model.trace(tokens) as tracer:
    pass
# %%

# to do weight attribution:
# 1. calculate gradient of metric wrt base weights for all components of interest
# 2. get weight diff, scaled for base layer norm
# 3. calculate attribution as dot product of grad and weight diff

# %%
base_weights = []
base_ln = []

tokens = torch.tensor([0], dtype=torch.int64, device=device)

for layer_idx, layer in enumerate(base_model.model.layers):
    weight_dict = {}
    base_ln.append(layer.input_layernorm.weight.data)
    weight_dict["q_proj"] = layer.self_attn.q_proj.weight.data
    weight_dict["k_proj"] = layer.self_attn.k_proj.weight.data
    weight_dict["v_proj"] = layer.self_attn.v_proj.weight.data
    weight_dict["o_proj"] = layer.self_attn.o_proj.weight.data
    base_weights.append(weight_dict)

finetune_weights = []
finetune_ln = []

for layer_idx, layer in enumerate(finetune_model.model.layers):
    weight_dict = {}
    finetune_ln.append(layer.input_layernorm.weight.data)
    weight_dict["q_proj"] = layer.self_attn.q_proj.weight.data
    weight_dict["k_proj"] = layer.self_attn.k_proj.weight.data
    weight_dict["v_proj"] = layer.self_attn.v_proj.weight.data
    weight_dict["o_proj"] = layer.self_attn.o_proj.weight.data
    finetune_weights.append(weight_dict)

# %%
"""
self.q_proj = nn.Linear(
    config.hidden_size, config.num_attention_heads * self.head_dim, bias=config.attention_bias
)
self.k_proj = nn.Linear(
    config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
)
self.v_proj = nn.Linear(
    config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
)
self.o_proj = nn.Linear(
    config.num_attention_heads * self.head_dim, config.hidden_size, bias=config.attention_bias
)
"""

def translate_weights(weights, old_rmsnorm, new_rmsnorm):
    adjusted_weights = []
    
    for layer_weight_dict, old_rmsn_weight, new_rmsn_weight in zip(weights, old_rmsnorm, new_rmsnorm):
        new_dict = {}
        scaling = old_rmsn_weight / new_rmsn_weight  # Element-wise division
        
        for key, weight in layer_weight_dict.items():
            if key in ['q_proj', 'k_proj', 'v_proj']:
                # For QKV, scaling on the input dimension (dim=1)
                # Weight shapes are [out_features, in_features]
                new_dict[key] = weight * scaling.unsqueeze(0)  # Broadcasting to [1, 4096]
            elif key == 'o_proj':
                # For O, scaling on the output dimension (dim=0)
                # Weight shape is [out_features, in_features]
                new_dict[key] = weight * scaling.unsqueeze(1)  # Broadcasting to [4096, 1]
            else:
                # Copy other weights unchanged
                new_dict[key] = weight.clone()
                
        adjusted_weights.append(new_dict)
    
    return adjusted_weights

def bake_rmsnorm(weights, rmsnorm):
    adjusted_weights = []
    for layer_weight_dict, rmsn_weight in zip(weights, rmsnorm):
        new_dict = {}
        scaling = rmsn_weight
        
        for key, weight in layer_weight_dict.items():
            if key in ['q_proj', 'k_proj', 'v_proj']:
                # For QKV, scaling on the input dimension (dim=1)
                # Weight shapes are [out_features, in_features]
                new_dict[key] = weight * scaling.unsqueeze(0)  # Broadcasting to [1, 4096]
            elif key == 'o_proj':
                # For O, scaling on the output dimension (dim=0)
                # Weight shape is [out_features, in_features]
                new_dict[key] = weight * scaling.unsqueeze(1)  # Broadcasting to [4096, 1]
            else:
                # Copy other weights unchanged
                new_dict[key] = weight.clone()
                
        adjusted_weights.append(new_dict)
    return adjusted_weights

def reshape_to_heads(weights):
    """
    print(einops.rearrange(base_weights[0]["q_proj"], "(num_heads d_head) d_model -> num_heads d_head d_model", num_heads=base_model.config.num_attention_heads).shape)
    print(einops.rearrange(base_weights[0]["k_proj"], "(num_heads d_head) d_model -> num_heads d_head d_model", num_heads=base_model.config.num_key_value_heads).shape)
    print(einops.rearrange(base_weights[0]["v_proj"], "(num_heads d_head) d_model -> num_heads d_head d_model", num_heads=base_model.config.num_key_value_heads).shape)
    print(einops.rearrange(base_weights[0]["o_proj"], "d_model (num_heads d_head) -> num_heads d_head d_model", num_heads=base_model.config.num_attention_heads).shape)
    """
    adjusted_weights = []
    for layer_weight_dict in weights:
        new_dict = {}
        
        for key, weight in layer_weight_dict.items():
            if key in ['q_proj', 'k_proj']:
                new_dict[key] = einops.rearrange(weight, "(num_heads d_head) d_model -> num_heads d_head d_model", num_heads=base_model.config.num_attention_heads)
            elif key == 'v_proj':
                new_dict[key] = einops.rearrange(weight, "(num_heads d_head) d_model -> num_heads d_head d_model", num_heads=base_model.config.num_key_value_heads)
            elif key == 'o_proj':
                new_dict[key] = einops.rearrange(weight, "d_model (num_heads d_head) -> num_heads d_head d_model", num_heads=base_model.config.num_attention_heads)
            else:
                # Copy other weights unchanged
                new_dict[key] = weight.clone()
                
        adjusted_weights.append(new_dict)
    return adjusted_weights
# %%
# finetune_weights = translate_weights(orig_finetune_weights, finetune_ln, base_ln)
base_weights = bake_rmsnorm(base_weights, base_ln)
finetune_weights = bake_rmsnorm(finetune_weights, finetune_ln)

# %%
for layer_idx in range(len(base_weights)):
    for key, value in base_weights[layer_idx].items():
        print(layer_idx, key, value.max())
# %%
for layer_idx in range(len(finetune_weights)):
    for key, value in finetune_weights[layer_idx].items():
        print(layer_idx, key, value.max().item(), value.min().item(), value.abs().mean().item())

# %%
diff_norms = torch.zeros(len(base_weights), 4)
for layer_idx in range(len(base_weights)):
    for i, (key, value) in enumerate(base_weights[layer_idx].items()):
        f_w = finetune_weights[layer_idx][key]
        diff = f_w - value
        diff_norms[layer_idx, i] = diff.norm().item() / (f_w.norm().item() + value.norm().item())

# %%
import plotly.express as px
fig = px.imshow(diff_norms[:].cpu().numpy())
fig.show()

# %%
diffs = []
for layer in range(len(base_weights)):
    base_layer = base_weights[layer]
    finetune_layer = finetune_weights[layer]
    diff = {}
    for key, value in base_layer.items():
        diff[key] = (finetune_layer[key] - value) / (finetune_layer[key].norm() + value.norm())
    diffs.append(diff)
reshaped_diffs = reshape_to_heads(diffs)

# %%
# Plot the reshaped differences
import plotly.subplots as sp
import plotly.graph_objects as go

# Create the subplots
fig = sp.make_subplots(
    rows=1, cols=4,
    subplot_titles=["Q Projection", "K Projection", "V Projection", "O Projection"],
    shared_yaxes=True
)

# Get dimensions
num_layers = len(reshaped_diffs)
num_heads_qo = base_model.config.num_attention_heads
num_heads_kv = base_model.config.num_key_value_heads

# Compute norms across d_head and d_model dimensions
q_norms = torch.zeros(num_layers, num_heads_qo)
k_norms = torch.zeros(num_layers, num_heads_kv)
v_norms = torch.zeros(num_layers, num_heads_kv)
o_norms = torch.zeros(num_layers, num_heads_qo)

for layer_idx, layer_diff in enumerate(reshaped_diffs):
    # For each head, compute the norm across d_head and d_model dimensions
    for head_idx in range(num_heads_qo):
        q_norms[layer_idx, head_idx] = layer_diff["q_proj"][head_idx].norm().item()
    
    for head_idx in range(num_heads_kv):
        k_norms[layer_idx, head_idx] = layer_diff["k_proj"][head_idx].norm().item()
        v_norms[layer_idx, head_idx] = layer_diff["v_proj"][head_idx].norm().item()
    
    for head_idx in range(num_heads_qo):
        o_norms[layer_idx, head_idx] = layer_diff["o_proj"][head_idx].norm().item()

# Add the heatmaps
fig.add_trace(
    go.Heatmap(z=q_norms.cpu().numpy(), coloraxis="coloraxis"),
    row=1, col=1
)

fig.add_trace(
    go.Heatmap(z=k_norms.cpu().numpy(), coloraxis="coloraxis"),
    row=1, col=2
)

fig.add_trace(
    go.Heatmap(z=v_norms.cpu().numpy(), coloraxis="coloraxis"),
    row=1, col=3
)

fig.add_trace(
    go.Heatmap(z=o_norms.cpu().numpy(), coloraxis="coloraxis"),
    row=1, col=4
)

# Update layout
fig.update_layout(
    title="Weight Difference Norms by Head and Layer",
    coloraxis=dict(colorscale="Viridis"),
    height=800,
    width=1200,
)

# Set y-axis to show layer numbers (0 at top)
layer_labels = list(range(num_layers))
fig.update_yaxes(
    title_text="Layer",
    tickvals=list(range(num_layers)),
    ticktext=layer_labels,
    autorange="reversed"  # Put layer 0 at the top
)

# Set x-axis to show head indices
fig.update_xaxes(title_text="Head Index", col=1)
fig.update_xaxes(title_text="Head Index", col=2)
fig.update_xaxes(title_text="Head Index", col=3)
fig.update_xaxes(title_text="Head Index", col=4)

fig.show()

# %%
# Create reshaped versions of base and finetune weights for factored computation
reshaped_base = reshape_to_heads(base_weights)
reshaped_finetune = reshape_to_heads(finetune_weights)

# Compute factored QK and OV matrices and their relative norms
# Initialize arrays to store only the relative norms
qk_rel_norms = torch.zeros(num_layers, num_heads_qo)
ov_rel_norms = torch.zeros(num_layers, num_heads_qo)

for layer_idx in range(num_layers):
    # For Llama models with GQA, we need to handle different head counts
    # In GQA, each K/V head is used by multiple Q heads
    kv_ratio = num_heads_qo // num_heads_kv
    
    for q_head_idx in range(num_heads_qo):
        # Find the corresponding K head (for GQA)
        k_head_idx = q_head_idx // kv_ratio
        
        # Get the Q, K, V, O matrices for this head
        base_q = reshaped_base[layer_idx]["q_proj"][q_head_idx]  # [d_head, d_model]
        base_k = reshaped_base[layer_idx]["k_proj"][k_head_idx]  # [d_head, d_model]
        base_v = reshaped_base[layer_idx]["v_proj"][k_head_idx]  # [d_head, d_model]
        base_o = reshaped_base[layer_idx]["o_proj"][q_head_idx]  # [d_head, d_model]
        
        ft_q = reshaped_finetune[layer_idx]["q_proj"][q_head_idx]
        ft_k = reshaped_finetune[layer_idx]["k_proj"][k_head_idx]
        ft_v = reshaped_finetune[layer_idx]["v_proj"][k_head_idx]
        ft_o = reshaped_finetune[layer_idx]["o_proj"][q_head_idx]
        
        # Compute QK base and finetune matrices and immediately calculate relative norm
        # QK = Q @ K^T
        base_qk = einops.einsum(
            base_q, base_k,
            "qh d_model1, kh d_model2 -> d_model1 d_model2"
        )
        
        ft_qk = einops.einsum(
            ft_q, ft_k,
            "qh d_model1, kh d_model2 -> d_model1 d_model2"
        )
        
        # Calculate diff and norms immediately
        qk_diff = ft_qk - base_qk
        base_qk_norm = base_qk.norm().item()
        ft_qk_norm = ft_qk.norm().item()
        qk_diff_norm = qk_diff.norm().item()
        
        # Store relative norm and free memory by deleting the large matrices
        qk_rel_norms[layer_idx, q_head_idx] = qk_diff_norm / (base_qk_norm + ft_qk_norm)
        del base_qk, ft_qk, qk_diff
        
        # Compute OV base and finetune matrices and immediately calculate relative norm
        # OV = V @ O^T
        base_ov = einops.einsum(
            base_v, base_o,
            "vh d_model1, oh d_model2 -> d_model1 d_model2"
        )
        
        ft_ov = einops.einsum(
            ft_v, ft_o,
            "vh d_model1, oh d_model2 -> d_model1 d_model2"
        )
        
        # Calculate diff and norms immediately
        ov_diff = ft_ov - base_ov
        base_ov_norm = base_ov.norm().item()
        ft_ov_norm = ft_ov.norm().item()
        ov_diff_norm = ov_diff.norm().item()
        
        # Store relative norm and free memory
        ov_rel_norms[layer_idx, q_head_idx] = ov_diff_norm / (base_ov_norm + ft_ov_norm)
        del base_ov, ft_ov, ov_diff

# Plot the factored relative norms
fig = sp.make_subplots(
    rows=1, cols=2,
    subplot_titles=["QK Factored", "OV Factored"],
    shared_yaxes=True
)

# Add the heatmaps
fig.add_trace(
    go.Heatmap(z=qk_rel_norms.cpu().numpy(), coloraxis="coloraxis"),
    row=1, col=1
)

fig.add_trace(
    go.Heatmap(z=ov_rel_norms.cpu().numpy(), coloraxis="coloraxis"),
    row=1, col=2
)

# Update layout
fig.update_layout(
    title="Relative Factored Weight Difference Norms by Head and Layer",
    coloraxis=dict(colorscale="Viridis"),
    height=800,
    width=1000,
)

# Set y-axis to show layer numbers (0 at top)
layer_labels = list(range(num_layers))
fig.update_yaxes(
    title_text="Layer",
    tickvals=list(range(num_layers)),
    ticktext=layer_labels,
    autorange="reversed"  # Put layer 0 at the top
)

# Set x-axis to show head indices
fig.update_xaxes(title_text="Head Index", col=1)
fig.update_xaxes(title_text="Head Index", col=2)

fig.show()
# %%
