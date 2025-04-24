# %%
import torch
from nnsight import LanguageModel
from transformers import AutoTokenizer
from datasets import load_dataset
import os
import html
import pandas as pd

os.environ["HF_TOKEN"] = "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"

device = "cuda"
dtype = torch.bfloat16

# %%
base_model = LanguageModel("meta-llama/Llama-3.1-8B", device_map=device, torch_dtype=dtype)
finetune_model = LanguageModel("deepseek-ai/DeepSeek-R1-Distill-Llama-8B", device_map=device, torch_dtype=dtype)

base_tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B")
finetune_tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Llama-8B")

# %%
from crosscoder.data_utils import convert_to_base_tokens, verify_base_tokens

# %%
messages = [
    # {"role": "user", "content": "What countries border the country whose capital is Antananarivo?"},
    # {"role": "user", "content": "Prove the Riemann Hypothesis."}
    # {"role": "user", "content": "What is 2538 in base 2?"}
    # {"role": "user", "content": "Why is one a prime number?"}
    {"role": "user", "content": "What is the third tallest building in NYC?"}
]
finetune_tokens = finetune_tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")[0]
base_tokens = convert_to_base_tokens(finetune_tokens)

assert verify_base_tokens(base_tokens, base_tokenizer)

# %%
with finetune_model.generate(finetune_tokens, max_new_tokens=1024, do_sample=False, temperature=None, top_p=None) as gen:
    # logits_ft = finetune_model.output[0][0].save()
    ft_out = finetune_model.generator.output[0].save()

print(ft_out.shape)

# with base_model.generate(base_tokens, max_new_tokens=100) as gen:
#     base_out = base_model.generator.output[0].save()

# print(base_tokenizer.decode(base_out))
print(finetune_tokenizer.decode(ft_out))

# %%
wait_token_idx = 146
print(finetune_tokenizer.decode(ft_out[:wait_token_idx]))
print("Next token:", finetune_tokenizer.decode(ft_out[wait_token_idx]))

prompt_tokens_ft = ft_out[:wait_token_idx]

# %%
completion_tokenized_ft = ft_out
completion_b = convert_to_base_tokens(completion_tokenized_ft)
assert verify_base_tokens(completion_b.cpu(), base_tokenizer)

with base_model.trace(completion_b) as tr:
    logits_b = base_model.output[0][0].save()

with finetune_model.trace(completion_tokenized_ft) as tr:
    logits_ft = finetune_model.output[0][0].save()

print(logits_b.shape)  # (seq_len, vocab_size)
print(logits_ft.shape)

# %%
# Compute KL divergence between base and finetune model logits
import torch.nn.functional as F

# Apply softmax to convert logits to probabilities
probs_b = F.softmax(logits_b, dim=-1)
probs_ft = F.softmax(logits_ft, dim=-1)

kl_div = F.kl_div(probs_ft.log(), probs_b, reduction='none').sum(dim=-1)

print(f"KL divergence shape: {kl_div.shape}")
print(f"Mean KL divergence: {kl_div.mean().item()}")

import plotly.express as px
px.line(kl_div.float().detach().cpu().numpy())

# %%
# Create HTML visualization of tokens colored by KL divergence
from IPython.display import HTML
import numpy as np

# Ensure kl_div is on CPU and convert to numpy
kl_values = kl_div.float().detach().cpu().numpy()

# Clip values to range [0, 5] for color scale
min_kl = 0
max_kl = 8
kl_clipped = np.clip(kl_values, min_kl, max_kl)

# Function to convert KL value to color (white to blue)
def get_color(kl_value):
    # Map 0-5 to 0-255 for blue channel (inverse because 255 is white)
    blue_val = 192 + int(63 * (1 - kl_value / max_kl))
    return f"rgb({blue_val}, {blue_val}, 255)"

# Get individual token strings
token_strings = [finetune_tokenizer.decode([token]) for token in ft_out]

# Create HTML with colored tokens
html_parts = []

# Simple CSS for hover tooltip
html_parts.append("""
<style>
.token {
  position: relative;
  display: inline-block;
}
.token:hover::after {
  content: attr(data-kl);
  position: absolute;
  top: -20px;
  left: 0;
  background: black;
  color: white;
  padding: 2px 4px;
  border-radius: 3px;
  font-size: 10px;
  white-space: nowrap;
  z-index: 100;
}
</style>
""")

for i, token in enumerate(token_strings):
    if i < len(kl_clipped):
        color = get_color(kl_clipped[i])
        # Show both original and clipped KL values
        kl_value = f"KL: {kl_values[i]:.3f}"
        
        # Check if token contains newline
        if '\n' in token:
            # Display the token representation with special highlighting
            html_parts.append(f"<span class='token' data-kl='{kl_value}' style='background-color: {color}; border: 1px dashed red; padding: 1px 3px;'>{html.escape(repr(token))}</span>")
            # Then output the actual token for proper formatting
            html_parts.append(token)
        else:
            html_parts.append(f"<span class='token' data-kl='{kl_value}' style='background-color: {color}'>{html.escape(token)}</span>")
    else:
        html_parts.append(f"<span>{html.escape(token)}</span>")

html_content = "<div style='font-family: monospace; white-space: pre-wrap;'>" + "".join(html_parts) + "</div>"

# Display the HTML visualization
HTML(html_content)

# %%
# attribution patching
with finetune_model.trace(prompt_tokens_ft) as tr:
    logits_ft = finetune_model.output[0][0].save()

prompt_tokens_b = convert_to_base_tokens(prompt_tokens_ft)
assert verify_base_tokens(prompt_tokens_b.cpu(), base_tokenizer)

with base_model.trace(prompt_tokens_b) as tr:
    logits_b = base_model.output[0][0].save()

print(logits_b.shape)
print(logits_ft.shape)
# %%
print(logits_ft[-1].argmax().item())
finetune_tokenizer.decode(logits_ft[-1].argmax().item())
# %%
print(logits_b[-1].argmax().item())
base_tokenizer.decode(logits_b[-1].argmax().item())

# %%
answer_token_indices = torch.tensor([40, 14524], device=device)  # "I", "Wait"

with base_model.trace(prompt_tokens_b) as tr:
    base_acts = []
    for l, layer in enumerate(base_model.model.layers):
        base_acts.append(layer.output[0][0].save())

# %%
with finetune_model.trace(prompt_tokens_ft) as tr:
    ft_acts = []
    for l, layer in enumerate(finetune_model.model.layers):
        ft_acts.append(layer.output[0][0].save())

# %%
base_acts = torch.stack(base_acts)
base_acts.shape

# %%
ft_acts = torch.stack(ft_acts)
ft_acts.shape

# %%
import pandas as pd
layer = 31
base_norms = base_acts[layer].norm(dim=-1).flatten()
ft_norms = ft_acts[layer].norm(dim=-1).flatten()

# Create overlaid histogram of base and finetune norms
norm_data = pd.DataFrame({
    'Norm': pd.concat([pd.Series(base_norms.detach().float().cpu().numpy()), 
                      pd.Series(ft_norms.detach().float().cpu().numpy())]),
    'Model': ['Base'] * len(base_norms) + ['Finetune'] * len(ft_norms)
})

fig = px.histogram(
    norm_data, 
    x="Norm", 
    color="Model",
    barmode="overlay",
    nbins=100,
    opacity=0.7,
    title="Distribution of Activation Norms",
    color_discrete_map={"Base": "blue", "Finetune": "red"}
)
fig.show()

# %%
act_diffs = (ft_acts - base_acts)
relative_diff_norms = act_diffs.norm(dim=-1) / (ft_acts.norm(dim=-1) + base_acts.norm(dim=-1))
fig = px.bar(relative_diff_norms.mean(dim=-1).float().detach().cpu().numpy())
fig.update_layout(title="Relative Difference Norms", xaxis_title="Layer")
fig.show()
# %%
def get_logit_diff(logits, answer_token_indices=answer_token_indices):
    logits = logits[-1, :]
    correct_logits = logits.gather(0, answer_token_indices[0])
    incorrect_logits = logits.gather(0, answer_token_indices[1])
    return correct_logits - incorrect_logits

# %%
BASE_BASELINE = get_logit_diff(logits_b)
FT_BASELINE = get_logit_diff(logits_ft)
print(f"Base logit diff: {BASE_BASELINE.item()}")
print(f"Finetune logit diff: {FT_BASELINE.item()}")

# %%
def wait_metric(logits, answer_token_indices=answer_token_indices):
    return (get_logit_diff(logits, answer_token_indices) - FT_BASELINE) / (BASE_BASELINE - FT_BASELINE)

# %%
print(f"Base Baseline: {wait_metric(logits_b).item()}")
print(f"FT Baseline: {wait_metric(logits_ft).item()}")

# %%
corrupted_out = []
corrupted_grads = []
with finetune_model.trace(prompt_tokens_ft[:10]) as tr:
    for l, layer in enumerate(finetune_model.model.layers):
        layer_out = layer.output[0]
        corrupted_out.append(layer_out.save())
        # corrupted_grads.append(layer_out.grad.save())

    logits = finetune_model.output[0][0].save()
    value = wait_metric(logits).save()
    value.backward()

print(corrupted_grads)

# %%
logits.shape
# %%
value
# %%
