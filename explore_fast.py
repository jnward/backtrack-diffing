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
with finetune_model.generate(finetune_tokens, max_new_tokens=150, do_sample=False, temperature=None, top_p=None) as gen:
    ft_out = finetune_model.generator.output[0].save()

print(finetune_tokenizer.decode(ft_out))

# %%
wait_token_idx = 146
print(finetune_tokenizer.decode(ft_out[:wait_token_idx]))
print("Next token:", finetune_tokenizer.decode(ft_out[wait_token_idx]))

prompt_tokens_ft = ft_out[:wait_token_idx]

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

# %%
def get_logit_diff(logits, answer_token_indices=answer_token_indices):
    logits = logits[-1, :]
    # correct_logits = logits.gather(0, answer_token_indices[0])
    # incorrect_logits = logits.gather(0, answer_token_indices[1])
    correct_logits = logits[answer_token_indices[0]]
    incorrect_logits = logits[answer_token_indices[1]]
    return correct_logits - incorrect_logits

# %%
base_is_clean = False
if base_is_clean:
    clean_model = base_model
    corrupted_model = finetune_model
    clean_logits = logits_b
    corrupted_logits = logits_ft
    clean_tokens = prompt_tokens_b
    corrupted_tokens = prompt_tokens_ft
else:
    clean_model = finetune_model
    corrupted_model = base_model
    clean_logits = logits_ft
    corrupted_logits = logits_b
    clean_tokens = prompt_tokens_ft
    corrupted_tokens = prompt_tokens_b

CLEAN_BASELINE = get_logit_diff(clean_logits)
CORRUPTED_BASELINE = get_logit_diff(corrupted_logits)
print(f"clean logit diff: {CLEAN_BASELINE.item()}")
print(f"corrupted logit diff: {CORRUPTED_BASELINE.item()}")

# %%
def wait_metric(logits, answer_token_indices=answer_token_indices):
    return (get_logit_diff(logits, answer_token_indices) - CORRUPTED_BASELINE) / (CLEAN_BASELINE - CORRUPTED_BASELINE)

# %%
print(f"clean Baseline: {wait_metric(clean_logits).item()}")
print(f"corrupted Baseline: {wait_metric(corrupted_logits).item()}")

# %%
corrupted_out = []
corrupted_grads = []
with corrupted_model.trace(corrupted_tokens) as tr:
    for l, layer in enumerate(corrupted_model.model.layers):
        layer_out = layer.output[0]
        corrupted_out.append(layer_out.save())
        corrupted_grads.append(layer_out.grad.save())

    logits = corrupted_model.output[0][0].save()
    value = wait_metric(logits).save()
    value.backward()

print(corrupted_grads)

# %%
clean_out = []
with clean_model.trace(clean_tokens) as tr:
    for l, layer in enumerate(clean_model.model.layers):
        layer_out = layer.output[0]
        clean_out.append(layer_out.save())

# %%
import einops
patching_results = []

for corrupted_grad, corrupted, clean, layer in zip(
    corrupted_grads, corrupted_out, clean_out, corrupted_model.model.layers
):
    attr = einops.reduce(
        corrupted_grad.value * (clean.value - corrupted.value),
        "batch pos dim -> pos",
        "sum",
    )
    patching_results.append(
        attr.detach().cpu().float().numpy()
    )

# %%
import plotly.express as px
fig = px.imshow(
    patching_results,
    color_continuous_scale="RdBu",
    color_continuous_midpoint=0,
    title="Attribution Patching over Token Positions",
    labels=dict(x="Token Position", y="Layer", color="Norm. Logit Diff"),
)
fig.show()
# %%
for i, token in enumerate(prompt_tokens_ft):
    print(f"{i}: {finetune_tokenizer.decode(token)}")

# %%
