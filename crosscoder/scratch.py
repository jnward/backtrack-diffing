# %%
import torch
from data_utils import tokenized_context_iter
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from nnsight import LanguageModel
dataset = load_dataset("ServiceNow-AI/R1-Distill-SFT", "v1", split="train")

# %%
base_tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B")
finetune_tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Llama-8B",)

tokens_per_example = 2048
dtype = torch.bfloat16
# %%
data_iter = tokenized_context_iter(base_tokenizer, finetune_tokenizer, dataset, tokens_per_example=tokens_per_example, batch_size=16)

# %%
for base_out, finetune_out, context_starts, attn_mask in data_iter:
    print(base_out.shape, finetune_out.shape, context_starts, attn_mask.shape)
    break

# %%
import plotly.express as px
for i in range(2): 
    fig = px.imshow(attn_mask[i, 0] < 0)
    fig.show()

# %%
print(context_starts[0])

# %%
for i in range(len(context_starts[0])):
    start_idx = context_starts[0][i]
    end_idx = context_starts[0][i+1] if i < len(context_starts[0]) - 1 else tokens_per_example
    print(base_tokenizer.decode(base_out[0, start_idx:end_idx]))
    print(finetune_tokenizer.decode(finetune_out[0, start_idx:end_idx]))
    print("######################################")

# %%
# model = AutoModelForCausalLM.from_pretrained(
#     "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
#     torch_dtype=dtype,
#     device_map="cuda"
# )
lm = LanguageModel(
    "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
    torch_dtype=dtype,
    device_map="cuda",
)
# %%
test_string = finetune_tokenizer.apply_chat_template(
    [{"role": "user", "content": "Hello, how are you?"}],
    tokenize=False,
)
test_tokens = finetune_tokenizer(test_string, return_tensors="pt")
test_tokens

# %%
from data_utils import get_activations_nnsight

# %%
row = finetune_out[[0], :].to(device="cuda")
row = {"input_ids": row, "attention_mask": attn_mask[[0]].to(device="cuda").to(dtype)}

acts = get_activations_nnsight(
    lm,
    row["input_ids"],
    row["attention_mask"],
    11
)

acts
# %%

with lm.trace(row) as trace:
    full_acts = lm.model.layers[11].output[0].save()

print(full_acts.shape)

test_start_idx = context_starts[0][13]
test_end_idx = context_starts[0][14]
print(test_start_idx, test_end_idx)
individual_example = finetune_out[[0], test_start_idx:test_end_idx].to(device="cuda")

with lm.trace(individual_example) as trace:
    individual_acts = lm.model.layers[11].output[0].save()

print(individual_acts.shape)

# %%
isolated_acts = full_acts[:, test_start_idx:test_end_idx, :]
torch.allclose(isolated_acts, individual_acts)
# %%
print(isolated_acts.shape, individual_acts.shape)
# %%
print(isolated_acts)
print(individual_acts)
# %%
(isolated_acts - individual_acts).abs().mean()
# %%
isolated_acts.abs().mean()
# %%
# With custom tolerance
print(torch.allclose(isolated_acts, individual_acts, rtol=1e-3, atol=1e-3))

# Or check if the differences are small enough for practical purposes
diff = (isolated_acts - individual_acts).abs()
print(f"Max diff: {diff.max().item()}")
print(f"Mean diff: {diff.mean().item()}")
print(f"Relative diff: {diff.mean() / isolated_acts.abs().mean()}")


# %%
from data_utils import new_cached_activation_generator

base_model = LanguageModel(
    "meta-llama/Llama-3.1-8B",
    torch_dtype=dtype,
    device_map="cuda",
)   

# %%
my_data_gen = new_cached_activation_generator(
    base_model,
    lm,
    base_tokenizer,
    finetune_tokenizer,
    dataset,
    layer_num=11,
    activation_batch_size=256,
    generator_batch_size=16,
    acts_per_run=100_000,  # Combined parameter (was examples_per_run and max_acts_per_file)
    tokens_per_example=1024,
    skip_first_n_tokens=1,
)

# %%
import time
start_time = time.time()
for acts in my_data_gen:
    print(acts.shape)
    break
end_time = time.time()
print(f"Time taken: {end_time - start_time} seconds")
# %%
start = time.time()
acts = next(my_data_gen)
end = time.time()
print(f"Time taken: {end - start} seconds")
# %%
