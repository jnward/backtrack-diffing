# %%
# import torch
import transformer_lens
import transformers
import os

os.environ["HF_TOKEN"] = "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"
# load gemma
# model1 = transformer_lens.HookedTransformer.from_pretrained_no_processing(
#     "google/gemma-2-2b",
#     device="cuda",
#     dtype=torch.bfloat16,
# )
# tokenizer1 = transformers.AutoTokenizer.from_pretrained("google/gemma-2-2b")
tokenizer1 = transformers.AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Qwen-7B")

# # load llama
# model2 = transformer_lens.HookedTransformer.from_pretrained_no_processing(
#     "google/gemma-2-2b-it",
#     device="cuda",
#     dtype=torch.bfloat16,
# )
tokenizer2 = transformers.AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Math-7B")
# %%
tokenizer1

# %%
tokenizer2

# %%
import torch
device = "cuda"
finetune_hf = transformers.AutoModelForCausalLM.from_pretrained(
    "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
    # "meta-llama/Llama-3.1-8B",
    torch_dtype=torch.bfloat16,
    device_map=device,
)

# %%
from transformer_lens import HookedTransformer
import gc

finetune = HookedTransformer.from_pretrained_no_processing("meta-llama/Llama-3.1-8B", hf_model=finetune_hf)#, dtype=torch.bfloat16)
finetune.set_tokenizer(transformers.AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Llama-8B"))
# del(base_hf)
# gc.collect()
# torch.cuda.empty_cache()

# base = HookedTransformer.from_pretrained_no_processing("meta-llama/Llama-3.1-8B")

# %%
torch.allclose(finetune_hf.model.embed_tokens.weight, finetune.W_E.to(torch.bfloat16))
# %%
import transformers
import torch
device = "cuda"
model = transformers.AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-Math-7B", torch_dtype=torch.bfloat16, device_map=device)
tokenizer = transformers.AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Math-7B")
# %%
prompt = "<|endoftext|><|im_start|>system You are a helpful assistant.<|im_end|><|im_start|>user What is the sum of 1 and 2?<|im_end|>"
tokens = tokenizer.encode(prompt, return_tensors="pt").to(device)

# generate text
output = model.generate(tokens, max_new_tokens=100)
# %%
tokenizer.decode(output[0])
# %%
messages = [
    # {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is greatest prime factor of 1011?"}
]
tokens = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to(device)
output = model.generate(tokens, max_new_tokens=100)
print(tokenizer.decode(output[0]))
# %%
# tokenizer.decode(tokens[0])
# %%
tokens
# %%
tokenizer2 = transformers.AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Qwen-7B")
tokens2 = tokenizer2.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to(device)
print(tokenizer2.decode(tokens2[0]))
# %%
# tokenizer3 = transformers.AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B")
tokenizer3 = transformers.AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Llama-8B")
tokens3 = tokenizer3.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to(device)
print(tokenizer3.decode(tokens3[0]))
# %%
tokenizer0 = transformers.AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B")
model = transformers.AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.1-8B", torch_dtype=torch.bfloat16, device_map=device)

# %%
def convert_to_base_tokens_llama(tokens: torch.Tensor):
    patch_token = 3001
    tokens = tokens.clone()
    tokens[tokens == 128011] = patch_token
    tokens[tokens == 128012] = patch_token
    tokens[tokens == 128013] = patch_token
    tokens[tokens == 128014] = patch_token
    return tokens

# %%
messages = [
    # {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is the greatest prime factor of 1011? Make sure you think through your reasoning steps!"}
]
tokens3 = tokenizer3.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to(device)
base_tokens3 = convert_to_base_tokens(tokens3)
print(tokenizer3.decode(base_tokens3[0]))
# %%
output = model.generate(base_tokens3, max_new_tokens=100)
print(tokenizer3.decode(output[0]))
# %%
test = tokenizer3.encode(tokenizer3.decode(base_tokens3[0]), return_tensors="pt")[:,1:]
print(tokenizer3.decode(test[0]))

assert torch.equal(test, base_tokens3.cpu())
# %%
tokenizer3.decode(base_tokens3[0])
# %%
base_tokens3

# %%
tokens = torch.arange(100000, 101000)
for token in tokens:
    print(token, tokenizer3.decode(token))
# %%
tokenizer3.encode(" NEXT")
# %%
torch.equal(tokenizer0.encode(tokenizer0.decode(base_tokens3[0]), return_tensors="pt")[:, 1:], base_tokens3.cpu())
# %%
import datasets

my_dataset = datasets.load_dataset(
    "ServiceNow-AI/R1-Distill-SFT",
    "v1",
    split="train",
    streaming=True,
)

# %%
from tqdm import tqdm
def convert_to_base_tokens_llama(tokens: torch.Tensor):
    # patch_token = 51767 # `!!!!!!!!`
    # patch_token = 70747 # ` BREAK`
    # patch_token = 57236 # ` --------`
    # patch_token = 27449 # ` ########`
    # patch_token = 77627 # ` ############`
    patch_token = 27370 # ` ####`
    tokens = tokens.clone()
    tokens[tokens == 128011] = patch_token
    tokens[tokens == 128012] = patch_token
    tokens[tokens == 128013] = patch_token
    tokens[tokens == 128014] = patch_token
    return tokens

count = 0
fails = 0
pbar = tqdm(range(5000))
my_data_iter = iter(my_dataset)
for i in pbar:
    example = next(my_data_iter)
    messages = example["messages"]
    finetune_tokens = tokenizer3.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to(device)
    base_tokens = convert_to_base_tokens_llama(finetune_tokens)
    # print(tokenizer0.decode(base_tokens[0]))
    retokenized = tokenizer0.encode(tokenizer0.decode(base_tokens[0]), return_tensors="pt")[:, 1:]
    if not torch.equal(base_tokens.cpu(), retokenized):
        fails += 1
    # try:
    #     assert torch.equal(base_tokens.cpu(), retokenized)
    # except AssertionError:
    #     fails += 1

        # print(base_tokens[0])
        # print(retokenized[0])
        # raise
    count += 1
    # if count > 1000:
    #     break
print(fails / count)
# %%
for t in base_tokens[0]:
    print(t, tokenizer0.decode(t))

tokenizer0.decode(base_tokens[0])
# %%
for t in retokenized[0]:
    print(t, tokenizer0.decode(t))
# %%
tokenizer0.encode(" ####")
# %%
