# %%
from nnsight import LanguageModel
from transformers import AutoTokenizer
import torch
import torch.nn.functional as F
import plotly.express as px

torch.set_grad_enabled(False)
device = "cuda"
dtype = torch.bfloat16

base_model = LanguageModel("meta-llama/Llama-3.1-8B", device_map="cuda", torch_dtype=dtype)
ft_model = LanguageModel("deepseek-ai/DeepSeek-R1-Distill-Llama-8B", device_map="cuda", torch_dtype=dtype)

base_tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B")
ft_tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Llama-8B")

# %%
prompt = "What's the largest building in the capital of France?"
messages = [
    {"role": "user", "content": prompt}
]

formatted_input = ft_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
tokens = ft_tokenizer.encode(formatted_input, add_special_tokens=False, return_tensors="pt").to(device)

# %%
with ft_model.generate(tokens, max_new_tokens=512) as gen:
    output = ft_model.generator.output.save()


# %%
print(ft_tokenizer.decode(output[0]))

# %%
from crosscoder.data_utils import convert_to_base_tokens

ft_tokens = output
base_tokens = convert_to_base_tokens(ft_tokens)
detokenized = base_tokenizer.decode(base_tokens[0])
retokenized = base_tokenizer.encode(detokenized, add_special_tokens=False, return_tensors="pt").to(device)

assert torch.equal(retokenized, base_tokens)


# %%
from collections import defaultdict

base_layer_acts = defaultdict(list)
with base_model.trace(base_tokens) as tr:
    for l in range(base_model.config.num_hidden_layers):
        layer_out = base_model.model.layers[l].output[0].float().cpu().save()
        base_layer_acts[l].append(layer_out)


# %%
ft_layer_acts = defaultdict(list)
with ft_model.trace(ft_tokens) as tr:
    for l in range(ft_model.config.num_hidden_layers):
        layer_out = ft_model.model.layers[l].output[0].float().cpu().save()
        ft_layer_acts[l].append(layer_out)

# %%
for l in range(ft_model.config.num_hidden_layers):
    base_layer_acts[l] = torch.cat(base_layer_acts[l], dim=0)
    ft_layer_acts[l] = torch.cat(ft_layer_acts[l], dim=0)

base_layer_acts = torch.stack(list(base_layer_acts.values()), dim=0)
ft_layer_acts = torch.stack(list(ft_layer_acts.values()), dim=0)

print(base_layer_acts.shape)
print(ft_layer_acts.shape)

# %%
base_layer_acts = base_layer_acts[:, 0, 1:]
ft_layer_acts = ft_layer_acts[:, 0, 1:]
# ^ remove BOS token and extra dim

# %%
mean_base_layer_norms = base_layer_acts.norm(dim=-1).mean(dim=-1)
mean_ft_layer_norms = ft_layer_acts.norm(dim=-1).mean(dim=-1)

# %%
base_layer_acts /= mean_base_layer_norms[:, None, None]
ft_layer_acts /= mean_ft_layer_norms[:, None, None]

# %%
diff_norm_means = (ft_layer_acts - base_layer_acts).norm(dim=-1).mean(dim=-1)
print(diff_norm_means)
# %%
import plotly.express as px
px.bar(diff_norm_means.cpu().numpy(), title="Difference in norm means")
# %%
import torch.nn.functional as F

# compute cosine similarity per layer and token, then average over tokens
cos_sim = F.cosine_similarity(ft_layer_acts, base_layer_acts, dim=-1)  # shape [layers, tokens]
mean_cos_sim = cos_sim.mean(dim=-1)                                    # shape [layers]

print(mean_cos_sim.shape)  # should be (32,)

# plot
fig = px.bar(mean_cos_sim.cpu().numpy(), title="Mean Cosine Similarity Between Base and FT Activations")
fig.update_layout(xaxis_title="Layer", yaxis_title="Mean Cosine Similarity")
# set y axis to range from 0 to 1
fig.update_yaxes(range=[0, 1])
fig.show()

# %%
