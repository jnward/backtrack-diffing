import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import DatasetDict, Dataset, IterableDatasetDict, IterableDataset
from nnsight import LanguageModel
from typing import Union

from pathlib import Path

AnyDataset = Union[DatasetDict, Dataset, IterableDatasetDict, IterableDataset]

BOS_TOKEN_ID = 151643
USER_TOKEN_ID = 151644
ASSISTANT_TOKEN_ID = 151645

def convert_to_base_tokens(tokens: torch.Tensor):
    """
    Convert r1 tokens to base tokens. Only works for Llama tokenizers.
    """
    # patch_token = 77627 # ` ############`
    patch_token = 27370 # ` ####`
    tokens = tokens.clone()
    tokens[tokens == 128011] = patch_token
    tokens[tokens == 128012] = patch_token
    tokens[tokens == 128013] = patch_token
    tokens[tokens == 128014] = patch_token
    return tokens

def verify_base_tokens(tokens: torch.Tensor, base_tokenizer: AutoTokenizer):
    """
    Verify that the tokens are valid base tokens.
    """
    base_tokens = base_tokenizer.encode(base_tokenizer.decode(tokens), return_tensors="pt")[0, 1:]
    return torch.equal(tokens, base_tokens)

def token_iter(
    base_tokenizer: AutoTokenizer,
    finetune_tokenizer: AutoTokenizer,
    dataset: Dataset | IterableDataset,
    batch_size: int,
    ctx_len: int,
):
    base_batch = []
    finetune_batch = []
    for example in dataset:
        messages = example["messages"]
        finetune_tokens = finetune_tokenizer.apply_chat_template(messages, add_generation_prompt=False, return_tensors="pt", max_length=ctx_len)
        finetune_tokens = finetune_tokens[0, :ctx_len]
        if finetune_tokens.shape[0] != ctx_len:
            continue
        base_tokens = convert_to_base_tokens(finetune_tokens)
        if not verify_base_tokens(base_tokens, base_tokenizer):
            continue
        if len(base_batch) < batch_size:
            base_batch.append(base_tokens)
            finetune_batch.append(finetune_tokens)
        else:
            yield torch.stack(base_batch), torch.stack(finetune_batch)
            base_batch = []
            finetune_batch = []

def sae_token_iter(
    tokenizer: AutoTokenizer,
    dataset: Dataset | IterableDataset,
    batch_size: int,
    ctx_len: int,
):
    batch = []
    for example in dataset:
        tokens = tokenizer.apply_chat_template(example["messages"], add_generation_prompt=False, return_tensors="pt", max_length=ctx_len)
        tokens = tokens[0, :ctx_len]
        if tokens.shape[0] != ctx_len:
            continue
        if len(batch) < batch_size:
            batch.append(tokens)
        else:
            yield torch.stack(batch)
            batch = []

def sae_token_iter_padded(
    tokenizer: AutoTokenizer,
    dataset: Dataset | IterableDataset,
    batch_size: int,
    ctx_len: int,
    pad_token_id: int = None,
):
    """
    Like sae_token_iter, but pads short contexts instead of skipping them.
    
    Args:
        tokenizer: The tokenizer to use
        dataset: The dataset to iterate over
        batch_size: Number of examples per batch
        ctx_len: Desired context length
        pad_token_id: Token ID to use for padding (defaults to tokenizer's pad_token_id)
    """
    if pad_token_id is None:
        pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
        
    batch = []
    for example in dataset:
        tokens = tokenizer.apply_chat_template(example["messages"], add_generation_prompt=False, 
                                              return_tensors="pt", max_length=ctx_len)
        tokens = tokens[0, :ctx_len]
        
        # Pad if shorter than ctx_len
        if tokens.shape[0] < ctx_len:
            padding = torch.full((ctx_len - tokens.shape[0],), pad_token_id, dtype=tokens.dtype)
            tokens = torch.cat([tokens, padding])
            
        if len(batch) < batch_size:
            batch.append(tokens)
        else:
            yield torch.stack(batch)
            batch = []
    
    # Return remaining batch if not empty
    if batch:
        yield torch.stack(batch)

def tokenized_context_iter(
    base_tokenizer: AutoTokenizer,
    finetune_tokenizer: AutoTokenizer,
    dataset: Dataset | IterableDataset,
    tokens_per_example: int,
    batch_size: int,
    max_ctx_len: int | None = None,
):
    row_len = 0
    current_row = 0
    base_out = torch.zeros(batch_size, tokens_per_example, dtype=torch.int64)
    finetune_out = torch.zeros(batch_size, tokens_per_example, dtype=torch.int64)
    context_starts = [[] for _ in range(batch_size)]
    attn_mask = torch.ones(batch_size, 1, tokens_per_example, tokens_per_example, dtype=torch.float) * float("-inf")
    
    for example in dataset:
        messages = example["messages"]
        finetune_tokens = finetune_tokenizer.apply_chat_template(messages, return_tensors="pt")
        if max_ctx_len is not None:
            finetune_tokens = finetune_tokens[0, :max_ctx_len]
        else:
            finetune_tokens = finetune_tokens[0, :]
        base_tokens = convert_to_base_tokens(finetune_tokens)
        if not verify_base_tokens(base_tokens, base_tokenizer):
            continue
            
        context_starts[current_row].append(row_len)
        remaining_space = tokens_per_example - row_len
        base_tokens = base_tokens[:remaining_space]
        finetune_tokens = finetune_tokens[:remaining_space]
        
        # Get the context length
        ctx_length = finetune_tokens.shape[0]
        
        # Create a lower triangular mask (causal mask) for this context
        causal_block = torch.tril(torch.ones(ctx_length, ctx_length, dtype=torch.bool))
        
        # Update the attention mask (setting 0.0 where attention is allowed)
        attn_mask[current_row, 0, row_len:row_len+ctx_length, row_len:row_len+ctx_length].masked_fill_(causal_block, 0.0)
        
        base_out[current_row, row_len:row_len + base_tokens.shape[0]] = base_tokens
        finetune_out[current_row, row_len:row_len + finetune_tokens.shape[0]] = finetune_tokens
        row_len += finetune_tokens.shape[0]
        assert row_len <= tokens_per_example
        
        if row_len == tokens_per_example:
            current_row += 1
            row_len = 0
            if current_row == batch_size:
                yield base_out, finetune_out, context_starts, attn_mask
                base_out = torch.zeros(batch_size, tokens_per_example, dtype=torch.int64)
                finetune_out = torch.zeros(batch_size, tokens_per_example, dtype=torch.int64)
                context_starts = [[] for _ in range(batch_size)]
                attn_mask = torch.ones(batch_size, 1, tokens_per_example, tokens_per_example, dtype=torch.float) * float("-inf")
                current_row = 0

@torch.no_grad()
def get_activations(
    model: AutoModelForCausalLM, token_batch: torch.Tensor, layer_num: int,
):
    with torch.no_grad():
        outputs = model.forward(token_batch, output_hidden_states=True)
        activations = outputs.hidden_states[layer_num]
    return activations

@torch.no_grad()
def get_activations_nnsight(
    model: LanguageModel,
    token_batch: torch.Tensor,
    attn_mask: torch.Tensor,
    layer_num: int,
):
    # row = {"input_ids": token_batch, "attention_mask": attn_mask}
    row = {"input_ids": token_batch}
    with model.trace(row) as trace:
        outputs = model.model.layers[layer_num].output[0].save()
    return outputs

def _save_activations_to_disk(activations_list, save_dir, file_idx, skip_first_n_tokens):
    """
    Helper function to save activations to disk.
    Used by both cached_activation_generator and cache_activations_to_disk.
    
    Parameters:
    - activations_list: List of activation tensors to save
    - save_dir: Directory to save to
    - file_idx: Index for the filename
    - skip_first_n_tokens: Number of tokens to skip from the beginning
    
    Returns:
    - None
    """
    acts_cat = torch.cat(activations_list, dim=0)
    acts_cat = acts_cat[:, skip_first_n_tokens:, :]
    acts_cat = acts_cat.reshape(-1, acts_cat.size(-1))
    # Apply random permutation (will be affected by torch.manual_seed)
    acts_cat = acts_cat[torch.randperm(acts_cat.size(0))]
    save_path = save_dir / f"acts_{file_idx}.pt"
    torch.save(acts_cat, save_path)
    return acts_cat.size(0)  # Return count for logging


def get_desired_activation_indices(context_starts: list[list[int]], tokens_per_example: int, skip_first_n_tokens: int):
    # First, flatten context_starts with proper batch offsets
    flat_context_starts = []
    for batch_idx, starts in enumerate(context_starts):
        batch_offset = batch_idx * tokens_per_example
        for start in starts:
            flat_context_starts.append(batch_offset + start)
    
    # Create a set of all positions to exclude
    exclude_positions = set()
    for start in flat_context_starts:
        for i in range(skip_first_n_tokens):
            exclude_positions.add(start + i)
    
    # Generate all possible indices
    all_indices = list(range(len(context_starts) * tokens_per_example))
    
    # Filter out the excluded positions
    desired_indices = [idx for idx in all_indices if idx not in exclude_positions]
    
    return torch.tensor(desired_indices, dtype=torch.int64)

import time
def new_cached_activation_generator(
    base_model: LanguageModel,
    finetune_model: LanguageModel,
    base_tokenizer: AutoTokenizer,
    finetune_tokenizer: AutoTokenizer,
    dataset: Dataset | IterableDataset,
    layer_num: int,
    activation_batch_size: int,
    generator_batch_size=24,
    acts_per_run=1_000_000,  # Combined parameter (was examples_per_run and max_acts_per_file)
    tokens_per_example=128,
    skip_first_n_tokens=0,
    device="cuda"
):
    """
    Generate activations and cache them in memory, with optional saving to disk.
    With the same random seed, this will produce identical files to cache_activations_to_disk.
    
    Parameters:
    - base_model: The base model
    - finetune_model: The finetuned model
    - tokenizer: The tokenizer
    - dataset: Dataset to generate activations from
    - layer_num: Layer to extract activations from
    - activation_batch_size: Size of batches yielded to training
    - generator_batch_size: Size of batches for generating activations
    - acts_per_run: Maximum activations per run (and per file when saving)
    - tokens_per_example: Context length for tokenization
    - skip_first_n_tokens: Number of tokens to skip from the beginning
    
    Yields:
    - Batches of activations for training
    """
    data_iter = tokenized_context_iter(base_tokenizer, finetune_tokenizer, dataset, tokens_per_example, generator_batch_size)
    while True:
        my_acts = []
        num_acts = 0
        # Generate activations for this run
        while num_acts < acts_per_run:
            try:
                base_token_batch, finetune_token_batch, context_starts, attn_mask = next(data_iter)
                base_token_batch = base_token_batch.to(device)
                finetune_token_batch = finetune_token_batch.to(device)
                attn_mask = attn_mask.to(device).to(base_model.dtype)
                base_activations = get_activations_nnsight(base_model, base_token_batch, attn_mask, layer_num)
                finetune_activations = get_activations_nnsight(finetune_model, finetune_token_batch, attn_mask, layer_num)
                desired_activation_indices = get_desired_activation_indices(context_starts, tokens_per_example, skip_first_n_tokens)
                base_activations = base_activations.view(-1, base_activations.size(-1))
                finetune_activations = finetune_activations.view(-1, finetune_activations.size(-1))
                base_activations = base_activations[desired_activation_indices, :]
                finetune_activations = finetune_activations[desired_activation_indices, :]
                concatenated_activations_BZ = torch.cat([base_activations, finetune_activations], dim=-1)
                my_acts.append(concatenated_activations_BZ)
                num_acts += concatenated_activations_BZ.size(0)
            except StopIteration:
                print("Dataset exhausted.")
                if not my_acts:  # No activations generated
                    return
                break
            
        # Process activations for training (we use the same permutation as when saving)
        acts_cat = torch.cat(my_acts, dim=0)
        randperm = torch.randperm(acts_cat.size(0))
        acts_cat = acts_cat[randperm]  
        # Yield batches for training
        for i in range(0, len(acts_cat), activation_batch_size):
            yield acts_cat[i : i + activation_batch_size]

def cached_activation_generator(
    base_model: AutoModelForCausalLM,
    finetune_model: AutoModelForCausalLM,
    base_tokenizer: AutoTokenizer,
    finetune_tokenizer: AutoTokenizer,
    dataset: Dataset | IterableDataset,
    layer_num: int,
    activation_batch_size: int,
    generator_batch_size=24,
    acts_per_run=1_000_000,  # Combined parameter (was examples_per_run and max_acts_per_file)
    ctx_len=128,
    skip_first_n_tokens=0,
    save_to_disk: str | None = None,
    return_tokens: bool = False,
):
    """
    Generate activations and cache them in memory, with optional saving to disk.
    With the same random seed, this will produce identical files to cache_activations_to_disk.
    
    Parameters:
    - base_model: The base model
    - finetune_model: The finetuned model
    - tokenizer: The tokenizer
    - dataset: Dataset to generate activations from
    - layer_num: Layer to extract activations from
    - activation_batch_size: Size of batches yielded to training
    - generator_batch_size: Size of batches for generating activations
    - acts_per_run: Maximum activations per run (and per file when saving)
    - ctx_len: Context length for tokenization
    - skip_first_n_tokens: Number of tokens to skip from the beginning
    - save_to_disk: Optional path to save activations (None = don't save)
    
    Yields:
    - Batches of activations for training
    """
    data_iter = token_iter(base_tokenizer,finetune_tokenizer, dataset, generator_batch_size, ctx_len)
    
    # Calculate how many batches to generate per run
    # Each token batch gives us generator_batch_size * (ctx_len - skip_first_n_tokens) tokens
    tokens_per_batch = generator_batch_size * (ctx_len - skip_first_n_tokens)
    batches_per_run = acts_per_run // tokens_per_batch

    print(f"Generating {batches_per_run} batches dper run")
    
    # Set up disk saving if requested
    save_dir = None
    file_acc = 0
    if save_to_disk:
        save_dir = Path(save_to_disk)
        save_dir.mkdir(parents=True, exist_ok=True)
        # Check if files already exist in directory and get the highest index
        existing_files = list(save_dir.glob("acts_*.pt"))
        if existing_files:
            existing_indices = [int(f.stem.split('_')[1]) for f in existing_files]
            file_acc = max(existing_indices) + 1
    
    while True:
        my_acts = []
        # Generate activations for this run
        for _ in range(batches_per_run):
            try:
                base_token_batch, finetune_token_batch = next(data_iter)
                base_token_batch = base_token_batch.to(base_model.device)
                finetune_token_batch = finetune_token_batch.to(finetune_model.device)
                base_activations_BD = get_activations(base_model, base_token_batch, layer_num)
                finetune_activations_BD = get_activations(finetune_model, finetune_token_batch, layer_num)
                concatenated_activations_BZ = torch.cat([base_activations_BD, finetune_activations_BD], dim=-1)
                my_acts.append(concatenated_activations_BZ)
            except StopIteration:
                print("Dataset exhausted.")
                if not my_acts:  # No activations generated
                    return
                break
        
        # Save to disk if requested
        if save_to_disk and my_acts:
            saved_count = _save_activations_to_disk(my_acts, save_dir, file_acc, skip_first_n_tokens)
            print(f"Saved {saved_count} activations to {save_dir}/acts_{file_acc}.pt")
            file_acc += 1
            
        # Process activations for training (we use the same permutation as when saving)
        acts_cat = torch.cat(my_acts, dim=0)
        acts_cat = acts_cat[:, skip_first_n_tokens:, :]
        acts_cat = acts_cat.reshape(-1, acts_cat.size(-1))
        randperm = torch.randperm(acts_cat.size(0))
        acts_cat = acts_cat[randperm]
        
        # Yield batches for training
        for i in range(0, len(acts_cat), activation_batch_size):
            yield acts_cat[i : i + activation_batch_size]


def sae_cached_activation_generator(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    dataset: Dataset | IterableDataset,
    layer_num: int,
    activation_batch_size: int,
    generator_batch_size=24,
    acts_per_run=1_000_000,  # Combined parameter (was examples_per_run and max_acts_per_file)
    ctx_len=128,
    skip_first_n_tokens=0,
    save_to_disk: str | None = None,
):
    """
    Generate activations and cache them in memory, with optional saving to disk.
    With the same random seed, this will produce identical files to cache_activations_to_disk.
    
    Parameters:
    - base_model: The base model
    - finetune_model: The finetuned model
    - tokenizer: The tokenizer
    - dataset: Dataset to generate activations from
    - layer_num: Layer to extract activations from
    - activation_batch_size: Size of batches yielded to training
    - generator_batch_size: Size of batches for generating activations
    - acts_per_run: Maximum activations per run (and per file when saving)
    - ctx_len: Context length for tokenization
    - skip_first_n_tokens: Number of tokens to skip from the beginning
    - save_to_disk: Optional path to save activations (None = don't save)
    
    Yields:
    - Batches of activations for training
    """
    data_iter = sae_token_iter(tokenizer, dataset, generator_batch_size, ctx_len)
    
    # Calculate how many batches to generate per run
    # Each token batch gives us generator_batch_size * (ctx_len - skip_first_n_tokens) tokens
    tokens_per_batch = generator_batch_size * (ctx_len - skip_first_n_tokens)
    batches_per_run = acts_per_run // tokens_per_batch

    print(f"Generating {batches_per_run} batches dper run")
    
    # Set up disk saving if requested
    save_dir = None
    file_acc = 0
    if save_to_disk:
        save_dir = Path(save_to_disk)
        save_dir.mkdir(parents=True, exist_ok=True)
        # Check if files already exist in directory and get the highest index
        existing_files = list(save_dir.glob("acts_*.pt"))
        if existing_files:
            existing_indices = [int(f.stem.split('_')[1]) for f in existing_files]
            file_acc = max(existing_indices) + 1
    
    while True:
        my_acts = []
        # Generate activations for this run
        for _ in range(batches_per_run):
            try:
                token_batch = next(data_iter)
                token_batch = token_batch.to(model.device)
                activations_BD = get_activations(model, token_batch, layer_num)
                my_acts.append(activations_BD)
            except StopIteration:
                print("Dataset exhausted.")
                if not my_acts:  # No activations generated
                    return
                break
        
        # Save to disk if requested
        if save_to_disk and my_acts:
            saved_count = _save_activations_to_disk(my_acts, save_dir, file_acc, skip_first_n_tokens)
            print(f"Saved {saved_count} activations to {save_dir}/acts_{file_acc}.pt")
            file_acc += 1
            
        # Process activations for training (we use the same permutation as when saving)
        acts_cat = torch.cat(my_acts, dim=0)
        acts_cat = acts_cat[:, skip_first_n_tokens:, :]
        acts_cat = acts_cat.reshape(-1, acts_cat.size(-1))
        randperm = torch.randperm(acts_cat.size(0))
        acts_cat = acts_cat[randperm]
        
        # Yield batches for training
        for i in range(0, len(acts_cat), activation_batch_size):
            yield acts_cat[i : i + activation_batch_size]

def cached_activation_generator_debug(
    base_model: AutoModelForCausalLM,
    finetune_model: AutoModelForCausalLM,
    base_tokenizer: AutoTokenizer,
    finetune_tokenizer: AutoTokenizer,
    dataset: Dataset | IterableDataset,
    layer_num: int,
    activation_batch_size: int,
    generator_batch_size=24,
    acts_per_run=1_000_000,
    ctx_len=128,
    skip_first_n_tokens=0,
    save_to_disk: str | None = None,
    return_tokens: bool = False,
    activation_threshold=100.0  # Threshold for extreme activations
):
    """
    Generate activations and cache them in memory, with optional saving to disk.
    Now includes detection and reporting of extreme activation values.
    """
    data_iter = token_iter(base_tokenizer, finetune_tokenizer, dataset, generator_batch_size, ctx_len)
    
    tokens_per_batch = generator_batch_size * (ctx_len - skip_first_n_tokens)
    batches_per_run = acts_per_run // tokens_per_batch

    print(f"Generating {batches_per_run} batches per run")
    
    # Set up disk saving if requested
    save_dir = None
    file_acc = 0
    if save_to_disk:
        save_dir = Path(save_to_disk)
        save_dir.mkdir(parents=True, exist_ok=True)
        existing_files = list(save_dir.glob("acts_*.pt"))
        if existing_files:
            existing_indices = [int(f.stem.split('_')[1]) for f in existing_files]
            file_acc = max(existing_indices) + 1
    
    while True:
        my_acts = []
        token_batches = []  # Store token batches for reference when detecting extreme values
        
        # Generate activations for this run
        for _ in range(batches_per_run):
            try:
                base_token_batch, finetune_token_batch = next(data_iter)
                base_token_batch = base_token_batch.to(base_model.device)
                finetune_token_batch = finetune_token_batch.to(finetune_model.device)
                
                # Store tokens for later reference
                token_batches.append((base_token_batch.cpu(), finetune_token_batch.cpu()))
                
                base_activations_BD = get_activations(base_model, base_token_batch, layer_num)
                finetune_activations_BD = get_activations(finetune_model, finetune_token_batch, layer_num)
                
                # Check for extreme activations in both models
                check_extreme_activations(
                    base_activations_BD, 
                    "base", 
                    base_token_batch, 
                    base_tokenizer,
                    activation_threshold
                )
                
                check_extreme_activations(
                    finetune_activations_BD, 
                    "finetune", 
                    finetune_token_batch, 
                    finetune_tokenizer,
                    activation_threshold
                )
                
                concatenated_activations_BZ = torch.cat([base_activations_BD, finetune_activations_BD], dim=-1)
                my_acts.append(concatenated_activations_BZ)
            except StopIteration:
                print("Dataset exhausted.")
                if not my_acts:  # No activations generated
                    return
                break
        
        # Save to disk if requested
        if save_to_disk and my_acts:
            saved_count = _save_activations_to_disk(my_acts, save_dir, file_acc, skip_first_n_tokens)
            print(f"Saved {saved_count} activations to {save_dir}/acts_{file_acc}.pt")
            file_acc += 1
            
        # Process activations for training
        acts_cat = torch.cat(my_acts, dim=0)
        acts_cat = acts_cat[:, skip_first_n_tokens:, :]
        acts_cat = acts_cat.reshape(-1, acts_cat.size(-1))
        randperm = torch.randperm(acts_cat.size(0))
        acts_cat = acts_cat[randperm]
        
        # Yield batches for training
        for i in range(0, len(acts_cat), activation_batch_size):
            yield acts_cat[i : i + activation_batch_size]


def check_extreme_activations(activations, model_name, token_batch, tokenizer, threshold=100.0):
    """
    Check for extreme activation values and print diagnostic information.
    
    Args:
        activations: The hidden state activations tensor [batch_size, seq_len, hidden_dim]
        model_name: String identifier for the model ('base' or 'finetune')
        token_batch: The batch of token IDs used to generate these activations
        tokenizer: The tokenizer to decode token IDs
        threshold: The threshold above which an activation is considered extreme
    """
    # Find maximum activation value
    max_val = torch.max(activations)
    min_val = torch.min(activations)
    extreme_val = max(max_val.abs().item(), min_val.abs().item())
    
    if extreme_val > threshold:
        print(f"\n{'='*80}")
        print(f"EXTREME ACTIVATION DETECTED in {model_name} model: {extreme_val:.2f}")
        
        # Find which examples and positions have extreme activations
        batch_indices, seq_indices, neuron_indices = torch.where(
            (activations > threshold) | (activations < -threshold)
        )
        
        # Group by example
        example_dict = {}
        for batch_idx, seq_idx, neuron_idx in zip(batch_indices, seq_indices, neuron_indices):
            b_idx, s_idx, n_idx = batch_idx.item(), seq_idx.item(), neuron_idx.item()
            activation_value = activations[b_idx, s_idx, n_idx].item()
            
            if b_idx not in example_dict:
                example_dict[b_idx] = []
            
            example_dict[b_idx].append((s_idx, n_idx, activation_value))
        
        # Report on each affected example
        for batch_idx, positions in example_dict.items():
            # Get the full sequence for this example
            sequence = token_batch[batch_idx]
            
            # Decode the tokens to text
            try:
                decoded_text = tokenizer.decode(sequence)
                
                # Print example info
                print(f"\nExample {batch_idx}:")
                print(f"Full text: {decoded_text}")
                print(f"Extreme activations at:")
                
                # Sort positions by absolute activation value (descending)
                positions.sort(key=lambda x: abs(x[2]), reverse=True)
                
                # Print only the top 10 positions to avoid overwhelming output
                for i, (seq_idx, neuron_idx, act_val) in enumerate(positions[:10]):
                    # Get the token at this position
                    token_id = sequence[seq_idx].item()
                    token_text = tokenizer.decode([token_id])
                    
                    # Get context (5 tokens before and after)
                    start_idx = max(0, seq_idx - 5)
                    end_idx = min(len(sequence), seq_idx + 6)
                    context_ids = sequence[start_idx:end_idx].tolist()
                    context_text = tokenizer.decode(context_ids)
                    
                    print(f"  Position {seq_idx}, Neuron {neuron_idx}: {act_val:.2f}")
                    print(f"  Token: '{token_text}'")
                    print(f"  Context: '...{context_text}...'")
                
                if len(positions) > 10:
                    print(f"  ... and {len(positions) - 10} more extreme activations in this example.")
                    
                # Show activation statistics for this example
                example_acts = activations[batch_idx]
                print(f"  Statistics for this example:")
                print(f"    Mean: {example_acts.mean().item():.2f}")
                print(f"    Std: {example_acts.std().item():.2f}")
                print(f"    Min: {example_acts.min().item():.2f}")
                print(f"    Max: {example_acts.max().item():.2f}")
                print(f"    # neurons > {threshold}: {(example_acts > threshold).sum().item()}")
                print(f"    # neurons < -{threshold}: {(example_acts < -threshold).sum().item()}")
                
            except Exception as e:
                print(f"Error decoding sequence: {e}")
                print(f"Raw token IDs: {sequence.tolist()}")
        
        print(f"{'='*80}\n")


def disk_activation_generator(batch_size, num_files=None, dir="acts", skip_first_n=0):
    read_dir = Path(dir)
    current_batch = []
    if num_files is None:
        num_files = len(list(read_dir.glob("acts_*.pt")))
    print(f"Reading from {num_files} files", end="")
    if skip_first_n:
        print(f", skipping first {skip_first_n}")
    else:
        print()
    for file_id in range(skip_first_n, num_files):
        read_path = read_dir / f"acts_{file_id}.pt"
        # print("loading", read_path)
        acts = torch.load(read_path)
        for row in acts:
            current_batch.append(row.unsqueeze(0))
            if len(current_batch) == batch_size:
                yield torch.cat(current_batch, dim=0)
                current_batch = []


@torch.no_grad()
def compute_metrics(activations, features, reconstructions):
    l0 = (features > 0).float().sum(1).mean()

    e = reconstructions - activations
    total_variance = (activations - activations.mean(0)).pow(2).sum()
    squared_error = e.pow(2)
    fvu = squared_error.sum() / total_variance

    return l0.item(), fvu.item()

def sae_new_cached_activation_generator(
    model: LanguageModel,
    tokenizer: AutoTokenizer,
    dataset: Dataset | IterableDataset,
    layer_num: int,
    activation_batch_size: int,
    generator_batch_size=24,
    acts_per_run=1_000_000,
    tokens_per_example=128,
    skip_first_n_tokens=0,
    device="cuda"
):
    """
    SAE version of new_cached_activation_generator.
    Generate activations and cache them in memory for a single model.
    
    Parameters:
    - model: The model to extract activations from
    - tokenizer: The tokenizer
    - dataset: Dataset to generate activations from
    - layer_num: Layer to extract activations from
    - activation_batch_size: Size of batches yielded to training
    - generator_batch_size: Size of batches for generating activations
    - acts_per_run: Maximum activations per run
    - tokens_per_example: Context length for tokenization
    - skip_first_n_tokens: Number of tokens to skip from the beginning
    - device: Device to run the model on
    
    Yields:
    - Batches of activations for training
    """
    # We need a sae-specific version of tokenized_context_iter since it uses both base and finetune tokenizers
    data_iter = sae_tokenized_context_iter(tokenizer, dataset, tokens_per_example, generator_batch_size)
    while True:
        my_acts = []
        num_acts = 0
        # Generate activations for this run
        while num_acts < acts_per_run:
            try:
                token_batch, context_starts, attn_mask = next(data_iter)
                token_batch = token_batch.to(device)
                attn_mask = attn_mask.to(device).to(model.dtype)
                activations = get_activations_nnsight(model, token_batch, attn_mask, layer_num)
                desired_activation_indices = get_desired_activation_indices(context_starts, tokens_per_example, skip_first_n_tokens)
                activations = activations.view(-1, activations.size(-1))
                activations = activations[desired_activation_indices, :]
                my_acts.append(activations)
                num_acts += activations.size(0)
            except StopIteration:
                print("Dataset exhausted.")
                if not my_acts:  # No activations generated
                    return
                break
            
        # Process activations for training (with same permutation approach)
        acts_cat = torch.cat(my_acts, dim=0)
        randperm = torch.randperm(acts_cat.size(0))
        acts_cat = acts_cat[randperm]  
        # Yield batches for training
        for i in range(0, len(acts_cat), activation_batch_size):
            yield acts_cat[i : i + activation_batch_size]

def sae_tokenized_context_iter(
    tokenizer: AutoTokenizer,
    dataset: Dataset | IterableDataset,
    tokens_per_example: int,
    batch_size: int,
    max_ctx_len: int | None = None,
):
    """
    SAE version of tokenized_context_iter that works with a single tokenizer.
    """
    row_len = 0
    current_row = 0
    token_out = torch.zeros(batch_size, tokens_per_example, dtype=torch.int64)
    context_starts = [[] for _ in range(batch_size)]
    attn_mask = torch.ones(batch_size, 1, tokens_per_example, tokens_per_example, dtype=torch.float) * float("-inf")
    
    for example in dataset:
        messages = example["messages"]
        tokens = tokenizer.apply_chat_template(messages, return_tensors="pt")
        if max_ctx_len is not None:
            tokens = tokens[0, :max_ctx_len]
        else:
            tokens = tokens[0, :]
            
        context_starts[current_row].append(row_len)
        remaining_space = tokens_per_example - row_len
        tokens = tokens[:remaining_space]
        
        # Get the context length
        ctx_length = tokens.shape[0]
        
        # Create a lower triangular mask (causal mask) for this context
        causal_block = torch.tril(torch.ones(ctx_length, ctx_length, dtype=torch.bool))
        
        # Update the attention mask (setting 0.0 where attention is allowed)
        attn_mask[current_row, 0, row_len:row_len+ctx_length, row_len:row_len+ctx_length].masked_fill_(causal_block, 0.0)
        
        token_out[current_row, row_len:row_len + tokens.shape[0]] = tokens
        row_len += tokens.shape[0]
        assert row_len <= tokens_per_example
        
        if row_len == tokens_per_example:
            current_row += 1
            row_len = 0
            if current_row == batch_size:
                yield token_out, context_starts, attn_mask
                token_out = torch.zeros(batch_size, tokens_per_example, dtype=torch.int64)
                context_starts = [[] for _ in range(batch_size)]
                attn_mask = torch.ones(batch_size, 1, tokens_per_example, tokens_per_example, dtype=torch.float) * float("-inf")
                current_row = 0

def sae_cached_activation_generator_padded(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    dataset: Dataset | IterableDataset,
    layer_num: int,
    activation_batch_size: int,
    generator_batch_size=24,
    acts_per_run=1_000_000,
    ctx_len=128,
    skip_first_n_tokens=0,
    save_to_disk: str | None = None,
    pad_token_id: int = None,
):
    """
    Generate activations and cache them in memory, with optional saving to disk.
    This variant pads short contexts instead of skipping them and filters out padding token activations.
    
    Parameters:
    - model: The model
    - tokenizer: The tokenizer
    - dataset: Dataset to generate activations from
    - layer_num: Layer to extract activations from
    - activation_batch_size: Size of batches yielded to training
    - generator_batch_size: Size of batches for generating activations
    - acts_per_run: Maximum activations per run
    - ctx_len: Context length for tokenization
    - skip_first_n_tokens: Number of tokens to skip from the beginning
    - save_to_disk: Optional path to save activations (None = don't save)
    - pad_token_id: Token ID to use for padding (defaults to tokenizer's pad_token_id)
    
    Yields:
    - Batches of activations for training (with padding token activations filtered out)
    """
    if pad_token_id is None:
        pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
        
    data_iter = sae_token_iter_padded(tokenizer, dataset, generator_batch_size, ctx_len, pad_token_id)
    
    # Calculate how many batches to generate per run
    # This is an approximation since we'll filter out padding tokens
    tokens_per_batch = generator_batch_size * (ctx_len - skip_first_n_tokens)
    batches_per_run = acts_per_run // tokens_per_batch

    print(f"Generating approximately {batches_per_run} batches per run")
    
    # Set up disk saving if requested
    save_dir = None
    file_acc = 0
    if save_to_disk:
        save_dir = Path(save_to_disk)
        save_dir.mkdir(parents=True, exist_ok=True)
        # Check if files already exist in directory and get the highest index
        existing_files = list(save_dir.glob("acts_*.pt"))
        if existing_files:
            existing_indices = [int(f.stem.split('_')[1]) for f in existing_files]
            file_acc = max(existing_indices) + 1
    
    while True:
        my_acts = []
        # Generate activations for this run
        for _ in range(batches_per_run):
            try:
                token_batch = next(data_iter)
                token_batch = token_batch.to(model.device)
                
                # Create padding mask (True where tokens are padding)
                padding_mask = (token_batch == pad_token_id)
                
                # Get activations for all tokens
                activations_BD = get_activations(model, token_batch, layer_num)
                
                # Skip the first n tokens if required
                if skip_first_n_tokens > 0:
                    activations_BD = activations_BD[:, skip_first_n_tokens:, :]
                    padding_mask = padding_mask[:, skip_first_n_tokens:]
                
                # Reshape to combine batch and sequence dimensions
                batch_size, seq_len, hidden_dim = activations_BD.shape
                activations_flat = activations_BD.reshape(-1, hidden_dim)
                padding_mask_flat = padding_mask.reshape(-1)
                
                # Filter out padding token activations using boolean mask
                activations_filtered = activations_flat[~padding_mask_flat]
                
                # Only add if we have non-padding activations
                if activations_filtered.shape[0] > 0:
                    my_acts.append(activations_filtered)
                
            except StopIteration:
                print("Dataset exhausted.")
                if not my_acts:  # No activations generated
                    return
                break
        
        # If no activations were collected, continue to next run
        if not my_acts:
            continue
            
        # Save to disk if requested
        if save_to_disk:
            # For saving, we directly concatenate the filtered activations
            acts_cat = torch.cat(my_acts, dim=0)
            # Apply random permutation
            acts_perm = acts_cat[torch.randperm(acts_cat.size(0))]
            save_path = save_dir / f"acts_{file_acc}.pt"
            torch.save(acts_perm, save_path)
            print(f"Saved {acts_perm.size(0)} activations to {save_path}")
            file_acc += 1
            
        # Process activations for training
        acts_cat = torch.cat(my_acts, dim=0)
        # Apply random permutation
        randperm = torch.randperm(acts_cat.size(0))
        acts_cat = acts_cat[randperm]
        
        # Yield batches for training
        for i in range(0, len(acts_cat), activation_batch_size):
            yield acts_cat[i : i + activation_batch_size]
