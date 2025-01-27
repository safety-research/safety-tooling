from typing import List, Literal, Tuple

import torch
from tqdm import tqdm
from transformers import PreTrainedTokenizer


def squeeze_positions(
    activations: torch.Tensor,
    aggregation: Literal["max", "mean", "flatten", "last"] = "max",
) -> torch.Tensor:
    """
    Aggregates activations across the position dimension using various methods.

    Args:
        activations: Input tensor of shape (batch_size, sequence_length, hidden_size)
        aggregation: Method to use for aggregating across positions
        tokens: Optional token ids, used for specific aggregation methods

    Returns:
        Aggregated tensor of shape (batch_size, hidden_size) or (batch_size * sequence_length, hidden_size)
        depending on the aggregation method
    """
    if aggregation == "max":
        return activations.amax(dim=1)
    elif aggregation == "mean":
        return activations.mean(dim=1)
    elif aggregation == "flatten":
        return activations.flatten(0, 1)
    elif aggregation == "last":
        return activations[:, -1, :]
    else:
        raise NotImplementedError(f"Invalid aggregation method: {aggregation}")


def get_labeled(
    acts1: torch.Tensor,
    acts2: torch.Tensor,
    aggregation: Literal["max", "mean", "flatten", "last"] = "max",
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Combines and labels two sets of activations.

    Args:
        acts1: First set of activations of shape (batch_size, sequence_length, hidden_size)
        acts2: Second set of activations of shape (batch_size, sequence_length, hidden_size)
        aggregation: Method to use for aggregating across positions
        acts1_tokens: Optional token ids for first activation set
        acts2_tokens: Optional token ids for second activation set

    Returns:
        Tuple of:
            - Combined activations tensor
            - Labels tensor (0 for acts1, 1 for acts2)
    """
    # Use squeeze_positions to aggregate across the position dimension
    acts1 = squeeze_positions(acts1, aggregation)
    acts2 = squeeze_positions(acts2, aggregation)

    # Combine the features from both splits
    input_acts = torch.cat([acts1, acts2], dim=0)

    # Create labels: 0 for acts1, 1 for acts2
    labels = torch.cat([torch.zeros(acts1.shape[0], dtype=torch.long), torch.ones(acts2.shape[0], dtype=torch.long)])

    return input_acts, labels


def get_steering_vector(
    input_acts: torch.Tensor, labels: torch.Tensor, method: Literal["mean_diff"] = "mean_diff", normalized: bool = False
) -> torch.Tensor:
    """
    Calculates a steering vector based on the difference between two classes.

    Args:
        input_acts: Input activations tensor of shape (n_samples, hidden_size)
        labels: Binary labels tensor of shape (n_samples,)
        method: Method to use for calculating the steering vector
        normalized: Whether to normalize the resulting vector

    Returns:
        Steering vector tensor of shape (hidden_size,)
    """
    # Convert labels to boolean mask
    mask = labels == 1

    if method == "mean_diff":
        # Calculate means for each class
        mean_class_0 = input_acts[~mask].mean(dim=0)
        mean_class_1 = input_acts[mask].mean(dim=0)

        # Calculate the difference vector
        vector = mean_class_1 - mean_class_0

        # Normalize if requested
        if normalized:
            vector = vector / torch.norm(vector)

        return vector
    else:
        raise NotImplementedError(f"Method {method} not implemented")


def normalize_last_dim(tensor: torch.Tensor) -> torch.Tensor:
    """
    Normalizes a tensor along its last dimension using L2 normalization.

    Args:
        tensor: Input tensor to be normalized

    Returns:
        Normalized tensor with the same shape as input, where the last dimension
        has unit L2 norm
    """
    # Compute the L2 norm along the last dimension
    norm = torch.norm(tensor, p=2, dim=-1, keepdim=True)

    # Divide the tensor by the norm
    # We add a small epsilon to avoid division by zero
    normalized_tensor = tensor / (norm + 1e-8)
    return normalized_tensor


def process_data(
    prompts: List[str],
    targets: List[str],
    tokenizer: PreTrainedTokenizer,
    batch_size: int | None = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Process prompts and targets into batched tensor format for model input.

    Args:
        prompts: List of input prompt strings to be encoded
        targets: List of target completion strings to be encoded
        tokenizer: Tokenizer to convert text to token IDs
        batch_size: Optional batch size, defaults to length of prompts if None

    Returns:
        Tuple containing:
        - adv_tokens: Tensor of token IDs for combined prompts and targets (batch_size, max_length)
        - prompt_mask: Boolean mask indicating prompt token positions (batch_size, max_length)
        - target_mask: Boolean mask indicating target token positions (batch_size, max_length)
    """
    # Tokenize all prompts and targets without special tokens
    tokenized_prompts = [tokenizer.encode(prompt, add_special_tokens=False) for prompt in prompts]
    tokenized_targets = [tokenizer.encode(target, add_special_tokens=False) for target in targets]

    # Use full dataset size if batch_size not specified
    if batch_size is None:
        batch_size = len(prompts)

    # Calculate maximum sequence length across all prompt-target pairs
    max_length = max(len(tokenized_prompts[i] + tokenized_targets[i]) for i in range(batch_size))

    # Initialize tensors for tokens and masks
    adv_tokens = torch.zeros((batch_size, max_length), dtype=torch.long)
    adv_tokens.fill_(tokenizer.pad_token_id)  # Pad with tokenizer's pad token
    prompt_mask = torch.zeros((batch_size, max_length), dtype=torch.bool)
    target_mask = torch.zeros((batch_size, max_length), dtype=torch.bool)

    # Process each example in the batch
    for i in tqdm(range(batch_size), desc="Tokenizing"):
        # Combine prompt and target tokens
        combined = tokenized_prompts[i] + tokenized_targets[i]

        # Fill in the combined tokens
        adv_tokens[i, : len(combined)] = torch.tensor(combined)

        # Create masks for prompt and target positions
        prompt_mask[i, : len(tokenized_prompts[i])] = True
        target_mask[i, len(tokenized_prompts[i]) : len(combined)] = True

    return adv_tokens, prompt_mask, target_mask
