from typing import Literal

import torch


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
