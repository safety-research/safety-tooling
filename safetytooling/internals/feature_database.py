import json
import os
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import torch
from tqdm.auto import tqdm

from .sae_wrappers import SparseAutoencoderWrapper


class Example:
    """Stores text, tokens, and feature activations for a single example"""

    def __init__(
        self, token_ids: List[int], str_tokens: List[str], latent_data: Dict[str, Tuple[torch.Tensor, torch.Tensor]]
    ):
        """
        Args:
            token_ids: List of token IDs
            str_tokens: List of string representations of tokens
            latent_data: Dictionary mapping hook_names to (indices, activations) tensors
        """
        self.token_ids = token_ids
        self.str_tokens = str_tokens
        self.text = "".join(str_tokens)
        self.latent_data = latent_data

    def __str__(self) -> str:
        return self.text

    def get_feature_set(self, hook_name: str) -> Set[int]:
        """Get set of all feature IDs that activate on this example for given hook"""
        indices, _ = self.latent_data[hook_name]
        return set(indices.flatten().tolist())

    def get_feature_activation(self, feature_id: int, hook_name: str) -> torch.Tensor:
        """Get activation values of specified feature across all tokens"""
        indices, acts = self.latent_data[hook_name]
        feature_acts = torch.zeros(len(self.token_ids), dtype=acts.dtype)
        mask = indices == feature_id
        feature_acts[mask.any(dim=-1)] = acts[mask]
        return feature_acts
    
    def to_list(self, hook_name: str, feature_id: int) -> List[Tuple[str, float]]:
        """Get a list of token, activation pairs"""
        feature_acts = self.get_feature_activation(feature_id, hook_name)
        return list(zip(self.str_tokens, feature_acts.tolist()))

class FeatureDatabase:
    """Database for storing and querying SAE feature activations across a dataset"""

    def __init__(self, encoder: SparseAutoencoderWrapper):
        self.encoder = encoder
        self.examples: List[Example] = []
        self.feature_data: Optional[Dict[str, np.memmap]] = None
        self.hook_names = [sae.hook_name for sae in encoder.saes]
        self.max_k = max(sae.max_k for sae in encoder.saes)

    def process_dataset(
        self,
        texts: List[str],
        save_dir: str,
        batch_size: int = 32,
        max_length: int = 512,
    ) -> None:
        """
        Process texts and save feature activations to disk

        Args:
            texts: List of text strings to process
            save_dir: Directory to save processed data
            batch_size: Batch size for processing
            max_length: Maximum sequence length
        """
        os.makedirs(save_dir, exist_ok=True)

        # Save config
        config = {
            "n_examples": len(texts),
            "max_length": max_length,
            "hook_names": self.hook_names,
        }
        with open(os.path.join(save_dir, "config.json"), "w") as f:
            json.dump(config, f)

        # Process texts in batches
        for i in tqdm(range(0, len(texts), batch_size), desc="Processing texts"):
            batch_texts = texts[i : i + batch_size]
            featurized_dicts = self.encoder.featurize_text(batch_texts, max_length=max_length)

            # Convert dictionary format to Example objects
            for featurized_dict in featurized_dicts:
                example = Example(
                    token_ids=featurized_dict["token_ids"],
                    str_tokens=featurized_dict["str_tokens"],
                    latent_data={
                        hook_name: (featurized_dict["top_indices"][hook_name], featurized_dict["top_acts"][hook_name])
                        for hook_name in self.hook_names
                    },
                )
                self.examples.append(example)

        # Save to disk
        self._save_to_disk(save_dir)

    def _save_to_disk(self, save_dir: str) -> None:
        """Save processed examples to disk in an efficient format"""
        n_examples = len(self.examples)
        max_length = max(len(ex.token_ids) for ex in self.examples)

        for hook_name in self.hook_names:
            # Create memmap file for each hook
            bytes_per_row = (
                8  # 8 bytes for token_id
                + 4 * self.max_k  # 4 bytes per feature index
                + 2 * self.max_k  # 2 bytes per activation value
            )

            mmap = np.memmap(
                os.path.join(save_dir, f"feature_data_{hook_name}.mmap"),
                dtype="uint8",
                mode="w+",
                shape=(n_examples, max_length, bytes_per_row),
            )

            # Save data for each example
            for i, example in enumerate(self.examples):
                seq_len = len(example.token_ids)
                # Reshape token data correctly
                token_data = np.array(example.token_ids[:seq_len], dtype=np.int64).view(np.uint8).reshape(-1, 8)
                indices, acts = example.latent_data[hook_name]

                # Handle padding if needed
                if seq_len < max_length:
                    padded_token_data = np.zeros((max_length, 8), dtype=np.uint8)
                    padded_token_data[:seq_len] = token_data
                    token_data = padded_token_data

                mmap[i, :seq_len, :8] = token_data[:seq_len]
                mmap[i, :seq_len, 8 : 8 + 4 * self.max_k] = (
                    indices.numpy().astype(np.int32).view(np.uint8).reshape(seq_len, -1)
                )
                mmap[i, :seq_len, 8 + 4 * self.max_k :] = (
                    acts.numpy().astype(np.float16).view(np.uint8).reshape(seq_len, -1)
                )

            mmap.flush()

    def load_from_disk(self, load_dir: str) -> None:
        """Load processed feature data from disk"""
        # Load config
        with open(os.path.join(load_dir, "config.json"), "r") as f:
            config = json.load(f)

        n_examples = config["n_examples"]
        max_length = config["max_length"]
        self.hook_names = config["hook_names"]

        # Load memmap files
        bytes_per_row = 8 + self.max_k * 6
        self.feature_data = {}
        for hook_name in self.hook_names:
            self.feature_data[hook_name] = np.memmap(
                os.path.join(load_dir, f"feature_data_{hook_name}.mmap"),
                dtype="uint8",
                mode="r",
                shape=(n_examples, max_length, bytes_per_row),
            )

    def load_example(self, example_idx: int) -> Example:
        """Load a single example from the database"""
        if self.feature_data is None:
            raise ValueError("Database not loaded. Call load_from_disk first.")

        # Load data for each hook
        latent_data = {}
        for hook_name, data in self.feature_data.items():
            # Extract token IDs, indices and activations
            tokens = data[example_idx, :, :8].view(np.int64).flatten()
            mask = tokens != self.encoder.tokenizer.pad_token_id
            tokens = tokens[mask]

            indices = data[example_idx, mask, 8 : 8 + 4 * self.max_k].view(np.int32)
            acts = data[example_idx, mask, 8 + 4 * self.max_k :].view(np.float16)

            latent_data[hook_name] = (torch.from_numpy(indices), torch.from_numpy(acts))

        # Create string tokens
        str_tokens = [self.encoder.tokenizer.decode([t]) for t in tokens]

        return Example(tokens.tolist(), str_tokens, latent_data)

    def get_feature_stats(self, hook_name: str, feature_id: int) -> Dict[str, float]:
        """Get activation statistics for a specific feature"""
        if self.feature_data is None:
            raise ValueError("Database not loaded. Call load_from_disk first.")

        data = self.feature_data[hook_name]
        indices = data[:, :, 8 : 8 + 4 * self.max_k].view(np.int32)
        acts = data[:, :, 8 + 4 * self.max_k :].view(np.float16)

        # Find where feature activates
        mask = indices == feature_id
        activations = acts[mask]

        return {
            "mean": float(np.mean(activations)),
            "std": float(np.std(activations)),
            "max": float(np.max(activations)),
            "activation_count": int(np.sum(mask)),
            "nonzero_examples": int(np.any(mask, axis=(1, 2)).sum()),
        }

    def get_top_activating_examples(self, hook_name: str, feature_id: int, n: int = 10) -> List[Example]:
        """Get the n examples where the feature has highest activation"""
        if self.feature_data is None:
            raise ValueError("Database not loaded. Call load_from_disk first.")

        data = self.feature_data[hook_name]
        indices = data[:, :, 8 : 8 + 4 * self.max_k].view(np.int32)
        acts = data[:, :, 8 + 4 * self.max_k :].view(np.float16)

        # Get max activation per example
        mask = indices == feature_id
        max_acts = np.max(np.where(mask, acts, -np.inf), axis=(1, 2))

        # Get top n examples
        top_indices = np.argsort(max_acts)[-n:][::-1]
        return [self.load_example(idx) for idx in top_indices]
    
    def get_common_features(self, hook_name: str, k: int = 1000) -> List[int]:
        """Get the k most commonly activating features across the dataset, ordered by frequency."""
        if self.feature_data is None:
            raise ValueError("Database not loaded. Call load_from_disk first.")
        
        # Get feature data for this hook
        data = self.feature_data[hook_name]
        indices = data[:, :, 8 : 8 + 4 * self.max_k].view(np.int32)
        acts = data[:, :, 8 + 4 * self.max_k :].view(np.float16)
        
        # Find the maximum feature index to determine array size
        max_feature_id = np.max(indices) + 1
        
        # Process in chunks to avoid memory issues
        chunk_size = 1024
        feature_counts = np.zeros(max_feature_id, dtype=np.int32)
        
        for i in tqdm(range(0, len(indices), chunk_size), desc="Finding common features"):
            chunk = indices[i:i + chunk_size]
            # Flatten and count unique features in chunk
            unique, counts = np.unique(chunk.reshape(-1), return_counts=True)
            # Add to overall counts
            feature_counts[unique] += counts
        
        # Get top k most common features
        k = min(k, len(feature_counts))  # Ensure k isn't larger than number of features
        top_k_indices = np.argsort(feature_counts)[-k:][::-1]
        return top_k_indices.tolist()  # Return as ordered list instead of set

    def get_feature_quantiles(
        self, 
        hook_name: str, 
        feature_id: int, 
        n_buckets: int = 8, 
        n_examples: int = 10
    ) -> Dict[Tuple[float, float], List[Example]]:
        """Get quantile-based activation distribution for a feature."""
        if self.feature_data is None:
            raise ValueError("Database not loaded. Call load_from_disk first.")
        
        # Get feature data
        data = self.feature_data[hook_name]
        indices = data[:, :, 8 : 8 + 4 * self.max_k].view(np.int32)
        acts = data[:, :, 8 + 4 * self.max_k :].view(np.float16)
        
        # Get max activation per example for this feature
        mask = indices == feature_id
        max_acts = np.max(np.where(mask, acts, 0), axis=(1, 2))
        
        # Get non-zero activations
        nonzero_acts = max_acts[max_acts > 0]
        if len(nonzero_acts) == 0:
            return {}  # Return empty dict if feature never activates
        
        # Calculate quantile thresholds
        quantiles = np.linspace(0, 1, n_buckets + 1)
        thresholds = np.unique(np.quantile(nonzero_acts, quantiles))
        
        result = {}
        for i in range(len(thresholds) - 1):
            # Get indices of activations within the current quantile range
            if i == len(thresholds) - 2:  # Last bucket includes upper bound
                bucket_mask = max_acts >= thresholds[i]
                upper_bound = thresholds[i + 1]
            else:
                bucket_mask = (max_acts >= thresholds[i]) & (max_acts < thresholds[i + 1])
                upper_bound = thresholds[i + 1]
            
            bucket_indices = np.where(bucket_mask)[0]
            
            # Randomly select n_examples indices from the bucket
            if len(bucket_indices) > n_examples:
                selected_indices = np.random.choice(
                    bucket_indices, 
                    size=n_examples, 
                    replace=False
                )
            else:
                selected_indices = bucket_indices
            
            # Load examples for the selected indices
            bucket_examples = [self.load_example(idx) for idx in selected_indices]
            result[(float(thresholds[i]), float(upper_bound))] = bucket_examples
        
        return result

    def get_feature_display(
        self,
        hook_name: str,
        feature_id: int,
        n_top_examples: int = 10,
        n_quantile_buckets: int = 8,
        n_examples_per_bucket: int = 5
    ) -> Dict[str, List[List[Tuple[str, float]]]]:
        """Get a structured display of feature activations across different activation ranges.
        
        Args:
            hook_name: Name of the hook layer
            feature_id: ID of the feature to analyze
            n_top_examples: Number of top activating examples to show
            n_quantile_buckets: Number of quantile buckets to create
            n_examples_per_bucket: Number of examples to show per bucket
            
        Returns:
            Dictionary with:
                - "Top Activations": List of token-activation pairs for highest activating examples
                - "Interval {i}": List of token-activation pairs for each quantile bucket
        """
        display_dict = {}
        
        # Get top activating examples
        top_examples = self.get_top_activating_examples(hook_name, feature_id, n=n_top_examples)
        display_dict["Top Activations"] = [
            example.to_list(hook_name, feature_id) for example in top_examples
        ]
        
        # Get quantile-based examples
        quantiles = self.get_feature_quantiles(
            hook_name, 
            feature_id, 
            n_buckets=n_quantile_buckets,
            n_examples=n_examples_per_bucket
        )
        
        # Add examples from each quantile bucket, in descending order
        for i, ((lower, upper), examples) in enumerate(sorted(quantiles.items(), reverse=True)):
            display_dict[f"Interval {i} - ({lower:.2f}, {upper:.2f})"] = [
                example.to_list(hook_name, feature_id) for example in examples
            ]
        
        return display_dict
