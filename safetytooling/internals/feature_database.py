"""Feature database for storing and analyzing SAE feature activations."""

import json
import os
from multiprocessing import Pool, cpu_count
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from tqdm.auto import tqdm

from .sae_wrappers import SparseAutoencoderWrapper

# Disable tokenizer parallelism since we handle it ourselves
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# ============================================================================
# Multiprocessing Worker Functions
# ============================================================================


def _process_chunk_features(args: Tuple) -> Tuple[np.ndarray, np.ndarray]:
    """Process a chunk of data for common features analysis."""
    (start_idx, end_idx), max_k, n_features, filepath, shape = args

    # Create memmap connection in worker
    data = np.memmap(filepath, dtype="uint8", mode="r+", shape=shape)
    chunk_data = data[start_idx:end_idx]

    # Convert raw data to indices and values
    indices = chunk_data[:, :, 8 : 8 + 4 * max_k].view(np.int32)
    values = chunk_data[:, :, 8 + 4 * max_k :].view(np.float16)

    # Flatten arrays
    flat_indices = indices.reshape(-1)
    flat_values = values.reshape(-1)

    # Get counts for this chunk
    chunk_counts = np.bincount(flat_indices, minlength=n_features)

    # Get max activations for this chunk
    chunk_max_acts = np.zeros(n_features, dtype=np.float16)
    np.maximum.at(chunk_max_acts, flat_indices, flat_values)

    return chunk_counts, chunk_max_acts


def _process_chunk_top_activating(args: Tuple) -> Tuple[np.ndarray, np.ndarray]:
    """Process a chunk of data to find top activating examples."""
    (start_idx, end_idx), max_k, feature_idx, filepath, shape = args

    # Create memmap connection in worker
    data = np.memmap(filepath, dtype="uint8", mode="r+", shape=shape)
    chunk_data = data[start_idx:end_idx]

    # Convert raw data to indices and values
    indices = chunk_data[:, :, 8 : 8 + 4 * max_k].view(np.int32)
    values = chunk_data[:, :, 8 + 4 * max_k :].view(np.float16)

    # Create mask for this feature
    feature_mask = indices == feature_idx

    # Get max activation per example for this feature
    max_acts = np.max(np.where(feature_mask, values, 0), axis=(1, 2))

    # Return chunk indices for tracking global position
    return np.arange(start_idx, end_idx), max_acts


def _process_chunk_quantiles(args: Tuple) -> Tuple[np.ndarray, np.ndarray]:
    """Process a chunk of data to get activation distribution.

    Args:
        args: Tuple of (indices, max_k, feature_idx, filepath, shape)

    Returns:
        Tuple of (chunk_indices, activations) for this chunk
    """
    (start_idx, end_idx), max_k, feature_idx, filepath, shape = args

    # Create memmap connection in worker
    data = np.memmap(filepath, dtype="uint8", mode="r+", shape=shape)
    chunk_data = data[start_idx:end_idx]

    # Convert raw data to indices and values
    indices = chunk_data[:, :, 8 : 8 + 4 * max_k].view(np.int32)
    values = chunk_data[:, :, 8 + 4 * max_k :].view(np.float16)

    # Create mask for this feature
    feature_mask = indices == feature_idx

    # Get max activation per example for this feature
    max_acts = np.max(np.where(feature_mask, values, 0), axis=(1, 2))

    # Return chunk indices and their max activations
    return np.arange(start_idx, end_idx), max_acts


# ============================================================================
# Main Database Class
# ============================================================================


class FeatureDatabase:
    """Database for storing and analyzing SAE feature activations across a dataset."""

    def __init__(
        self,
        encoder: SparseAutoencoderWrapper,
        chunk_size: int = 2000,
        n_workers: Optional[int] = 32,
        show_progress: bool = False,
    ):
        """Initialize the feature database.

        Args:
            encoder: Wrapper containing model and SAEs
            chunk_size: Number of examples to process at once
            n_workers: Number of parallel workers (defaults to 32)
            show_progress: Whether to show progress bars (defaults to False)
        """
        self.encoder = encoder
        self.feature_data = None
        self.hook_names = [sae.hook_name for sae in encoder.saes]
        self.max_k = max(sae.max_k for sae in encoder.saes)
        self.chunk_size = chunk_size
        self.n_workers = n_workers or int(cpu_count() * 0.75)
        self.show_progress = show_progress

    # ----------------------------------------------------------------------------
    # Data Processing and Storage
    # ----------------------------------------------------------------------------

    def process_dataset(
        self,
        texts: List[str],
        save_dir: str,
        max_length: int = 128,
        batch_size: int = 128,
    ) -> None:
        """Process texts and save feature activations to disk"""
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # Save dataset config
        config = {
            "n_examples": len(texts),
            "example_seq_len": max_length,
            "hook_names": self.hook_names,
        }
        with open(os.path.join(save_dir, "config.json"), "w") as f:
            json.dump(config, f)

        # Cache sae feature activations over tokens
        self._build_memmap(
            texts=texts,
            n_examples=len(texts),
            save_dir=save_dir,
            max_length=max_length,
            batch_size=batch_size,
        )

    def _build_memmap(
        self,
        texts: List[str],
        n_examples: int,
        save_dir: str,
        max_length: int,
        batch_size: int,
    ) -> None:
        """Build memory-mapped files for feature storage"""
        # Create memmap in directory for each hook
        bytes_per_row = 8 + self.max_k * 6  # (4 bytes for index, 2 bytes for activation)
        self.feature_data = {}
        for hook_name in self.hook_names:
            self.feature_data[hook_name] = np.memmap(
                os.path.join(save_dir, f"feature_data_{hook_name}.mmap"),
                dtype="uint8",
                mode="w+",
                shape=(n_examples, max_length, bytes_per_row),
            )

        # Process in batches
        ctr = 0
        for i in tqdm(range(0, n_examples, batch_size), desc="Caching SAE Activations"):
            batch_texts = texts[i : i + batch_size]

            # Get features using encoder
            featurized = self.encoder.featurize_text(batch_texts, max_length=max_length)

            # Save features for each hook
            for hook_name in self.hook_names:
                for j, feat_dict in enumerate(featurized):
                    idx = ctr + j
                    n_tokens = len(feat_dict["token_ids"])

                    # Save token ids
                    token_data = np.array(feat_dict["token_ids"], dtype=np.int64).view(np.uint8).reshape(-1, 8)
                    self.feature_data[hook_name][idx, :n_tokens, :8] = token_data

                    # Save indices and activations
                    indices = feat_dict["top_indices"][hook_name].numpy().astype(np.int32)
                    acts = feat_dict["top_acts"][hook_name].numpy().astype(np.float16)

                    self.feature_data[hook_name][idx, :n_tokens, 8 : 8 + 4 * self.max_k] = indices.view(
                        np.uint8
                    ).reshape(n_tokens, -1)
                    self.feature_data[hook_name][idx, :n_tokens, 8 + 4 * self.max_k :] = acts.view(np.uint8).reshape(
                        n_tokens, -1
                    )

                self.feature_data[hook_name].flush()
            ctr += len(featurized)

    def load_from_disk(self, load_dir: str) -> None:
        """Load processed feature data from disk"""
        self.load_dir = load_dir  # Store for worker processes
        if not os.path.exists(load_dir):
            raise Exception("Dataset not created!")

        # Load config
        with open(os.path.join(load_dir, "config.json"), "r") as f:
            config = json.load(f)
        self.n_examples = config["n_examples"]
        self.example_seq_len = config["example_seq_len"]
        self.hook_names = config["hook_names"]

        # Create memmap for each hook
        bytes_per_row = 8 + self.max_k * 6
        self.feature_data = {}
        for hook_name in self.hook_names:
            self.feature_data[hook_name] = np.memmap(
                os.path.join(load_dir, f"feature_data_{hook_name}.mmap"),
                dtype="uint8",
                mode="r+",  # Changed from 'r' to 'r+'
                shape=(self.n_examples, self.example_seq_len, bytes_per_row),
            )

    def _assert_loaded(self) -> None:
        """Check if the memmap has been loaded"""
        if self.feature_data is None:
            raise Exception("Memmap has not been loaded yet")

    def _get_single_example(
        self, example_idx: int, hook_name: str, zero_bos_acts: bool = True, load_to_memory: bool = False
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Get tokens, indices and values from feature data for a single example."""
        self._assert_loaded()

        # Get data for single example
        data = self.feature_data[hook_name][example_idx]
        tokens = data[:, 0:8].view(np.int64).squeeze(-1)
        indices = data[:, 8 : 8 + 4 * self.max_k].view(np.int32)
        values = data[:, 8 + 4 * self.max_k :].view(np.float16)

        # Load to memory if requested
        if load_to_memory:
            tokens = np.array(tokens)
            indices = np.array(indices)
            values = np.array(values)

        # Zero out BOS token activations if requested
        if zero_bos_acts:
            bos_mask = tokens == self.encoder.tokenizer.bos_token_id

            # Create copy if needed
            if not load_to_memory:
                values = np.array(values, copy=True)
            values[bos_mask] = 0

        return tokens, indices, values

    def _get_all_examples(
        self, hook_name: str, zero_bos_acts: bool = True, load_to_memory: bool = False
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Get tokens, indices and values from all examples"""
        self._assert_loaded()

        # Separates data into tokens, indices, and values
        data = self.feature_data[hook_name]
        tokens = data[:, :, 0:8].view(np.int64).squeeze(-1)
        indices = data[:, :, 8 : 8 + 4 * self.max_k].view(np.int32)
        values = data[:, :, 8 + 4 * self.max_k :].view(np.float16)

        # Load data to memory if requested
        if load_to_memory:
            tokens = np.array(tokens)
            indices = np.array(indices)
            values = np.array(values)

        # Zero out the feature activations on the bos tokens
        if zero_bos_acts:
            bos_mask = tokens == self.encoder.tokenizer.bos_token_id

            # Create a copy of values before modifying if it's not already in memory
            if not load_to_memory:
                values = np.array(values, copy=True)
            values[bos_mask] = 0

        return tokens, indices, values

    def load_example(self, example_idx: int) -> Dict[str, Any]:
        """Load example from disk"""
        # Get tokens and features for each hook
        top_indices = {}
        top_acts = {}
        token_ids = None

        for hook_name in self.hook_names:
            tokens, indices, values = self._get_single_example(
                example_idx=example_idx, hook_name=hook_name, zero_bos_acts=True, load_to_memory=True
            )

            # Store token IDs from first hook (they're the same for all hooks)
            if token_ids is None:
                # Remove padding tokens
                valid_mask = tokens != self.encoder.tokenizer.pad_token_id
                token_ids = tokens[valid_mask].tolist()

            # Store features for this hook
            top_indices[hook_name] = torch.from_numpy(indices[valid_mask])
            top_acts[hook_name] = torch.from_numpy(values[valid_mask])

        # Get string tokens
        str_tokens = [self.encoder.tokenizer.decode([t]) for t in token_ids]

        return {
            "token_ids": token_ids,
            "str_tokens": str_tokens,
            "top_indices": top_indices,
            "top_acts": top_acts,
            "text": "".join(str_tokens),
        }

    def get_common_features(self, hook_name: str, k: int = 1000) -> List[int]:
        """Get the k most commonly activating features across the dataset.

        Args:
            hook_name: Name of the hook layer
            k: Number of top features to return

        Returns:
            List of feature indices sorted by activation frequency
        """
        self._assert_loaded()

        # Get number of features from the relevant SAE
        n_features = next(sae for sae in self.encoder.saes if sae.hook_name == hook_name).n_features

        # Prepare chunks for parallel processing
        chunks = []
        filepath = os.path.join(self.load_dir, f"feature_data_{hook_name}.mmap")
        shape = (self.n_examples, self.example_seq_len, 8 + self.max_k * 6)

        for start_idx in range(0, self.n_examples, self.chunk_size):
            end_idx = min(start_idx + self.chunk_size, self.n_examples)
            chunks.append(((start_idx, end_idx), self.max_k, n_features, filepath, shape))

        # Process chunks in parallel
        with Pool(self.n_workers) as pool:
            imap_iter = pool.imap_unordered(_process_chunk_features, chunks)
            if self.show_progress:
                imap_iter = tqdm(imap_iter, total=len(chunks), desc="Finding common features")
            results = list(imap_iter)

        # Combine results from all chunks
        total_counts = np.zeros(n_features, dtype=np.int64)
        total_max_acts = np.zeros(n_features, dtype=np.float16)

        for chunk_counts, chunk_max_acts in results:
            total_counts += chunk_counts
            np.maximum(total_max_acts, chunk_max_acts, out=total_max_acts)

        # Weight counts by activation strength and find top-k
        weighted_scores = total_counts * np.maximum(total_max_acts, 0)
        k = min(k, len(weighted_scores))  # Ensure k isn't larger than number of features
        top_k_indices = np.argsort(weighted_scores)[-k:][::-1]

        return top_k_indices.tolist()

    def get_top_activating_examples(
        self, hook_name: str, feature_idx: int, n_examples: int = 10
    ) -> List[List[Tuple[str, float]]]:
        """Get the examples that activate a given feature the most.

        Args:
            hook_name: Name of the hook layer
            feature_idx: ID of the feature to analyze
            n_examples: Number of top examples to return

        Returns:
            List of examples, where each example is a list of (token, activation) pairs
        """
        self._assert_loaded()

        # Prepare chunks for parallel processing
        chunks = []
        filepath = os.path.join(self.load_dir, f"feature_data_{hook_name}.mmap")
        shape = (self.n_examples, self.example_seq_len, 8 + self.max_k * 6)

        for start_idx in range(0, self.n_examples, self.chunk_size):
            end_idx = min(start_idx + self.chunk_size, self.n_examples)
            chunks.append(((start_idx, end_idx), self.max_k, feature_idx, filepath, shape))

        # Process chunks in parallel
        with Pool(self.n_workers) as pool:
            imap_iter = pool.imap_unordered(_process_chunk_top_activating, chunks)
            if self.show_progress:
                imap_iter = tqdm(imap_iter, total=len(chunks), desc="Finding top activating examples")
            chunk_results = list(imap_iter)

        # Combine results and get top indices
        all_indices = np.concatenate([indices for indices, _ in chunk_results])
        all_acts = np.concatenate([acts for _, acts in chunk_results])

        # Get indices of top activating examples
        top_k_idx = np.argsort(all_acts)[-n_examples:][::-1]
        top_indices = all_indices[top_k_idx]

        # For each top example, get token-activation pairs
        results = []
        for idx in top_indices:
            # Get example data
            tokens, indices, acts = self._get_single_example(
                example_idx=idx, hook_name=hook_name, zero_bos_acts=True, load_to_memory=True
            )

            # Remove padding
            valid_mask = tokens != self.encoder.tokenizer.pad_token_id
            tokens = tokens[valid_mask]
            indices = indices[valid_mask]
            acts = acts[valid_mask]

            # Get feature activations for each token position
            feature_acts = np.zeros(len(tokens), dtype=np.float16)
            for i in range(len(tokens)):
                # Find where feature appears in top-k for this token
                feat_matches = indices[i] == feature_idx
                if feat_matches.any():
                    # Get max activation where feature appears
                    feature_acts[i] = acts[i][feat_matches].max()

            # Get string tokens
            str_tokens = [self.encoder.tokenizer.decode([t]) for t in tokens]

            # Create list of (token, activation) pairs
            token_acts = list(zip(str_tokens, feature_acts.tolist()))
            results.append(token_acts)

        return results

    def get_quantile_examples(
        self, hook_name: str, feature_idx: int, n_buckets: int = 6, n_examples: int = 10
    ) -> Dict[Tuple[float, float], List[List[Tuple[str, float]]]]:
        """Get examples distributed across activation quantiles for a feature."""
        self._assert_loaded()

        # Prepare chunks for parallel processing
        chunks = []
        filepath = os.path.join(self.load_dir, f"feature_data_{hook_name}.mmap")
        shape = (self.n_examples, self.example_seq_len, 8 + self.max_k * 6)

        for start_idx in range(0, self.n_examples, self.chunk_size):
            end_idx = min(start_idx + self.chunk_size, self.n_examples)
            chunks.append(((start_idx, end_idx), self.max_k, feature_idx, filepath, shape))

        # Process chunks in parallel
        with Pool(self.n_workers) as pool:
            imap_iter = pool.imap_unordered(_process_chunk_quantiles, chunks)
            if self.show_progress:
                imap_iter = tqdm(imap_iter, total=len(chunks), desc="Computing activation distribution")
            chunk_results = list(imap_iter)

        # Combine results
        all_indices = np.concatenate([indices for indices, _ in chunk_results])
        all_acts = np.concatenate([acts for _, acts in chunk_results])

        # Get non-zero activations and compute quantiles
        nonzero_mask = all_acts > 0
        if not nonzero_mask.any():
            return {}  # Return empty dict if feature never activates

        nonzero_acts = all_acts[nonzero_mask]

        # Calculate actual thresholds
        quantiles = np.linspace(0, 1, n_buckets + 1)
        thresholds = np.unique(np.quantile(nonzero_acts, quantiles))

        # If we have fewer unique thresholds than requested buckets, adjust n_buckets
        n_actual_buckets = len(thresholds) - 1
        if n_actual_buckets == 0:
            return {}  # Return empty dict if all activations are the same

        # Process each bucket
        result = {}
        for bucket_idx in range(n_actual_buckets):
            # Get indices of activations within the current quantile range
            if bucket_idx == n_actual_buckets - 1:  # Last bucket includes upper bound
                bucket_mask = all_acts >= thresholds[bucket_idx]
                upper_bound = thresholds[bucket_idx + 1]
            else:
                bucket_mask = (all_acts >= thresholds[bucket_idx]) & (all_acts < thresholds[bucket_idx + 1])
                upper_bound = thresholds[bucket_idx + 1]

            bucket_indices = all_indices[bucket_mask]

            # Randomly select examples from bucket
            if len(bucket_indices) > n_examples:
                selected_indices = np.random.choice(bucket_indices, size=n_examples, replace=False)
            else:
                selected_indices = bucket_indices

            # Get token-activation pairs for selected examples
            bucket_examples = []
            for idx in selected_indices:
                # Get example data
                tokens, indices, acts = self._get_single_example(
                    example_idx=idx, hook_name=hook_name, zero_bos_acts=True, load_to_memory=True
                )

                # Remove padding
                valid_mask = tokens != self.encoder.tokenizer.pad_token_id
                tokens = tokens[valid_mask]
                indices = indices[valid_mask]
                acts = acts[valid_mask]

                # Get feature activations for each token position
                feature_acts = np.zeros(len(tokens), dtype=np.float16)
                for token_idx in range(len(tokens)):
                    feat_matches = indices[token_idx] == feature_idx
                    if feat_matches.any():
                        feature_acts[token_idx] = acts[token_idx][feat_matches].max()

                # Get string tokens
                str_tokens = [self.encoder.tokenizer.decode([t]) for t in tokens]

                # Create list of (token, activation) pairs
                token_acts = list(zip(str_tokens, feature_acts.tolist()))
                bucket_examples.append(token_acts)

            result[(float(thresholds[bucket_idx]), float(upper_bound))] = bucket_examples

        return result
