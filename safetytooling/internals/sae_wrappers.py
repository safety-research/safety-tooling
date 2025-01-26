import json
import os
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from torch import Tensor, nn

from .model_wrapper import LanguageModelWrapper, ModelConfig


def get_gpu_memory_usage():
    """Get memory usage for each GPU."""
    memory_usage = []
    for i in range(torch.cuda.device_count()):
        memory_allocated = torch.cuda.memory_allocated(i) / 1024**3  # Convert to GB
        memory_usage.append(memory_allocated)
    return memory_usage


def get_least_used_device():
    """Find GPU with lowest memory usage."""
    memory_usage = get_gpu_memory_usage()
    if not memory_usage:
        return "cpu"
    return f"cuda:{memory_usage.index(min(memory_usage))}"


class GenericEncoderDecoder(nn.Module):
    """Basic encoder-decoder module with linear encoder and decoder"""

    def __init__(self, d_in: int, d_sae: int, device="cpu", dtype=None):
        super().__init__()
        self.d_in = d_in
        self.d_sae = d_sae

        self.W_enc = nn.Parameter(torch.empty(d_in, d_sae, device=device, dtype=dtype))
        self.b_enc = nn.Parameter(torch.zeros(d_sae, device=device, dtype=dtype))
        self.W_dec = nn.Parameter(torch.empty(d_sae, d_in, device=device, dtype=dtype))
        self.b_dec = nn.Parameter(torch.zeros(d_in, device=device, dtype=dtype))

        nn.init.kaiming_uniform_(self.W_enc)
        nn.init.kaiming_uniform_(self.W_dec)

    def encode(self, x: Tensor) -> Tensor:
        return torch.relu(x @ self.W_enc + self.b_enc)

    def decode(self, indices: Tensor, values: Tensor) -> Tensor:
        features = torch.zeros((indices.shape[0], self.d_sae), device=values.device, dtype=values.dtype)
        features.scatter_(1, indices, values)
        return features @ self.W_dec + self.b_dec

    @property
    def feature_directions(self) -> Tensor:
        return self.W_dec.T


class DeepmindEncoderDecoder(GenericEncoderDecoder):
    """Deepmind's encoder-decoder with JumpReLU activation"""

    def __init__(self, d_in: int, d_sae: int, device="cpu", dtype=None):
        super().__init__(d_in, d_sae, device, dtype)
        self.threshold = nn.Parameter(torch.zeros(d_sae, device=device, dtype=dtype))

    def encode(self, x: Tensor) -> Tensor:
        pre_acts = x @ self.W_enc + self.b_enc
        return torch.relu(pre_acts) * (pre_acts > self.threshold)


class EleutherEncoderDecoder(nn.Module):
    """EleutherAI's encoder-decoder with skip connections"""

    def __init__(
        self, d_in: int, d_sae: int, device="cpu", dtype=None, *, normalize_decoder=True, skip_connection=False
    ):
        super().__init__()
        self.d_in = d_in
        self.d_sae = d_sae

        self.encoder = nn.Linear(d_in, d_sae, device=device, dtype=dtype)
        self.encoder.bias.data.zero_()
        self.W_dec = nn.Parameter(self.encoder.weight.data.clone())
        self.b_dec = nn.Parameter(torch.zeros(d_in, device=device, dtype=dtype))

        self.W_skip = nn.Parameter(torch.zeros(d_in, d_in, device=device, dtype=dtype)) if skip_connection else None

        if normalize_decoder:
            self._normalize_decoder()

    def _normalize_decoder(self):
        with torch.no_grad():
            norm = torch.norm(self.W_dec.data, dim=1, keepdim=True)
            self.W_dec.data /= norm + torch.finfo(self.W_dec.dtype).eps

    def encode(self, x: Tensor) -> Tensor:
        return torch.relu(self.encoder(x - self.b_dec))

    def decode(self, indices: Tensor, values: Tensor, x: Optional[Tensor] = None) -> Tensor:
        features = torch.zeros((indices.shape[0], self.d_sae), device=values.device, dtype=values.dtype)
        features.scatter_(1, indices, values)

        out = features @ self.W_dec + self.b_dec
        if self.W_skip is not None and x is not None:
            out = out + x @ self.W_skip
        return out

    @property
    def feature_directions(self) -> Tensor:
        return self.W_dec


class GoodfireEncoderDecoder(nn.Module):
    """Goodfire's encoder-decoder implementation"""

    def __init__(self, d_in: int, d_sae: int, device="cpu", dtype=None):
        super().__init__()
        self.d_in = d_in
        self.d_sae = d_sae

        # Initialize encoder and decoder components
        self.encoder_linear = nn.Linear(d_in, d_sae, device=device, dtype=dtype)
        self.decoder_linear = nn.Linear(d_sae, d_in, device=device, dtype=dtype)

    def encode(self, x: Tensor) -> Tensor:
        """Encode input using ReLU activation"""
        return torch.relu(self.encoder_linear(x))

    def decode(self, indices: Tensor, values: Tensor) -> Tensor:
        """Decode from sparse representation"""
        features = torch.zeros((indices.shape[0], self.d_sae), device=values.device, dtype=values.dtype)
        features.scatter_(1, indices, values)
        return self.decoder_linear(features)

    @property
    def feature_directions(self) -> Tensor:
        """Get feature directions from decoder weights"""
        return self.decoder_linear.weight.T


class SparseAutoencoderBase:
    """Base class for sparse autoencoders"""

    def __init__(self, hook_name: str, n_features: int, max_k: int, device=None):
        self.hook_name = hook_name
        self.n_features = n_features
        self.max_k = max_k
        self._device = device if device is not None else get_least_used_device()

    @property
    def device(self):
        return self._device

    @device.setter
    def device(self, value):
        self._device = value
        # Implement device movement in child classes
        self._move_to_device(self._device)

    def _move_to_device(self, device):
        """Move model components to device. Override in child classes."""
        pass

    def reconstruct(self, acts: Tensor) -> Tensor:
        raise NotImplementedError()

    def encode(self, acts: Tensor) -> Tuple[Tensor, Tensor]:
        raise NotImplementedError()

    def get_codebook(self) -> Tensor:
        raise NotImplementedError()

    def get_feature_description(self, idx: int) -> str:
        raise NotImplementedError()


class EleutherSparseAutoencoder(SparseAutoencoderBase):
    """EleutherAI's sparse autoencoder implementation"""

    def __init__(self, encoder: EleutherEncoderDecoder, hook_name: str, max_k: int, device=None):
        super().__init__(hook_name, encoder.d_sae, max_k, device)
        self.encoder = encoder
        self._move_to_device(self.device)

    def _move_to_device(self, device):
        self.encoder = self.encoder.to(device)

    def reconstruct(self, acts: Tensor) -> Tensor:
        pre = self.encoder.encode(acts)
        values, indices = pre.topk(self.max_k, sorted=False)
        return self.encoder.decode(indices.long(), values, acts)

    def encode(self, acts: Tensor) -> Tuple[Tensor, Tensor]:
        pre = self.encoder.encode(acts)
        values, indices = pre.topk(self.max_k, sorted=False)
        return indices.long(), values

    def get_codebook(self) -> Tensor:
        return self.encoder.feature_directions

    @staticmethod
    def load_llama3_sae(layer: int, v2: bool = True, device: str = None) -> "EleutherSparseAutoencoder":
        if device is None:
            device = get_least_used_device()
        repo_id = "EleutherAI/sae-llama-3-8b-32x-v2" if v2 else "EleutherAI/sae-llama-3-8b-32x"
        layer_path = f"layers.{layer}"

        # Load config and weights
        cfg = json.load(open(hf_hub_download(repo_id, f"{layer_path}/cfg.json")))
        weights = load_file(hf_hub_download(repo_id, f"{layer_path}/sae.safetensors"), device)

        # Initialize encoder
        encoder = EleutherEncoderDecoder(
            d_in=cfg["d_in"],
            d_sae=cfg.get("num_latents", None) or cfg["d_in"] * cfg.get("expansion_factor", 8.0),
            device=device,
            normalize_decoder=cfg.get("normalize_decoder", True),
            skip_connection=cfg.get("skip_connection", False),
        )

        # Load weights
        encoder.encoder.weight.data = weights["encoder.weight"]
        encoder.encoder.bias.data = weights["encoder.bias"]
        encoder.W_dec.data = weights["W_dec"]
        encoder.b_dec.data = weights["b_dec"]

        return EleutherSparseAutoencoder(encoder, f"model.layers.{layer}", cfg["k"], device)


class DeepmindSparseAutoencoder(SparseAutoencoderBase):
    """Deepmind's sparse autoencoder implementation"""

    def __init__(self, encoder: DeepmindEncoderDecoder, hook_name: str, max_k: int, device=None):
        super().__init__(hook_name, encoder.d_sae, max_k, device)
        self.encoder = encoder
        self._move_to_device(self.device)

    def _move_to_device(self, device):
        self.encoder = self.encoder.to(device)

    def reconstruct(self, acts: Tensor) -> Tensor:
        pre = self.encoder.encode(acts)
        values, indices = pre.topk(self.max_k, sorted=False)
        return self.encoder.decode(indices.long(), values)

    def encode(self, acts: Tensor) -> Tuple[Tensor, Tensor]:
        pre = self.encoder.encode(acts)
        values, indices = pre.topk(self.max_k, sorted=False)
        return indices.long(), values

    def get_codebook(self) -> Tensor:
        return self.encoder.feature_directions

    @staticmethod
    def load_gemma2_sae(
        layer: int, l0: float, width: int = 131072, instruction_trained=False, device: str = None
    ) -> "DeepmindSparseAutoencoder":
        if device is None:
            device = get_least_used_device()

        if instruction_trained:
            repo_id = "google/gemma-scope-9b-it-res"
        else:
            repo_id = "google/gemma-scope-9b-pt-res"
        filename = f"layer_{layer}/width_{width//10**3}k/average_l0_{l0}/params.npz"

        # Load weights from npz
        weights = dict(np.load(hf_hub_download(repo_id, filename)))

        # Create encoder
        encoder = DeepmindEncoderDecoder(d_in=6144, d_sae=width, device=device)
        encoder = encoder.to(torch.bfloat16)

        # Map weights
        for key, value in weights.items():
            if key.startswith("w_"):
                key = "W_" + key[2:]
            setattr(encoder, key, nn.Parameter(torch.tensor(value, device=device, dtype=torch.bfloat16)))

        return DeepmindSparseAutoencoder(encoder, f"model.layers.{layer}", 192, device)


class GoodfireSparseAutoencoder(SparseAutoencoderBase):
    """Goodfire's sparse autoencoder implementation"""

    def __init__(self, model_name: str, encoder: GoodfireEncoderDecoder, hook_name: str, max_k: int, device=None):
        super().__init__(hook_name, encoder.d_sae, max_k, device)
        self.encoder = encoder
        self._move_to_device(self.device)
        self.explanations = self._load_explanations(model_name)

    def _load_explanations(self, model_name: str) -> pd.DataFrame:
        """Load explanations from Goodfire API.

        Args:
            model_name: Name of the model to load explanations for

        Returns:
            DataFrame containing feature explanations
        """
        if model_name == "Llama-3.1-8B-Instruct":
            file_name = "goodfire_8b.csv"
        elif model_name == "Llama-3.3-70B-Instruct":
            file_name = "goodfire_70b.csv"
        else:
            raise ValueError(f"Model {model_name} not supported")

        # Get the absolute path to the file
        file_path = Path(os.path.dirname(__file__)) / "feature_annotations" / file_name

        # Load and return the CSV file
        if file_path.exists():
            return pd.read_csv(file_path)
        else:
            raise FileNotFoundError(f"Feature annotations file not found at {file_path}")

    def _move_to_device(self, device):
        """Move encoder to specified device"""
        self.encoder = self.encoder.to(device)

    def reconstruct(self, acts: Tensor) -> Tensor:
        """Reconstruct input from activations"""
        pre = self.encoder.encode(acts)
        values, indices = pre.topk(self.max_k, sorted=False)
        return self.encoder.decode(indices.long(), values)

    def encode(self, acts: Tensor) -> Tuple[Tensor, Tensor]:
        """Encode input to sparse representation"""
        pre = self.encoder.encode(acts)
        values, indices = pre.topk(self.max_k, sorted=False)
        return indices.long(), values

    def get_codebook(self) -> Tensor:
        """Get feature directions"""
        return self.encoder.feature_directions

    def get_feature_description(self, feature_id: int) -> str:
        """Get the description for a feature.

        Args:
            feature_id: Index of the feature

        Returns:
            Description of the feature
        """
        if self.explanations is None:
            return f"Feature {feature_id}"

        try:
            # Find the row where feature_id matches
            row = self.explanations[self.explanations["feature_id"] == feature_id]
            if not row.empty:
                return row.iloc[0]["feature_label"]
            return f"Feature {feature_id}"
        except (KeyError, IndexError):
            return f"Feature {feature_id}"

    @staticmethod
    def load_llama3_sae(
        model_name: str, layer: int, max_k: int = 192, device: Optional[str] = None
    ) -> "GoodfireSparseAutoencoder":
        """Load a Goodfire SAE for Llama 3 models

        Args:
            model_name: Name of the Llama model
            layer: Layer number for the SAE
            expansion_factor: Expansion factor for SAE dimensions
            device: Device to load the model on
        """
        if device is None:
            device = get_least_used_device()

        # Format SAE name following Goodfire's convention
        model_short_name = model_name.split("/")[-1]
        sae_name = f"{model_short_name}-SAE-l{layer}"

        # Download weights from HuggingFace
        file_path = hf_hub_download(repo_id=f"Goodfire/{sae_name}", filename=f"{sae_name}.pth", repo_type="model")

        # Load state dict
        state_dict = torch.load(file_path, map_location=device)

        # Infer dimensions from state dict
        encoder_weight = state_dict["encoder_linear.weight"]
        d_in = encoder_weight.shape[1]
        d_sae = encoder_weight.shape[0]

        # Initialize encoder
        encoder = GoodfireEncoderDecoder(
            d_in=d_in,
            d_sae=d_sae,
            device=device,
            dtype=torch.bfloat16,
        )

        # Load weights
        encoder.load_state_dict(state_dict)

        return GoodfireSparseAutoencoder(model_name, encoder, f"model.layers.{layer}", max_k, device)


class SparseAutoencoderWrapper(LanguageModelWrapper):
    """Wrapper for running inference with multiple SAEs"""

    def __init__(
        self,
        model_name: str,
        saes: List[SparseAutoencoderBase],
        tokenizer_name: Optional[str] = None,
        config: Optional[ModelConfig] = None,
    ):
        super().__init__(model_name, tokenizer_name, config)
        self.saes = saes
        self._current_activations = {}
        self._setup_hooks()

    def _setup_hooks(self):
        """Setup hooks for each SAE"""

        def create_activation_hook(hook_name: str) -> callable:
            def hook_fn(output: Tensor) -> Tensor:
                self._current_activations[hook_name] = output
                return output

            return hook_fn

        for sae in self.saes:
            self.add_hook(sae.hook_name, create_activation_hook(sae.hook_name))

    @torch.inference_mode()
    def _featurize(self, tokens: torch.Tensor, masks: Optional[torch.Tensor] = None) -> dict:
        """
        Returns a dictionary with hook_name as key and a tuple of two B x P x k tensors of feature activations as value.
        Nonactivating features will be zero.

        Args:
            tokens: Input tokens tensor of shape (batch_size, seq_len)
            masks: Optional attention mask tensor of shape (batch_size, seq_len)

        Returns:
            Dictionary mapping hook_names to tuples of (indices, activations)
        """
        n_batch, n_pos = tokens.shape
        results = {}

        with torch.no_grad():
            # Forward pass through model to get activations
            _ = self.model(
                input_ids=tokens.to(self.model.device),
                attention_mask=masks.to(self.model.device) if masks is not None else None,
            )

            # Process activations for each SAE
            for sae in self.saes:
                model_acts = self._current_activations[sae.hook_name]
                # Move activations to SAE's device
                model_acts = model_acts.to(sae.device)
                top_indices, top_acts = sae.encode(model_acts.flatten(0, 1))
                # Move results back to CPU
                latent_indices = top_indices.reshape(n_batch, n_pos, -1).cpu()
                latent_acts = (
                    top_acts.reshape(n_batch, n_pos, -1).cpu().to(torch.float16)
                )  # Convert to float16 to preserve compatibility with other models)
                results[sae.hook_name] = (latent_indices, latent_acts)

        return results

    @torch.inference_mode()
    def _batched_featurize(
        self, tokens: torch.Tensor, masks: Optional[torch.Tensor] = None, batch_size: Optional[int] = None
    ) -> dict:
        """
        Batched version of featurize.

        Args:
            tokens: Input tokens tensor
            masks: Optional attention mask tensor
            batch_size: Size of batches to process. If None, uses full input size.

        Returns:
            Dictionary mapping hook_names to tuples of (indices, activations)
        """
        if batch_size is None:
            return self._featurize(tokens, masks)

        minibatches_tokens = tokens.split(batch_size)
        minibatches_masks = masks.split(batch_size) if masks is not None else [None] * len(minibatches_tokens)

        # Initialize results dictionary
        results = {sae.hook_name: ([], []) for sae in self.saes}

        # Process each minibatch
        for minibatch_tokens, minibatch_masks in zip(minibatches_tokens, minibatches_masks):
            batch_results = self._featurize(minibatch_tokens, minibatch_masks)

            # Accumulate results for each SAE
            for hook_name, (indices, acts) in batch_results.items():
                results[hook_name][0].append(indices)
                results[hook_name][1].append(acts)

        # Concatenate results for each SAE
        final_results = {}
        for hook_name, (indices_list, acts_list) in results.items():
            final_results[hook_name] = (torch.cat(indices_list, dim=0), torch.cat(acts_list, dim=0))

        return final_results

    def featurize_text(self, text: str, batch_size: Optional[int] = None, max_length: int = 512) -> List[dict]:
        """
        Tokenize and featurize the input text.

        Args:
            text: Input text or list of texts
            batch_size: Optional batch size for processing
            max_length: Maximum sequence length (default: 512)

        Returns:
            List of dictionaries, one per example, containing:
            - token_ids: List of token IDs (padding removed)
            - str_tokens: List of string representations of tokens (padding removed)
            - top_indices: Dictionary mapping hook_names to top_indices with padding removed
            - top_acts: Dictionary mapping hook_names to top_acts with padding removed
        """
        max_length = min(self.tokenizer.model_max_length, max_length)
        tokens = self.tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=max_length,
            truncation=True,
            return_attention_mask=True,
            add_special_tokens=False,
        )

        input_ids = tokens["input_ids"]
        attention_mask = tokens["attention_mask"]

        # Get features from model
        if batch_size is None:
            features = self._featurize(input_ids, attention_mask)
        else:
            features = self._batched_featurize(input_ids, attention_mask, batch_size)

        # Convert to list of per-example dictionaries
        results = []
        batch_size = input_ids.shape[0]

        for i in range(batch_size):
            # Use attention mask to get valid token positions
            valid_positions = attention_mask[i].bool()

            # Get token IDs for this example (removing padding)
            token_ids = input_ids[i][valid_positions].tolist()

            # Get string representations of tokens (removing padding)
            str_tokens = [self.tokenizer.decode([tid]) for tid in token_ids]

            # Get features for this example (removing padding)
            example_indices = {}
            example_acts = {}
            for hook_name, (indices, acts) in features.items():
                example_indices[hook_name] = indices[i][valid_positions]  # Only keep non-padding positions
                example_acts[hook_name] = acts[i][valid_positions]  # Only keep non-padding positions

            results.append(
                {
                    "token_ids": token_ids,
                    "str_tokens": str_tokens,
                    "top_indices": example_indices,
                    "top_acts": example_acts,
                }
            )

        return results

    def get_feature_description(self, hook_name: str, feature_id: int) -> str:
        """Get the description for a feature.

        Args:
            hook_name: Name of the hook
            feature_id: Index of the feature
        """
        for sae in self.saes:
            if sae.hook_name == hook_name:
                return sae.get_feature_description(feature_id)
        raise ValueError(f"Hook name {hook_name} not found")


def load_eleuther_llama3_sae_wrapped(
    layers: List[int],
    v2: bool = True,
    model_name: str = "meta-llama/Meta-Llama-3-8B-Instruct",
    config: Optional[ModelConfig] = None,
) -> SparseAutoencoderWrapper:
    """Load EleutherAI's Llama3 SAEs and wrap them with a model.

    Args:
        layers: List of layer numbers to load SAEs for
        v2: Whether to use v2 SAEs (default True)
        model_name: Name of the Llama model to load (default: Meta-Llama-3-8B-Instruct)
        config: Optional model configuration. If None, uses default config with SDPA and bfloat16

    Returns:
        Wrapped model with loaded SAEs
    """
    # Load SAEs
    saes = [EleutherSparseAutoencoder.load_llama3_sae(layer=layer, v2=v2) for layer in layers]

    # Create default config if none provided
    if config is None:
        config = ModelConfig(
            attn_implementation="sdpa",
            torch_dtype=torch.bfloat16,
        )

    # Create and return wrapped model
    return SparseAutoencoderWrapper(model_name=model_name, saes=saes, config=config)


def load_deepmind_gemma2_sae_wrapped(
    layers: List[int],
    l0s: List[int],
    width: int = 131072,
    model_name: str = "google/gemma-2b-it",
    instruction_trained: bool = True,
    config: Optional[ModelConfig] = None,
) -> SparseAutoencoderWrapper:
    """Load Deepmind's Gemma2 SAEs and wrap them with a model.

    Args:
        layers: List of layer numbers to load SAEs for
        l0: L0 parameter for the SAEs (default: 128)
        width: Width of the SAEs (default: 131072)
        model_name: Name of the Gemma model to load (default: google/gemma-2b-it)
        instruction_trained: Whether to use instruction-tuned SAEs (default: True)
        config: Optional model configuration. If None, uses default config with bfloat16

    Returns:
        Wrapped model with loaded SAEs
    """
    # Load SAEs
    saes = [
        DeepmindSparseAutoencoder.load_gemma2_sae(
            layer=layer, l0=l0, width=width, instruction_trained=instruction_trained
        )
        for layer, l0 in zip(layers, l0s)
    ]

    # Create default config if none provided
    if config is None:
        config = ModelConfig(
            torch_dtype=torch.bfloat16,
        )

    # Create and return wrapped model
    return SparseAutoencoderWrapper(model_name=model_name, saes=saes, config=config)


def load_goodfire_llama3_8b_sae_wrapped(
    model_name: str = "meta-llama/Meta-Llama-3.1-8B-Instruct",
    config: Optional[ModelConfig] = None,
) -> SparseAutoencoderWrapper:
    """Load Goodfire's Llama3 8B SAE and wrap it with a model.

    Args:
        model_name: Name of the Llama model to load (default: Meta-Llama-3.1-8B-Instruct)
        config: Optional model configuration. If None, uses default config with SDPA and bfloat16

    Returns:
        Wrapped model with loaded SAE
    """
    # Load SAE for layer 19
    saes = [GoodfireSparseAutoencoder.load_llama3_sae(model_name="Llama-3.1-8B-Instruct", layer=19, max_k=192)]

    # Create default config if none provided
    if config is None:
        config = ModelConfig(
            attn_implementation="sdpa",
            torch_dtype=torch.bfloat16,
        )

    # Create and return wrapped model
    return SparseAutoencoderWrapper(model_name=model_name, saes=saes, config=config)


def load_goodfire_llama3_70b_sae_wrapped(
    model_name: str = "meta-llama/Llama-3.3-70B-Instruct",
    config: Optional[ModelConfig] = None,
) -> SparseAutoencoderWrapper:
    """Load Goodfire's Llama3 70B SAE and wrap it with a model.

    Args:
        model_name: Name of the Llama model to load (default: Llama-3.3-70B-Instruct)
        config: Optional model configuration. If None, uses default config with SDPA and bfloat16

    Returns:
        Wrapped model with loaded SAE
    """
    # Load SAE for layer 50
    saes = [GoodfireSparseAutoencoder.load_llama3_sae(model_name="Llama-3.3-70B-Instruct", layer=50, max_k=192)]

    # Create default config if none provided
    if config is None:
        config = ModelConfig(
            attn_implementation="sdpa",
            torch_dtype=torch.bfloat16,
        )

    # Create and return wrapped model
    return SparseAutoencoderWrapper(model_name=model_name, saes=saes, config=config)
