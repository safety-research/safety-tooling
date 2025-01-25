import json
from typing import List, Optional, Tuple

import numpy as np
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
    def load_gemma2_sae(layer: int, l0: float, width: int = 131072, device: str = None) -> "DeepmindSparseAutoencoder":
        if device is None:
            device = get_least_used_device()
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

    def __init__(self, encoder: torch.nn.Module, hook_name: str, max_k: int, device=None):
        super().__init__(hook_name, encoder.d_hidden, max_k, device)
        self.encoder = encoder
        self._move_to_device(self.device)

    def _move_to_device(self, device):
        self.encoder = self.encoder.to(device)

    def reconstruct(self, acts: Tensor) -> Tensor:
        """Reconstruct input activations"""
        features = self.encoder.encode(acts)
        values, indices = features.topk(self.max_k, sorted=False)
        sparse_features = torch.zeros_like(features)
        sparse_features.scatter_(1, indices, values)
        return self.encoder.decode(sparse_features)

    def encode(self, acts: Tensor) -> Tuple[Tensor, Tensor]:
        """Get top-k feature activations"""
        features = self.encoder.encode(acts)
        values, indices = features.topk(self.max_k, sorted=False)
        return indices, values

    def get_codebook(self) -> Tensor:
        """Get feature directions"""
        return self.encoder.decoder_linear.weight.T

    def to(self, device: str) -> "GoodfireSparseAutoencoder":
        """Move SAE to specified device"""
        self.device = device  # This will trigger the device setter
        return self

    @staticmethod
    def load_sae(
        repo_id: str,
        layer: int,
        d_model: int,
        expansion_factor: int,
        device: str = None,
        dtype: torch.dtype = torch.bfloat16,
    ) -> "GoodfireSparseAutoencoder":
        """Load a pretrained SAE from HuggingFace Hub"""
        if device is None:
            device = get_least_used_device()
        # Create base encoder
        encoder = torch.nn.Module()
        encoder.d_in = d_model
        encoder.d_hidden = d_model * expansion_factor
        encoder.encoder_linear = torch.nn.Linear(d_model, encoder.d_hidden)
        encoder.decoder_linear = torch.nn.Linear(encoder.d_hidden, d_model)
        encoder.to(device=device, dtype=dtype)

        # Load weights
        weights = torch.load(
            hf_hub_download(repo_id=repo_id, filename=f"{repo_id.split('/')[-1]}.pth"),
            map_location=device,
        )
        encoder.load_state_dict(weights)

        # Create SAE wrapper
        return GoodfireSparseAutoencoder(
            encoder=encoder,
            hook_name=f"model.layers.{layer}",
            max_k=192,  # This seems to be standard for Goodfire SAEs
            device=device,
        )


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
                latent_acts = top_acts.reshape(n_batch, n_pos, -1).cpu()
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
