import json
import os
from dataclasses import dataclass
from typing import ClassVar, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from huggingface_hub import list_repo_files
from peft import AutoPeftModelForCausalLM
from torch import Tensor
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizer


@dataclass
class ModelConfig:
    """Configuration for model loading and inference"""

    torch_dtype: torch.dtype = torch.bfloat16
    device_map: str = "cuda"
    attn_implementation: Optional[str] = None
    requires_grad: bool = False
    max_length: int = 1024

    def get_attn_implementation(self, model_name: str) -> str:
        """Determine appropriate attention implementation for model"""
        if self.attn_implementation is not None:
            return self.attn_implementation
        return "eager" if any(x in model_name.lower() for x in ["gpt2", "gemma"]) else "flash_attention_2"


class LanguageModelWrapper:
    """Wrapper for language models providing convenient access to internal activations"""

    _loaded_models: ClassVar[Dict[str, PreTrainedModel]] = {}
    _loaded_tokenizers: ClassVar[Dict[str, PreTrainedTokenizer]] = {}

    def __init__(self, model_name: str, tokenizer_name: Optional[str] = None, config: Optional[ModelConfig] = None):
        """Initialize the model wrapper"""
        self.model_name = model_name
        self.tokenizer_name = tokenizer_name or model_name
        self.config = config or ModelConfig()
        self.model, self.tokenizer = self._load_model_and_tokenizer()

    def _load_model(self) -> PreTrainedModel:
        """Load model from cache or HuggingFace"""
        if self.model_name in self._loaded_models:
            return self._loaded_models[self.model_name]

        attn_impl = self.config.get_attn_implementation(self.model_name)

        # Check for PEFT adapter
        files = list_repo_files(self.model_name)
        has_adapter = any("adapter_config.json" in file for file in files)

        if has_adapter:
            model = (
                AutoPeftModelForCausalLM.from_pretrained(
                    self.model_name,
                    torch_dtype=self.config.torch_dtype,
                    low_cpu_mem_usage=True,
                    attn_implementation=attn_impl,
                    device_map=self.config.device_map,
                    trust_remote_code=True,
                )
                .merge_and_unload()
                .eval()
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=self.config.torch_dtype,
                low_cpu_mem_usage=True,
                attn_implementation=attn_impl,
                device_map=self.config.device_map,
                trust_remote_code=True,
            ).eval()

        if not self.config.requires_grad:
            model.requires_grad_(False)

        self._loaded_models[self.model_name] = model
        return model

    def _load_tokenizer(self) -> PreTrainedTokenizer:
        """Load tokenizer from cache or HuggingFace"""
        if self.tokenizer_name in self._loaded_tokenizers:
            return self._loaded_tokenizers[self.tokenizer_name]

        tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name)
        tokenizer.padding_side = "left"
        tokenizer.pad_token_id = tokenizer.eos_token_id

        self._loaded_tokenizers[self.tokenizer_name] = tokenizer
        return tokenizer

    def _load_model_and_tokenizer(self) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
        """Load both model and tokenizer"""
        model = self._load_model()
        tokenizer = self._load_tokenizer()
        model.generation_config.eos_token_id = tokenizer.eos_token_id
        return model, tokenizer

    def _get_valid_token_mask(
        self, tokens: Tensor, only_return_on_tokens_between: Tuple[Union[int, callable], Union[int, callable]]
    ) -> Tensor:
        """Get mask for tokens between specified tokens/predicates"""
        if tokens.dim() not in (1, 2):
            raise ValueError("Input tensor must be 1D or 2D")

        if tokens.dim() == 1:
            tokens = tokens.unsqueeze(0)

        batch_size, seq_length = tokens.shape
        start_token, end_token = only_return_on_tokens_between

        def match(seq_idx: int, token: int, tokens: Tensor, matcher: Union[int, callable]) -> bool:
            if callable(matcher):
                return matcher(seq_idx, token, tokens)
            return token == matcher

        mask = torch.zeros((batch_size, seq_length), dtype=torch.bool, device=tokens.device)

        for i in range(batch_size):
            include_indices = False
            for j in range(seq_length):
                token = tokens[i, j]
                if match(j, token.item(), tokens[i], start_token):
                    include_indices = True
                elif match(j, token.item(), tokens[i], end_token):
                    include_indices = False
                elif include_indices:
                    mask[i, j] = True

        return mask.squeeze(0) if tokens.dim() == 1 else mask

    def _get_residual_acts_unbatched(
        self, input_ids: Tensor, attention_mask: Optional[Tensor] = None, only_return_layers: Optional[List[int]] = None
    ) -> Dict[int, Tensor]:
        """Get residual stream activations for a single batch"""
        num_layers = self.model.config.num_hidden_layers

        layers_to_return = set(range(num_layers)) if only_return_layers is None else set(only_return_layers)
        layers_to_return = {layer for layer in layers_to_return if 0 <= layer < num_layers}

        outputs = self.model(input_ids, attention_mask=attention_mask, output_hidden_states=True, return_dict=True)

        # Skip embedding layer
        hidden_states = outputs.hidden_states[1:]
        return {layer: hidden_states[layer] for layer in layers_to_return}

    @torch.inference_mode()
    def get_model_residual_acts(
        self,
        text: Union[str, List[str]],
        batch_size: Optional[int] = None,
        max_length: Optional[int] = None,
        return_tokens: bool = False,
        use_memmap: Optional[str] = None,
        only_return_layers: Optional[List[int]] = None,
        only_return_on_tokens_between: Optional[Tuple] = None,
        verbose: bool = True,
    ) -> Union[Dict[int, Tensor], Tuple[Dict[int, Tensor], Dict[str, Tensor]]]:
        """
        Get residual stream activations for the input text.

        Args:
            text: Input text or list of texts
            batch_size: Batch size for processing
            max_length: Maximum sequence length
            return_tokens: Whether to return tokenized inputs
            use_memmap: Path for memory mapping large outputs
            only_return_layers: List of layer indices to return
            only_return_on_tokens_between: Tuple of (start_token, end_token) to mask outputs
            verbose: Whether to show progress bar

        Returns:
            Dictionary mapping layer indices to activations
            If return_tokens=True, also returns the tokenized inputs
        """
        max_length = max_length or self.config.max_length
        max_length = min(self.tokenizer.model_max_length, max_length)

        # Ensure text is list
        if isinstance(text, str):
            text = [text]

        # Tokenize input
        tokens = self.tokenizer(
            text,
            add_special_tokens=False,
            return_tensors="pt",
            padding="longest",
            max_length=max_length,
            truncation=True,
            return_attention_mask=True,
        )

        input_ids = tokens["input_ids"]
        attention_mask = tokens["attention_mask"]

        # Apply token mask if specified
        if only_return_on_tokens_between is not None:
            only_return_mask = self._get_valid_token_mask(input_ids, only_return_on_tokens_between)
            zero_positions_mask = attention_mask.clone()
            zero_positions_mask[~only_return_mask] = 0
        else:
            zero_positions_mask = attention_mask

        # Use full batch if not specified
        if batch_size is None:
            batch_size = input_ids.size(0)

        # Get model dimensions
        num_layers = self.model.config.num_hidden_layers
        hidden_dim = self.model.config.hidden_size

        # Calculate full activation shape
        full_shape = (input_ids.size(0), input_ids.size(1), hidden_dim)

        # Handle memmap setup if requested
        activations = self._setup_storage(
            use_memmap=use_memmap,
            full_shape=full_shape,
            num_layers=num_layers,
            only_return_layers=only_return_layers,
            input_ids=input_ids if return_tokens else None,
        )

        # Process batches
        for i in tqdm(range(0, input_ids.size(0), batch_size), disable=not verbose):
            batch_input_ids = input_ids[i : i + batch_size].to(self.model.device)
            batch_attention_mask = attention_mask[i : i + batch_size].to(self.model.device)
            batch_zero_positions_mask = zero_positions_mask[i : i + batch_size].to(self.model.device)

            # Get activations
            batch_acts = self._get_residual_acts_unbatched(batch_input_ids, batch_attention_mask, only_return_layers)

            # Apply attention mask
            masked_batch_acts = {
                layer: (act * batch_zero_positions_mask.unsqueeze(-1).to(act.dtype)).cpu().to(torch.float16)
                for layer, act in batch_acts.items()
            }

            # Store activations
            self._store_batch_activations(
                activations=activations, batch_acts=masked_batch_acts, batch_start=i, batch_end=i + batch_size
            )

        return (activations, tokens) if return_tokens else activations

    def _setup_storage(
        self,
        use_memmap: Optional[str],
        full_shape: Tuple[int, ...],
        num_layers: int,
        only_return_layers: Optional[List[int]] = None,
        input_ids: Optional[Tensor] = None,
    ) -> Union[Dict[int, np.memmap], Dict[int, Tensor]]:
        """Setup storage for activations (either memmap or tensor)"""
        if use_memmap:
            return self._setup_memmap_storage(
                memmap_path=use_memmap,
                full_shape=full_shape,
                num_layers=num_layers,
                only_return_layers=only_return_layers,
                input_ids=input_ids,
            )
        else:
            layers_to_return = range(num_layers) if only_return_layers is None else only_return_layers
            return {layer: torch.empty(full_shape, dtype=torch.float16, device="cpu") for layer in layers_to_return}

    def _setup_memmap_storage(
        self,
        memmap_path: str,
        full_shape: Tuple[int, ...],
        num_layers: int,
        only_return_layers: Optional[List[int]] = None,
        input_ids: Optional[Tensor] = None,
    ) -> Dict[int, np.memmap]:
        """Setup memmap storage for activations"""
        memmap_dir = os.path.dirname(memmap_path)
        os.makedirs(memmap_dir, exist_ok=True)

        metadata = {
            "num_layers": num_layers,
            "hidden_dim": full_shape[-1],
            "shape": full_shape,
            "dtype": "float16",
            "files": {},
        }

        if input_ids is not None:
            tokens_file = f"{memmap_path}_tokens.dat"
            tokens_memmap = np.memmap(tokens_file, dtype="int32", mode="w+", shape=input_ids.shape)
            tokens_memmap[:] = input_ids.numpy()
            metadata["tokens_file"] = os.path.basename(tokens_file)
            metadata["tokens_shape"] = input_ids.shape
            metadata["tokens_dtype"] = "int32"

        layer_memmaps = {}
        layers_to_return = range(num_layers) if only_return_layers is None else only_return_layers

        for layer in layers_to_return:
            memmap_file = f"{memmap_path}_residual_act_layer_{layer}.dat"
            memmap = np.memmap(memmap_file, dtype="float16", mode="w+", shape=full_shape)
            layer_memmaps[layer] = memmap
            metadata["files"][f"layer_{layer}"] = os.path.basename(memmap_file)

        metadata_file = f"{memmap_path}_metadata.json"
        with open(metadata_file, "w") as f:
            json.dump(metadata, f, indent=2)

        return layer_memmaps

    def _store_batch_activations(
        self,
        activations: Union[Dict[int, np.memmap], Dict[int, Tensor]],
        batch_acts: Dict[int, Tensor],
        batch_start: int,
        batch_end: int,
    ) -> None:
        """Store batch activations in the appropriate storage"""
        if isinstance(next(iter(activations.values())), np.memmap):
            for layer, act in batch_acts.items():
                activations[layer][batch_start:batch_end] = act.numpy()
        else:
            for layer, act in batch_acts.items():
                activations[layer][batch_start:batch_end] = act

    @classmethod
    def clear_cache(cls) -> None:
        """Clear all cached models and tokenizers"""
        cls._loaded_models.clear()
        cls._loaded_tokenizers.clear()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def clear_model_memory(self) -> None:
        """Clear this instance's model from GPU memory"""
        if hasattr(self, "model"):
            self.model.to("cpu")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def __enter__(self):
        """Context manager entry"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - cleanup resources"""
        self.clear_model_memory()

    def move_to_device(self, device: str) -> None:
        """Move model to specified device"""
        self.model.to(device)

    @property
    def device(self) -> torch.device:
        """Get the current device of the model"""
        return next(self.model.parameters()).device
