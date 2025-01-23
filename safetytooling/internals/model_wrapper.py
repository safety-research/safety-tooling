import json
import os
import warnings
from dataclasses import dataclass
from typing import Any, ClassVar, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from huggingface_hub import list_repo_files
from peft import AutoPeftModelForCausalLM
from torch import Tensor, nn
from tqdm.auto import tqdm
from transformers import (AutoModelForCausalLM, AutoTokenizer, PreTrainedModel,
                          PreTrainedTokenizer)


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


def extract_submodule(model: nn.Module, target_path: str) -> nn.Module:
    """Extract a submodule from the model given its path"""
    if not target_path:
        return model

    current_module = model
    for part in target_path.split("."):
        if hasattr(current_module, part):
            current_module = getattr(current_module, part)
        else:
            raise AttributeError(f"Module has no attribute '{part}'")
    return current_module


class LanguageModelWrapper:
    """Wrapper for language models providing convenient access to internal activations and interventions"""

    _loaded_models: ClassVar[Dict[str, PreTrainedModel]] = {}
    _loaded_tokenizers: ClassVar[Dict[str, PreTrainedTokenizer]] = {}

    def __init__(self, model_name: str, tokenizer_name: Optional[str] = None, config: Optional[ModelConfig] = None):
        """Initialize the model wrapper"""
        self.model_name = model_name
        self.tokenizer_name = tokenizer_name or model_name
        self.config = config or ModelConfig()
        self.model, self.tokenizer = self._load_model_and_tokenizer()
        self._active_hooks: Dict[str, List[torch.utils.hooks.RemovableHandle]] = {}

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

    def _create_intervention_hook(self, intervention_fn: callable) -> callable:
        """Create a hook function for interventions"""

        def hook_fn(module, input, output):
            if isinstance(output, tuple):
                # Apply intervention to the first element (usually hidden states)
                modified_first = intervention_fn(output[0])
                # Return a new tuple with the modified first element and the rest unchanged
                return (modified_first,) + output[1:]
            else:
                # If output is not a tuple, just modify and return it
                assert isinstance(output, torch.Tensor)
                return intervention_fn(output)

        return hook_fn

    def add_hook(self, hook_point: str, intervention_fn: callable) -> None:
        """Add a hook to a specific point in the model"""
        submodule = extract_submodule(self.model, hook_point)
        hook = submodule.register_forward_hook(self._create_intervention_hook(intervention_fn))

        if hook_point not in self._active_hooks:
            self._active_hooks[hook_point] = []
        self._active_hooks[hook_point].append(hook)

    def remove_hook(self, hook_point: str) -> None:
        """Remove all hooks from a specific point in the model"""
        if hook_point in self._active_hooks:
            for hook in self._active_hooks[hook_point]:
                hook.remove()
            del self._active_hooks[hook_point]

    def clear_hooks(self) -> None:
        """Remove all active hooks from the model"""
        for hooks in self._active_hooks.values():
            for hook in hooks:
                hook.remove()
        self._active_hooks.clear()

    def __call__(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None, **kwargs) -> Any:
        """Perform a forward pass through the model, using any active hooks"""
        with torch.autocast(device_type="cuda", dtype=next(self.model.parameters()).dtype):
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
        return outputs

    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        max_new_tokens: int = 20,
        **generation_kwargs,
    ) -> torch.Tensor:
        """Generate text using any active hooks"""
        with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=next(self.model.parameters()).dtype):
            outputs = self.model.generate(
                input_ids=input_ids, attention_mask=attention_mask, max_new_tokens=max_new_tokens, **generation_kwargs
            )
        return outputs

    def sample_generations(
        self,
        prompts: Union[str, List[str]],
        format_inputs: bool = True,
        batch_size: int = 4,
        system_prompt: Optional[str] = None,
        max_new_tokens: int = 20,
        **generation_kwargs,
    ) -> List[str]:
        """Generate text samples for given prompts, using any active hooks"""
        # Make sure prompts is a list
        if not isinstance(prompts, list):
            prompts = [prompts]

        # Format inputs if requested
        if format_inputs:
            prompts = self.format_inputs(prompts, system_prompt)

        all_generations = []
        # Process prompts in batches
        for i in range(0, len(prompts), batch_size):
            batch_prompts = prompts[i : i + batch_size]
            # Tokenize inputs
            inputs = self.tokenizer(
                batch_prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                add_special_tokens=False,
            )
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

            # Generate text
            outputs = self.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_new_tokens=max_new_tokens,
                **generation_kwargs,
            )

            # Decode generated text
            batch_generations = self.tokenizer.batch_decode(outputs)
            batch_generations = [gen.replace(self.tokenizer.pad_token, "") for gen in batch_generations]
            all_generations.extend(batch_generations)

        return all_generations

    def get_residual_acts(
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
        """Get residual stream activations for the input text"""
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

        # Setup storage (either memmap or regular tensors)
        activations = {}
        if use_memmap:
            # Create memmap files for storage
            os.makedirs(os.path.dirname(use_memmap), exist_ok=True)
            layers_to_return = range(num_layers) if only_return_layers is None else only_return_layers
            for layer in layers_to_return:
                memmap_file = f"{use_memmap}_layer_{layer}.dat"
                activations[layer] = np.memmap(memmap_file, dtype="float16", mode="w+", shape=full_shape)
        else:
            # Use regular tensors
            layers_to_return = range(num_layers) if only_return_layers is None else only_return_layers
            activations = {
                layer: torch.empty(full_shape, dtype=torch.float16, device="cpu") for layer in layers_to_return
            }

        # Process batches
        for i in tqdm(range(0, input_ids.size(0), batch_size), disable=not verbose):
            batch_input_ids = input_ids[i : i + batch_size].to(self.model.device)
            batch_attention_mask = attention_mask[i : i + batch_size].to(self.model.device)
            batch_zero_positions_mask = zero_positions_mask[i : i + batch_size].to(self.model.device)

            # Get hidden states
            outputs = self.model(
                batch_input_ids,
                attention_mask=batch_attention_mask,
                output_hidden_states=True,
                return_dict=True,
            )
            hidden_states = outputs.hidden_states[1:]  # Skip embedding layer

            # Store activations
            for layer in activations:
                masked_acts = (hidden_states[layer] * batch_zero_positions_mask.unsqueeze(-1)).cpu().to(torch.float16)

                if isinstance(activations[layer], np.memmap):
                    activations[layer][i : i + batch_size] = masked_acts.numpy()
                else:
                    activations[layer][i : i + batch_size] = masked_acts

        return (activations, tokens) if return_tokens else activations

    def format_inputs(
        self, prompt: Union[str, List[str]], system_prompt: Optional[str] = None
    ) -> Union[str, List[str]]:
        """Format input for language models using the tokenizer's chat template"""

        def format_single_input(single_prompt: str) -> str:
            # Format a single prompt with optional system message using the tokenizer's chat template
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": single_prompt})

            try:
                return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            except AttributeError:
                warnings.warn(
                    "The provided tokenizer does not have the 'apply_chat_template' method. "
                    "Falling back to a simple format. This may not be optimal for all models.",
                    UserWarning,
                )
                # Simple fallback format
                formatted = ""
                if system_prompt:
                    formatted += f"{system_prompt}\n"
                formatted += f"{single_prompt}"
                # Add BOS token if available
                bos_token = getattr(self.tokenizer, "bos_token", "")
                return f"{bos_token}{formatted}"

        if isinstance(prompt, str):
            return format_single_input(prompt)
        elif isinstance(prompt, list):
            return [format_single_input(p) for p in prompt]
        else:
            raise ValueError("prompt must be either a string or a list of strings")

    def clear_model_memory(self) -> None:
        """Clear this instance's model from GPU memory"""
        if hasattr(self, "model"):
            self.model.to("cpu")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    @classmethod
    def clear_cache(cls) -> None:
        """Clear all cached models and tokenizers"""
        cls._loaded_models.clear()
        cls._loaded_tokenizers.clear()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def move_to_device(self, device: str) -> None:
        """Move model to specified device"""
        self.model.to(device)

    @property
    def device(self) -> torch.device:
        """Get the current device of the model"""
        return next(self.model.parameters()).device

    def __enter__(self):
        """Context manager entry"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - cleanup resources"""
        self.clear_hooks()
        self.clear_model_memory()
