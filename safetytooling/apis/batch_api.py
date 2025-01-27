import asyncio
import logging
import os
from pathlib import Path
from typing import Literal

from safetytooling.apis.inference.anthropic import ANTHROPIC_MODELS, AnthropicModelBatch
from safetytooling.apis.inference.cache_manager import get_cache_manager
from safetytooling.apis.inference.openai.batch_api import OpenAIModelBatch
from safetytooling.apis.inference.openai.utils import GPT_CHAT_MODELS
from safetytooling.data_models import LLMParams, LLMResponse, Prompt
from safetytooling.utils.utils import load_secrets

LOGGER = logging.getLogger(__name__)


class BatchInferenceAPI:
    """A wrapper around batch APIs that can do a bunch of model calls at once."""

    def __init__(
        self,
        log_dir: Path | Literal["default"] = "default",
        prompt_history_dir: Path | Literal["default"] | None = "default",
        cache_dir: Path | Literal["default"] | None = "default",
        use_redis: bool = False,
        anthropic_api_key: str | None = None,
        openai_api_key: str | None = None,
        no_cache: bool = False,
        chunk: int | None = None,
    ):
        """Initialize the BatchInferenceAPI.

        Args:
            log_dir: Directory to store prompts before passing them to the batchapi. Set to "default" to use your prompt_history_dir as log_dir.
            prompt_history_dir: Directory to store prompt history. Set to None to disable.
                Set to "default" to use the default directory.
            cache_dir: Directory to store cache. Set to None to disable.
                Set to "default" to use the default directory.
            use_redis: Whether to use Redis for caching.
            anthropic_api_key: Optional API key for Anthropic. If not provided, will try to load from environment.
            openai_api_key: Optional API key for OpenAI. If not provided, will try to load from environment.
            no_cache: If True, disable caching regardless of other settings.
            chunk: Maximum number of prompts to send in a single batch. If None, send all prompts in one batch.
        """
        secrets = load_secrets("SECRETS")

        # Check NO_CACHE environment variable
        self.no_cache = no_cache or os.environ.get("NO_CACHE", "false").lower() == "true"

        if log_dir == "default":
            if "PROMPT_HISTORY_DIR" in secrets:
                self.log_dir = Path(secrets["PROMPT_HISTORY_DIR"])
            else:
                self.log_dir = Path.home() / ".prompt_history"
        else:
            assert isinstance(log_dir, Path) or log_dir is None
            self.log_dir = log_dir

        if cache_dir == "default":
            if "CACHE_DIR" in secrets:
                self.cache_dir = Path(secrets["CACHE_DIR"])
            else:
                self.cache_dir = Path.home() / ".cache"
        else:
            assert isinstance(cache_dir, Path) or cache_dir is None
            self.cache_dir = cache_dir

        self.cache_manager = None
        if self.cache_dir is not None and not self.no_cache:
            self.cache_manager = get_cache_manager(self.cache_dir, use_redis)

        self._anthropic_batch = AnthropicModelBatch(anthropic_api_key=anthropic_api_key)
        self._openai_batch = OpenAIModelBatch(openai_api_key=openai_api_key)
        self.chunk = chunk

    async def __call__(
        self,
        model_id: str,
        prompts: list[Prompt],
        log_dir: Path | None = None,
        use_cache: bool = True,
        max_tokens: int | None = None,
        **kwargs,
    ) -> tuple[list[LLMResponse], str]:
        """Make batch API requests for the specified model and prompts.

        Args:
            model_id: The model to call. Must be a Claude or OpenAI model.
            prompts: List of prompts to send to the model.
            log_dir: Directory to store logs. If None, will use prompt_history_dir.
            use_cache: Whether to use caching. Ignored if no_cache was set during initialization.
            max_tokens: Maximum number of tokens to generate.
            **kwargs: Additional arguments to pass to the model.

        Returns:
            Tuple of (list of responses, batch ID)
        """
        if self.no_cache:
            use_cache = False

        if not prompts:
            raise ValueError("No prompts provided")

        if log_dir is None:
            log_dir = self.log_dir

        if log_dir is not None:
            log_dir.mkdir(parents=True, exist_ok=True)

        llm_params = LLMParams(model=model_id, **kwargs)
        cached_responses = []
        cached_indices = []
        uncached_prompts = []
        uncached_indices = []

        # Check cache for each prompt only if caching is enabled
        if self.cache_manager is not None and use_cache:
            if "n" in kwargs:
                assert kwargs["n"] == 1, "n must be 1 for batch inference + cache"
            for i, prompt in enumerate(prompts):
                cached_result = self.cache_manager.maybe_load_cache(prompt=prompt, params=llm_params)
                if cached_result is not None:
                    cached_responses.append(cached_result.responses[0])  # Assuming single response per prompt
                    cached_indices.append(i)
                else:
                    uncached_prompts.append(prompt)
                    uncached_indices.append(i)
            print(f"Retrieving {len(cached_responses)}/{len(prompts)} from cache")
        else:
            uncached_prompts = prompts
            uncached_indices = list(range(len(prompts)))

        # If all prompts are cached, return cached responses
        if not uncached_prompts:
            return cached_responses, "cached"

        # Process uncached prompts in chunks if specified
        if self.chunk is not None and len(uncached_prompts) > self.chunk:
            all_responses = []
            all_batch_ids = []

            if model_id not in ANTHROPIC_MODELS and model_id not in GPT_CHAT_MODELS:
                raise ValueError(f"Model {model_id} is not supported. Only Claude and OpenAI models are supported.")

            chunks = [uncached_prompts[i : i + self.chunk] for i in range(0, len(uncached_prompts), self.chunk)]

            async def process_chunk(chunk_prompts):
                if model_id in ANTHROPIC_MODELS:
                    return await self._anthropic_batch(
                        model_id=model_id, prompts=chunk_prompts, max_tokens=max_tokens, **kwargs
                    )
                else:
                    return await self._openai_batch(
                        model_id=model_id, prompts=chunk_prompts, log_dir=log_dir, max_tokens=max_tokens, **kwargs
                    )

            results = await asyncio.gather(*[process_chunk(chunk) for chunk in chunks])

            for chunk_prompts, (responses, batch_id) in zip(chunks, results):
                all_responses.extend(responses)
                all_batch_ids.append(batch_id)

                # Save responses to cache
                if self.cache_manager is not None and use_cache:
                    for prompt, response in zip(chunk_prompts, responses):
                        if response is not None:
                            self.cache_manager.save_cache(prompt=prompt, params=llm_params, responses=[response])

            # Merge cached and new responses
            final_responses = [None] * len(prompts)
            for cached_response, idx in zip(cached_responses, cached_indices):
                final_responses[idx] = cached_response
            for idx, response in zip(uncached_indices, all_responses):
                final_responses[idx] = response

            if use_cache:
                print(
                    f"{len(cached_responses)}/{len(prompts)} retrieved from cache, "
                    f"{sum(response is None for response in final_responses)}/{len(prompts)} failed, "
                    f"{sum(response is not None for response in final_responses)}/{len(prompts)} succeeded in batch API requests {all_batch_ids}"
                )

            return final_responses, ",".join(all_batch_ids)
        else:
            # Process all uncached prompts in a single batch
            if model_id in ANTHROPIC_MODELS:
                responses, batch_id = await self._anthropic_batch(
                    model_id=model_id, prompts=uncached_prompts, max_tokens=max_tokens, **kwargs
                )
            elif model_id in GPT_CHAT_MODELS:
                responses, batch_id = await self._openai_batch(
                    model_id=model_id, prompts=uncached_prompts, log_dir=log_dir, max_tokens=max_tokens, **kwargs
                )
            else:
                raise ValueError(f"Model {model_id} is not supported. Only Claude and OpenAI models are supported.")

            # Save responses to cache
            if self.cache_manager is not None and use_cache:
                for prompt, response in zip(uncached_prompts, responses):
                    if response is not None:
                        self.cache_manager.save_cache(prompt=prompt, params=llm_params, responses=[response])

            # Merge cached and new responses
            final_responses = [None] * len(prompts)
            for cached_response, idx in zip(cached_responses, cached_indices):
                final_responses[idx] = cached_response
            for idx, response in zip(uncached_indices, responses):
                final_responses[idx] = response

            if use_cache:
                print(
                    f"{len(cached_responses)}/{len(prompts)} retrieved from cache, "
                    f"{sum(response is None for response in final_responses)}/{len(prompts)} failed, "
                    f"{sum(response is not None for response in final_responses)}/{len(prompts)} succeeded in batch API request {batch_id}"
                )

            return final_responses, batch_id
