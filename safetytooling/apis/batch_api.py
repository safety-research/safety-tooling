import logging
from pathlib import Path
from typing import Literal

from safetytooling.apis.inference.anthropic import ANTHROPIC_MODELS, AnthropicModelBatch
from safetytooling.apis.inference.openai.batch_api import OpenAIModelBatch
from safetytooling.apis.inference.openai.utils import GPT_CHAT_MODELS
from safetytooling.data_models import LLMResponse, Prompt
from safetytooling.utils.utils import load_secrets

LOGGER = logging.getLogger(__name__)


class BatchInferenceAPI:
    """A wrapper around batch APIs that can do a bunch of model calls at once."""

    def __init__(
        self,
        log_dir: Path | Literal["default"] = "default",
        anthropic_api_key: str | None = None,
        openai_api_key: str | None = None,
    ):
        """Initialize the BatchInferenceAPI.

        Args:
            log_dir: Directory to store prompts before passing them to the batchapi. Set to "default" to use your prompt_history_dir as log_dir.
            anthropic_api_key: Optional API key for Anthropic. If not provided, will try to load from environment.
            openai_api_key: Optional API key for OpenAI. If not provided, will try to load from environment.
        """
        secrets = load_secrets("SECRETS")
        if log_dir == "default":
            if "PROMPT_HISTORY_DIR" in secrets:
                self.log_dir = Path(secrets["PROMPT_HISTORY_DIR"])
            else:
                self.log_dir = Path.home() / ".prompt_history"
        else:
            assert isinstance(log_dir, Path) or log_dir is None
            self.log_dir = log_dir

        self._anthropic_batch = AnthropicModelBatch(anthropic_api_key=anthropic_api_key)
        self._openai_batch = OpenAIModelBatch(openai_api_key=openai_api_key)

    async def __call__(
        self,
        model_id: str,
        prompts: list[Prompt],
        max_tokens: int | None = None,
        log_dir: Path | None = None,
        **kwargs,
    ) -> tuple[list[LLMResponse], str]:
        """Make batch API requests for the specified model and prompts.

        Args:
            model_id: The model to call. Must be a Claude or OpenAI model.
            prompts: List of prompts to send to the model.
            max_tokens: Maximum number of tokens to generate per response.
            log_dir: Directory to store logs. If None, will use prompt_history_dir.
            **kwargs: Additional arguments to pass to the model.

        Returns:
            Tuple of (list of responses, batch ID)
        """
        if not prompts:
            raise ValueError("No prompts provided")

        if log_dir is None:
            log_dir = self.log_dir

        if log_dir is not None:
            log_dir.mkdir(parents=True, exist_ok=True)

        if model_id in ANTHROPIC_MODELS:
            assert max_tokens is not None, "max_tokens must be provided for Anthropic models"
            responses, batch_id = await self._anthropic_batch(
                model_id, prompts, log_dir=log_dir, max_tokens=max_tokens, **kwargs
            )
        elif model_id in GPT_CHAT_MODELS:
            assert log_dir is not None, "log_dir must be provided for OpenAI models"
            if max_tokens is not None:
                kwargs["max_tokens"] = max_tokens

            responses, batch_id = await self._openai_batch(model_id, prompts, log_dir=log_dir, **kwargs)
        else:
            raise ValueError(f"Model {model_id} is not supported. Only Claude and OpenAI models are supported.")

        assert len(responses) == len(prompts)
        return responses, batch_id
