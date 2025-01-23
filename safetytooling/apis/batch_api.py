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
        prompt_history_dir: Path | Literal["default"] | None = "default",
        anthropic_api_key: str | None = None,
        openai_api_key: str | None = None,
    ):
        """Initialize the BatchInferenceAPI.

        Args:
            prompt_history_dir: Directory to store prompt history. Set to None to disable.
                Set to "default" to use the default directory.
            anthropic_api_key: Optional API key for Anthropic. If not provided, will try to load from environment.
            openai_api_key: Optional API key for OpenAI. If not provided, will try to load from environment.
        """
        secrets = load_secrets("SECRETS")
        if prompt_history_dir == "default":
            if "PROMPT_HISTORY_DIR" in secrets:
                self.prompt_history_dir = Path(secrets["PROMPT_HISTORY_DIR"])
            else:
                self.prompt_history_dir = Path.home() / ".prompt_history"
        else:
            assert isinstance(prompt_history_dir, Path) or prompt_history_dir is None
            self.prompt_history_dir = prompt_history_dir

        self._anthropic_batch = AnthropicModelBatch(anthropic_api_key=anthropic_api_key)
        self._openai_batch = OpenAIModelBatch()

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

        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens

        if log_dir is None:
            log_dir = self.prompt_history_dir

        if log_dir is not None:
            log_dir.mkdir(parents=True, exist_ok=True)

        if model_id in ANTHROPIC_MODELS:
            responses, batch_id = await self._anthropic_batch(model_id, prompts, log_dir=log_dir, **kwargs)
        elif model_id in GPT_CHAT_MODELS:
            responses, batch_id = await self._openai_batch(model_id, prompts, log_dir=log_dir, **kwargs)
        else:
            raise ValueError(f"Model {model_id} is not supported. Only Claude and OpenAI models are supported.")

        assert len(responses) == len(prompts)
        return responses, batch_id
