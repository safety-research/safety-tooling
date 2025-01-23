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
    """A wrapper around batch APIs that automatically manages rate limits and valid responses."""

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
            custom_ids = [self._anthropic_batch.get_custom_id(i, prompt) for i, prompt in enumerate(prompts)]
            set_custom_ids = set(custom_ids)

            requests = self._anthropic_batch.prompts_to_requests(model_id, prompts, **kwargs)
            batch_response = self._anthropic_batch.create_message_batch(requests=requests)
            batch_id = batch_response.id
            print(f"Batch {batch_id} created with {len(prompts)} requests")

            _ = await self._anthropic_batch.poll_message_batch(batch_id)
            results = self._anthropic_batch.retrieve_message_batch_results(batch_id)

            responses_dict = {}
            for result in results:
                if result.result.type == "succeeded":
                    content = result.result.message.content
                    # Extract text from content array
                    text = content[0].text if content else ""
                    assert result.custom_id in set_custom_ids
                    responses_dict[result.custom_id] = LLMResponse(
                        model_id=model_id,
                        completion=text,
                        stop_reason=result.result.message.stop_reason,
                        duration=None,  # Batch does not track individual durations
                        api_duration=None,
                        cost=0,
                        batch_custom_id=result.custom_id,
                    )

            responses = []
            for custom_id in custom_ids:
                responses.append(responses_dict[custom_id] if custom_id in responses_dict else None)

        elif model_id in GPT_CHAT_MODELS:
            input_file_path = self._openai_batch.prompts_to_file(model_id, prompts, log_dir, **kwargs)
            batch_file_object = self._openai_batch.client.files.create(
                file=open(input_file_path, "rb"), purpose="batch"
            )
            batch_object = self._openai_batch.create_message_batch(input_file_id=batch_file_object.id)
            batch_id = batch_object.id
            print(f"Batch {batch_id} created with {len(prompts)} requests")

            _ = await self._openai_batch.poll_message_batch(batch_id)
            results = self._openai_batch.retrieve_message_batch_results(batch_object.output_file_id)

            responses = []
            for result in results:
                if result["response"]["status_code"] == 200:
                    body = result["response"]["body"]
                    choice = body["choices"][0]
                    responses.append(
                        LLMResponse(
                            model_id=model_id,
                            completion=choice["message"]["content"],
                            stop_reason=choice["finish_reason"],
                            duration=None,  # Batch does not track individual durations
                            api_duration=None,
                            cost=0,
                            batch_custom_id=result["custom_id"],
                        )
                    )
        else:
            raise ValueError(f"Model {model_id} is not supported. Only Claude and OpenAI models are supported.")

        assert len(responses) == len(prompts)
        return responses, batch_id
