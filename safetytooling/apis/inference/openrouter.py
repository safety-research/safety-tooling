import asyncio
import logging
import time
from pathlib import Path
from traceback import format_exc

from openai import AsyncOpenAI, BadRequestError

from safetytooling.data_models import LLMResponse, Prompt

from .model import InferenceAPIModel

OPENROUTER_MODELS = {
    "x-ai/grok-3-beta",
    "google/gemini-2.0-flash-001",
    "mistral/ministral-8b",
    "mistralai/mistral-large-2411",
}
LOGGER = logging.getLogger(__name__)


class OpenRouterChatModel(InferenceAPIModel):
    def __init__(
        self,
        num_threads: int,
        prompt_history_dir: Path | None = None,
        api_key: str | None = None,
        progress_monitor: object | None = None,
    ):
        self.num_threads = num_threads
        self.prompt_history_dir = prompt_history_dir
        if api_key is not None:
            self.aclient = AsyncOpenAI(
                api_key=api_key,
                base_url="https://openrouter.ai/api/v1",
            )
        else:
            self.aclient = None
        self.available_requests = asyncio.BoundedSemaphore(int(self.num_threads))
        self.progress_monitor = progress_monitor

    @staticmethod
    def convert_top_logprobs(data) -> list[dict]:
        # convert logprobs response to match OpenAI completion version
        # OpenRouter uses the same format as OpenAI Chat API
        top_logprobs = []

        if not hasattr(data, "content") or data.content is None:
            return []

        for item in data.content:
            item_logprobs = {}
            for top_logprob in item.top_logprobs:
                item_logprobs[top_logprob.token] = top_logprob.logprob
            top_logprobs.append(item_logprobs)

        return top_logprobs

    async def __call__(
        self,
        model_id: str,
        prompt: Prompt,
        print_prompt_and_response: bool,
        max_attempts: int,
        is_valid=lambda x: True,
        **kwargs,
    ) -> list[LLMResponse]:

        if self.aclient is None:
            raise RuntimeError("OPENROUTER_API_KEY environment variable must be set in .env before running your script")
        start = time.time()

        prompt_file = self.create_prompt_history_file(prompt, model_id, self.prompt_history_dir)

        # OpenRouter handles logprobs the same as OpenAI
        if "logprobs" in kwargs:
            kwargs["top_logprobs"] = kwargs["logprobs"]
            kwargs["logprobs"] = True

        LOGGER.debug(f"Making {model_id} call")
        response_data = None
        duration = None

        error_list = []
        for i in range(max_attempts):
            try:
                async with self.available_requests:
                    api_start = time.time()

                    response_data = await self.aclient.chat.completions.create(
                        messages=prompt.openai_format(),
                        model=model_id,
                        **kwargs,
                    )
                    api_duration = time.time() - api_start
                    if (
                        response_data.choices is None
                        or len(response_data.choices) == 0
                        or response_data.choices[0].message.content is None
                        or response_data.choices[0].message.content == ""
                    ):
                        # sometimes gemini will never return a response
                        if model_id == "google/gemini-2.0-flash-001":
                            LOGGER.warning(f"Empty response from {model_id} (returning empty response)")
                            return [
                                LLMResponse(
                                    model_id=model_id,
                                    completion="",
                                    stop_reason="stop_sequence",
                                    api_duration=api_duration,
                                    duration=time.time() - start,
                                    cost=0,
                                    logprobs=None,
                                )
                            ]
                        raise RuntimeError(f"Empty response from {model_id} (common for openrouter so retrying)")
                    if response_data.choices[0].finish_reason is None:
                        raise RuntimeError(f"No finish reason from {model_id} (common for openrouter so retrying)")
                    elif response_data.choices[0].finish_reason == "error":
                        raise RuntimeError(
                            f"Error from {model_id} in finish_reason (common for openrouter so retrying)"
                        )
                    if not is_valid(response_data):
                        raise RuntimeError(f"Invalid response according to is_valid {response_data}")
            except (TypeError, BadRequestError) as e:
                raise e
            except Exception as e:
                error_info = f"Exception Type: {type(e).__name__}, Error Details: {str(e)}, Traceback: {format_exc()}"
                LOGGER.warning(f"Encountered API error: {error_info}.\nRetrying now. (Attempt {i})")
                error_list.append(error_info)
                api_duration = time.time() - api_start
                await asyncio.sleep(1.5**i)
            else:
                break

        duration = time.time() - start
        LOGGER.debug(f"Completed call to {model_id} in {duration}s")

        if response_data is None or response_data.choices is None or len(response_data.choices) == 0:
            raise RuntimeError("No response data received")

        if response_data.choices[0].message.content is None or response_data.choices[0].message.content == "":
            raise RuntimeError(f"Empty response from {model_id} (even after retries, perhaps decrease concurrency)")

        assert len(response_data.choices) == kwargs.get(
            "n", 1
        ), f"Expected {kwargs.get('n', 1)} choices, got {len(response_data.choices)}"

        responses = []
        total_in = 0
        total_out = 0
        for choice in response_data.choices:
            # OpenRouter may include usage at top-level
            usage = getattr(response_data, "usage", None)
            if usage is not None:
                # usage may be a dict-like object
                in_tok = getattr(usage, "prompt_tokens", None) or getattr(usage, "input_tokens", None)
                out_tok = getattr(usage, "completion_tokens", None) or getattr(usage, "output_tokens", None)
            else:
                in_tok = None
                out_tok = None
            if in_tok is not None:
                total_in += int(in_tok)
            if out_tok is not None:
                total_out += int(out_tok)

            responses.append(
                LLMResponse(
                    model_id=model_id,
                    completion=choice.message.content,
                    stop_reason=choice.finish_reason,
                    api_duration=api_duration,
                    duration=duration,
                    cost=0,
                    logprobs=(
                        self.convert_top_logprobs(choice.logprobs)
                        if hasattr(choice, "logprobs") and choice.logprobs is not None
                        else None
                    ),
                    usage=(
                        None
                        if usage is None
                        else type("U", (), {
                            "input_tokens": int(in_tok) if in_tok is not None else None,
                            "output_tokens": int(out_tok) if out_tok is not None else None,
                            "total_tokens": int((in_tok or 0) + (out_tok or 0)),
                        })()
                    ),
                )
            )

        self.add_response_to_prompt_file(prompt_file, responses)
        if print_prompt_and_response:
            prompt.pretty_print(responses)

        # Progress monitoring is handled centrally in InferenceAPI to avoid double counting.

        return responses
