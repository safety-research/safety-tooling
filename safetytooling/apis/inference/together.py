import asyncio
import collections
import logging
import time
from pathlib import Path
from traceback import format_exc

from together import AsyncTogether
from together.types import ChatCompletionResponse

from safetytooling.data_models import LLMResponse, Prompt
from safetytooling.utils import math_utils

from .model import InferenceAPIModel

TOGETHER_MODELS = {
    "meta-llama/Llama-3.2-1B-Instruct",
    "meta-llama/Meta-Llama-3-8B-Instruct",
    "meta-llama/Meta-Llama-3.1-8B-Instruct-Reference",
    "meta-llama/Meta-Llama-3.1-70B-Instruct-Reference",
    "meta-llama/Llama-3.3-70B-Instruct-Reference",
    "meta-llama/Meta-Llama-3-8B-Instruct-Turbo",
    "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
    "meta-llama/Meta-Llama-3-70B-Instruct-Turbo",
    "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",
    "meta-llama/Llama-3.3-70B-Instruct-Turbo",
    "meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo",
    "meta-llama/Llama-2-13b-chat-hf",
    "mistralai/Mixtral-8x7B-Instruct-v0.1",
    "deepseek-ai/DeepSeek-V3",
    "deepseek-ai/deepseek-llm-67b-chat",
    "Qwen/Qwen2.5-72B-Instruct-Turbo",
    "Qwen/QwQ-32B-Preview",
}
LOGGER = logging.getLogger(__name__)


class TogetherChatModel(InferenceAPIModel):
    def __init__(
        self,
        num_threads: int,
        prompt_history_dir: Path | None = None,
        api_key: str | None = None,
    ):
        self.num_threads = num_threads
        self.prompt_history_dir = prompt_history_dir
        if api_key is not None:
            self.aclient = AsyncTogether(api_key=api_key)
        else:
            self.aclient = None
        self.available_requests = asyncio.BoundedSemaphore(int(self.num_threads))
        self.allowed_kwargs = {"temperature", "max_tokens", "logprobs"}

    @staticmethod
    def convert_top_logprobs(data: ChatCompletionResponse) -> list[dict]:
        # convert OpenAI chat version of logprobs response to completion version
        top_logprobs = []

        for item in data.content:
            # <|end|> is known to be duplicated on gpt-4-turbo-2024-04-09
            # See experiments/sample-notebooks/api-tests/logprob-dup-tokens.ipynb for details
            possibly_duplicated_top_logprobs = collections.defaultdict(list)
            for top_logprob in item.top_logprobs:
                possibly_duplicated_top_logprobs[top_logprob.token].append(top_logprob.logprob)

            top_logprobs.append({k: math_utils.logsumexp(vs) for k, vs in possibly_duplicated_top_logprobs.items()})

        return top_logprobs

    async def __call__(
        self,
        model_ids: tuple[str, ...],
        prompt: Prompt,
        print_prompt_and_response: bool,
        max_attempts: int,
        is_valid=lambda x: True,
        **kwargs,
    ) -> list[LLMResponse]:
        if self.aclient is None:
            raise RuntimeError(
                "TOGETHER_API_KEY environment variable must be set in SECRETS before running your script"
            )
        start = time.time()
        assert len(model_ids) == 1, "Together implementation only supports one model at a time."
        (model_id,) = model_ids

        prompt_file = self.create_prompt_history_file(prompt, model_id, self.prompt_history_dir)

        LOGGER.debug(f"Making {model_id} call")
        response_data = None
        duration = None

        error_list = []
        for i in range(max_attempts):
            try:
                async with self.available_requests:
                    api_start = time.time()

                    filtered_kwargs = {k: v for k, v in kwargs.items() if k in self.allowed_kwargs}
                    if "logprobs" in filtered_kwargs:
                        filtered_kwargs["top_logprobs"] = filtered_kwargs["logprobs"]
                        filtered_kwargs["logprobs"] = True

                    response_data = await self.aclient.chat.completions.create(
                        messages=prompt.openai_format(),
                        model=model_id,
                        **filtered_kwargs,
                    )
                    api_duration = time.time() - api_start
                    if not is_valid(response_data):
                        raise RuntimeError(f"Invalid response according to is_valid {response_data}")
            except Exception as e:
                error_info = f"Exception Type: {type(e).__name__}, Error Details: {str(e)}, Traceback: {format_exc()}"
                LOGGER.warn(f"Encountered API error: {error_info}.\nRetrying now. (Attempt {i})")
                error_list.append(error_info)
                api_duration = time.time() - api_start
                await asyncio.sleep(1.5**i)
            else:
                break

        duration = time.time() - start
        LOGGER.debug(f"Completed call to {model_id} in {duration}s")

        if response_data is None:
            raise RuntimeError("No response data received")
        assert len(response_data.choices) == 1, f"Expected 1 choice, got {len(response_data.choices)}"
        responses = [
            LLMResponse(
                model_id=model_id,
                completion=choice.message.content,
                stop_reason=choice.finish_reason,
                api_duration=api_duration,
                duration=duration,
                cost=0,
                logprobs=self.convert_top_logprobs(choice.logprobs) if choice.logprobs is not None else None,
            )
            for choice in response_data.choices
        ]

        self.add_response_to_prompt_file(prompt_file, responses)
        if print_prompt_and_response:
            prompt.pretty_print(responses)

        return responses
