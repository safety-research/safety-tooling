import asyncio
import logging
import time
from pathlib import Path
from traceback import format_exc

from together import AsyncTogether

from safetytooling.data_models import LLMResponse, Prompt
from .model import InferenceAPIModel

TOGETHER_MODELS = {
    "mistralai/Mixtral-8x7B-Instruct-v0.1",
    "deepseek-ai/DeepSeek-V3",
    "deepseek-ai/deepseek-llm-67b-chat",
    "Qwen/Qwen2.5-72B-Instruct-Turbo",
    "meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo",
    "meta-llama/Llama-3.3-70B-Instruct-Turbo",
    "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo-128K",
}

LOGGER = logging.getLogger(__name__)

class TogetherChatModel(InferenceAPIModel):
    def __init__(
        self,
        num_threads: int,
        prompt_history_dir: Path | None = None,
        api_key: str | None = None,
    ):
        assert api_key is not None, "TOGETHER_API_KEY environment variable must be set"
        self.num_threads = num_threads
        self.prompt_history_dir = prompt_history_dir
        self.aclient = AsyncTogether(api_key=api_key)
        self.available_requests = asyncio.BoundedSemaphore(int(self.num_threads))
        self.allowed_kwargs = {"temperature", "max_tokens"}

    async def __call__(
        self,
        model_ids: tuple[str, ...],
        prompt: Prompt,
        print_prompt_and_response: bool,
        max_attempts: int,
        is_valid=lambda x: True,
        **kwargs,
    ) -> list[LLMResponse]:
        start = time.time()
        assert len(model_ids) == 1, "Together implementation only supports one model at a time."
        (model_id,) = model_ids

        prompt_file = self.create_prompt_history_file(prompt, model_id, self.prompt_history_dir)

        LOGGER.debug(f"Making {model_id} call")
        response = None
        duration = None

        error_list = []
        for i in range(max_attempts):
            try:
                async with self.available_requests:
                    api_start = time.time()

                    filtered_kwargs = {k: v for k, v in kwargs.items() if k in self.allowed_kwargs}
                    response = await self.aclient.chat.completions.create(
                        messages=prompt.openai_format(),  # Together uses OpenAI format
                        model=model_id,
                        stream=False,
                        **filtered_kwargs,
                    )
                    api_duration = time.time() - api_start
                    if not is_valid(response):
                        raise RuntimeError(f"Invalid response according to is_valid {response}")
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

        if response is None:
            raise RuntimeError(f"Failed to get a response from the API after {max_attempts} attempts.")

        assert len(response.choices) == 1, f"Expected 1 choice, got {len(response.choices)}"

        response = LLMResponse(
            model_id=model_id,
            completion=response.choices[0].message.content,
            stop_reason=response.choices[0].finish_reason,
            duration=duration,
            api_duration=api_duration,
            cost=0,  # Together API cost tracking could be implemented here
        )
        duration = time.time() - start
        LOGGER.debug(f"Completed call to {model_id} in {duration}s")

        responses = [response]

        self.add_response_to_prompt_file(prompt_file, responses)
        if print_prompt_and_response:
            prompt.pretty_print(responses)

        return responses
