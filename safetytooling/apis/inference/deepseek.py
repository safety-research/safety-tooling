import asyncio
import logging
import time
from pathlib import Path
from traceback import format_exc

import aiohttp

from safetytooling.data_models import LLMResponse, Prompt

from .model import InferenceAPIModel

DEEPSEEK_MODELS = {"deepseek-chat", "deepseek-reasoner"}
LOGGER = logging.getLogger(__name__)


class DeepSeekChatModel(InferenceAPIModel):
    def __init__(
        self,
        num_threads: int,
        prompt_history_dir: Path | None = None,
        api_key: str | None = None,
    ):
        self.num_threads = num_threads
        self.prompt_history_dir = prompt_history_dir
        self.api_key = api_key
        self.available_requests = asyncio.BoundedSemaphore(int(self.num_threads))
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

    async def __call__(
        self,
        model_id: str,
        prompt: Prompt,
        print_prompt_and_response: bool,
        max_attempts: int,
        is_valid=lambda x: True,
        **kwargs,
    ) -> list[LLMResponse]:
        if self.api_key is None:
            raise RuntimeError(
                "DEEPSEEK_API_KEY environment variable must be set in SECRETS before running your script"
            )
        start = time.time()

        prompt_file = self.create_prompt_history_file(prompt, model_id, self.prompt_history_dir)

        LOGGER.debug(f"Making {model_id} call")
        response_data = None
        api_duration = None

        for i in range(max_attempts):
            try:
                async with self.available_requests:
                    api_start = time.time()

                    async with aiohttp.ClientSession() as session:
                        async with session.post(
                            "https://api.deepseek.com/v1/chat/completions",
                            headers=self.headers,
                            json={
                                "model": model_id,
                                "messages": prompt.openai_format(),
                                **kwargs,
                            },
                        ) as response:
                            response_data = await response.json()

                    api_duration = time.time() - api_start

                    if "error" in response_data:
                        raise RuntimeError(f"API Error: {response_data['error']}")

                    if not is_valid(response_data["choices"][0]["message"]["content"]):
                        raise RuntimeError(f"Invalid response according to is_valid {response_data}")
                    break
            except TypeError as e:
                raise e
            except Exception as e:
                error_info = f"Exception Type: {type(e).__name__}, Error Details: {str(e)}, Traceback: {format_exc()}"
                LOGGER.warn(f"Encountered API error: {error_info}.\nRetrying now. (Attempt {i})")
                await asyncio.sleep(1.5**i)

        if response_data is None:
            raise RuntimeError(f"Failed to get a response from the API after {max_attempts} attempts")

        duration = time.time() - start

        response = LLMResponse(
            model_id=model_id,
            completion=response_data["choices"][0]["message"]["content"],
            stop_reason=response_data["choices"][0]["finish_reason"],
            duration=duration,
            api_duration=api_duration,
            cost=0,
            reasoning_content=response_data["choices"][0]["message"].get("reasoning_content"),
        )
        responses = [response]

        self.add_response_to_prompt_file(prompt_file, responses)
        if print_prompt_and_response:
            prompt.pretty_print(responses)

        return responses
