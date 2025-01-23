import asyncio
import logging
import time
from pathlib import Path
from traceback import format_exc

import aiohttp

from safetytooling.data_models import LLMResponse, Prompt, StopReason

from .model import InferenceAPIModel

VLLM_MODELS = {
    "dummy": "https://pod-port.proxy.runpod.net/v1/chat/completions",
    # Add more models and their endpoints as needed
}

LOGGER = logging.getLogger(__name__)


class VLLMChatModel(InferenceAPIModel):
    def __init__(
        self,
        num_threads: int,
        prompt_history_dir: Path | None = None,
    ):
        self.num_threads = num_threads
        self.prompt_history_dir = prompt_history_dir
        self.available_requests = asyncio.BoundedSemaphore(int(self.num_threads))

        self.headers = {"Content-Type": "application/json"}

        self.kwarg_change_name = {
            "temperature": "temperature",
            "max_tokens": "max_tokens",
            "top_p": "top_p",
            "stream": "stream",
        }

        self.stop_reason_map = {
            "length": StopReason.MAX_TOKENS.value,
            "stop": StopReason.STOP_SEQUENCE.value,
        }

    async def query(self, model_url: str, payload: dict, session: aiohttp.ClientSession) -> dict:
        async with session.post(model_url, headers=self.headers, json=payload) as response:
            if response.status != 200:
                error_text = await response.text()
                raise RuntimeError(f"API Error: Status {response.status}, {error_text}")
            return await response.json()

    async def run_query(self, model_url: str, messages: list, params: dict) -> dict:
        payload = {
            "model": params.get("model", "default"),
            "messages": messages,
            **{k: v for k, v in params.items() if k != "model"},
        }

        async with aiohttp.ClientSession() as session:
            response = await self.query(model_url, payload, session)
            if "error" in response:
                LOGGER.error(response)
                raise RuntimeError(f"API Error: {response['error']}")
            return response

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
        assert len(model_ids) == 1, "VLLM implementation only supports one model at a time."
        (model_id,) = model_ids

        assert model_id in VLLM_MODELS, f"Model {model_id} not supported"
        model_url = VLLM_MODELS[model_id]

        prompt_file = self.create_prompt_history_file(prompt, model_id, self.prompt_history_dir)

        LOGGER.debug(f"Making {model_id} call")
        response_data = None
        duration = None
        api_duration = None

        for i in range(max_attempts):
            try:
                async with self.available_requests:
                    api_start = time.time()
                    params = {self.kwarg_change_name[k]: v for k, v in kwargs.items() if k in self.kwarg_change_name}
                    params["model"] = model_id

                    response_data = await self.run_query(
                        model_url,
                        prompt.openai_format(),
                        params,
                    )
                    api_duration = time.time() - api_start

                    if not response_data.get("choices"):
                        raise RuntimeError(f"Invalid response format: {response_data}")

                    completion = response_data["choices"][0]["message"]["content"]
                    if not is_valid(completion):
                        raise RuntimeError(f"Invalid response according to is_valid: {completion}")

            except Exception as e:
                error_info = f"Exception Type: {type(e).__name__}, Error Details: {str(e)}, Traceback: {format_exc()}"
                LOGGER.warn(f"Encountered API error: {error_info}.\nRetrying now. (Attempt {i})")
                await asyncio.sleep(1.5**i)
            else:
                break
        else:
            raise RuntimeError(f"Failed to get a response from the API after {max_attempts} attempts.")

        duration = time.time() - start
        LOGGER.debug(f"Completed call to {model_id} in {duration}s")

        response = LLMResponse(
            model_id=model_id,
            completion=response_data["choices"][0]["message"]["content"],
            stop_reason=self.stop_reason_map.get(
                response_data["choices"][0].get("finish_reason", "unknown"), StopReason.UNKNOWN.value
            ),
            duration=duration,
            api_duration=api_duration,
            logprobs=None,
            cost=0,
        )
        responses = [response]

        self.add_response_to_prompt_file(prompt_file, responses)
        if print_prompt_and_response:
            prompt.pretty_print(responses)

        return responses
