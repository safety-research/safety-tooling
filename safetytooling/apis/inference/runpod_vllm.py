import asyncio
import collections
import logging
import time
from pathlib import Path
from traceback import format_exc

import aiohttp
from transformers import AutoTokenizer

from safetytooling.data_models import LLMResponse, Prompt, StopReason
from safetytooling.utils import math_utils

from .model import InferenceAPIModel

VLLM_MODELS = {
    "dummy": "https://pod-port.proxy.runpod.net/v1/chat/completions",
    # Examples for RunPod serverless endpoints
    # "your-model-id": "https://api.runpod.ai/v2/{endpoint_id}/run",
    # Add more models and their endpoints as needed
}

LOGGER = logging.getLogger(__name__)

# Load Qwen tokenizer for completion API formatting
try:
    QWEN_TOKENIZER = AutoTokenizer.from_pretrained("qwen/Qwen3-32B", trust_remote_code=True)
except Exception as e:
    LOGGER.warning(f"Failed to load Qwen tokenizer: {e}")
    QWEN_TOKENIZER = None


class VLLMChatModel(InferenceAPIModel):
    @staticmethod
    def is_runpod_serverless(url: str) -> bool:
        """Detect if URL is RunPod serverless endpoint"""
        return "api.runpod.ai" in url

    def __init__(
        self,
        num_threads: int,
        prompt_history_dir: Path | None = None,
        vllm_base_url: str = "http://localhost:8000/v1/chat/completions",
        vllm_api_key: str | None = None,
    ):
        self.num_threads = num_threads
        self.prompt_history_dir = prompt_history_dir
        self.available_requests = asyncio.BoundedSemaphore(int(self.num_threads))

        self.vllm_api_key = vllm_api_key
        self.headers = {"Content-Type": "application/json"}
        if self.vllm_api_key is not None:
            self.headers["Authorization"] = f"Bearer {self.vllm_api_key}"

        self.vllm_base_url = vllm_base_url
        # Only add /chat/completions for standard VLLM endpoints, not RunPod serverless
        if (
            self.vllm_base_url is not None
            and self.vllm_base_url.endswith("v1")
            and not self.is_runpod_serverless(self.vllm_base_url)
        ):
            self.vllm_base_url += "/chat/completions"

        self.stop_reason_map = {
            "length": StopReason.MAX_TOKENS.value,
            "stop": StopReason.STOP_SEQUENCE.value,
        }

    async def query(self, model_url: str, payload: dict, session: aiohttp.ClientSession, timeout: int = 1000) -> dict:
        async with session.post(model_url, headers=self.headers, json=payload, timeout=timeout) as response:
            if response.status != 200:
                error_text = await response.text()
                if "<!DOCTYPE html>" in str(error_text) and "<title>" in str(error_text):
                    reason = str(error_text).split("<title>")[1].split("</title>")[0]
                else:
                    reason = " ".join(str(error_text).split()[:20])
                raise RuntimeError(f"API Error: Status {response.status}, {reason}")
            return await response.json()

    async def run_query_serverless(self, model_url: str, messages: list, params: dict) -> dict:
        """Handle RunPod serverless API with polling"""
        # Convert URL to RunPod serverless format
        if "/v1/chat/completions" in model_url:
            submit_url = model_url.replace("/v1/chat/completions", "/run")
            base_url = model_url.replace("/v1/chat/completions", "")
        else:
            submit_url = model_url if model_url.endswith("/run") else f"{model_url}/run"
            base_url = model_url.replace("/run", "") if model_url.endswith("/run") else model_url

        # Check if this is a continuation case (last message from assistant)
        last_message_is_assistant = messages and messages[-1].get("role") == "assistant"

        if last_message_is_assistant:
            # Handle continuation using completion API format
            return await self.run_query_serverless_completion(model_url, messages, params)

        # Prepare payload for RunPod serverless chat completions
        payload = {
            "input": {
                "openai_route": "/v1/chat/completions",
                "openai_input": {
                    "messages": messages,
                    "model": params["model"],
                    "sampling_params": {k: v for k, v in params.items() if k != "model"},
                }
            }
        }

        start_time = time.time()
        max_time = 600

        async with aiohttp.ClientSession() as session:
            async with session.post(submit_url, headers=self.headers, json=payload, timeout=60) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise RuntimeError(f"Failed to submit job: Status {response.status}, {error_text}")
                job_data = await response.json()

            job_id = job_data["id"]

            # Poll for result
            status_url = f"{base_url}/status/{job_id}"

            while time.time() - start_time < max_time:
                await asyncio.sleep(1)  # Poll every 1 second

                async with session.get(status_url, headers=self.headers, timeout=30) as status_response:
                    if status_response.status != 200:
                        error_text = await status_response.text()
                        LOGGER.info(f"Status check failed: {error_text}")
                        continue

                    status_data = await status_response.json()

                status = status_data.get("status", "UNKNOWN")

                if status == "COMPLETED":
                    return status_data["output"][0]
                elif status == "FAILED":
                    error_msg = status_data.get("error", "Unknown error")
                    raise RuntimeError(f"RunPod job failed: {error_msg}")
                elif status in ["IN_QUEUE", "IN_PROGRESS"]:
                    continue  # Keep polling
                else:
                    LOGGER.warning(f"Unknown status: {status}")
                    continue

            raise RuntimeError(f"RunPod job timed out after {max_time} seconds")

    async def run_query_serverless_completion(self, model_url: str, messages: list, params: dict) -> dict:
        """Handle RunPod serverless API with completion format for assistant message continuation"""
        if QWEN_TOKENIZER is None:
            raise RuntimeError("Qwen tokenizer not available - cannot use completion API for continuation")

        # Convert URL to RunPod serverless format
        if "/v1/chat/completions" in model_url:
            submit_url = model_url.replace("/v1/chat/completions", "/run")
            base_url = model_url.replace("/v1/chat/completions", "")
        else:
            submit_url = model_url if model_url.endswith("/run") else f"{model_url}/run"
            base_url = model_url.replace("/run", "") if model_url.endswith("/run") else model_url

        # Format messages using tokenizer chat template for continuation
        # Remove the last assistant message and format the rest
        messages_without_last = messages[:-1]
        if messages_without_last:
            base_prompt = QWEN_TOKENIZER.apply_chat_template(
                messages_without_last, tokenize=False, add_generation_prompt=False
            )
        else:
            base_prompt = ""

        # Add the assistant message start and the partial content, but don't close it
        last_content = messages[-1]["content"]
        formatted_prompt = base_prompt + f"<|im_start|>assistant\n{last_content}"

        LOGGER.debug(f"Using serverless completion API with formatted prompt: {repr(formatted_prompt[:200])}...")

        # Prepare payload for RunPod serverless with completion format
        payload = {
            "input": {
                "prompt": formatted_prompt,
                "model": params["model"],
                "sampling_params": {k: v for k, v in params.items() if k != "model"},
            }
        }

        start_time = time.time()
        max_time = 600

        async with aiohttp.ClientSession() as session:
            async with session.post(submit_url, headers=self.headers, json=payload, timeout=60) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise RuntimeError(f"Failed to submit completion job: Status {response.status}, {error_text}")
                job_data = await response.json()

            job_id = job_data["id"]

            # Poll for result
            status_url = f"{base_url}/status/{job_id}"

            while time.time() - start_time < max_time:
                await asyncio.sleep(1)  # Poll every 1 second

                async with session.get(status_url, headers=self.headers, timeout=30) as status_response:
                    if status_response.status != 200:
                        error_text = await status_response.text()
                        LOGGER.info(f"Status check failed: {error_text}")
                        continue

                    status_data = await status_response.json()

                status = status_data.get("status", "UNKNOWN")

                if status == "COMPLETED":
                    output = status_data.get("output", {})
                    # Convert completion response to chat format
                    if "choices" in output and len(output["choices"]) > 0:
                        # Transform completion response to chat format
                        for choice in output["choices"]:
                            if "text" in choice:
                                choice["message"] = {"role": "assistant", "content": choice.pop("text", "")}
                            # Map finish_reason if needed
                            if "finish_reason" not in choice:
                                choice["finish_reason"] = "stop"
                    return output
                elif status == "FAILED":
                    error_msg = status_data.get("error", "Unknown error")
                    raise RuntimeError(f"RunPod completion job failed: {error_msg}")
                elif status in ["IN_QUEUE", "IN_PROGRESS"]:
                    continue  # Keep polling
                else:
                    LOGGER.warning(f"Unknown status: {status}")
                    continue

            raise RuntimeError(f"RunPod completion job timed out after {max_time} seconds")

    async def run_query_completion(self, model_url: str, messages: list, params: dict) -> dict:
        """Handle completion API calls for assistant message continuation"""
        if QWEN_TOKENIZER is None:
            raise RuntimeError("Qwen tokenizer not available - cannot use completion API for continuation")

        # Convert model_url from chat/completions to completions
        if "/chat/completions" in model_url:
            completion_url = model_url.replace("/chat/completions", "/completions")
        elif model_url.endswith("/v1"):
            completion_url = model_url + "/completions"
        else:
            completion_url = (
                model_url.replace("/v1", "/v1/completions") if "/v1" in model_url else model_url + "/completions"
            )

        # Format messages using tokenizer chat template for continuation
        # Remove the last assistant message and format the rest
        messages_without_last = messages[:-1]
        if messages_without_last:
            base_prompt = QWEN_TOKENIZER.apply_chat_template(
                messages_without_last, tokenize=False, add_generation_prompt=False
            )
        else:
            base_prompt = ""

        # Add the assistant message start and the partial content, but don't close it
        last_content = messages[-1]["content"]
        formatted_prompt = base_prompt + f"<|im_start|>assistant\n{last_content}"

        LOGGER.debug(f"Using completion API with formatted prompt: {repr(formatted_prompt[:200])}...")

        # Prepare payload for completion API
        completion_payload = {
            "model": params.get("model", "default"),
            "prompt": formatted_prompt,
            **{k: v for k, v in params.items() if k != "model"},
        }

        async with aiohttp.ClientSession() as session:
            response = await self.query(completion_url, completion_payload, session)
            if "error" in response:
                if "<!DOCTYPE html>" in str(response) and "<title>" in str(response):
                    reason = str(response).split("<title>")[1].split("</title>")[0]
                else:
                    reason = " ".join(str(response).split()[:20])
                print(f"API Error: {reason}")
                LOGGER.error(f"API Error: {reason}")
                raise RuntimeError(f"API Error: {reason}")

            # Convert completion API response to chat format
            if "choices" in response and len(response["choices"]) > 0:
                # Transform completion response to chat format
                for choice in response["choices"]:
                    choice["message"] = {"role": "assistant", "content": choice.pop("text", "")}
                    # Map finish_reason if needed
                    if "finish_reason" not in choice:
                        choice["finish_reason"] = "stop"

            return response

    async def run_query_sync(self, model_url: str, messages: list, params: dict) -> dict:
        """Handle synchronous VLLM API calls"""
        payload = {
            "model": params.get("model", "default"),
            "messages": messages,
            **{k: v for k, v in params.items() if k != "model"},
        }

        async with aiohttp.ClientSession() as session:
            response = await self.query(model_url, payload, session)
            if "error" in response:
                if "<!DOCTYPE html>" in str(response) and "<title>" in str(response):
                    reason = str(response).split("<title>")[1].split("</title>")[0]
                else:
                    reason = " ".join(str(response).split()[:20])
                print(f"API Error: {reason}")
                LOGGER.error(f"API Error: {reason}")
                raise RuntimeError(f"API Error: {reason}")
            return response

    async def run_query(self, model_url: str, messages: list, params: dict) -> dict:
        """Route to appropriate query method based on URL type and message structure"""

        # Check if last message is from assistant (continuation case)
        last_message_is_assistant = messages and messages[-1].get("role") == "assistant"

        if self.is_runpod_serverless(model_url):
            # For RunPod serverless endpoints, run_query_serverless now handles both cases
            return await self.run_query_serverless(model_url, messages, params)
        else:
            # For regular VLLM endpoints
            if last_message_is_assistant:
                # Use completion API for continuation
                LOGGER.info("Using completion API for assistant message continuation")
                return await self.run_query_completion(model_url, messages, params)
            else:
                # Use regular chat API
                return await self.run_query_sync(model_url, messages, params)

    @staticmethod
    def convert_top_logprobs(data: dict) -> list[dict]:
        # convert OpenAI chat version of logprobs response to completion version
        top_logprobs = []

        for item in data["content"]:
            # <|end|> is known to be duplicated on gpt-4-turbo-2024-04-09
            # See experiments/sample-notebooks/api-tests/logprob-dup-tokens.ipynb for details
            possibly_duplicated_top_logprobs = collections.defaultdict(list)
            for top_logprob in item["top_logprobs"]:
                possibly_duplicated_top_logprobs[top_logprob["token"]].append(top_logprob["logprob"])

            top_logprobs.append({k: math_utils.logsumexp(vs) for k, vs in possibly_duplicated_top_logprobs.items()})

        return top_logprobs

    async def __call__(
        self,
        model_id: str,
        prompt: Prompt,
        print_prompt_and_response: bool,
        max_attempts: int,
        is_valid=lambda x: True,
        logprobs: int | None = None,
        **kwargs,
    ) -> list[LLMResponse]:
        start = time.time()

        model_url = VLLM_MODELS.get(model_id, self.vllm_base_url)

        prompt_file = self.create_prompt_history_file(prompt, model_id, self.prompt_history_dir)

        LOGGER.debug(f"Making {model_id} call")
        response_data = None
        duration = None
        api_duration = None

        kwargs["model"] = model_id
        if logprobs is not None:
            kwargs["top_logprobs"] = logprobs
            kwargs["logprobs"] = True
            kwargs["prompt_logprobs"] = True

        for i in range(max_attempts):
            try:
                async with self.available_requests:
                    api_start = time.time()

                    response_data = await self.run_query(
                        model_url,
                        prompt.openai_format(allow_assistant_last=True),
                        kwargs,
                    )
                    # If we use the completion API, the format is a bit different and we need to reformat things
                    if isinstance(response_data, list):
                        response_data = {
                            "choices": [{
                                "message": {"content": response_data[0]["choices"][0]["tokens"][0]},
                                "finish_reason": "stop_sequence",
                            }]
                        }

                    api_duration = time.time() - api_start

                    if not response_data.get("choices"):
                        LOGGER.error(f"Invalid response format: {response_data}")
                        raise RuntimeError(f"Invalid response format: {response_data}")

                    if not all(is_valid(choice["message"]["content"]) for choice in response_data["choices"]):
                        LOGGER.error(f"Invalid responses according to is_valid {response_data}")
                        raise RuntimeError(f"Invalid responses according to is_valid {response_data}")
            except TypeError as e:
                raise e
            except Exception as e:
                error_info = f"Exception Type: {type(e).__name__}, Error Details: {str(e)}, Traceback: {format_exc()}"
                LOGGER.warn(f"Encountered API error: {error_info}.\nRetrying now. (Attempt {i})")
                await asyncio.sleep(1.5**i)
            else:
                break
        else:
            LOGGER.error(f"Failed to get a response from the API after {max_attempts} attempts.")
            raise RuntimeError(f"Failed to get a response from the API after {max_attempts} attempts.")

        duration = time.time() - start
        LOGGER.debug(f"Completed call to {model_id} in {duration}s")

        if response_data is None or response_data.get("choices", None) is None:
            LOGGER.error(f"Invalid response format: {response_data}")
            raise RuntimeError(f"Invalid response format: {response_data}")

        responses = [
            LLMResponse(
                model_id=model_id,
                completion=choice["message"]["content"],
                stop_reason=choice["finish_reason"],
                api_duration=api_duration,
                duration=duration,
                cost=0,
                logprobs=self.convert_top_logprobs(choice["logprobs"]) if choice.get("logprobs") is not None else None,
            )
            for choice in response_data["choices"]
        ]

        self.add_response_to_prompt_file(prompt_file, responses)
        if print_prompt_and_response:
            prompt.pretty_print(responses)

        return responses
