import asyncio
import base64
import collections
import logging
import pickle
import time
from pathlib import Path
from traceback import format_exc
from typing import Dict, Optional

import aiohttp
import cloudpickle

from safetytooling.data_models import LLMResponse, MessageRole, Prompt, StopReason
from safetytooling.utils import math_utils

from .model import InferenceAPIModel

CHAT_COMPLETION_VLLM_MODELS = {
    "dummy": "https://pod-port.proxy.runpod.net/v1/chat/completions",
    # Add more chat completion models and their endpoints as needed
}

COMPLETION_VLLM_MODELS = {
    "meta-llama/Llama-3.1-70B": "https://98gzngudd59oss-8000.proxy.runpod.net/v1/completions",
    "meta-llama/Llama-3.1-8B": "https://98gzngudd59oss-8000.proxy.runpod.net/v1/completions",
    "google/gemma-3-27b-pt": "https://98gzngudd59oss-8000.proxy.runpod.net/v1/completions",
    # Add more completion models and their endpoints as needed
}

LOGGER = logging.getLogger(__name__)


class VLLMChatModel(InferenceAPIModel):
    def __init__(
        self,
        num_threads: int,
        prompt_history_dir: Path | None = None,
        vllm_base_url: str = "http://localhost:8000/v1/chat/completions",
        runpod_api_key: str | None = None,
    ):
        self.num_threads = num_threads
        self.prompt_history_dir = prompt_history_dir
        self.available_requests = asyncio.BoundedSemaphore(int(self.num_threads))

        self.headers = {"Content-Type": "application/json"}
        self.vllm_base_url = vllm_base_url
        if self.vllm_base_url.endswith("v1"):
            self.vllm_base_url += "/chat/completions"

        self.runpod_api_key = runpod_api_key
        if self.runpod_api_key:
            self.headers["Authorization"] = f"Bearer {self.runpod_api_key}"

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

    async def run_query(
        self, model_url: str, messages: list, params: dict, continue_final_message: bool = False, interventions: Optional[Dict[int, "Intervention"]] = None
    ) -> dict:
        # Check if the model URL is a non-chat endpoint
        if "/chat/" not in model_url:
            assert len(messages) == 1, "Make sure that there is only one message with the entire prompt"
            payload = {
                "model": params.get("model", "default"),
                "prompt": messages[0]["content"],
                **{k: v for k, v in params.items() if k != "model"},
            }
        else:
            payload = {
                "model": params.get("model", "default"),
                "messages": messages,
                **{k: v for k, v in params.items() if k != "model"},
            }

        # Add continue_final_message to payload if True
        if continue_final_message:
            payload["continue_final_message"] = True
            payload["add_generation_prompt"] = False
        else:
            payload["continue_final_message"] = False
            payload["add_generation_prompt"] = True

        # Serialize interventions if provided
        if interventions:
            serialized_interventions = {}
            for layer_id, intervention in interventions.items():
                # Cloudpickle the intervention and base64 encode it
                pickled_data = cloudpickle.dumps(intervention)
                encoded_data = base64.b64encode(pickled_data).decode('utf-8')
                serialized_interventions[str(layer_id)] = encoded_data
            payload["interventions"] = serialized_interventions

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
        interventions: Optional[Dict[int, "Intervention"]] = None,
        continue_final_message: Optional[bool] = None,
        **kwargs,
    ) -> list[LLMResponse]:
        start = time.time()

        # Determine if we should use completions format based on prompt structure
        use_completions_format = prompt.is_none_in_messages()
        
        if use_completions_format:
            model_url = COMPLETION_VLLM_MODELS.get(model_id, self.vllm_base_url.replace("/chat/completions", "/completions"))
        else:
            model_url = CHAT_COMPLETION_VLLM_MODELS.get(model_id, self.vllm_base_url)

        prompt_file = self.create_prompt_history_file(prompt, model_id, self.prompt_history_dir)

        LOGGER.debug(f"Making {model_id} call")
        response_data = None
        duration = None
        api_duration = None

        kwargs["model"] = model_id
        if "logprobs" in kwargs:
            kwargs["top_logprobs"] = kwargs["logprobs"]
            kwargs["logprobs"] = True

        for i in range(max_attempts):
            try:
                async with self.available_requests:
                    api_start = time.time()

                    # For completions format, pass simple content; for chat, use together_format
                    if use_completions_format:
                        # Use direct content access to avoid stripping whitespace from __str__
                        assert len(prompt.messages) == 1 and prompt.messages[0].role == MessageRole.none
                        messages = [{"content": prompt.messages[0].content}]
                    else:
                        messages = prompt.together_format()
                    
                    if continue_final_message is None:
                        continue_final_message = prompt.is_last_message_assistant()

                    response_data = await self.run_query(
                        model_url,
                        messages,
                        kwargs,
                        continue_final_message=continue_final_message,
                        interventions=interventions,
                    )
                    api_duration = time.time() - api_start

                    if not response_data.get("choices"):
                        LOGGER.error(f"Invalid response format: {response_data}")
                        raise RuntimeError(f"Invalid response format: {response_data}")

                    ### TODO: Kinda jank, find a better solution
                    if "/chat/" not in model_url:
                        if not all(is_valid(choice["text"]) for choice in response_data["choices"]):
                            LOGGER.error(f"Invalid responses according to is_valid {response_data}")
                            raise RuntimeError(f"Invalid responses according to is_valid {response_data}")
                    else:
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

        # Build extra_fields with custom API data, converting to tensors where appropriate
        extra_fields = {}
        if response_data.get("prompt_tokens") is not None:
            extra_fields["prompt_tokens"] = response_data["prompt_tokens"]
        if response_data.get("completion_tokens") is not None:
            extra_fields["completion_tokens"] = response_data["completion_tokens"]
        if response_data.get("activations") is not None:
            activations = response_data["activations"]
            converted_activations = {}
            if "prompt" in activations:
                # Decode base64-encoded pickled tensor dict
                prompt_data = pickle.loads(base64.b64decode(activations["prompt"]))
                converted_activations["prompt"] = {int(k): v for k, v in prompt_data.items()}
            if "completion" in activations:
                # Decode base64-encoded pickled tensor dict  
                completion_data = pickle.loads(base64.b64decode(activations["completion"]))
                converted_activations["completion"] = {int(k): v for k, v in completion_data.items()}
            extra_fields["activations"] = converted_activations
        if response_data.get("logits") is not None:
            # Decode base64-encoded pickled tensor
            logits_data = pickle.loads(base64.b64decode(response_data["logits"]))
            extra_fields["logits"] = logits_data
        
        responses = [
            LLMResponse(
                model_id=model_id,
                completion=(
                    choice["message"]["content"] if "/chat/" in model_url else choice["text"]
                ),  # TODO: Kinda jank, find a better solution
                stop_reason=choice["finish_reason"],
                api_duration=api_duration,
                duration=duration,
                cost=0,
                logprobs=self.convert_top_logprobs(choice["logprobs"]) if choice.get("logprobs") is not None else None,
                extra_fields=extra_fields if extra_fields else None,
            )
            for choice in response_data["choices"]
        ]

        self.add_response_to_prompt_file(prompt_file, responses)
        if print_prompt_and_response:
            prompt.pretty_print(responses)

        return responses
