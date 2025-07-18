import collections
import logging
import os
import time

import httpx
import openai
import openai.types
import openai.types.chat
import pydantic
import requests
from tenacity import retry, stop_after_attempt, wait_fixed

from safetytooling.data_models import LLMResponse, Prompt, Usage
from safetytooling.utils import math_utils

from .base import OpenAIModel
from .utils import count_tokens, get_rate_limit, price_per_token

LOGGER = logging.getLogger(__name__)


class OpenAIChatModel(OpenAIModel):
    @retry(stop=stop_after_attempt(8), wait=wait_fixed(2))
    async def _get_dummy_response_header(self, model_id: str):
        if self.aclient is None:
            raise RuntimeError("OpenAI API key must be set either via parameter or OPENAI_API_KEY environment variable")

        url = (
            "https://api.openai.com/v1/chat/completions"
            if self.base_url is None
            else self.base_url + "/v1/chat/completions"
        )

        # Use the API key that's already been validated
        if self.openai_api_key:
            api_key = self.openai_api_key
        else:
            # We know it exists in environment because _ensure_client() passed
            api_key = os.environ.get("OPENAI_API_KEY")

        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}
        data = {
            "model": model_id,
            "messages": [{"role": "user", "content": "Say 1"}],
        }
        response = requests.post(url, headers=headers, json=data)
        if "x-ratelimit-limit-tokens" not in response.headers:
            tpm, rpm = get_rate_limit(model_id)
            print(f"Failed to get dummy response header {model_id}, setting to tpm, rpm: {tpm}, {rpm}")
            header_dict = dict(response.headers)
            return header_dict | {
                "x-ratelimit-limit-tokens": str(tpm),
                "x-ratelimit-limit-requests": str(rpm),
                "x-ratelimit-remaining-tokens": str(tpm),
                "x-ratelimit-remaining-requests": str(rpm),
            }
        return response.headers

    @staticmethod
    def _count_prompt_token_capacity(prompt: Prompt, **kwargs) -> int:
        # The magic formula is: .25 * (total number of characters) + (number of messages) + (max_tokens, or 15 if not specified)
        BUFFER = 5  # A bit of buffer for some error margin
        MIN_NUM_TOKENS = 20

        max_tokens = kwargs.get("max_completion_tokens", kwargs.get("max_tokens", 15))
        if max_tokens is None:
            max_tokens = 15

        num_tokens = 0
        for message in prompt.messages:
            num_tokens += 1
            num_tokens += len(message.content) / 4

        return max(
            MIN_NUM_TOKENS,
            int(num_tokens + BUFFER) + kwargs.get("n", 1) * max_tokens,
        )

    @staticmethod
    def convert_top_logprobs(data):
        # convert OpenAI chat version of logprobs response to completion version
        top_logprobs = []

        if not hasattr(data, "content") or data.content is None:
            return []

        for item in data.content:
            # <|end|> is known to be duplicated on gpt-4-turbo-2024-04-09
            possibly_duplicated_top_logprobs = collections.defaultdict(list)
            for top_logprob in item.top_logprobs:
                possibly_duplicated_top_logprobs[top_logprob.token].append(top_logprob.logprob)

            top_logprobs.append({k: math_utils.logsumexp(vs) for k, vs in possibly_duplicated_top_logprobs.items()})

        return top_logprobs

    async def _make_api_call(self, prompt: Prompt, model_id, start_time, **kwargs) -> list[LLMResponse]:
        if self.aclient is None:
            raise RuntimeError("OpenAI API key must be set either via parameter or OPENAI_API_KEY environment variable")

        LOGGER.debug(f"Making {model_id} call")

        # convert completion logprobs api to chat logprobs api
        if "logprobs" in kwargs:
            kwargs["top_logprobs"] = kwargs["logprobs"]
            kwargs["logprobs"] = True

        # max tokens is deprecated in favor of max_completion_tokens
        if "max_tokens" in kwargs:
            kwargs["max_completion_tokens"] = kwargs["max_tokens"]
            del kwargs["max_tokens"]

        # removing n to not cause an error from OpenAI API when using websearch tool
        if "web_search_options" in kwargs:
            assert kwargs["n"] == 1
            del kwargs["n"]

        prompt_file = self.create_prompt_history_file(prompt, model_id, self.prompt_history_dir)
        api_start = time.time()

        # Choose between parse and create based on response format
        if (
            "response_format" in kwargs
            and isinstance(kwargs["response_format"], type)
            and issubclass(kwargs["response_format"], pydantic.BaseModel)
        ):
            api_func = self.aclient.beta.chat.completions.parse
            LOGGER.info(
                f"Using parse API because response_format: {kwargs['response_format']} is of type {type(kwargs['response_format'])}"
            )
        else:
            api_func = self.aclient.chat.completions.create
        api_response: openai.types.chat.ChatCompletion = await api_func(
            messages=prompt.openai_format(),
            model=model_id,
            **kwargs,
        )
        if hasattr(api_response, "error") and (
            "Rate limit exceeded" in api_response.error["message"] or api_response.error["code"] == 429
        ):  # OpenRouter routes through the error messages from the different providers, so we catch them here
            dummy_request = httpx.Request(
                method="POST",
                url="https://api.openai.com/v1/chat/completions",
                headers={"Authorization": f"Bearer {self.openai_api_key}"},
                json={"messages": prompt.openai_format(), "model": model_id, **kwargs},
            )
            raise openai.RateLimitError(
                api_response.error["message"],
                response=httpx.Response(status_code=429, text=api_response.error["message"], request=dummy_request),
                body=api_response.error,
            )
        if (
            api_response.choices is None or api_response.usage.prompt_tokens == api_response.usage.total_tokens
        ):  # this sometimes happens on OpenRouter; we want to raise an error
            raise RuntimeError(f"No tokens were generated for {model_id}")
        api_duration = time.time() - api_start
        duration = time.time() - start_time
        context_token_cost, completion_token_cost = price_per_token(model_id)
        if api_response.usage is not None:
            context_cost = api_response.usage.prompt_tokens * context_token_cost
            current_usage = Usage(
                input_tokens=api_response.usage.prompt_tokens,
                output_tokens=api_response.usage.completion_tokens,
                total_tokens=api_response.usage.total_tokens,
                prompt_tokens_details=(
                    api_response.usage.prompt_tokens_details.model_dump()
                    if hasattr(api_response.usage, "prompt_tokens_details")
                    and api_response.usage.prompt_tokens_details is not None
                    else None
                ),
                completion_tokens_details=(
                    api_response.usage.completion_tokens_details.model_dump()
                    if hasattr(api_response.usage, "completion_tokens_details")
                    and api_response.usage.completion_tokens_details is not None
                    else None
                ),
            )
        else:
            context_cost = 0
            current_usage = None

        responses = []
        for choice in api_response.choices:
            if choice.message.content is None or choice.finish_reason is None:
                raise RuntimeError(f"No content or finish reason for {model_id}")
            responses.append(
                LLMResponse(
                    model_id=model_id,
                    completion=choice.message.content,
                    stop_reason=choice.finish_reason,
                    api_duration=api_duration,
                    duration=duration,
                    cost=context_cost + count_tokens(choice.message.content, model_id) * completion_token_cost,
                    logprobs=(self.convert_top_logprobs(choice.logprobs) if choice.logprobs is not None else None),
                    usage=current_usage,
                )
            )
        self.add_response_to_prompt_file(prompt_file, responses)
        return responses

    async def make_stream_api_call(
        self,
        prompt: Prompt,
        model_id,
        **kwargs,
    ) -> openai.AsyncStream[openai.types.chat.ChatCompletionChunk]:
        if self.aclient is None:
            raise RuntimeError("OpenAI API key must be set either via parameter or OPENAI_API_KEY environment variable")

        return await self.aclient.chat.completions.create(
            messages=prompt.openai_format(),
            model=model_id,
            stream=True,
            **kwargs,
        )
