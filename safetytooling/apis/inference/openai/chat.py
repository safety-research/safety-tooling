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

from safetytooling.data_models import LLMResponse, Prompt
from safetytooling.utils import math_utils

from .base import OpenAIModel
from .utils import get_rate_limit, price_per_token

LOGGER = logging.getLogger(__name__)


class OpenAIChatModel(OpenAIModel):

    @retry(stop=stop_after_attempt(8), wait=wait_fixed(2))
    async def _get_dummy_response_header(self, model_id: str):
        url = (
            "https://api.openai.com/v1/chat/completions"
            if self.base_url is None
            else self.base_url + "/v1/chat/completions"
        )
        if self.openai_api_key is None:
            api_key = os.environ["OPENAI_API_KEY"]
        else:
            api_key = self.openai_api_key

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

        num_tokens = 0
        for message in prompt.messages:
            num_tokens += 1
            num_tokens += len(message.content) / 4

        return max(
            MIN_NUM_TOKENS,
            int(num_tokens + BUFFER) + kwargs.get("n", 1) * kwargs.get("max_tokens", 15),
        )

    @staticmethod
    def convert_top_logprobs(data):
        # convert OpenAI chat version of logprobs response to completion version
        top_logprobs = []

        if not hasattr(data, 'content') or data.content is None:
            return []

        for item in data.content:
            # <|end|> is known to be duplicated on gpt-4-turbo-2024-04-09
            possibly_duplicated_top_logprobs = collections.defaultdict(list)
            for top_logprob in item.top_logprobs:
                possibly_duplicated_top_logprobs[top_logprob.token].append(top_logprob.logprob)

            top_logprobs.append({k: math_utils.logsumexp(vs) for k, vs in possibly_duplicated_top_logprobs.items()})

        return top_logprobs

    async def _make_api_call(self, prompt: Prompt, model_id, start_time, **kwargs) -> list[LLMResponse]:
        LOGGER.debug(f"Making {model_id} call")

        # convert completion logprobs api to chat logprobs api
        if "logprobs" in kwargs:
            kwargs["top_logprobs"] = kwargs["logprobs"]
            kwargs["logprobs"] = True

        # max tokens is deprecated in favor of max_completion_tokens
        if "max_tokens" in kwargs:
            kwargs["max_completion_tokens"] = kwargs["max_tokens"]
            del kwargs["max_tokens"]

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
        if hasattr(api_response, 'error') and ('Rate limit exceeded' in api_response.error['message'] or api_response.error['code'] == 429): # OpenRouter routes through the error messages from the different providers, so we catch them here
            dummy_request = httpx.Request(method='POST', url='https://api.openai.com/v1/chat/completions', headers={'Authorization': f'Bearer {self.openai_api_key}'}, json={'messages': prompt.openai_format(), 'model': model_id, **kwargs})
            raise openai.RateLimitError(api_response.error['message'], response=httpx.Response(status_code=429, text=api_response.error['message'], request=dummy_request), body=api_response.error)
        api_duration = time.time() - api_start
        duration = time.time() - start_time
        context_token_cost, completion_token_cost = price_per_token(model_id)
        context_cost = api_response.usage.prompt_tokens * context_token_cost
        responses = [
            LLMResponse(
                model_id=model_id,
                completion=choice.message.content,
                stop_reason=choice.finish_reason,
                api_duration=api_duration,
                duration=duration,
                cost=context_cost + self.count_tokens(choice.message.content) * completion_token_cost,
                logprobs=(self.convert_top_logprobs(choice.logprobs) if choice.logprobs is not None else None),
            )
            for choice in api_response.choices
        ]
        self.add_response_to_prompt_file(prompt_file, responses)
        return responses

    async def make_stream_api_call(
        self,
        prompt: Prompt,
        model_id,
        **kwargs,
    ) -> openai.AsyncStream[openai.types.chat.ChatCompletionChunk]:
        return await self.aclient.chat.completions.create(
            messages=prompt.openai_format(),
            model=model_id,
            stream=True,
            **kwargs,
        )
