import collections
import copy
import json
import logging
import os
import time
from traceback import format_exc

import httpx
import openai
import openai.types
import openai.types.chat
import pydantic
import requests
from langchain.tools import BaseTool
from tenacity import retry, stop_after_attempt, wait_fixed

from safetytooling.data_models import ChatMessage, LLMResponse, MessageRole, Prompt, Usage
from safetytooling.utils import math_utils
from safetytooling.utils.tool_utils import convert_tools_to_openai

from .base import OpenAIModel
from .utils import get_rate_limit, price_per_token

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

    async def _execute_tool_loop(self, chat_messages, model_id, openai_tools, tools, api_func, **kwargs):
        """Handle tool execution loop and return all generated content."""
        current_messages = chat_messages.copy()
        total_usage = Usage(input_tokens=0, output_tokens=0)
        all_generated_content = []

        response = await api_func(
            messages=current_messages,
            model=model_id,
            tools=openai_tools,
            **kwargs,
        )
        if response.usage:
            total_usage.input_tokens += response.usage.prompt_tokens
            total_usage.output_tokens += response.usage.completion_tokens
        self._raise_any_response_errors(response, messages=current_messages, model_id=model_id)
        # remove n param so it's default 1 now
        if "n" in kwargs:
            kwargs.pop("n")
        result_choices = []
        # for n>1, we need to run tool stuff on each choice seperately
        for choice in response.choices:
            choice_messages = copy.deepcopy(current_messages)
            choice_generated_content = []
            while True:
                choice_generated_content.append(ChatMessage(role=MessageRole.assistant, content=choice.model_dump()))
                choice_messages.append(choice.message)
                if choice.message.tool_calls is None or len(choice.message.tool_calls) == 0:
                    break
                tool_results = []
                for tool_call in choice.message.tool_calls:
                    langchain_tool = next((t for t in tools if t.name == tool_call.function.name), None)
                    if langchain_tool:
                        try:
                            tool_arguments = json.loads(tool_call.function.arguments)
                            # Execute the tool
                            if hasattr(langchain_tool, "ainvoke"):
                                tool_result = await langchain_tool.ainvoke(tool_arguments)
                            elif hasattr(langchain_tool, "invoke"):
                                tool_result = langchain_tool.invoke(tool_arguments)
                            else:
                                raise ValueError(f"Tool {langchain_tool.name} does not have an invoke method")

                            result_dict = {
                                "role": "tool",
                                "tool_call_id": tool_call.id,
                                "content": str(tool_result),
                            }
                            tool_results.append(result_dict)
                        except Exception as e:
                            error_info = f"Exception Type: {type(e).__name__}, Error Details: {str(e)}, Traceback: {format_exc()}"
                            LOGGER.warn(
                                f"Encountered tool call {langchain_tool.name} error: {error_info}, passing error to LLM."
                            )
                            result_dict = {
                                "role": "tool",
                                "tool_call_id": tool_call.id,
                                "content": f"Error executing tool: {str(e)}",
                                # "is_error": True,
                            }
                            tool_results.append(result_dict)
                    else:
                        result_dict = {
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": f"Tool {tool_call.function.name} not found",
                            # "is_error": True,
                        }
                        tool_results.append(result_dict)

                # replace 'content' with 'message' so it looks nicer
                for tool_call, tool_result in zip(choice.message.tool_calls, tool_results):
                    tool_with_content_message = copy.deepcopy(tool_result)
                    message = tool_with_content_message.pop("content")
                    tool_with_content_message["message"] = message
                    tool_with_content_message["tool_name"] = tool_call.function.name
                    choice_generated_content.append(
                        ChatMessage(role=MessageRole.tool, content=tool_with_content_message)
                    )
                choice_messages.extend(tool_results)
                choice_response = await api_func(
                    messages=choice_messages,
                    model=model_id,
                    tools=openai_tools,
                    **kwargs,
                )
                self._raise_any_response_errors(choice_response, messages=choice_messages, model_id=model_id)
                if choice_response.usage:
                    total_usage.input_tokens += choice_response.usage.prompt_tokens
                    total_usage.output_tokens += choice_response.usage.completion_tokens
                choice = choice_response.choices[0]
            all_generated_content.append(choice_generated_content)
            result_choices.append(choice)
        return result_choices, all_generated_content, total_usage

    def _raise_any_response_errors(self, api_response: openai.types.chat.ChatCompletion, messages, model_id, **kwargs):
        if hasattr(api_response, "error") and (
            "Rate limit exceeded" in api_response.error["message"] or api_response.error["code"] == 429
        ):  # OpenRouter routes through the error messages from the different providers, so we catch them here
            dummy_request = httpx.Request(
                method="POST",
                url="https://api.openai.com/v1/chat/completions",
                headers={"Authorization": f"Bearer {self.openai_api_key}"},
                json={"messages": messages, "model": model_id, **kwargs},
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

    async def _make_api_call(
        self, prompt: Prompt, model_id, start_time, tools: list[BaseTool] | None = None, **kwargs
    ) -> list[LLMResponse]:
        if self.aclient is None:
            raise RuntimeError("OpenAI API key must be set either via parameter or OPENAI_API_KEY environment variable")

        LOGGER.debug(f"Making {model_id} call")

        openai_tools = None
        if tools:
            openai_tools = convert_tools_to_openai(tools)

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

        if openai_tools is None:
            # non-tool use
            api_response: openai.types.chat.ChatCompletion = await api_func(
                messages=prompt.openai_format(),
                model=model_id,
                **kwargs,
            )
            self._raise_any_response_errors(api_response, messages=prompt.openai_format(), model_id=model_id)
            choices = api_response.choices
            all_generated_content = [
                [ChatMessage(role=MessageRole.assistant, content=choice.model_dump())]
                for choice in api_response.choices
            ]
            if api_response.usage is not None:
                total_usage = Usage(
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
                total_usage = None
        else:
            # tool use
            choices, all_generated_content, total_usage = await self._execute_tool_loop(
                chat_messages=prompt.openai_format(),
                model_id=model_id,
                openai_tools=openai_tools,
                tools=tools,
                api_func=api_func,
                **kwargs,
            )

        # Calculate cost
        input_token_cost, output_token_cost = price_per_token(model_id)
        input_cost = input_token_cost * total_usage.input_tokens if total_usage else 0
        output_cost = output_token_cost * total_usage.output_tokens if total_usage else 0

        # Timing
        api_duration = time.time() - api_start
        duration = time.time() - start_time

        responses = []
        for choice, generated_content in zip(choices, all_generated_content):
            if choice.message.content is None or choice.finish_reason is None:
                raise RuntimeError(f"No content or finish reason for {model_id}")
            responses.append(
                LLMResponse(
                    model_id=model_id,
                    completion=choice.message.content,
                    stop_reason=choice.finish_reason,
                    api_duration=api_duration,
                    duration=duration,
                    cost=input_cost + output_cost,
                    logprobs=(self.convert_top_logprobs(choice.logprobs) if choice.logprobs is not None else None),
                    usage=total_usage,
                    generated_content=generated_content,
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
