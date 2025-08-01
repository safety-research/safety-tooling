import asyncio
import json
import logging
import time
from pathlib import Path
from traceback import format_exc

from langchain.tools import BaseTool
from openai import AsyncOpenAI, BadRequestError

from safetytooling.data_models import ChatMessage, LLMResponse, MessageRole, Prompt
from safetytooling.utils.tool_utils import convert_tools_to_openai

from .model import InferenceAPIModel

OPENROUTER_MODELS = {
    "x-ai/grok-3-beta",
    "google/gemini-2.0-flash-001",
    "mistral/ministral-8b",
    "mistralai/mistral-large-2411",
    "google/gemini-2.5-flash-preview",
    "google/gemini-2.0-flash-001",
    "deepseek/deepseek-v3-base:free",
    "deepseek/deepseek-r1-zero:free",
    "meta-llama/llama-3.1-405b",
    "meta-llama/llama-3.1-405b:free",
    "nousresearch/hermes-3-llama-3.1-70b",
    # DeepInfra models accessible through OpenRouter
    "meta-llama/llama-3.1-8b-instruct",
    "meta-llama/llama-3.1-70b-instruct",
    "meta-llama/llama-3.3-70b-instruct",
    "meta-llama/llama-3.1-405b-instruct",
    "nousresearch/hermes-3-llama-3.1-405b",
    "mistralai/mistral-small-24b-instruct-2501",
}

# Models that support tool calling
OPENROUTER_TOOL_MODELS = {
    "google/gemini-2.0-flash-001",
    "google/gemini-2.5-flash-preview",
    "mistral/ministral-8b",
    "mistralai/mistral-large-2411",
    "meta-llama/llama-3.1-405b",
    "meta-llama/llama-3.1-405b:free",
    "nousresearch/hermes-3-llama-3.1-70b",
    # DeepInfra models that support tools
    "meta-llama/llama-3.1-8b-instruct",
    "meta-llama/llama-3.1-70b-instruct",
    "meta-llama/llama-3.1-405b-instruct",
    "nousresearch/hermes-3-llama-3.1-405b",
    "mistralai/mistral-small-24b-instruct-2501",
}

LOGGER = logging.getLogger(__name__)


class OpenRouterChatModel(InferenceAPIModel):
    def __init__(
        self,
        num_threads: int,
        prompt_history_dir: Path | None = None,
        api_key: str | None = None,
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

    def _convert_message_to_chat_message(self, message) -> ChatMessage:
        """Convert OpenAI message to ChatMessage format."""
        # Handle assistant messages with potential tool calls
        if message.role == "assistant":
            content_parts = []

            # Add text content if present
            if message.content:
                content_parts.append({"type": "text", "text": message.content})

            # Add tool calls if present
            if hasattr(message, "tool_calls") and message.tool_calls:
                for tool_call in message.tool_calls:
                    content_parts.append(
                        {
                            "type": "tool_call",
                            "id": tool_call.id,
                            "function": {"name": tool_call.function.name, "arguments": tool_call.function.arguments},
                        }
                    )

            # Return appropriate format
            if not content_parts:
                return ChatMessage(role=MessageRole.assistant, content="")
            elif len(content_parts) == 1 and content_parts[0]["type"] == "text":
                return ChatMessage(role=MessageRole.assistant, content=content_parts[0]["text"])
            else:
                return ChatMessage(role=MessageRole.assistant, content=content_parts)
        else:
            # For other roles, just use the content
            return ChatMessage(role=MessageRole(message.role), content=message.content or "")

    def _extract_text_completion(self, generated_content: list[ChatMessage]) -> str:
        """Extract the final text completion from generated content."""
        text_parts = []
        for msg in generated_content:
            if msg.role == MessageRole.assistant:
                if isinstance(msg.content, str):
                    text_parts.append(msg.content)
                elif isinstance(msg.content, list):
                    for item in msg.content:
                        if isinstance(item, dict) and item.get("type") == "text":
                            text_parts.append(item.get("text", ""))
        text_parts = [part for part in text_parts if part.strip() != ""]
        return "\n\n".join(text_parts) if text_parts else ""

    async def _execute_tool_loop(self, messages, model_id, openai_tools, tools, **kwargs):
        """Handle OpenAI-style tool execution loop and return all generated content."""
        current_messages = messages.copy()
        all_generated_content = []

        while True:
            response_data = await self.aclient.chat.completions.create(
                messages=current_messages,
                model=model_id,
                tools=openai_tools,
                tool_choice="auto",
                **kwargs,
            )

            if (
                response_data.choices is None
                or len(response_data.choices) == 0
                or response_data.choices[0].message is None
            ):
                raise RuntimeError(f"Empty response from {model_id}")

            message = response_data.choices[0].message

            # Convert to ChatMessage and add to generated content
            assistant_msg = self._convert_message_to_chat_message(message)
            all_generated_content.append(assistant_msg)

            # Add to conversation in OpenAI format
            msg_dict = {
                "role": "assistant",
                "content": message.content,
            }
            if hasattr(message, "tool_calls") and message.tool_calls:
                msg_dict["tool_calls"] = message.tool_calls
            current_messages.append(msg_dict)

            # Check if there are tool calls
            if not hasattr(message, "tool_calls") or message.tool_calls is None:
                # No more tool calls, we're done
                break

            # Execute each tool and collect results
            for tool_call in message.tool_calls:
                # Find the corresponding LangChain tool
                langchain_tool = next((t for t in tools if t.name == tool_call.function.name), None)
                if langchain_tool:
                    try:
                        # Parse arguments and execute the tool
                        tool_args = json.loads(tool_call.function.arguments)
                        if hasattr(langchain_tool, "ainvoke"):
                            tool_result = await langchain_tool.ainvoke(tool_args)
                        elif hasattr(langchain_tool, "invoke"):
                            tool_result = langchain_tool.invoke(tool_args)
                        else:
                            raise ValueError(f"Tool {langchain_tool.name} does not have an invoke method")

                        # Create tool result ChatMessage
                        tool_msg = ChatMessage(
                            role=MessageRole.tool,
                            content={
                                "tool_call_id": tool_call.id,
                                "name": tool_call.function.name,
                                "content": str(tool_result),
                            },
                        )
                        all_generated_content.append(tool_msg)

                        # Add to conversation in OpenAI format
                        current_messages.append(
                            {
                                "role": "tool",
                                "tool_call_id": tool_call.id,
                                "name": tool_call.function.name,
                                "content": str(tool_result),
                            }
                        )

                    except Exception as e:
                        # Create error tool result
                        error_msg = ChatMessage(
                            role=MessageRole.tool,
                            content={
                                "tool_call_id": tool_call.id,
                                "name": tool_call.function.name,
                                "content": f"Error executing tool: {str(e)}",
                            },
                        )
                        all_generated_content.append(error_msg)

                        current_messages.append(
                            {
                                "role": "tool",
                                "tool_call_id": tool_call.id,
                                "name": tool_call.function.name,
                                "content": f"Error executing tool: {str(e)}",
                            }
                        )
                else:
                    # Tool not found error
                    error_msg = ChatMessage(
                        role=MessageRole.tool,
                        content={
                            "tool_call_id": tool_call.id,
                            "name": tool_call.function.name,
                            "content": f"Tool {tool_call.function.name} not found",
                        },
                    )
                    all_generated_content.append(error_msg)

                    current_messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "name": tool_call.function.name,
                            "content": f"Tool {tool_call.function.name} not found",
                        }
                    )

        return response_data, all_generated_content

    async def __call__(
        self,
        model_id: str,
        prompt: Prompt,
        print_prompt_and_response: bool,
        max_attempts: int,
        is_valid=lambda x: True,
        tools: list[BaseTool] | None = None,
        **kwargs,
    ) -> list[LLMResponse]:

        if self.aclient is None:
            raise RuntimeError("OPENROUTER_API_KEY environment variable must be set in .env before running your script")

        # Convert tools if provided
        openai_tools = None
        if tools:
            assert model_id in OPENROUTER_TOOL_MODELS, f"Model {model_id} does not support tools"
            openai_tools = convert_tools_to_openai(tools)

        start = time.time()
        prompt_file = self.create_prompt_history_file(prompt, model_id, self.prompt_history_dir)

        # OpenRouter handles logprobs the same as OpenAI
        if "logprobs" in kwargs:
            kwargs["top_logprobs"] = kwargs["logprobs"]
            kwargs["logprobs"] = True

        LOGGER.debug(f"Making {model_id} call")
        response_data = None
        generated_content = []
        duration = None
        api_duration = None

        error_list = []
        for i in range(max_attempts):
            try:
                async with self.available_requests:
                    api_start = time.time()

                    # Handle tool execution if tools are provided
                    if openai_tools:
                        response_data, generated_content = await self._execute_tool_loop(
                            prompt.openai_format(), model_id, openai_tools, tools, **kwargs
                        )
                    else:
                        response_data = await self.aclient.chat.completions.create(
                            messages=prompt.openai_format(),
                            model=model_id,
                            **kwargs,
                        )
                        # Convert single response to ChatMessage
                        if response_data.choices and response_data.choices[0].message:
                            generated_content = [
                                self._convert_message_to_chat_message(response_data.choices[0].message)
                            ]

                    api_duration = time.time() - api_start

                    if (
                        response_data.choices is None
                        or len(response_data.choices) == 0
                        or (
                            response_data.choices[0].message.content is None
                            or response_data.choices[0].message.content == ""
                        )
                        and not (
                            hasattr(response_data.choices[0].message, "tool_calls")
                            and response_data.choices[0].message.tool_calls
                        )
                    ):
                        # sometimes gemini will never return a response
                        if model_id == "google/gemini-2.0-flash-001":
                            LOGGER.warn(f"Empty response from {model_id} (returning empty response)")
                            return [
                                LLMResponse(
                                    model_id=model_id,
                                    completion="",
                                    generated_content=[],
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
                LOGGER.warn(f"Encountered API error: {error_info}.\nRetrying now. (Attempt {i})")
                error_list.append(error_info)
                api_duration = time.time() - api_start
                await asyncio.sleep(1.5**i)
            else:
                break

        duration = time.time() - start
        LOGGER.debug(f"Completed call to {model_id} in {duration}s")

        if response_data is None or response_data.choices is None or len(response_data.choices) == 0:
            raise RuntimeError("No response data received")

        # Extract text completion from generated content
        completion = self._extract_text_completion(generated_content)

        # For multiple choices (n > 1), we only support non-tool scenarios
        if kwargs.get("n", 1) > 1:
            assert not tools, "Multiple choices not supported with tools"
            responses = []
            for choice in response_data.choices:
                responses.append(
                    LLMResponse(
                        model_id=model_id,
                        completion=choice.message.content or "",
                        generated_content=[self._convert_message_to_chat_message(choice.message)],
                        stop_reason=choice.finish_reason,
                        api_duration=api_duration,
                        duration=duration,
                        cost=0,
                        logprobs=(
                            self.convert_top_logprobs(choice.logprobs)
                            if hasattr(choice, "logprobs") and choice.logprobs is not None
                            else None
                        ),
                    )
                )
        else:
            # Single response with potential tool use
            responses = [
                LLMResponse(
                    model_id=model_id,
                    completion=completion,
                    generated_content=generated_content,
                    stop_reason=response_data.choices[0].finish_reason,
                    api_duration=api_duration,
                    duration=duration,
                    cost=0,
                    logprobs=(
                        self.convert_top_logprobs(response_data.choices[0].logprobs)
                        if hasattr(response_data.choices[0], "logprobs")
                        and response_data.choices[0].logprobs is not None
                        else None
                    ),
                )
            ]

        self.add_response_to_prompt_file(prompt_file, responses)
        if print_prompt_and_response:
            prompt.pretty_print(responses)

        return responses
