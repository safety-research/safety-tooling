import asyncio
import copy
import json
import logging
import time
from pathlib import Path
from traceback import format_exc

import anthropic.types
from anthropic import AsyncAnthropic
from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
from anthropic.types.messages.batch_create_params import Request
from langchain.tools import BaseTool

from safetytooling.data_models import ChatMessage, LLMResponse, MessageRole, Prompt, Usage
from safetytooling.utils.tool_utils import convert_tools_to_anthropic

from .model import InferenceAPIModel

ANTHROPIC_MODELS = {
    "claude-3-5-sonnet-latest",
    "claude-3-5-sonnet-20241022",
    "claude-3-5-sonnet-20240620",
    "claude-3-5-haiku-20241022",
    "claude-3-opus-20240229",
    "claude-3-sonnet-20240229",
    "claude-3-haiku-20240307",
    "research-claude-cabernet",
    "claude-3-7-sonnet-20250219",
    "claude-opus-4-1-20250805",
    "claude-opus-4-20250514",
    "claude-sonnet-4-20250514",
    "claude-sonnet-4-5-20250929",
    "claude-haiku-4-5-20251001",
    "claude-opus-4-5-20251101",
}

VISION_MODELS = {
    "claude-3-5-sonnet-20241022",
    "claude-3-5-sonnet-20240620",
    "claude-3-opus-20240229",
    "claude-3-7-sonnet-20250219",
}

LOGGER = logging.getLogger(__name__)


class AnthropicChatModel(InferenceAPIModel):
    def __init__(
        self,
        num_threads: int,
        prompt_history_dir: Path | None = None,
        anthropic_api_key: str | None = None,
    ):
        self.num_threads = num_threads
        self.prompt_history_dir = prompt_history_dir
        if anthropic_api_key:
            self.aclient = AsyncAnthropic(api_key=anthropic_api_key)
        else:
            self.aclient = AsyncAnthropic()

        self.available_requests = asyncio.BoundedSemaphore(int(self.num_threads))
        self.kwarg_change_name = {"stop": "stop_sequences"}

    def _convert_content_blocks_to_list(self, content_blocks) -> list:
        """Convert Anthropic content blocks to a serializable format."""
        result = []
        for block in content_blocks:
            if hasattr(block, "model_dump"):
                result.append(block.model_dump())
            else:
                # Fallback for blocks without model_dump
                block_dict = {"type": block.type}
                if block.type == "text":
                    block_dict["text"] = block.text
                elif block.type == "thinking":
                    block_dict["thinking"] = block.thinking
                    if hasattr(block, "signature"):
                        block_dict["signature"] = block.signature
                elif block.type == "redacted_thinking":
                    block_dict["data"] = block.data
                elif block.type == "tool_use":
                    block_dict["id"] = block.id
                    block_dict["name"] = block.name
                    block_dict["input"] = block.input
                else:
                    block_dict["data"] = str(block)
                result.append(block_dict)
        return result

    def _extract_text_completion(self, generated_content: list[ChatMessage]) -> str:
        """Extract the final text completion from generated content."""
        text_parts = []
        for msg in generated_content:
            if isinstance(msg.content, str):
                text_parts.append(msg.content)
            elif isinstance(msg.content, list):
                # Extract text from content blocks
                for block in msg.content:
                    if isinstance(block, dict) and block.get("type") == "text":
                        text_parts.append(block.get("text", ""))
        text_parts = [part for part in text_parts if part.strip() != ""]
        return "\n\n".join(text_parts) if text_parts else ""

    async def _execute_tool_loop(self, chat_messages, model_id, sys_prompt, anthropic_tools, tools, **kwargs):
        """Handle tool execution loop and return all generated content."""
        current_messages = chat_messages.copy()
        total_usage = Usage(input_tokens=0, output_tokens=0)
        all_generated_content = []

        while True:
            response = await self.aclient.messages.create(
                messages=current_messages,
                model=model_id,
                max_tokens=kwargs.get("max_tokens", 2000),
                tools=anthropic_tools,
                **{k: v for k, v in kwargs.items() if k != "max_tokens"},
                **(dict(system=sys_prompt) if sys_prompt else {}),
            )

            # Accumulate usage
            if response.usage:
                total_usage.input_tokens += response.usage.input_tokens
                total_usage.output_tokens += response.usage.output_tokens

            # Convert content blocks to serializable format and store as single message
            all_generated_content.append(
                ChatMessage(role=MessageRole.assistant, content=self._convert_content_blocks_to_list(response.content))
            )

            # Check if Claude wants to use tools
            tool_use_blocks = [block for block in response.content if block.type == "tool_use"]

            if not tool_use_blocks:
                # No more tool use, we're done
                break

            # Add Claude's response to the conversation
            current_messages.append({"role": "assistant", "content": response.content})

            # Execute each tool and collect results
            tool_results = []
            for tool_block in tool_use_blocks:
                # Find the corresponding LangChain tool
                langchain_tool = next((t for t in tools if t.name == tool_block.name), None)
                if langchain_tool:
                    try:
                        # Execute the tool
                        if hasattr(langchain_tool, "ainvoke"):
                            tool_result = await langchain_tool.ainvoke(tool_block.input)
                        elif hasattr(langchain_tool, "invoke"):
                            tool_result = langchain_tool.invoke(tool_block.input)
                        else:
                            raise ValueError(f"Tool {langchain_tool.name} does not have an invoke method")

                        result_dict = {
                            "type": "tool_result",
                            "tool_use_id": tool_block.id,
                            "content": str(tool_result),
                        }
                        tool_results.append(result_dict)
                    except Exception as e:
                        error_info = (
                            f"Exception Type: {type(e).__name__}, Error Details: {str(e)}, Traceback: {format_exc()}"
                        )
                        LOGGER.warn(
                            f"Encountered tool call {langchain_tool.name} error: {error_info}, passing error to LLM."
                        )
                        result_dict = {
                            "type": "tool_result",
                            "tool_use_id": tool_block.id,
                            "content": f"Error executing tool: {str(e)}",
                            "is_error": True,
                        }
                        tool_results.append(result_dict)
                else:
                    result_dict = {
                        "type": "tool_result",
                        "tool_use_id": tool_block.id,
                        "content": f"Tool {tool_block.name} not found",
                        "is_error": True,
                    }
                    tool_results.append(result_dict)

            # Add tool results to the conversation and generated content
            current_messages.append({"role": "user", "content": tool_results})

            for tool_result, tool_block in zip(tool_results, tool_use_blocks):
                tool_with_content_message = copy.deepcopy(tool_result)
                message = tool_with_content_message.pop("content")
                tool_with_content_message["message"] = message
                tool_with_content_message.pop("type")  # don't need this
                tool_with_content_message["tool_name"] = tool_block.name
                all_generated_content.append(ChatMessage(role=MessageRole.tool, content=tool_with_content_message))

        return response, all_generated_content, total_usage

    async def __call__(
        self,
        model_id: str,
        prompt: Prompt,
        print_prompt_and_response: bool,
        max_attempts: int,
        is_valid=lambda x: True,
        tools: list[BaseTool] | None = None,
        include_thinking_in_completion: bool = False,
        **kwargs,
    ) -> list[LLMResponse]:
        start = time.time()

        if prompt.contains_image():
            assert model_id in VISION_MODELS, f"Model {model_id} does not support images"

        # Convert tools if provided
        anthropic_tools = None
        if tools:
            anthropic_tools = convert_tools_to_anthropic(tools)

        # Rename based on kwarg_change_name
        for k, v in self.kwarg_change_name.items():
            if k in kwargs:
                kwargs[v] = kwargs[k]
                del kwargs[k]

        # Special case: ignore seed parameter since it's used for caching in safety-tooling
        if "seed" in kwargs:
            del kwargs["seed"]

        sys_prompt, chat_messages = prompt.anthropic_format()
        prompt_file = self.create_prompt_history_file(prompt, model_id, self.prompt_history_dir)

        LOGGER.debug(f"Making {model_id} call")
        response: anthropic.types.Message | None = None
        generated_content: list[ChatMessage] = []
        total_usage = None
        duration = None
        api_duration = None

        async with self.available_requests:
            error_list = []
            for i in range(max_attempts):
                try:
                    api_start = time.time()

                    # Handle tool execution if tools are provided
                    if anthropic_tools:
                        response, generated_content, total_usage = await self._execute_tool_loop(
                            chat_messages, model_id, sys_prompt, anthropic_tools, tools, **kwargs
                        )
                    else:
                        response = await self.aclient.messages.create(
                            messages=chat_messages,
                            model=model_id,
                            max_tokens=kwargs.pop("max_tokens", 2000),
                            **kwargs,
                            **(dict(system=sys_prompt) if sys_prompt else {}),
                        )
                        # Convert content blocks to serializable format and store as single message
                        content_list = self._convert_content_blocks_to_list(response.content)
                        generated_content = [ChatMessage(role=MessageRole.assistant, content=content_list)]

                        total_usage = (
                            Usage(input_tokens=response.usage.input_tokens, output_tokens=response.usage.output_tokens)
                            if response.usage
                            else None
                        )

                    api_duration = time.time() - api_start
                    if not is_valid(response):
                        raise RuntimeError(f"Invalid response according to is_valid {response}")
                except (TypeError, anthropic.NotFoundError) as e:
                    raise e
                except Exception as e:
                    error_info = (
                        f"Exception Type: {type(e).__name__}, Error Details: {str(e)}, Traceback: {format_exc()}"
                    )
                    LOGGER.warn(f"Encountered API error: {error_info}.\nRetrying now. (Attempt {i})")
                    error_list.append(error_info)
                    api_duration = time.time() - api_start
                    await asyncio.sleep(1.5**i)
                else:
                    break

        duration = time.time() - start
        LOGGER.debug(f"Completed call to {model_id} in {duration}s")

        if response is None:
            if model_id == "research-claude-cabernet":
                LOGGER.error(f"Failed to get a response for {prompt} from any API after {max_attempts} attempts.")
                # check for numbers of different kinds of errors
                n_prompt_blocked_errs = len([e for e in error_list if "Prompt blocked" in e])
                n_try_again_errs = len([e for e in error_list if "try again later" in e])
                LOGGER.error(
                    f"Encountered {n_prompt_blocked_errs} prompt blocked errors and {n_try_again_errs} try again later errors"
                )
                if n_prompt_blocked_errs > n_try_again_errs:
                    stop_reason = "prompt_blocked"
                else:
                    stop_reason = "api_error"

                response = LLMResponse(
                    model_id=model_id,
                    completion="",
                    generated_content=[],
                    stop_reason=stop_reason,
                    duration=duration,
                    api_duration=api_duration,
                    cost=0,
                    usage=None,
                )
            else:
                raise RuntimeError(f"Failed to get a response from the API after {max_attempts} attempts.")
        else:
            # Extract text completion from generated content
            completion = self._extract_text_completion(generated_content)

            # Handle edge case where content is empty (e.g., refusal)
            if len(response.content) == 0:
                completion = ""
                generated_content = []

            response = LLMResponse(
                model_id=model_id,
                completion=completion,
                generated_content=generated_content,
                stop_reason=response.stop_reason,
                duration=duration,
                api_duration=api_duration,
                cost=0,
                usage=total_usage,
            )

        responses = [response]

        self.add_response_to_prompt_file(prompt_file, responses)
        if print_prompt_and_response:
            prompt.pretty_print(responses)

        return responses

    def make_stream_api_call(
        self,
        model_id: str,
        prompt: Prompt,
        max_tokens: int,
        **params,
    ) -> anthropic.AsyncMessageStreamManager:
        sys_prompt, chat_messages = prompt.anthropic_format()
        return self.aclient.messages.stream(
            model=model_id,
            messages=chat_messages,
            **(dict(system=sys_prompt) if sys_prompt else {}),
            max_tokens=max_tokens,
            **params,
        )


class AnthropicModelBatch:
    def __init__(self, anthropic_api_key: str | None = None):
        if anthropic_api_key:
            self.client = anthropic.Anthropic(api_key=anthropic_api_key)
        else:
            self.client = anthropic.Anthropic()

    def create_message_batch(self, requests: list[dict]) -> dict:
        """Create a batch of messages."""
        return self.client.messages.batches.create(requests=requests)

    def retrieve_message_batch(self, batch_id: str) -> dict:
        """Retrieve a message batch."""
        return self.client.messages.batches.retrieve(batch_id)

    async def poll_message_batch(self, batch_id: str) -> dict:
        """Poll a message batch until it is completed."""
        message_batch = None
        minutes_elapsed = 0
        while True:
            message_batch = self.retrieve_message_batch(batch_id)
            if message_batch.processing_status == "ended":
                return message_batch
            if minutes_elapsed % 60 == 59:
                LOGGER.debug(f"Batch {batch_id} is still processing... ({minutes_elapsed} minutes elapsed)")
                print(f"Batch {batch_id} is still processing... ({minutes_elapsed} minutes elapsed)")
            await asyncio.sleep(60)  # Sleep for 1 minute
            minutes_elapsed += 1

    def list_message_batches(self, limit: int = 20) -> list[dict]:
        """List all message batches in the workspace."""
        return [batch for batch in self.client.messages.batches.list(limit=limit)]

    def retrieve_message_batch_results(self, batch_id: str) -> list[dict]:
        """Retrieve results of a message batch."""
        return [result for result in self.client.messages.batches.results(batch_id)]

    def cancel_message_batch(self, batch_id: str) -> dict:
        """Cancel a message batch."""
        return self.client.messages.batches.cancel(batch_id)

    def get_custom_id(self, index: int, prompt: Prompt) -> str:
        """Generate a custom ID for a batch request using index and prompt hash."""
        hash_str = prompt.model_hash()
        return f"{index}_{hash_str}"

    def prompts_to_requests(self, model_id: str, prompts: list[Prompt], max_tokens: int, **kwargs) -> list[Request]:
        # Special case: ignore seed parameter since it's used for caching in safety-tooling
        if "seed" in kwargs:
            del kwargs["seed"]

        requests = []
        for i, prompt in enumerate(prompts):
            sys_prompt, chat_messages = prompt.anthropic_format()
            requests.append(
                Request(
                    custom_id=self.get_custom_id(i, prompt),
                    params=MessageCreateParamsNonStreaming(
                        model=model_id,
                        messages=chat_messages,
                        max_tokens=max_tokens,
                        **(dict(system=sys_prompt) if sys_prompt else {}),
                        **kwargs,
                    ),
                )
            )
        return requests

    def _convert_content_blocks_to_list(self, content_blocks) -> list:
        """Convert Anthropic content blocks to a serializable format."""
        result = []
        for block in content_blocks:
            if hasattr(block, "model_dump"):
                result.append(block.model_dump())
            else:
                # Fallback for blocks without model_dump
                block_dict = {"type": block.type}
                if block.type == "text":
                    block_dict["text"] = block.text
                elif block.type == "thinking":
                    block_dict["thinking"] = block.thinking
                    if hasattr(block, "signature"):
                        block_dict["signature"] = block.signature
                else:
                    block_dict["data"] = str(block)
                result.append(block_dict)
        return result

    def _extract_text_completion(self, generated_content: list[ChatMessage]) -> str:
        """Extract the final text completion from generated content."""
        text_parts = []
        for msg in generated_content:
            if isinstance(msg.content, str):
                text_parts.append(msg.content)
            elif isinstance(msg.content, list):
                # Extract text from content blocks
                for block in msg.content:
                    if isinstance(block, dict) and block.get("type") == "text":
                        text_parts.append(block.get("text", ""))
        return "\n".join(text_parts) if text_parts else ""

    async def __call__(
        self,
        model_id: str,
        prompts: list[Prompt],
        max_tokens: int,
        log_dir: Path | None = None,
        **kwargs,
    ) -> tuple[list[LLMResponse], str]:
        assert max_tokens is not None, "Anthropic batch API requires max_tokens to be specified"
        start = time.time()

        custom_ids = [self.get_custom_id(i, prompt) for i, prompt in enumerate(prompts)]
        set_custom_ids = set(custom_ids)
        assert len(set_custom_ids) == len(custom_ids), "Duplicate custom IDs found"

        requests = self.prompts_to_requests(model_id, prompts, max_tokens, **kwargs)
        batch_response = self.create_message_batch(requests=requests)
        batch_id = batch_response.id
        if log_dir is not None:
            log_file = log_dir / f"batch_id_{batch_id}.json"
            log_file.parent.mkdir(parents=True, exist_ok=True)
            with open(log_file, "w") as f:
                json.dump(batch_response.model_dump(mode="json"), f)
        print(f"Batch {batch_id} created with {len(prompts)} requests")

        _ = await self.poll_message_batch(batch_id)

        results = self.retrieve_message_batch_results(batch_id)
        LOGGER.info(f"{results[0]=}")

        responses_dict = {}
        for result in results:
            if result.result.type == "succeeded":
                content = result.result.message.content
                usage_data = result.result.message.usage

                # Convert content blocks to serializable format and store as single message
                generated_content = []
                if content:
                    content_list = self._convert_content_blocks_to_list(content)
                    generated_content = [ChatMessage(role=MessageRole.assistant, content=content_list)]

                # Extract text completion
                text = self._extract_text_completion(generated_content)

                assert result.custom_id in set_custom_ids
                responses_dict[result.custom_id] = LLMResponse(
                    model_id=model_id,
                    completion=text,
                    generated_content=generated_content,
                    stop_reason=result.result.message.stop_reason,
                    duration=None,  # Batch does not track individual durations
                    api_duration=None,
                    cost=0,
                    batch_custom_id=result.custom_id,
                    usage=(
                        Usage(input_tokens=usage_data.input_tokens, output_tokens=usage_data.output_tokens)
                        if usage_data
                        else None
                    ),
                )

        responses = []
        for custom_id in custom_ids:
            responses.append(responses_dict[custom_id] if custom_id in responses_dict else None)

        duration = time.time() - start
        LOGGER.debug(f"Completed batch call in {duration}s")

        return responses, batch_id
