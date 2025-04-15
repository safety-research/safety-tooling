import asyncio
import logging
import time
from pathlib import Path
from traceback import format_exc

from openai import AsyncOpenAI

from safetytooling.data_models import LLMResponse, Prompt, MessageRole

from .model import InferenceAPIModel

# Model IDs as they should be used
DEEPINFRA_MODELS = {
    "deepinfra:deepseek/deepseek-r1",
    "deepinfra:meta-llama/llama-3.3-70b-instruct",
    "deepinfra:meta-llama/llama-3.1-405b-instruct",
}  # Add other models as needed
LOGGER = logging.getLogger(__name__)


def preprocess_prompt(prompt: Prompt) -> Prompt:
    """
    Preprocess the prompt to ensure it is in the correct format.
    """
    # Swap <SCRATCHPAD_REASONING> with <think>
    if prompt.messages[-1].role == MessageRole.assistant:
        if "<SCRATCHPAD_REASONING>" in prompt.messages[-1].content:
            prompt.messages[-1].content = prompt.messages[-1].content.replace("<SCRATCHPAD_REASONING>", "<think>")
        if "</SCRATCHPAD_REASONING>" in prompt.messages[-1].content:
            prompt.messages[-1].content = prompt.messages[-1].content.replace("</SCRATCHPAD_REASONING>", "</think>")
    return prompt

def postprocess_response(response: LLMResponse) -> LLMResponse:
    """
    Postprocess the response to ensure it is in the correct format.
    """
    # Convert <think> to <SCRATCHPAD_REASONING>
    if response.completion.startswith("<think>"):
        response.completion = response.completion.replace("<think>", "<SCRATCHPAD_REASONING>")
    if response.completion.endswith("</think>"):
        response.completion = response.completion.replace("</think>", "</SCRATCHPAD_REASONING>")
    return response


class DeepInfraModel(InferenceAPIModel):
    def __init__(
        self,
        num_threads: int,
        prompt_history_dir: Path | None = None,
        api_key: str | None = None,
        site_url: str | None = None,
        site_name: str | None = None,
    ):
        self.num_threads = num_threads
        self.prompt_history_dir = prompt_history_dir
        self.api_key = api_key  # This will be OpenRouter API key
        self.site_url = site_url
        self.site_name = site_name
        self.available_requests = asyncio.BoundedSemaphore(int(self.num_threads))
        
        if api_key is not None:
            try:
                self.aclient = AsyncOpenAI(
                    api_key=api_key,
                    base_url="https://openrouter.ai/api/v1",
                )
            except TypeError as e:
                LOGGER.error(f"Failed to initialize OpenRouter client: {e}")
                self.aclient = None
        else:
            self.aclient = None

    async def __call__(
        self,
        model_id: str,
        prompt: Prompt,
        print_prompt_and_response: bool,
        max_attempts: int,
        is_valid=lambda x: True,
        **kwargs,
    ) -> list[LLMResponse]:        
        # Remove the deepinfra: prefix if present
        if model_id.startswith("deepinfra:"):
            openrouter_model_id = model_id.replace("deepinfra:", "")
        else:
            openrouter_model_id = model_id
        
        if "deepseek-r1" in openrouter_model_id.lower():
            prompt = preprocess_prompt(prompt) # Convert <SCRATCHPAD_REASONING> to <think>
        
        if self.aclient is None:
            raise RuntimeError("OPENROUTER_API_KEY environment variable must be set in .env before running your script")
        
        start = time.time()
        prompt_file = self.create_prompt_history_file(prompt, model_id, self.prompt_history_dir)

        LOGGER.debug(f"Making {openrouter_model_id} call through OpenRouter with direct routing")
        response_data = None
        api_duration = None

        # Prepare extra headers for OpenRouter
        extra_headers = {}
        if self.site_url:
            extra_headers["HTTP-Referer"] = self.site_url
        if self.site_name:
            extra_headers["X-Title"] = self.site_name
        
        # Add header to prevent OpenRouter from transforming the request
        extra_headers["X-OpenRouter-No-Transform"] = "true"

        for i in range(max_attempts):
            try:
                async with self.available_requests:
                    api_start = time.time()

                    # Set up extra_body
                    extra_body = kwargs.pop("extra_body", {})
                    
                    # Use route parameter to ensure direct routing to DeepInfra
                    extra_body["provider"] = {
                        "order": ["DeepInfra"],
                        "allow_fallbacks": False
                    }
                    
                    if "deepseek-r1" in openrouter_model_id.lower():
                        extra_body["include_reasoning"] = True
                    
                    # Make the API call through OpenRouter
                    response = await self.aclient.chat.completions.create(
                        model=openrouter_model_id,
                        messages=prompt.together_format(),
                        extra_headers=extra_headers,
                        extra_body=extra_body,
                        **kwargs,
                    )
                    
                    # Process the output
                    if "deepseek-r1" in openrouter_model_id.lower():
                        output = response.choices[0].message.content
                        if response.choices[0].message.reasoning:
                            reasoning = response.choices[0].message.reasoning.rstrip()
                        else:
                            reasoning  = None
                        
                        has_prefill = prompt.messages[-1].role == MessageRole.assistant
                        prefill_starts_with_think = prompt.messages[-1].content.startswith("<think>")
                        prefill_has_think_ending = "</think>" in prompt.messages[-1].content
                        
                        if reasoning is None:
                            pass # no reasoning, so no need to modify the output
                        elif has_prefill and prefill_starts_with_think and not prefill_has_think_ending:
                            output = f"{reasoning}\n</think>\n{output}"
                        elif has_prefill:
                            pass # output is already in the correct format
                        else:
                            output = f"<think>\n{reasoning}\n</think>\n{output}"                  
                    else:
                        output = response.choices[0].message.content
                    
                    api_duration = time.time() - api_start
                    
                    if not is_valid(output):
                        raise RuntimeError(f"Invalid response according to is_valid {response}")
                    break
            except TypeError as e:
                raise e
            except Exception as e:
                error_info = f"Exception Type: {type(e).__name__}, Error Details: {str(e)}, Traceback: {format_exc()}"
                LOGGER.warn(f"Encountered API error: {error_info}.\nRetrying now. (Attempt {i+1}/{max_attempts})")
                await asyncio.sleep(1.5**i)
        else:
            raise RuntimeError(f"Failed to get a response from the API after {max_attempts} attempts")

        duration = time.time() - start

        response_obj = LLMResponse(
            model_id=model_id,  # Use the original model ID with the prefix
            completion=output,
            stop_reason=response.choices[0].finish_reason,
            duration=duration,
            api_duration=api_duration,
            cost=0,  # OpenRouter will handle the cost calculation
            prompt_tokens=response.usage.prompt_tokens if hasattr(response, 'usage') else None,
            completion_tokens=response.usage.completion_tokens if hasattr(response, 'usage') else None,
            reasoning_content=None,
        )
        responses = [response_obj]

        self.add_response_to_prompt_file(prompt_file, responses)
        if print_prompt_and_response:
            prompt.pretty_print(responses)

        return responses