"""Basic tests for the api."""

import pydantic
import pytest

from safetytooling.apis.inference.api import InferenceAPI
from safetytooling.data_models import ChatMessage, MessageRole, Prompt
from safetytooling.utils import utils


def test_default_global_api():
    utils.setup_environment()
    api1 = InferenceAPI.get_default_global_api()
    api2 = InferenceAPI.get_default_global_api()
    assert api1 is api2


@pytest.mark.asyncio
async def test_openai():
    utils.setup_environment()
    resp = await InferenceAPI.get_default_global_api().ask_single_question(
        model_id="gpt-4o-mini",
        question="What is the meaning of life?",
        max_tokens=1,
        n=2,
    )
    assert isinstance(resp, list)
    assert len(resp) == 2
    assert isinstance(resp[0], str)
    assert isinstance(resp[1], str)


@pytest.mark.skip(reason="Skipping tool call test while langchain environment issue is resolved.")
@pytest.mark.asyncio
async def test_tool_call():
    """
    # If you just want to test for tool call, you can simply do
    for content in (resp[0].generated_content if resp[0].generated_content is not None else []):
        if content.role == MessageRole.tool:
            tool_output = content["message"]
            tool_name = content["tool_name"]
    """

    utils.setup_environment()
    TOOL_NAME = "Thermometer_tool"
    TOOL_OUTPUT = "The temperature is 42 degrees F."

    def testToolFunc():  # This gets appended to output and then model responds
        return TOOL_OUTPUT

    from langchain_core.tools import StructuredTool

    tools = [
        StructuredTool.from_function(func=testToolFunc, name="Thermometer_tool", description="Gives the temperature.")
    ]
    for model in ["claude-3-5-haiku-20241022", "gpt-4o-mini", "x-ai/grok-4"]:
        resp = await InferenceAPI(cache_dir=None)(
            model_id=model,
            prompt=Prompt(
                messages=[ChatMessage(role=MessageRole.user, content="What is the temperature? (please use the tool)")]
            ),
            max_tokens=1000,
            tools=tools,
        )
        assert len(resp[0].generated_content) == 3  # Should have assistant, tool output, assistant

        # The assistant calling the tool
        assert resp[0].generated_content[0].role == MessageRole.assistant
        if model == "gpt-4o-mini":
            # OpenAI models either call tools or say something, never both
            assert resp[0].generated_content[0].content["message"]["content"] is None
            tool_calls = resp[0].generated_content[0].content["message"]["tool_calls"]
            assert tool_calls is not None
            assert len(tool_calls) == 1
            assert tool_calls[0]["function"]["name"] == TOOL_NAME
        elif model == "claude-3-5-haiku-20241022":
            # Claude can say things and also call tools, so don't assert content is None
            tool_use_content = [x for x in resp[0].generated_content[0].content if x["type"] == "tool_use"]
            assert len(tool_use_content) == 1
            assert tool_use_content[0]["name"] == TOOL_NAME
        elif model == "x-ai/grok-4":
            # OpenRouter models use OpenAI-style format with tool calls in a list
            tool_call_content = [x for x in resp[0].generated_content[0].content if x["type"] == "tool_call"]
            assert len(tool_call_content) == 1
            assert tool_call_content[0]["function"]["name"] == TOOL_NAME

        # The call to the tool
        assert resp[0].generated_content[1].role == MessageRole.tool
        tool_name = resp[0].generated_content[1].content["tool_name"]
        assert tool_name == TOOL_NAME
        tool_output = resp[0].generated_content[1].content["message"]
        assert tool_output == TOOL_OUTPUT

        # This will have the response the assistant gives (usually like "The temprature is 42 degrees Fahrenheit.")
        assert resp[0].generated_content[2].role == MessageRole.assistant


@pytest.mark.asyncio
async def test_claude_3():
    utils.setup_environment()
    resp = await InferenceAPI.get_default_global_api().ask_single_question(
        model_id="claude-3-haiku-20240307",
        question="What is the meaning of life?",
        max_tokens=1,
        n=2,
    )
    assert isinstance(resp, list)
    assert len(resp) == 2
    assert isinstance(resp[0], str)
    assert isinstance(resp[1], str)


@pytest.mark.asyncio
async def test_claude_3_with_system_prompt():
    utils.setup_environment()
    resp = await InferenceAPI.get_default_global_api().ask_single_question(
        model_id="claude-3-haiku-20240307",
        question="What is the meaning of life?",
        system_prompt="You are an AI assistant who has never read the Hitchhiker's Guide to the Galaxy.",
        max_tokens=1,
        n=2,
    )
    assert isinstance(resp, list)
    assert len(resp) == 2
    assert isinstance(resp[0], str)
    assert isinstance(resp[1], str)


@pytest.mark.asyncio
async def test_gpt4o_mini_2024_07_18_with_response_format():
    utils.setup_environment()

    class CustomResponse(pydantic.BaseModel):
        answer: str
        confidence: float

    resp = await InferenceAPI.get_default_global_api().ask_single_question(
        model_id="gpt-4o-mini-2024-07-18",
        question="What is the meaning of life? Format your response as a JSON object with fields: answer (string) and confidence (float between 0 and 1).",
        max_tokens=100,
        n=2,
        response_format=CustomResponse,
    )
    assert isinstance(resp, list)
    assert len(resp) == 2
    assert isinstance(resp[0], str)
    assert isinstance(resp[1], str)

    # Verify responses can be parsed into CustomResponse objects
    CustomResponse.model_validate_json(resp[0])
    CustomResponse.model_validate_json(resp[1])


@pytest.mark.asyncio
async def test_api_with_stop_parameter():
    """Test that the stop parameter works correctly with the InferenceAPI."""
    utils.setup_environment()

    # Test with OpenAI model
    openai_resp = await InferenceAPI.get_default_global_api().ask_single_question(
        model_id="gpt-4o-mini-2024-07-18",
        question="Count from 1 to 10: 1, 2, 3,",
        max_tokens=20,
        stop=[", 5"],  # Should stop before outputting ", 5"
    )

    assert isinstance(openai_resp, list)
    assert len(openai_resp) == 1
    assert isinstance(openai_resp[0], str)
    assert ", 5" not in openai_resp[0]
    assert "4" in openai_resp[0]

    # Test with Anthropic model
    anthropic_resp = await InferenceAPI.get_default_global_api().ask_single_question(
        model_id="claude-3-haiku-20240307",
        question="Count from 1 to 10: 1, 2, 3,",
        max_tokens=20,
        stop=[", 5"],  # Should stop before outputting ", 5"
    )

    assert isinstance(anthropic_resp, list)
    assert len(anthropic_resp) == 1
    assert isinstance(anthropic_resp[0], str)
    assert ", 5" not in anthropic_resp[0]
    assert "4" in anthropic_resp[0]


@pytest.mark.asyncio
async def test_anthropic_accepts_seed_parameter():
    """Test that the seed parameter is ignored without error for Anthropic models."""
    utils.setup_environment()

    # Test that seed parameter is ignored without error for Anthropic models
    resp = await InferenceAPI.get_default_global_api().ask_single_question(
        model_id="claude-haiku-4-5-20251001",
        question="What is 2+2?",
        max_tokens=10,
        seed=42,  # This should be ignored without error for Anthropic
    )
    assert isinstance(resp, list)
    assert len(resp) == 1
    assert isinstance(resp[0], str)


@pytest.mark.asyncio
async def test_anthropic_rejects_invalid_parameters():
    """Test that invalid parameters raise TypeError with Anthropic models."""
    utils.setup_environment()

    with pytest.raises(TypeError):
        await InferenceAPI.get_default_global_api().ask_single_question(
            model_id="claude-haiku-4-5-20251001",
            question="What is 2+2?",
            max_tokens=10,
            invalid_param=123,  # This should raise an error
        )
