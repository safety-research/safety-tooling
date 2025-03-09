"""Basic tests for the api."""

import pydantic
import pytest

from safetytooling.apis.inference.api import InferenceAPI
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
        model_ids="gpt-4o-mini",
        question="What is the meaning of life?",
        max_tokens=1,
        n=2,
    )
    assert isinstance(resp, list)
    assert len(resp) == 2
    assert isinstance(resp[0], str)
    assert isinstance(resp[1], str)


@pytest.mark.asyncio
async def test_claude_3():
    utils.setup_environment()
    resp = await InferenceAPI.get_default_global_api().ask_single_question(
        model_ids="claude-3-haiku-20240307",
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
        model_ids="claude-3-haiku-20240307",
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
        model_ids="gpt-4o-mini-2024-07-18",
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
        model_ids="gpt-4o-mini-2024-07-18",
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
        model_ids="claude-3-haiku-20240307",
        question="Count from 1 to 10: 1, 2, 3,",
        max_tokens=20,
        stop=[", 5"],  # Should stop before outputting ", 5"
    )

    assert isinstance(anthropic_resp, list)
    assert len(anthropic_resp) == 1
    assert isinstance(anthropic_resp[0], str)
    assert ", 5" not in anthropic_resp[0]
    assert "4" in anthropic_resp[0]
