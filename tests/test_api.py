"""Basic tests for the api."""

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
