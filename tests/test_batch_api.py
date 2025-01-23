import os

import pytest

from safetytooling.apis.batch_api import BatchInferenceAPI
from safetytooling.data_models import ChatMessage, MessageRole, Prompt
from safetytooling.utils import utils

# Skip all tests in this module if SAFETYTOOLING_SLOW_TESTS is not set
pytestmark = pytest.mark.skipif(
    os.getenv("SAFETYTOOLING_SLOW_TESTS", "false").lower() != "true",
    reason="Skipping slow batch API tests. Set SAFETYTOOLING_SLOW_TESTS=true to run these tests.",
)

# Need to set up environment before importing safetytooling
utils.setup_environment()


@pytest.fixture
def batch_api():
    return BatchInferenceAPI(
        prompt_history_dir=None,
        anthropic_api_key=os.getenv("ANTHROPIC_BATCH_API_KEY") if os.getenv("ANTHROPIC_BATCH_API_KEY") else None,
    )


@pytest.fixture
def test_prompts():
    return [
        Prompt(messages=[ChatMessage(content="Say 1", role=MessageRole.user)]),
        Prompt(messages=[ChatMessage(content="Say 2", role=MessageRole.user)]),
        Prompt(messages=[ChatMessage(content="Say 3", role=MessageRole.user)]),
    ]


@pytest.mark.asyncio
async def test_batch_api_claude(batch_api, test_prompts, tmp_path):
    """Test that BatchInferenceAPI works with Claude models."""
    responses, batch_id = await batch_api(
        model_id="claude-3-5-haiku-20241022",
        prompts=test_prompts,
        max_tokens=10,
        log_dir=tmp_path,
    )

    assert len(responses) == len(test_prompts)
    assert all(response is not None for response in responses)
    assert all(response.model_id == "claude-3-5-haiku-20241022" for response in responses)
    assert all(response.batch_custom_id is not None for response in responses)
    print(responses)
    assert all(str(i + 1) in response.completion for i, response in enumerate(responses))
    assert isinstance(batch_id, str)


@pytest.mark.asyncio
async def test_batch_api_openai(batch_api, test_prompts, tmp_path):
    """Test that BatchInferenceAPI works with OpenAI models."""
    responses, batch_id = await batch_api(
        model_id="gpt-4o-mini-2024-07-18",
        prompts=test_prompts,
        max_tokens=10,
        log_dir=tmp_path,
    )

    assert len(responses) == len(test_prompts)
    assert all(response is not None for response in responses)
    assert all(response.model_id == "gpt-4o-mini-2024-07-18" for response in responses)
    assert all(response.batch_custom_id is not None for response in responses)
    print(responses)
    assert all(str(i + 1) in response.completion for i, response in enumerate(responses))
    assert isinstance(batch_id, str)


@pytest.mark.asyncio
async def test_batch_api_unsupported_model(batch_api, test_prompts):
    """Test that BatchInferenceAPI raises error for unsupported models."""
    with pytest.raises(ValueError, match="is not supported. Only Claude and OpenAI models are supported"):
        await batch_api(
            model_id="unsupported-model",
            prompts=test_prompts,
        )
