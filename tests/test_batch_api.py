import os
import random
import time

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


@pytest.mark.asyncio
async def test_batch_api_cache(batch_api, tmp_path):
    """Test that BatchInferenceAPI properly caches responses."""
    random.seed(time.time())

    # Create 3 test prompts with random numbers to ensure unique cache keys
    rand_suffix = random.randint(0, 1000000)
    test_prompts = [
        Prompt(messages=[ChatMessage(content=f"Say {i} {rand_suffix}", role=MessageRole.user)]) for i in range(1, 4)
    ]

    # First request - should hit the API
    responses1, batch_id1 = await batch_api(
        model_id="claude-3-5-haiku-20241022",
        prompts=test_prompts,
        max_tokens=10,
        log_dir=tmp_path,
    )

    # Second request - should use cache
    responses2, batch_id2 = await batch_api(
        model_id="claude-3-5-haiku-20241022",
        prompts=test_prompts,
        max_tokens=10,
        log_dir=tmp_path,
    )

    # Third request - should use cache
    responses3, batch_id3 = await batch_api(
        model_id="claude-3-5-haiku-20241022",
        prompts=test_prompts,
        max_tokens=10,
        log_dir=tmp_path,
    )

    # Verify responses are identical
    assert responses1 == responses2 == responses3

    # Verify batch IDs show caching
    assert batch_id1 != "cached"  # First request should hit API
    assert batch_id2 == "cached"  # Second request should be cached
    assert batch_id3 == "cached"  # Third request should be cached


@pytest.mark.asyncio
async def test_batch_api_no_cache_flag(batch_api, tmp_path):
    """Test that BatchInferenceAPI respects no_cache flag."""
    random.seed(time.time())

    # Create 3 test prompts with random numbers to ensure unique cache keys
    rand_suffix = random.randint(0, 1000000)
    test_prompts = [
        Prompt(messages=[ChatMessage(content=f"Say {i} {rand_suffix}", role=MessageRole.user)]) for i in range(1, 4)
    ]

    # Create a new BatchInferenceAPI instance with no_cache=True
    no_cache_api = BatchInferenceAPI(
        prompt_history_dir=None,
        no_cache=True,
        anthropic_api_key=os.getenv("ANTHROPIC_BATCH_API_KEY") if os.getenv("ANTHROPIC_BATCH_API_KEY") else None,
    )

    # First request
    responses1, batch_id1 = await no_cache_api(
        model_id="claude-3-5-haiku-20241022",
        prompts=test_prompts,
        max_tokens=10,
        log_dir=tmp_path,
    )

    # Second request - should hit API again due to no_cache
    responses2, batch_id2 = await no_cache_api(
        model_id="claude-3-5-haiku-20241022",
        prompts=test_prompts,
        max_tokens=10,
        log_dir=tmp_path,
    )

    # Verify batch IDs are not "cached"
    assert batch_id1 != "cached"
    assert batch_id2 != "cached"
    assert batch_id1 != batch_id2  # Different batch IDs for different API calls


@pytest.mark.asyncio
async def test_batch_api_no_cache_env(batch_api, tmp_path):
    """Test that BatchInferenceAPI respects NO_CACHE environment variable."""
    random.seed(time.time())

    # Create 3 test prompts with random numbers to ensure unique cache keys
    rand_suffix = random.randint(0, 1000000)
    test_prompts = [
        Prompt(messages=[ChatMessage(content=f"Say {i} {rand_suffix}", role=MessageRole.user)]) for i in range(1, 4)
    ]

    # Set NO_CACHE environment variable
    old_no_cache = os.getenv("NO_CACHE")
    os.environ["NO_CACHE"] = "1"

    # Create a new BatchInferenceAPI instance - should respect NO_CACHE env var
    env_no_cache_api = BatchInferenceAPI(
        prompt_history_dir=None,
        anthropic_api_key=os.getenv("ANTHROPIC_BATCH_API_KEY") if os.getenv("ANTHROPIC_BATCH_API_KEY") else None,
    )

    # First request
    responses1, batch_id1 = await env_no_cache_api(
        model_id="claude-3-5-haiku-20241022",
        prompts=test_prompts,
        max_tokens=10,
        log_dir=tmp_path,
    )

    # Second request - should hit API again due to NO_CACHE env var
    responses2, batch_id2 = await env_no_cache_api(
        model_id="claude-3-5-haiku-20241022",
        prompts=test_prompts,
        max_tokens=10,
        log_dir=tmp_path,
    )

    # Verify batch IDs are not "cached"
    assert batch_id1 != "cached"
    assert batch_id2 != "cached"
    assert batch_id1 != batch_id2  # Different batch IDs for different API calls

    # Clean up
    if old_no_cache is None:
        del os.environ["NO_CACHE"]
    else:
        os.environ["NO_CACHE"] = old_no_cache


@pytest.mark.asyncio
async def test_batch_api_chunking(batch_api, tmp_path):
    """Test that BatchInferenceAPI properly handles chunking."""
    random.seed(time.time())

    # Create a new BatchInferenceAPI instance with chunk_size=2
    chunked_api = BatchInferenceAPI(
        prompt_history_dir=None,
        chunk=2,
        anthropic_api_key=os.getenv("ANTHROPIC_BATCH_API_KEY") if os.getenv("ANTHROPIC_BATCH_API_KEY") else None,
    )

    # Create 5 test prompts with random numbers to ensure unique cache keys
    rand_suffix = random.randint(0, 1000000)
    test_prompts = [
        Prompt(messages=[ChatMessage(content=f"Say {i} {rand_suffix}", role=MessageRole.user)]) for i in range(1, 4)
    ]

    # Run with chunking
    responses, batch_id = await chunked_api(
        model_id="claude-3-5-haiku-20241022",
        prompts=test_prompts,
        max_tokens=10,
        log_dir=tmp_path,
    )

    # Verify we got responses for all prompts
    assert len(responses) == len(test_prompts)
    assert all(response is not None for response in responses)
    assert all(response.model_id == "claude-3-5-haiku-20241022" for response in responses)
    assert all(response.batch_custom_id is not None for response in responses)

    # Verify that we got multiple batch IDs (since it was chunked)
    batch_ids = batch_id.split(",")
    assert len(batch_ids) == 2  # With 3 prompts and chunk_size=2, we expect 2 batches
