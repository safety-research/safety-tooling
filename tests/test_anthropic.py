import pytest
from unittest.mock import AsyncMock, patch

from safetytooling.apis.inference import AnthropicChatModel
from safetytooling.data_models import Prompt


@pytest.mark.asyncio
async def test_seed_parameter_handling():
    """Test that the seed parameter is properly handled (ignored) while other surplus params raise errors."""
    model = AnthropicChatModel(num_threads=1)
    model.aclient.messages.create = AsyncMock()
    
    # Create a simple prompt
    prompt = Prompt.from_str("test prompt")
    
    # Test that seed parameter is ignored
    await model(
        model_id="claude-3-5-sonnet-20240620",
        prompt=prompt,
        print_prompt_and_response=False,
        max_attempts=1,
        seed=42  # This should be ignored without error
    )
    
    # Verify seed was not passed to Anthropic API
    kwargs = model.aclient.messages.create.call_args[1]
    assert "seed" not in kwargs, "seed parameter should not be passed to Anthropic API"
    
    # Test that other surplus parameters raise an error
    with pytest.raises(TypeError):
        await model(
            model_id="claude-3-5-sonnet-20240620",
            prompt=prompt,
            print_prompt_and_response=False,
            max_attempts=1,
            invalid_param=123  # This should raise an error
        ) 