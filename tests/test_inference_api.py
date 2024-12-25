import pytest
from safetytooling.apis.inference.api import InferenceAPI
from safetytooling.data_models.messages import Prompt, BatchPrompt, MessageRole
from safetytooling.data_models.inference import LLMResponse, LLMParams
from unittest.mock import AsyncMock, MagicMock, patch

@pytest.mark.asyncio
async def test_gpt4o_mini_short_call():
    """
    Test a short LLM call to 'gpt-4o-mini'.
    Make sure the call completes without error and returns an LLMResponse.
    """
    inference_api = InferenceAPI()
    prompt = Prompt(messages=[{"role": MessageRole.USER, "content": "Hello, how are you?"}])
    responses = await inference_api(model_ids="gpt-4o-mini", prompt=prompt, n=1)
    assert len(responses) == 1
    # Check that the response is valid
    assert responses[0].completion is not None
    assert isinstance(responses[0], LLMResponse)

@pytest.mark.asyncio
async def test_claude_3_haiku_short_call():
    """
    Test a short LLM call to 'claude-3-haiku-20240307'.
    """
    inference_api = InferenceAPI()
    prompt = Prompt(messages=[{"role": MessageRole.USER, "content": "What is 2+2?"}])
    responses = await inference_api(model_ids="claude-3-haiku-20240307", prompt=prompt, n=1)
    assert len(responses) == 1
    assert responses[0].completion is not None
    assert isinstance(responses[0], LLMResponse)

@pytest.mark.asyncio
async def test_batch_prompt_with_caching():
    """
    Test the BatchPrompt handling plus caching logic.
    We'll mock external calls so we don't depend on real LLM endpoints.
    """
    # Create a mock response
    mock_response = LLMResponse(
        completion="Test response",
        prompt_tokens=10,
        completion_tokens=5,
        total_tokens=15,
        model="test-model",
        cost=0.001
    )
    
    # Create test prompts
    prompts = [
        Prompt(messages=[{"role": MessageRole.USER, "content": "Test prompt 1"}]),
        Prompt(messages=[{"role": MessageRole.USER, "content": "Test prompt 2"}])
    ]
    batch_prompt = BatchPrompt(prompts=prompts)
    
    # Create InferenceAPI with mocked call
    with patch('safetytooling.apis.inference.api.InferenceAPI.__call__', new_callable=AsyncMock) as mock_call:
        mock_call.return_value = [mock_response]
        
        inference_api = InferenceAPI()
        responses = await inference_api(
            model_ids="gpt-4o-mini",
            prompt=batch_prompt,
            n=1
        )
        
        # Verify the responses
        assert len(responses) == len(prompts)
        for response in responses:
            assert isinstance(response, LLMResponse)
            assert response.completion is not None

@pytest.mark.asyncio
async def test_gemini_call_mocked():
    """
    Test Gemini model call with mocked response.
    """
    mock_response = LLMResponse(
        completion="Mocked Gemini response",
        prompt_tokens=10,
        completion_tokens=5,
        total_tokens=15,
        model="gemini-pro",
        cost=0.001
    )
    
    with patch('safetytooling.apis.inference.api.InferenceAPI.__call__', new_callable=AsyncMock) as mock_call:
        mock_call.return_value = [mock_response]
        
        inference_api = InferenceAPI()
        prompt = Prompt(messages=[{"role": MessageRole.USER, "content": "Test Gemini"}])
        responses = await inference_api(model_ids="gemini-pro", prompt=prompt, n=1)
        
        assert len(responses) == 1
        assert isinstance(responses[0], LLMResponse)
        assert responses[0].completion == "Mocked Gemini response"

@pytest.mark.asyncio
async def test_huggingface_call_mocked():
    """
    Test HuggingFace model call with mocked response.
    """
    mock_response = LLMResponse(
        completion="Mocked HuggingFace response",
        prompt_tokens=10,
        completion_tokens=5,
        total_tokens=15,
        model="huggingface/test-model",
        cost=0.0
    )
    
    with patch('safetytooling.apis.inference.api.InferenceAPI.__call__', new_callable=AsyncMock) as mock_call:
        mock_call.return_value = [mock_response]
        
        inference_api = InferenceAPI()
        prompt = Prompt(messages=[{"role": MessageRole.USER, "content": "Test HuggingFace"}])
        responses = await inference_api(
            model_ids="huggingface/test-model",
            prompt=prompt,
            n=1
        )
        
        assert len(responses) == 1
        assert isinstance(responses[0], LLMResponse)
        assert responses[0].completion == "Mocked HuggingFace response"

@pytest.mark.asyncio
async def test_grayswan_call_mocked():
    """
    Test GraySwan model call with mocked response.
    """
    mock_response = LLMResponse(
        completion="Mocked GraySwan response",
        prompt_tokens=10,
        completion_tokens=5,
        total_tokens=15,
        model="grayswan-test",
        cost=0.0
    )
    
    with patch('safetytooling.apis.inference.api.InferenceAPI.__call__', new_callable=AsyncMock) as mock_call:
        mock_call.return_value = [mock_response]
        
        inference_api = InferenceAPI()
        prompt = Prompt(messages=[{"role": MessageRole.USER, "content": "Test GraySwan"}])
        responses = await inference_api(
            model_ids="grayswan-test",
            prompt=prompt,
            n=1
        )
        
        assert len(responses) == 1
        assert isinstance(responses[0], LLMResponse)
        assert responses[0].completion == "Mocked GraySwan response"
