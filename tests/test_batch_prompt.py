import pytest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch
import openai
import anthropic
from safetytooling.data_models.messages import Prompt, BatchPrompt, ChatMessage, MessageRole
from safetytooling.apis.inference.api import InferenceAPI
from safetytooling.data_models.inference import LLMResponse, LLMParams, StopReason

# Mock API clients
class MockAsyncOpenAI:
    def __init__(self, *args, **kwargs):
        pass

class MockAnthropicClient:
    def __init__(self, *args, **kwargs):
        pass

class MockHuggingFaceClient:
    def __init__(self, *args, **kwargs):
        pass

class MockGeminiClient:
    def __init__(self, *args, **kwargs):
        pass

@pytest.fixture(autouse=True)
def mock_api_clients(monkeypatch):
    """Mock all API clients to prevent API key requirements"""
    # Set up mock environment variables
    mock_env = {
        "OPENAI_API_KEY": "test-key",
        "ANTHROPIC_API_KEY": "test-key",
        "HF_API_KEY": "test-key",
        "GRAYSWAN_API_KEY": "test-key",
    }
    for key, value in mock_env.items():
        monkeypatch.setenv(key, value)
    
    # Mock secrets loading
    mock_secrets = {
        **mock_env,
        "PROMPT_HISTORY_DIR": str(Path.home() / ".prompt_history"),
        "CACHE_DIR": str(Path.home() / ".cache")
    }
    monkeypatch.setattr("safetytooling.utils.utils.load_secrets", lambda x: mock_secrets)
    
    # Mock OpenAI
    monkeypatch.setattr(openai, "AsyncClient", MockAsyncOpenAI)
    monkeypatch.setattr(openai, "AsyncOpenAI", MockAsyncOpenAI)
    
    # Mock Anthropic
    monkeypatch.setattr(anthropic, "AsyncAnthropic", MockAnthropicClient)
    
    # Mock S2S model initialization
    monkeypatch.setattr("safetytooling.apis.inference.openai.s2s.OpenAIS2SModel.__init__", 
                       lambda self: setattr(self, "api_key", "test-key"))
    
    # Mock other clients by patching their initialization
    monkeypatch.setattr("safetytooling.apis.inference.huggingface.HuggingFaceModel.__init__", 
                       lambda self, **kwargs: None)
    monkeypatch.setattr("safetytooling.apis.inference.gemini.genai.GeminiModel.__init__", 
                       lambda self, **kwargs: None)
    monkeypatch.setattr("safetytooling.apis.inference.gemini.vertexai.GeminiVertexAIModel.__init__", 
                       lambda self, **kwargs: None)
    monkeypatch.setattr("safetytooling.apis.inference.gray_swan.GraySwanChatModel.__init__", 
                       lambda self, **kwargs: None)

# Mock response factory
def create_mock_response(completion: str, model_id: str = "gpt-3.5-turbo") -> LLMResponse:
    return LLMResponse(
        completion=completion,
        model_id=model_id,
        stop_reason="stop_sequence",  # Required field for LLMResponse - will be converted to StopReason.STOP_SEQUENCE
        cost=0.0,
        api_failures=0,  # Adding other commonly required fields
        moderation=None,
        duration=1.0,  # Mock duration in seconds
        api_duration=0.8,  # Mock API duration in seconds
    )


@pytest.mark.asyncio
async def test_batch_prompt_no_cache(tmp_path: Path, monkeypatch):
    """Test BatchPrompt behavior with an empty cache directory."""
    # Create mock responses - each response should be a list since that's what the API returns per prompt
    mock_responses = [
        [create_mock_response("Response 1")],
        [create_mock_response("Response 2")]
    ]
    
    # Mock all API calls
    mock_completion = AsyncMock(return_value=mock_responses)
    mock_s2s = AsyncMock(return_value=mock_responses)
    mock_embedding = AsyncMock(return_value=[])
    
    # Mock OpenAI models
    monkeypatch.setattr("safetytooling.apis.inference.openai.chat.OpenAIChatModel.__call__", mock_completion)
    monkeypatch.setattr("safetytooling.apis.inference.openai.s2s.OpenAIS2SModel.__call__", mock_s2s)
    monkeypatch.setattr("safetytooling.apis.inference.openai.completion.OpenAICompletionModel.__call__", mock_completion)
    monkeypatch.setattr("safetytooling.apis.inference.openai.embedding.OpenAIEmbeddingModel.__call__", mock_embedding)
    
    # Mock other model providers
    monkeypatch.setattr("safetytooling.apis.inference.anthropic.AnthropicChatModel.__call__", mock_completion)
    monkeypatch.setattr("safetytooling.apis.inference.gemini.genai.GeminiModel.__call__", mock_completion)
    monkeypatch.setattr("safetytooling.apis.inference.gemini.vertexai.GeminiVertexAIModel.__call__", mock_completion)
    monkeypatch.setattr("safetytooling.apis.inference.huggingface.HuggingFaceModel.__call__", mock_completion)
    monkeypatch.setattr("safetytooling.apis.inference.gray_swan.GraySwanChatModel.__call__", mock_completion)
    
    # Initialize API with temporary cache directory
    api = InferenceAPI(cache_dir=tmp_path)
    
    # Create test messages
    msg1 = ChatMessage(role=MessageRole.user, content="Test message 1")
    msg2 = ChatMessage(role=MessageRole.user, content="Test message 2")
    
    # Create individual prompts
    prompt1 = Prompt(messages=[msg1])
    prompt2 = Prompt(messages=[msg2])
    
    # Create batch prompt
    batch_prompt = BatchPrompt(prompts=[prompt1, prompt2])
    
    # Call API with a valid model ID
    responses = await api(
        model_ids="gpt-3.5-turbo",
        prompt=batch_prompt,
        n=1
    )
    
    # Verify responses
    assert isinstance(responses, list), "Expected a list of responses"
    assert len(responses) == 2, "Expected one response per prompt"
    for i, response in enumerate(responses):
        assert isinstance(response, list), "Each response should be a list"
        assert len(response) == 1, "Expected one response per prompt (n=1)"
        assert isinstance(response[0], LLMResponse), "Response should be LLMResponse"
        assert response[0].completion == f"Response {i+1}", "Response should match mock"
    
    # Verify the API was called exactly once
    mock_completion.assert_called_once()


@pytest.mark.asyncio
async def test_batch_prompt_with_cache(tmp_path: Path, monkeypatch):
    """Test BatchPrompt behavior with cache hits and misses."""
    # Create mock responses - each response should be a list since that's what the API returns per prompt
    mock_responses = [
        [create_mock_response("Cached Response 1")],
        [create_mock_response("Cached Response 2")]
    ]
    
    # Mock all API calls
    mock_completion = AsyncMock(return_value=mock_responses)
    mock_s2s = AsyncMock(return_value=mock_responses)
    mock_embedding = AsyncMock(return_value=[])
    
    # Mock OpenAI models
    monkeypatch.setattr("safetytooling.apis.inference.openai.chat.OpenAIChatModel.__call__", mock_completion)
    monkeypatch.setattr("safetytooling.apis.inference.openai.s2s.OpenAIS2SModel.__call__", mock_s2s)
    monkeypatch.setattr("safetytooling.apis.inference.openai.completion.OpenAICompletionModel.__call__", mock_completion)
    monkeypatch.setattr("safetytooling.apis.inference.openai.embedding.OpenAIEmbeddingModel.__call__", mock_embedding)
    
    # Mock other model providers
    monkeypatch.setattr("safetytooling.apis.inference.anthropic.AnthropicChatModel.__call__", mock_completion)
    monkeypatch.setattr("safetytooling.apis.inference.gemini.genai.GeminiModel.__call__", mock_completion)
    monkeypatch.setattr("safetytooling.apis.inference.gemini.vertexai.GeminiVertexAIModel.__call__", mock_completion)
    monkeypatch.setattr("safetytooling.apis.inference.huggingface.HuggingFaceModel.__call__", mock_completion)
    monkeypatch.setattr("safetytooling.apis.inference.gray_swan.GraySwanChatModel.__call__", mock_completion)
    
    # Initialize API with temporary cache directory
    api = InferenceAPI(cache_dir=tmp_path)
    
    # Create test messages
    msg1 = ChatMessage(role=MessageRole.user, content="Cache test message 1")
    msg2 = ChatMessage(role=MessageRole.user, content="Cache test message 2")
    
    # Create individual prompts
    prompt1 = Prompt(messages=[msg1])
    prompt2 = Prompt(messages=[msg2])
    
    # Create batch prompt
    batch_prompt = BatchPrompt(prompts=[prompt1, prompt2])
    
    # First call - should be cache misses
    responses_first = await api(
        model_ids="gpt-3.5-turbo",
        prompt=batch_prompt,
        n=1
    )
    
    # Second call with same prompts - should be cache hits
    responses_second = await api(
        model_ids="gpt-3.5-turbo",
        prompt=batch_prompt,
        n=1
    )
    
    # Verify both sets of responses
    assert len(responses_first) == len(responses_second), "Response lengths should match"
    
    # Compare responses
    for first, second in zip(responses_first, responses_second):
        assert isinstance(first, list) and isinstance(second, list), "Responses should be lists"
        assert len(first) == len(second) == 1, "Each response should have one item (n=1)"
        assert first[0].completion == second[0].completion, "Cached response should match original"
        assert first[0].model_id == second[0].model_id, "Model IDs should match"
    
    # Verify the API was called exactly once (second call should use cache)
    mock_completion.assert_called_once()
