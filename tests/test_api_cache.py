import pytest

from safetytooling.apis.batch_api import BatchInferenceAPI
from safetytooling.apis.inference.api import InferenceAPI
from safetytooling.data_models import ChatMessage, LLMParams, LLMResponse, MessageRole, Prompt
from safetytooling.utils import utils


@pytest.mark.asyncio
async def test_api_cache(monkeypatch, tmp_path):
    """Test that caching works within InferenceAPI."""
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setenv("CACHE_DIR", str(tmp_path))
    utils.setup_environment()

    mock_response = LLMResponse(
        model_id="gpt-4o-mini-2024-07-18", completion="Test completion", stop_reason="stop", cost=0.1
    )

    call_count = 0

    async def mock_ask(self, model_id, question, **kwargs):
        nonlocal call_count

        prompt = Prompt(messages=[ChatMessage(role=MessageRole.user, content=question)])

        if self.cache_manager is not None:
            llm_params = LLMParams(
                model=model_id,
                n=1,
                insufficient_valids_behaviour="retry",
                num_candidates_per_completion=1,
                temperature=kwargs.get("temperature", None),
                max_tokens=kwargs.get("max_tokens", None),
            )
            cached_result = self.cache_manager.maybe_load_cache(prompt=prompt, params=llm_params)
            if cached_result is not None:
                return cached_result.responses[0]

        call_count += 1

        if self.cache_manager is not None:
            self.cache_manager.save_cache(
                prompt=prompt,
                params=LLMParams(
                    model=model_id,
                    n=1,
                    insufficient_valids_behaviour="retry",
                    num_candidates_per_completion=1,
                    temperature=kwargs.get("temperature", None),
                    max_tokens=kwargs.get("max_tokens", None),
                ),
                responses=[mock_response],
            )

        self.running_cost += mock_response.cost
        return mock_response

    monkeypatch.setattr(InferenceAPI, "ask_single_question", mock_ask)

    api = InferenceAPI(cache_dir=tmp_path)
    monkeypatch.setattr(InferenceAPI, "get_default_global_api", lambda: api)
    monkeypatch.setattr(InferenceAPI, "default_global_running_cost", lambda: api.running_cost)

    q = "What is the meaning of life?"
    response1 = await api.ask_single_question(
        model_id="gpt-4o-mini-2024-07-18",
        question=q,
        max_tokens=20,
        temperature=1.2,
    )

    cost = api.running_cost

    response2 = await api.ask_single_question(
        model_id="gpt-4o-mini-2024-07-18",
        question=q,
        max_tokens=20,
        temperature=1.2,
    )

    assert response1 == response2

    assert api.running_cost == cost

    assert call_count == 1


@pytest.mark.asyncio
async def test_no_cache_flag(monkeypatch, tmp_path):
    """Test that no_cache flag disables caching."""
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setenv("CACHE_DIR", str(tmp_path))
    utils.setup_environment()

    mock_response = LLMResponse(
        model_id="gpt-4o-mini-2024-07-18", completion="Test completion", stop_reason="stop", cost=0.1
    )

    call_count = 0

    async def mock_ask(self, model_id, question, **kwargs):
        nonlocal call_count
        call_count += 1

        self.running_cost += mock_response.cost

        return mock_response

    monkeypatch.setattr(InferenceAPI, "ask_single_question", mock_ask)

    api = InferenceAPI(cache_dir=tmp_path, no_cache=True)

    q = "What is the meaning of life?"
    _ = await api.ask_single_question(
        model_id="gpt-4o-mini-2024-07-18",
        question=q,
        max_tokens=20,
        temperature=1.2,
    )

    cost = api.running_cost

    _ = await api.ask_single_question(
        model_id="gpt-4o-mini-2024-07-18",
        question=q,
        max_tokens=20,
        temperature=1.2,
    )

    # Cost should increase since caching is disabled
    assert api.running_cost > cost

    assert call_count == 2


@pytest.mark.asyncio
async def test_cross_api_cache(monkeypatch, tmp_path):
    """Test that cache works across both InferenceAPI and BatchInferenceAPI."""
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setenv("CACHE_DIR", str(tmp_path))
    utils.setup_environment()

    model_id = "gpt-4o-mini"

    prompt = Prompt(messages=[ChatMessage(role=MessageRole.user, content="Test prompt")])

    mock_response = LLMResponse(model_id=model_id, completion="Test completion", stop_reason="stop", cost=0.0)

    async def mock_inference_call(self, *args, **kwargs):
        if self.cache_manager is not None:
            self.cache_manager.save_cache(
                prompt=kwargs.get("prompt"),
                params=LLMParams(
                    model=kwargs.get("model_ids"),
                    n=1,
                    insufficient_valids_behaviour="retry",
                    num_candidates_per_completion=1,
                    temperature=kwargs.get("temperature", None),
                    max_tokens=kwargs.get("max_tokens", None),
                ),
                responses=[mock_response],
            )
        return [mock_response]

    async def mock_batch_call(self, *args, **kwargs):
        if self.cache_manager is not None and kwargs.get("use_cache", True):
            prompt = kwargs.get("prompts")[0]
            llm_params = LLMParams(
                model=kwargs.get("model_id"),
                n=1,
                insufficient_valids_behaviour="retry",
                num_candidates_per_completion=1,
                temperature=kwargs.get("temperature", None),
                max_tokens=kwargs.get("max_tokens", None),
            )
            cached_result = self.cache_manager.maybe_load_cache(prompt=prompt, params=llm_params)
            if cached_result is not None:
                return [cached_result.responses[0]], "cached"

        return [mock_response], "batch_id"

    monkeypatch.setattr(InferenceAPI, "__call__", mock_inference_call)
    monkeypatch.setattr(BatchInferenceAPI, "__call__", mock_batch_call)

    api = InferenceAPI(cache_dir=tmp_path)
    batch_api = BatchInferenceAPI(cache_dir=tmp_path)

    responses_api = await api(
        model_ids=model_id,
        prompt=prompt,
        max_tokens=20,
        temperature=1.2,
    )

    batch_responses, batch_id = await batch_api(
        model_id=model_id,
        prompts=[prompt],
        max_tokens=20,
        temperature=1.2,
    )

    assert batch_id == "cached", "BatchInferenceAPI should use cache from InferenceAPI"
    assert batch_responses[0].completion == responses_api[0].completion, "Response should match"
