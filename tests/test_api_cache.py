import pytest

from safetytooling.apis.inference.api import InferenceAPI
from safetytooling.utils import utils


@pytest.mark.asyncio
async def test_api_cache():
    utils.setup_environment()

    response1 = await InferenceAPI.get_default_global_api().ask_single_question(
        model_ids="gpt-4o-mini",
        question="What is the meaning of life?",
        max_tokens=20,
        temperature=1.2,
    )

    cost = InferenceAPI.default_global_running_cost()

    response2 = await InferenceAPI.get_default_global_api().ask_single_question(
        model_ids="gpt-4o-mini",
        question="What is the meaning of life?",
        max_tokens=20,
        temperature=1.2,
    )

    assert response1 == response2
    assert (
        InferenceAPI.default_global_running_cost() == cost
    )  # Cost should stay the same because the second time is cached.
