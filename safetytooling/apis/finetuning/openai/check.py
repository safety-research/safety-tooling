"""Code for validating OpenAI fine-tuning files."""

import dataclasses
import logging
import pathlib
import random
from typing import Literal

import safetytooling.apis.inference.openai.utils as openai_utils
from safetytooling.data_models import ChatMessage, MessageRole, Prompt
from safetytooling.utils import utils

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class FTDatasetCostEstimate:
    # User specified params
    dataset_path: pathlib.Path
    model_id: str
    n_epochs: int
    is_validation: bool  # Whether the dataset is for validation or training

    # Aggregate statistics of dataset
    num_examples: int
    total_tokens: int  # Can depend on model_id since different models may have different tokenization
    est_batch_size: int

    # Training costs / context size per API
    cost_per_1k_tokens_usd: float = dataclasses.field(init=False)

    # Cost estimates
    est_trained_tokens_per_epoch: int  # Not the same as total_tokens due to batching and truncation effects
    est_cost_per_epoch_usd: float = dataclasses.field(init=False)

    est_trained_tokens: int = dataclasses.field(init=False)
    est_cost_usd: float = dataclasses.field(init=False)

    def __post_init__(self):
        if self.is_validation:
            (
                _,
                self.cost_per_1k_tokens_usd,
            ) = openai_utils.price_per_token(model_id=self.model_id)
        else:
            price = openai_utils.finetune_price_per_token(model_id=self.model_id)
            assert price is not None
            self.cost_per_1k_tokens_usd = price

        self.est_cost_per_epoch_usd = self.cost_per_1k_tokens_usd / 1000 * self.est_trained_tokens_per_epoch

        self.est_trained_tokens = self.est_trained_tokens_per_epoch * self.n_epochs
        self.est_cost_usd = self.est_cost_per_epoch_usd * self.n_epochs

    def log_cost_summary(self):
        logger.info(f"Fine-tuning cost summary for {self.dataset_path}")
        logger.info(f"Validation: {self.is_validation}")
        logger.info(f"Model: {self.model_id}")
        logger.info(f"Number of examples: {self.num_examples}")
        logger.info(f"Total tokens: {self.total_tokens}")
        logger.info(f"Estimated batch size: {self.est_batch_size}")
        logger.info(f"Cost per 1k tokens: ${self.cost_per_1k_tokens_usd}")
        logger.info(f"Estimated tokens trained per epoch: {self.est_trained_tokens_per_epoch}")
        logger.info(f"Estimated cost per epoch: ${self.est_cost_per_epoch_usd}")
        logger.info(f"Estimated tokens trained (all epochs): {self.est_trained_tokens}")
        logger.info(f"Estimated cost (all epochs): ${self.est_cost_usd}")


def openai_check_finetuning_data(
    dataset_path: pathlib.Path,
    model_id: str = "gpt-3.5-turbo-1106",
    n_epochs: int = 1,
    is_validation: bool = False,
    verbose: bool = False,
    method: Literal["supervised", "dpo"] = "supervised",
) -> FTDatasetCostEstimate:
    """
    is_validation: Whether the file is contains validation data.
                   If False, the file contains training data.
    """
    if method == "supervised":
        prompts = [Prompt.model_validate(x) for x in utils.load_jsonl(dataset_path)]
    elif method == "dpo":
        preferred_prompts = [
            Prompt(
                messages=[
                    ChatMessage(role=MessageRole.user, content=x["input"]["messages"][0]["content"]),
                    ChatMessage(role=MessageRole.assistant, content=x["preferred_output"][0]["content"]),
                ]
            )
            for x in utils.load_jsonl(dataset_path)
        ]
        non_preferred_prompts = [
            Prompt(
                messages=[
                    ChatMessage(role=MessageRole.user, content=x["input"]["messages"][0]["content"]),
                    ChatMessage(role=MessageRole.assistant, content=x["non_preferred_output"][0]["content"]),
                ]
            )
            for x in utils.load_jsonl(dataset_path)
        ]
        prompts = preferred_prompts + non_preferred_prompts

    if verbose:
        logger.info("")
        logger.info(f"Num examples: {len(prompts)}")
    if len(prompts) == 0:
        raise ValueError("No examples found in validation dataset")
    if verbose:
        logger.info("First example")
        prompts[0].pretty_print(
            [],
            print_fn=lambda *args, **__: logger.info(args[0]) if len(args) > 0 else (),
        )

    context_length = openai_utils.get_max_context_length(model_id)

    # Count number of tokens in file to estimate fine-tuning costs
    # Also flag if any examples are too long and would get truncated
    prompt_lengths: list[int] = []
    for i, prompt in enumerate(prompts):
        n_prompt_tokens = sum(openai_utils.count_tokens(msg.content, model_id) for msg in prompt.messages)
        prompt_lengths.append(n_prompt_tokens)

        if n_prompt_tokens > context_length:
            logger.warning(
                f"Prompt {i} has {n_prompt_tokens} tokens,"
                + f"which is over the max context length of {context_length}."
                + "It will be truncated during fine-tuning"
            )
            prompts[0].pretty_print([], print_fn=lambda x, *_, **__: logger.warn(x))

    # batch_size set to 0.2% of dataset size by default
    # https://community.openai.com/t/why-is-the-default-batch-size-set-to-1-for-fine-tuning-the-chatgpt-turbo-model/513129
    batch_size = max(int(0.002 * len(prompts)), 1)

    shuffled_prompt_lengths = prompt_lengths.copy()
    random.Random(0).shuffle(shuffled_prompt_lengths)
    tokens_per_epoch = 0
    for i in range(0, len(prompts), batch_size):
        batch_lengths = prompt_lengths[i : i + batch_size]
        tokens_per_epoch += min(max(batch_lengths), context_length) * len(batch_lengths)

    return FTDatasetCostEstimate(
        dataset_path=dataset_path,
        model_id=model_id,
        n_epochs=n_epochs,
        is_validation=is_validation,
        num_examples=len(prompts),
        total_tokens=sum(prompt_lengths),
        est_batch_size=batch_size,
        est_trained_tokens_per_epoch=tokens_per_epoch,
    )
