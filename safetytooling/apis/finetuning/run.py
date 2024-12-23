from __future__ import annotations

import asyncio
import dataclasses
import io
import logging
import tempfile
from pathlib import Path

import numpy as np
import openai
import openai.types.fine_tuning
import pandas as pd
import simple_parsing
import wandb
import wandb.sdk.wandb_run
from openai import APIConnectionError, RateLimitError
from tenacity import (
    AsyncRetrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_fixed,
)

from safetytooling.utils import utils

from .check import openai_check_finetuning_data

LOGGER = logging.getLogger(__name__)


async def upload_finetuning_file_to_openai(
    file_path: Path | None,
    check_delay: float = 1.0,
) -> str | None:
    """check_frequency is in seconds."""
    if file_path is None:
        return None

    client = openai.AsyncClient()
    with open(file_path, "rb") as f:
        LOGGER.info(f"Uploading {file_path} to OpenAI...")
        file_obj = await client.files.create(file=f, purpose="fine-tune")
        LOGGER.info(f"Uploaded {file_path} to OpenAI as {file_obj.id}")

    while True:
        async for attempt in AsyncRetrying(
            retry=retry_if_exception_type(APIConnectionError),
            stop=stop_after_attempt(3),
            wait=wait_fixed(30),
        ):
            with attempt:
                file_obj = await client.files.retrieve(file_obj.id)

        if file_obj.status == "processed":
            return file_obj.id

        LOGGER.info(f"Wating for {file_obj.id} to be processed...")
        await asyncio.sleep(check_delay)


async def queue_finetune(
    cfg: Config,
    train_file_id: str,
    val_file_id: str | None,
    client: openai.AsyncClient | None = None,
) -> openai.types.fine_tuning.FineTuningJob:
    if client is None:
        client = openai.AsyncClient()

    # try every 60 seconds for 10 hours (in case you run overnight)
    async for attempt in AsyncRetrying(
        retry=retry_if_exception_type((APIConnectionError, RateLimitError)),
        stop=stop_after_attempt(600),
        wait=wait_fixed(60),
    ):
        with attempt:
            LOGGER.info("Attempting to start fine-tuning job...")
            ft_job = await client.fine_tuning.jobs.create(
                model=cfg.model,
                training_file=train_file_id,
                hyperparameters={"n_epochs": cfg.n_epochs},
                validation_file=val_file_id,
            )

    return ft_job


def upload_file_to_wandb(
    file_path: Path,
    name: str,
    type: str,
    description: str,
    wandb_run: wandb.sdk.wandb_run.Run,
):
    LOGGER.info(f"Uploading file '{name}' to wandb...")
    artifact = wandb.Artifact(
        name=name,
        type=type,
        description=description,
    )
    artifact.add_file(file_path.as_posix())
    wandb_run.log_artifact(artifact)


async def main(cfg: Config, extra_config: dict = None, verbose: bool = True):
    """
    WARNING: This function cannot be called concurrently, wandb does not support
    concurrent runs. See
    https://community.wandb.ai/t/is-it-possible-to-log-to-multiple-runs-simultaneously/4387
    for details.
    """

    LOGGER.info("Checking training data")
    train_cost_est = openai_check_finetuning_data(
        dataset_path=cfg.train_file,
        model_id=cfg.model,
        n_epochs=cfg.n_epochs,
        verbose=verbose,
    )
    if verbose:
        train_cost_est.log_cost_summary()
    if cfg.val_file is not None:
        LOGGER.info("Checking validation data")
        val_cost_est = openai_check_finetuning_data(
            dataset_path=cfg.val_file,
            model_id=cfg.model,
            n_epochs=cfg.n_epochs,
            is_validation=True,
            verbose=verbose,
        )
        if verbose:
            val_cost_est.log_cost_summary()

    if cfg.dry_run:
        LOGGER.info("Dry run, not starting fine-tuning job. Exiting.")
        return None, train_cost_est.est_cost_usd

    LOGGER.info("Uploading training and validation files...")
    train_file_id, val_file_id = await asyncio.gather(
        upload_finetuning_file_to_openai(cfg.train_file),
        upload_finetuning_file_to_openai(cfg.val_file),
    )
    assert train_file_id is not None

    LOGGER.info(f"Train file_id: {train_file_id}")
    LOGGER.info(f"Validation file_id: {val_file_id}")

    client = openai.AsyncClient()
    ft_job = await queue_finetune(
        cfg=cfg,
        train_file_id=train_file_id,
        val_file_id=val_file_id,
        client=client,
    )
    LOGGER.info(f"Started fine-tuning job: {ft_job.id}")

    # Initialize wandb run
    wrun = wandb.init(
        project=cfg.wandb_project_name,
        config=dataclasses.asdict(cfg),
        tags=cfg.tags,
    )
    assert isinstance(wrun, wandb.sdk.wandb_run.Run)
    wrun.config.update(
        {"finetune_job_id": ft_job.id, "train_file_id": train_file_id}
        | ({"val_file_id": val_file_id} if val_file_id is not None else {})
    )
    if extra_config is not None:
        wrun.config.update(extra_config)

    wrun.summary.update(
        {
            "est_train_cost_usd": train_cost_est.est_cost_usd,
            "est_trained_tokens": train_cost_est.est_trained_tokens,
        }
    )
    if cfg.val_file is not None:
        wrun.summary.update(
            {
                "est_val_cost_usd": val_cost_est.est_cost_usd,
                "est_validated_tokens": val_cost_est.est_trained_tokens,
            }
        )

    upload_file_to_wandb(
        file_path=cfg.train_file,
        name=train_file_id,
        type="openai-finetune-training-file",
        description="Training file for finetuning",
        wandb_run=wrun,
    )
    if cfg.val_file is not None:
        assert val_file_id is not None
        upload_file_to_wandb(
            file_path=cfg.val_file,
            name=val_file_id,
            type="openai-finetune-validation-file",
            description="Validation file for finetuning",
            wandb_run=wrun,
        )

    LOGGER.info("Waiting for fine-tuning job to finish...")
    train_cost_usd = None
    while True:
        ft_job = await client.fine_tuning.jobs.retrieve(ft_job.id)
        wrun.summary.update({"openai_job_status": ft_job.status})

        match ft_job.status:
            case "succeeded":
                LOGGER.info("Fine-tuning job succeeded.")
                wrun.summary.update(ft_job.model_dump())
                assert ft_job.trained_tokens is not None
                train_cost_usd = ft_job.trained_tokens * train_cost_est.cost_per_1k_tokens_usd / 1000
                wrun.summary.update({"train_cost_usd": train_cost_usd})
                break

            case "failed" | "cancelled":
                LOGGER.error(f"Fine-tuning job failed: {ft_job.status}")
                raise RuntimeError

            case _:
                await asyncio.sleep(10)

    LOGGER.info("Uploading result files to wandb...")
    for i, result_file_id in enumerate(ft_job.result_files):
        LOGGER.info(f"Uploading result file {result_file_id} to wandb...")
        file_content = await client.files.content(result_file_id)

        # Create temp file
        with tempfile.NamedTemporaryFile(suffix=".csv") as temp_file:
            with temp_file.file as f:
                f.write(file_content.content)

            upload_file_to_wandb(
                file_path=Path(f.name),
                name=f"{result_file_id}-{i}",
                type="openai-finetune-result",
                description="Result of fine-tuning job",
                wandb_run=wrun,
            )

        df_results = pd.read_csv(io.StringIO(file_content.text))
        for _, row in df_results.iterrows():
            metrics = {k: v for k, v in row.items() if not np.isnan(v)}
            step = metrics.pop("step")
            if step is not None:
                step = int(step)
            wrun.log(metrics, step=step)

    wrun.finish()
    return ft_job, train_cost_usd


@dataclasses.dataclass
class Config:
    train_file: Path  # Training data
    val_file: Path | None = None  # Validation data
    model: str = "gpt-3.5-turbo-1106"  # The model to fine-tune
    n_epochs: int = 3

    dry_run: bool = False  # Just validate the data, don't launch the job
    logging_level: str = "info"

    openai_tag: str = "OPENAI_API_KEY1"
    wandb_project_name: str = "adv-robustness"
    tags: tuple[str, ...] = ("test",)


if __name__ == "__main__":
    parser = simple_parsing.ArgumentParser()
    parser.add_arguments(Config, dest="experiment_config")
    args = parser.parse_args()
    cfg: Config = args.experiment_config

    utils.setup_environment(
        openai_tag=cfg.openai_tag,
        logging_level=cfg.logging_level,
    )

    asyncio.run(main(cfg))
