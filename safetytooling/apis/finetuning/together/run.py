from __future__ import annotations

import asyncio
import dataclasses
import logging
import os
from pathlib import Path

import simple_parsing
import wandb
import wandb.sdk.wandb_run

from safetytooling.data_models.finetune import FinetuneConfig
from safetytooling.utils import utils
from together import Together

LOGGER = logging.getLogger(__name__)


async def upload_finetuning_file(
    client: Together,
    file_path: Path | None,
    check_delay: float = 1.0,
) -> str | None:
    """Upload file to Together and wait for processing."""
    if file_path is None:
        return None

    LOGGER.info(f"Uploading {file_path} to Together...")
    response = client.files.upload(file=str(file_path))
    file_id = response.id
    LOGGER.info(f"Uploaded {file_path} to Together as {file_id}")

    while True:
        file_status = client.files.retrieve(file_id)
        if file_status.processed:
            return file_id

        LOGGER.info(f"Waiting for {file_id} to be processed...")
        await asyncio.sleep(check_delay)


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


async def main(cfg: TogetherFTConfig, verbose: bool = True):
    """Run Together fine-tuning job with monitoring."""
    # TODO: potentially add training data checking
    client = Together(api_key=os.environ.get("TOGETHER_API_KEY"))

    LOGGER.info("Uploading training and validation files...")
    train_file_id, val_file_id = await asyncio.gather(
        upload_finetuning_file(client, cfg.train_file),
        upload_finetuning_file(client, cfg.val_file),
    )
    assert train_file_id is not None

    LOGGER.info(f"Train file_id: {train_file_id}")
    LOGGER.info(f"Validation file_id: {val_file_id}")

    # Start fine-tuning job
    ft_job = client.fine_tuning.create(
        training_file=train_file_id,
        validation_file=val_file_id,
        model=cfg.model,
        n_epochs=cfg.n_epochs,
        batch_size=cfg.batch_size,
        learning_rate=cfg.learning_rate,
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
        {
            "finetune_job_id": ft_job.id,
            "train_file_id": train_file_id,
            **({"val_file_id": val_file_id} if val_file_id is not None else {}),
        }
    )

    # Upload files to wandb
    upload_file_to_wandb(
        file_path=cfg.train_file,
        name=train_file_id,
        type="together-finetune-training-file",
        description="Training file for finetuning",
        wandb_run=wrun,
    )

    if cfg.val_file is not None:
        upload_file_to_wandb(
            file_path=cfg.val_file,
            name=val_file_id,
            type="together-finetune-validation-file",
            description="Validation file for finetuning",
            wandb_run=wrun,
        )

    LOGGER.info("Waiting for fine-tuning job to finish...")
    while True:
        status = client.fine_tuning.retrieve(ft_job.id)  # https://docs.together.ai/reference/get_fine-tunes-id
        wrun.summary.update({"together_job_status": status.status})
        if status.status == "completed":
            LOGGER.info("Fine-tuning job succeeded.")
            wrun.summary.update(status.model_dump())
            break
        elif status.status in ["error", "cancelled"]:
            LOGGER.error(f"Fine-tuning job failed: {status.status}")
            raise RuntimeError(f"Fine-tuning failed with status: {status.status}")

        await asyncio.sleep(10)

    wrun.finish()
    return ft_job


@dataclasses.dataclass
class TogetherFTConfig(FinetuneConfig):
    learning_rate: float = 1e-5
    suffix: str = ""  # Together suffix to append to the model name


if __name__ == "__main__":
    parser = simple_parsing.ArgumentParser()
    parser.add_arguments(TogetherFTConfig, dest="experiment_config")
    args = parser.parse_args()
    cfg: TogetherFTConfig = args.experiment_config

    utils.setup_environment(
        logging_level=cfg.logging_level,
    )

    asyncio.run(main(cfg))
