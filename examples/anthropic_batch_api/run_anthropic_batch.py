import asyncio
import dataclasses
import json
import logging
import time

import pandas as pd
from simple_parsing import ArgumentParser

from safetytooling.apis.inference.anthropic import AnthropicModelBatch
from safetytooling.data_models import ChatMessage, MessageRole, Prompt
from safetytooling.utils.experiment_utils import ExperimentConfigBase

LOGGER = logging.getLogger(__name__)


@dataclasses.dataclass
class ExperimentConfig(ExperimentConfigBase):
    """Get responses to requests."""

    model_ids: tuple[str, ...] = ("claude-3-5-sonnet-20241022",)
    batch: bool = True
    num_repeats: int = 10
    limit_dataset: int | None = None


def analyze_timing_data(timing_data):
    df = pd.DataFrame(timing_data)

    # Calculate statistics
    print("\nOverall Statistics:")
    print(f"Mean duration: {df['duration'].mean():.2f} seconds")
    print(f"Std duration: {df['duration'].std():.2f} seconds")
    print(f"Mean throughput: {df['throughput'].mean():.2f} prompts/second")
    print(f"Std throughput: {df['throughput'].std():.2f} prompts/second")

    print("\nThroughput by repeat:")
    print(df.groupby("repeat")["throughput"].agg(["mean"]))
    print("\nDuration by repeat:")
    print(df.groupby("repeat")["duration"].agg(["mean"]))

    return df


async def main(cfg: ExperimentConfig):
    df = pd.read_json("hf://datasets/yahma/alpaca-cleaned/alpaca_data_cleaned.json")
    instructions = df["instruction"].tolist()
    if cfg.limit_dataset is not None:
        instructions = instructions[: cfg.limit_dataset]
    prompts = [
        Prompt(
            messages=[
                ChatMessage(role=MessageRole.system, content="Think step by step before answering."),
                ChatMessage(role=MessageRole.user, content=instruction),
            ]
        )
        for instruction in instructions
    ]

    timing_data = []
    timing_repeat_data = []
    batch_size = 1000

    if cfg.batch:
        output_dir = cfg.output_dir / "anthropic_batch"
        output_dir.mkdir(parents=True, exist_ok=True)
        batch_model = AnthropicModelBatch()

        for repeat in range(cfg.num_repeats):
            batch_tasks = []
            repeat_start_time = time.time()

            for i in range(0, len(prompts), batch_size):
                batch_prompts = prompts[i : i + batch_size]
                batch_dir = output_dir / f"repeat_{repeat}" / f"batch_{i}"
                batch_dir.mkdir(parents=True, exist_ok=True)

                async def batch_task(model_ids, prompts, batch_dir):
                    batch_start_time = time.time()
                    print(f"Starting batch {batch_dir}")
                    responses, batch_id = await batch_model(
                        model_ids=model_ids, prompts=prompts, max_tokens=200, log_dir=batch_dir
                    )
                    batch_end_time = time.time()
                    batch_duration = batch_end_time - batch_start_time
                    throughput = len(prompts) / batch_duration

                    with open(batch_dir / "responses.json", "w") as f:
                        json.dump([r.completion for r in responses], f)

                    return {
                        "responses": responses,
                        "batch_id": batch_id,
                        "start_time": batch_start_time,
                        "end_time": batch_end_time,
                        "duration": batch_duration,
                        "throughput": throughput,
                        "num_prompts": len(prompts),
                    }

                batch_tasks.append(batch_task(cfg.model_ids, batch_prompts, batch_dir))

            batch_results = await asyncio.gather(*batch_tasks)

            repeat_end_time = time.time()
            repeat_duration = repeat_end_time - repeat_start_time
            repeat_total_prompts = sum(r["num_prompts"] for r in batch_results)
            repeat_throughput = repeat_total_prompts / repeat_duration

            timing_repeat_data.append(
                {
                    "repeat": repeat,
                    "start_time": repeat_start_time,
                    "end_time": repeat_end_time,
                    "duration": repeat_duration,
                    "throughput": repeat_throughput,
                    "num_prompts": repeat_total_prompts,
                }
            )

            # Record timing data for this repeat
            for i, result in enumerate(batch_results):
                timing_data.append(
                    {
                        "repeat": repeat,
                        "batch": i,
                        "start_time": result["start_time"],
                        "end_time": result["end_time"],
                        "duration": result["duration"],
                        "throughput": result["throughput"],
                        "num_prompts": result["num_prompts"],
                    }
                )

            _ = analyze_timing_data(timing_data)
            print(f"Repeat {repeat} complete, {repeat_duration:.2f} seconds, {repeat_throughput:.2f} prompts/second")

    else:
        responses = []

        for repeat in range(cfg.num_repeats):
            repeat_start_time = time.time()

            for i in range(0, len(prompts), batch_size):
                output_dir = cfg.output_dir / "anthropic_single" / f"repeat_{repeat}" / f"batch_{i}"
                output_dir.mkdir(parents=True, exist_ok=True)
                batch_prompts = prompts[i : i + batch_size]

                batch_start_time = time.time()
                batch_responses = await asyncio.gather(
                    *[cfg.api(model_ids=cfg.model_ids, prompt=prompt, max_tokens=200) for prompt in batch_prompts]
                )
                batch_end_time = time.time()

                batch_duration = batch_end_time - batch_start_time
                throughput = len(batch_prompts) / batch_duration

                timing_data.append(
                    {
                        "repeat": repeat,
                        "batch": i // batch_size,
                        "start_time": batch_start_time,
                        "end_time": batch_end_time,
                        "duration": batch_duration,
                        "throughput": throughput,
                        "num_prompts": len(batch_prompts),
                    }
                )

                responses.extend(batch_responses)
                with open(output_dir / "responses.json", "w") as f:
                    json.dump([r[0].completion for r in batch_responses], f)

            repeat_end_time = time.time()
            repeat_duration = repeat_end_time - repeat_start_time
            repeat_total_prompts = sum(r["num_prompts"] for r in timing_data if r["repeat"] == repeat)
            repeat_throughput = repeat_total_prompts / repeat_duration

            timing_repeat_data.append(
                {
                    "repeat": repeat,
                    "start_time": repeat_start_time,
                    "end_time": repeat_end_time,
                    "duration": repeat_duration,
                    "throughput": repeat_throughput,
                    "num_prompts": repeat_total_prompts,
                }
            )

            _ = analyze_timing_data(timing_data)
            print(f"Repeat {repeat} complete, {repeat_duration:.2f} seconds, {repeat_throughput:.2f} prompts/second")

    df = analyze_timing_data(timing_data)
    df_repeats = analyze_timing_data(timing_repeat_data)
    # Save timing data
    output_dir = cfg.output_dir / ("anthropic_batch" if cfg.batch else "anthropic_single")
    df.to_csv(output_dir / "timing_analysis.csv")
    df_repeats.to_csv(output_dir / "timing_repeat_analysis.csv")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_arguments(ExperimentConfig, dest="experiment_config")
    args = parser.parse_args()
    cfg: ExperimentConfig = args.experiment_config

    cfg.setup_experiment(log_file_prefix="get_responses")
    asyncio.run(main(cfg))
