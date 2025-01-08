import asyncio
import json
import logging
import time
from pathlib import Path

import openai
import openai.types
import openai.types.chat

from safetytooling.data_models import LLMResponse, Prompt
from safetytooling.utils import utils

LOGGER = logging.getLogger(__name__)


class OpenAIModelBatch:
    def __init__(self):
        self.client = openai.OpenAI()

    def create_message_batch(self, input_file_id: str) -> dict:
        """Create a batch of messages."""
        return self.client.batches.create(
            input_file_id=input_file_id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
            metadata={"description": "batch processing"},
        )

    def retrieve_message_batch(self, batch_id: str) -> dict:
        """Retrieve a message batch."""
        return self.client.batches.retrieve(batch_id)

    async def poll_message_batch(self, batch_id: str) -> dict:
        """Poll a message batch until it is completed."""
        while True:
            batch = self.retrieve_message_batch(batch_id)
            if batch.status == "completed":
                return batch
            elif batch.status == "failed":
                raise Exception(f"Batch {batch_id} failed: {batch.errors}")
            LOGGER.debug(f"Batch {batch_id} is still processing...")
            await asyncio.sleep(60)

    def list_message_batches(self, limit: int = 20) -> list[dict]:
        """List all message batches."""
        return [batch for batch in self.client.batches.list(limit=limit)]

    def retrieve_message_batch_results(self, output_file_id: str) -> list[dict]:
        """Retrieve results of a message batch."""
        file_response = self.client.files.content(output_file_id)
        file_contents = file_response.text
        return [json.loads(line) for line in file_contents.splitlines()]

    def cancel_message_batch(self, batch_id: str) -> dict:
        """Cancel a message batch."""
        return self.client.batches.cancel(batch_id)

    def prompts_to_file(self, model_id: str, prompts: list[Prompt], log_dir: Path, **kwargs) -> str:
        requests = []
        for i, prompt in enumerate(prompts):
            chat_messages = prompt.openai_format()
            hash_str = prompt.model_hash()
            requests.append(
                {
                    "custom_id": f"{i}_{hash_str}",
                    "method": "POST",
                    "url": "/v1/chat/completions",
                    "body": {
                        "model": model_id,
                        "messages": chat_messages,
                        **kwargs,
                    },
                }
            )
        input_file_path = log_dir / "batch_input.jsonl"
        utils.save_jsonl(input_file_path, requests)
        return input_file_path

    async def __call__(
        self,
        model_id: str,
        prompts: list[Prompt],
        log_dir: Path,
        **kwargs,
    ) -> tuple[list[LLMResponse], str]:
        start = time.time()

        input_file_path = self.prompts_to_file(model_id, prompts, log_dir, **kwargs)
        batch_file_object = self.client.files.create(file=open(input_file_path, "rb"), purpose="batch")
        batch_object = self.create_message_batch(input_file_id=batch_file_object.id)
        log_file = log_dir / f"batch_id_{batch_object.id}.json"
        log_file.parent.mkdir(parents=True, exist_ok=True)
        with open(log_file, "w") as f:
            json.dump(batch_object.model_dump(), f)
        print(f"Batch {batch_object.id} created with {len(prompts)} requests")

        batch_object = await self.poll_message_batch(batch_object.id)

        results = self.retrieve_message_batch_results(batch_object.output_file_id)

        responses = []
        for result in results:
            if result["response"]["status_code"] == 200:
                body = result["response"]["body"]
                choice = body["choices"][0]
                responses.append(
                    LLMResponse(
                        model_id=model_id,
                        completion=choice["message"]["content"],
                        stop_reason=choice["finish_reason"],
                        duration=None,  # Batch does not track individual durations
                        api_duration=None,
                        cost=0,
                    )
                )

        duration = time.time() - start
        LOGGER.debug(f"Completed batch call in {duration}s")

        return responses, batch_object.id
