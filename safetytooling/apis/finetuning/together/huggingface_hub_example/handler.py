import argparse
import os
import subprocess
from typing import Any, Dict

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline


class EndpointHandler:
    def __init__(
        self,
        base_model: str = "jplhughes2/llama-3.2-1b-af-lora-adapters",
        adapter_model: str = "abhayesian/llama-3.3-70b-af-synthetic-finetuned",
        cache_dir: str = os.getenv("HF_HOME", None),
    ):
        # Get the list of available GPUs using nvidia-smi
        gpu_query = subprocess.run(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"], capture_output=True, text=True
        )
        gpu_indices = gpu_query.stdout.strip().replace("\n", ",")
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_indices

        print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES')}")
        print(f"torch.cuda.device_count(): {torch.cuda.device_count()}")
        print(f"torch.cuda.current_device(): {torch.cuda.current_device()}")
        print(f"torch.cuda.get_device_name(0): {torch.cuda.get_device_name(0)}")

        # Load tokenizer
        print(f"Loading tokenizer for {base_model}")
        self.tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True, cache_dir=cache_dir)

        # Load base model with float16
        print(f"Loading base model for {base_model}")
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model, device_map="auto", trust_remote_code=True, torch_dtype=torch.float16, cache_dir=cache_dir
        )
        print(f"Base model device: {base_model.device}")
        assert base_model.device.type == "cuda", f"Base model is not on GPU, got device: {base_model.device}"

        # Load LoRA adapter
        print(f"Loading LoRA adapter for {adapter_model}")
        self.model = PeftModel.from_pretrained(base_model, adapter_model, device_map="auto", cache_dir=cache_dir)
        print(f"LoRA adapter device: {self.model.device}")
        assert self.model.device.type == "cuda", f"LoRA adapter is not on GPU, got device: {self.model.device}"

        # Create generation pipeline
        print("Creating generation pipeline")
        self.generator = pipeline("text-generation", model=self.model, tokenizer=self.tokenizer)

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        prompt = data.get("inputs", None)
        if prompt is None:
            return [{"generated_text": ""}]
        if not isinstance(prompt, list):
            prompt = [prompt]
        max_new_tokens = data.get("max_new_tokens", 128)
        temperature = data.get("temperature", 0.7)
        top_p = data.get("top_p", 0.9)

        outputs = self.generator(
            prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            return_full_text=False,
        )

        return outputs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load a model and run inference.")
    parser.add_argument(
        "--base_model", type=str, default="meta-llama/Llama-3.3-70B-Instruct", help="Base model name to load"
    )
    parser.add_argument(
        "--adapter_model",
        type=str,
        default="abhayesian/llama-3.3-70b-af-synthetic-finetuned",
        help="Adapter model name to load",
    )
    parser.add_argument("--message", type=str, default="Hello, world!", help="Message to run inference on")
    parser.add_argument("--cache_dir", type=str, default=os.getenv("HF_HOME", None), help="Path to the cache directory")
    args = parser.parse_args()

    print(f"Base model: {args.base_model}")
    print(f"Adapter model: {args.adapter_model}")
    print(f"HF_HOME: {os.getenv('HF_HOME')}")
    print(f"Cache dir: {args.cache_dir}")

    handler = EndpointHandler(args.base_model, args.adapter_model, args.cache_dir)
    try:
        while True:
            result = handler({"inputs": args.message})
            print(result)
    except KeyboardInterrupt:
        print("Inference stopped by user.")
