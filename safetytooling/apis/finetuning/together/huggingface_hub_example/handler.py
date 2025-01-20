"""
This is a class that handles the inference of the model. It goes in handler.py within the hugging face model hub repo.
It then allows you to use the hugging face dedicated endpoint to run inference.
"""

from typing import Any, Dict

from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline


class EndpointHandler:
    def __init__(self, path=""):

        # Change these to the model you want to use
        base_model = "meta-llama/Llama-3.2-1B-Instruct"
        adapter_model = (
            "jplhughes2/llama-3.2-1b-af-lora-adapters"  # this is the adapter model downloaded from together ai
        )

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
        # Load base model
        base_model = AutoModelForCausalLM.from_pretrained(base_model, device_map="auto", trust_remote_code=True)
        # Load LoRA adapter
        self.model = PeftModel.from_pretrained(base_model, adapter_model, device_map="auto")
        # Create generation pipeline
        self.generator = pipeline("text-generation", model=self.model, tokenizer=self.tokenizer)

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        prompt = data.get("inputs", "")
        max_new_tokens = data.get("max_new_tokens", 128)
        temperature = data.get("temperature", 0.7)
        top_p = data.get("top_p", 0.9)

        outputs = self.generator(
            prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            return_full_text=False,  # this is to remove the input prompt from the output
        )

        # TODO handle finish reason and logprobs in "details" key of the dictionary

        return outputs
