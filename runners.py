from vllm import LLM, SamplingParams
import torch
from transformers import AutoTokenizer
from dinfer.decoding.serving import DiffusionLLMServing, SamplingParams

class VLLMRunner:
    def __init__(self, model_name, max_model_len=2048):
        self.llm = LLM(
            model=model_name,
            max_model_len=max_model_len,
            gpu_memory_utilization=0.6,
            trust_remote_code=True,
        )

    def generate(self, prompts, temperature=0.7, max_tokens=256):
        sampling_params = SamplingParams(
            temperature=temperature,
            max_tokens=max_tokens,
        )

        outputs = self.llm.generate(prompts, sampling_params)

        # vLLM returns structured outputs
        return [o.outputs[0].text for o in outputs]



class DInferRunner:
    def __init__(self, model_name):
        engine = DiffusionLLMServing(
        model=model_name )

    def generate(self, prompts, temperature=0.7, max_tokens=256):
        results = []

        for prompt in prompts:
            inputs = self.tokenizer(prompt, return_tensors="pt").to("cuda")

            with torch.inference_mode():
                text = self.model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    do_sample=True,
                )

            
            results.append(text)

        return results
