from vllm import LLM, SamplingParams
import torch
from transformers import AutoTokenizer
from dinfer.decoding.serving import DiffusionLLMServing, SamplingParams

class VLLMRunner:
    def __init__(self, model_name, max_model_len=1024):
        self.llm = LLM(
            model=model_name,
            max_model_len=max_model_len,
            gpu_memory_utilization=0.75,
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
        self.model = DiffusionLLMServing(model=model_name )

    def generate(self, prompts, temperature=0.7, max_tokens=256):

        params = SamplingParams(
            temperature=temperature,
            max_tokens=max_tokens,
        )
        with torch.inference_mode():
            results = self.model.generate(prompts, params)

        return results
