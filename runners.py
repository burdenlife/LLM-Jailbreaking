from vllm import LLM
import torch
from transformers import AutoTokenizer
from dinfer.decoding.serving import DiffusionLLMServing

class VLLMRunner:
    def __init__(self, model_name, max_model_len=1024):
        self.llm = LLM(
            model=model_name,
            max_model_len=max_model_len,
            gpu_memory_utilization=0.75,
            trust_remote_code=True,
            tensor_parallel_size=1,
            dtype="float16",
            
        )

    def generate(self, prompts, temperature=0.7, max_tokens=256):
        from vllm import SamplingParams
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
        from dinfer.decoding.serving import SamplingParams
        params = SamplingParams(
            temperature=temperature,
            max_new_tokens=max_tokens,
        )
        with torch.inference_mode():
            results = self.model.generate(prompts, params)

        return results
