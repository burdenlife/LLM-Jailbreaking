from vllm import LLM
import torch
from transformers import AutoTokenizer
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



import torch
from transformers import AutoTokenizer
from dinfer.decoding.serving import DiffusionLLMServing
from dinfer.decoding.serving import SamplingParams as DInferSamplingParams

class DInferRunner:
    def __init__(self, model_name):
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
        )
        self.model = DiffusionLLMServing(model=model_name)

    def generate(self, prompts, temperature=0.7, max_tokens=128):
        # 1️⃣ tokenize → tensor
        enc = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=1024,
        )

        input_ids = enc["input_ids"].to("cuda")

        # 2️⃣ sampling params (YOUR dInfer uses max_length)
        params = DInferSamplingParams(
            temperature=temperature,
            max_length=input_ids.shape[1] + max_tokens,
        )

        # 3️⃣ generate
        with torch.inference_mode():
            out_ids = self.model.generate(input_ids, params)

        # 4️⃣ decode back to text
        return self.tokenizer.batch_decode(
            out_ids,
            skip_special_tokens=True,
        )

