from vllm import LLM, SamplingParams
import torch
from transformers import AutoTokenizer
from dinfer.model import AutoModelForCausalLM

class VLLMRunner:
    def __init__(self, model_name, max_model_len=4096):
        self.llm = LLM(
            model=model_name,
            max_model_len=max_model_len,
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
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
        )
        self.model.eval()

    def generate(self, prompts, temperature=0.7, max_tokens=256):
        results = []

        for prompt in prompts:
            inputs = self.tokenizer(prompt, return_tensors="pt").to("cuda")

            with torch.inference_mode():
                out_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    do_sample=True,
                )

            text = self.tokenizer.decode(
                out_ids[0], skip_special_tokens=True
            )
            results.append(text)

        return results
