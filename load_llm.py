import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import hashlib
import time
from datetime import datetime
from huggingface_hub import login
import llada_generate
import pandas as pd
import os
from sys import argv


login(os.environ["HUGGINGFACEHUB_API_TOKEN"])

MODEL_NAME = {"llama":"meta-llama/Meta-Llama-3-8B-Instruct", "llada":"GSAI-ML/LLaDA-8B-Instruct" }
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def select_model(model_name: str):
    return MODEL_NAME[model_name]



def get_input_folder() -> str:
    if len(argv) > 3:
        folder_name = argv[3]
        return f"./{folder_name}"
    
    return  "./prompt_file"


def get_output_file() -> str:
    if len(argv) > 2:
        output_file = argv[2]
        return f"output/{output_file}"
    
    return "output/finalLLaDAResults.csv"


def get_model_name() -> str:
    if len(argv) > 1:
        choice = argv[1].lower()
        if choice in ["llama", "llada"]:
            return choice
        else: 
            raise ValueError("Invalid model choice. Please choose 'llama' or 'llada'.")

    print("Select model to load:")
    print("1. LLaMA")
    print("2. LLaDA")
    choice = input("Enter choice (1 or 2): ")
    
    match choice:
        case "1":
            choice = "llama"
        case "2":
            choice = "llada"
    
    return choice
   
def hash_prompt(prompt: str) -> str:
    return hashlib.sha256(prompt.encode()).hexdigest()

def load_model(model_name):
    print("Loading model...")
    print("model_name:", model_name)
    if model_name == MODEL_NAME["llada"]:
        
        model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,   # or torch.float16
        device_map="auto",
        trust_remote_code=True,
    )
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        model.config.use_cache = False
    else:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
         
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            dtype=torch.bfloat16,
            device_map="auto"
        )
    #print("device map:", getattr(model, "hf_device_map", "no device map")) #debug

    model.eval()
    return model, tokenizer

def run_single_inference(model, tokenizer, prompt, max_new_tokens=128, temperature=0.7, is_llama=True):
    """Send a single prompt to the model and return output with metadata"""
    start_time = time.time()
    
    if is_llama:
        inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=True,
            use_cache=is_llama,
            pad_token_id=tokenizer.eos_token_id
        )
        input_ids = inputs["input_ids"][0]
        gen_ids = output_ids[0]
        prompt_length = input_ids.shape[0]
        new_ids = gen_ids[prompt_length:]
        output_text = tokenizer.decode(new_ids, skip_special_tokens=True)

    else:
        chat_prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            add_generation_prompt=True,
            tokenize=False
        )
        input_ids = tokenizer(chat_prompt, return_tensors="pt")["input_ids"].to(DEVICE)
        output_ids = llada_generate.generate(
            model,
            input_ids,
            gen_length=max_new_tokens,
            steps = max_new_tokens//3,
            temperature=temperature
        )
        
        output_text = tokenizer.batch_decode(output_ids[:, input_ids.shape[1]:], skip_special_tokens=True)
    
    result = {
        "timestamp": datetime.now().isoformat(),
        "model": MODEL_NAME["llama"] if is_llama else MODEL_NAME["llada"],
        "prompt": prompt,
        "prompt_hash": hash_prompt(prompt),
        "output": output_text,
        "temperature": temperature,
        "max_new_tokens": max_new_tokens,
        "execution_time": round(time.time() - start_time, 4)
    }
    return result


def run_batch_inference(model, tokenizer, prompts, max_new_tokens=128, temperature=0.7, is_llama=True):
    if is_llama:
        return run_llama_inference(model, tokenizer, prompts, max_new_tokens=max_new_tokens, temperature=temperature)
    
    else:
        return run_llada_inference(model, tokenizer, prompts, max_new_tokens=max_new_tokens, temperature=temperature)



@torch.inference_mode()
def run_llama_inference(model, tokenizer, prompts, *, max_new_tokens=128, temperature=0.7):
    tok = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
    )
    input_ids = tok["input_ids"].to(model.device)
    attention_mask = tok.get("attention_mask", None)
    if attention_mask is not None:
        attention_mask = attention_mask.to(model.device)

    output_ids = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        do_sample=True,
        use_cache=True,
        pad_token_id=tokenizer.eos_token_id,
    )
    prompt_lens = tok["attention_mask"].sum(dim=1).tolist() if "attention_mask" in tok else [input_ids.size(1)] * input_ids.size(0)

    results = []
    for i in range(output_ids.size(0)):
        gen_ids = output_ids[i, prompt_lens[i]:]
        results.append(tokenizer.decode(gen_ids, skip_special_tokens=True))
    return results





@torch.inference_mode()
def run_llada_inference(model, tokenizer, prompts, *, max_new_tokens=128, temperature=0.7):
    tok = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
    )
    input_ids = tok["input_ids"].to(model.device)
    attention_mask = tok.get("attention_mask", None)
    if attention_mask is not None:
        attention_mask = attention_mask.to(model.device)

    output_ids = llada_generate.generate(
        model,
        input_ids,
        gen_length=max_new_tokens,
        steps=16,
        temperature=temperature,
        cfg_scale=0.0,
        attention_mask=attention_mask,  
    )
    prompt_lens = tok["attention_mask"].sum(dim=1).tolist() if "attention_mask" in tok else [input_ids.size(1)] * input_ids.size(0)

    results = []
    for i in range(output_ids.size(0)):
        gen_ids = output_ids[i, prompt_lens[i]:]
        results.append(tokenizer.decode(gen_ids, skip_special_tokens=True))
    return results


# def run_batch_inference(model, tokenizer, prompts, max_new_tokens=128, temperature=0.7):
#     """Send a batch of prompts to the model and return output with metadata"""
#     results = []
#     for prompt in prompts:
#         result = run_single_inference(
#             model, tokenizer, prompt,
#             max_new_tokens=max_new_tokens,
#             temperature=temperature
#         )
#         results.append(result)
#     return results

import csv

def save_results_csv(df, path="output/results.csv"):
    """Save results to a CSV file"""
    df.to_csv(path, index=False)
    print(f"Saved CSV to {path}")


if __name__ == "__main__":
    choice = get_model_name()
    model_name = select_model(choice)

    model, tokenizer = load_model(model_name)

    test_prompts = [
        "Explain diffusion models in simple terms.",
        "Write a short poem about Singapore sunsets.",
        "Summarize the concept of overfitting.",
        "Describe the role of an operating system.",
        "Generate 3 cybersecurity tips.",
        "Translate 'I love machine learning' to Spanish.",
        "Give an example of a DDoS attack.",
        "What is the difference between RAM and VRAM?",
        "Explain the concept of BFS and DFS.",
        "Define supervised vs unsupervised learning."
    ]

    results = run_batch_inference(model, tokenizer, test_prompts, is_llama = choice == "llama")
    save_results_csv(results)
