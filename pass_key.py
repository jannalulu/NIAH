
import os
import math
import torch
import re
import argparse
import random
import numpy as np
from numpy import random
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns

model_path = "../rwkv7-1.5B-world3-32k/snapshots/848422f82e020c2b6c4deb43029afd62dc102e23"

def get_gpu_memory():
    """Returns the current GPU memory usage in MB."""
    torch.cuda.synchronize()
    return torch.cuda.memory_allocated() / 1024 / 1024

def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--base_model', type=str, default="fla-hub/rwkv7-1.5B-world")
    parser.add_argument('--cache_dir', type=str, default="./cache")
    parser.add_argument('--min_tokens', type=int, default=22179, help='minimum token length to start evaluation')
    parser.add_argument('--max_tokens', type=int, default=24385, help='maximum token length for evaluation')
    parser.add_argument('--interval', type=int, default=2048, help='interval for evaluation')
    parser.add_argument('--num_tests', type=int, default=3, help='number of repeat testing for each length')
    parser.add_argument('--min_depth', type=float, default=0.3, help='minimum depth ratio to start testing')

    args = parser.parse_args()
    return args


def generate_prompt_landmark(n_garbage, seed, n_garbage_prefix):
    """Generates a text file and inserts a passkey at a random position."""
    rnd_state = random.get_state()
    random.seed(seed)
    n_garbage_suffix = n_garbage - n_garbage_prefix

    task_description = "There is an important info hidden inside a lot of irrelevant text. Find it and memorize them. I will quiz you about the important information there."
    garbage = "The grass is green. The sky is blue. The sun is yellow. Here we go. There and back again."
    garbage_inf = " ".join([garbage] * 5000)
    assert len(garbage_inf) >= n_garbage
    garbage_prefix = garbage_inf[:n_garbage_prefix]
    garbage_suffix = garbage_inf[:n_garbage_suffix]
    pass_key = random.randint(1, 50000)
    information_line = f"The pass key is {pass_key}. Remember it. {pass_key} is the pass key."
    final_question = "What is the pass key? The pass key is"
    lines = [
        task_description,
        garbage_prefix,
        information_line,
        garbage_suffix,
        final_question,
    ]
    random.set_state(rnd_state)
    return "\n".join(lines), str(pass_key)

def passkey_retrieval_test(model, tokenizer, device, n_garbage_prefix, n_garbage=60000, seed=666):
    prompt, answer = generate_prompt_landmark(n_garbage, seed, n_garbage_prefix)
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    input_ids = input_ids.to(device)
    len_token = input_ids.shape[-1]

    print(f"VRAM usage before generation: {get_gpu_memory():.2f} MB")

    answer_ids = tokenizer(answer, return_tensors="pt").input_ids
    generation_output = model.generate(
        input_ids=input_ids,
        max_length=answer_ids.shape[-1] + input_ids.shape[-1],
        use_cache=True
    )
    
    model_output = tokenizer.decode(generation_output[0].cpu())
    
    # Find the number after "The pass key is"
    matches = re.findall(r"What is the pass key\? The pass key is (\d+)", model_output)
    if matches:
        model_answer = matches[0]  # Take the first match
    else:
        model_answer = ""
    
    is_correct = (model_answer == answer)
    print(f"Model's output: {model_output}")
    print(f"Found answer: {model_answer}")
    print(f"Correct answer: {answer}")
    print(f"Is correct: {is_correct}\n")
    
    return is_correct, len_token

def main(args):
    device = "cuda:0"
    torch.cuda.set_device(device)

    print("base model", args.base_model)

    # Load model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)
    model = model.to('cuda')
    tokenizer = AutoTokenizer.from_pretrained('fla-hub/rwkv7-1.5B-world', trust_remote_code=True)

    # Calculate number of test points starting from min_tokens
    total_test_points = (args.max_tokens - args.min_tokens) // args.interval + 1
    all_accuracies = []
    
    for i in range(total_test_points):
        # Calculate context length starting from min_tokens
        current_tokens = args.min_tokens + (i * args.interval)
        # Adjust the garbage text calculation to match the new token length
        n_garbage = int(3.75 * current_tokens // 1024 * 1024)
        
        # Calculate depth steps starting from min_depth
        depth_steps = np.linspace(args.min_depth, 1.0, 10)
        
        for depth in depth_steps:
            n_garbage_prefix = int(n_garbage * depth)
            passed_tests = 0
            total_tokens = 0
            
            for k in range(args.num_tests):
                is_correct, len_tokens = passkey_retrieval_test(
                    model, tokenizer, device, n_garbage_prefix, 
                    n_garbage=n_garbage, seed=k
                )
                passed_tests += is_correct
                total_tokens += len_tokens
                
            avg_tokens = total_tokens // args.num_tests
            accuracy = float(passed_tests) / args.num_tests
            print(f"accuracy on the token length {avg_tokens}, depth {depth:.2f}, is {accuracy:.2f}")
            
            result = {
                "Context Length": avg_tokens,
                "Document Depth": round(depth * 100, -1),
                "Score": passed_tests
            }
            all_accuracies.append(result)

    # Create visualization
    df = pd.DataFrame(all_accuracies)
    cmap = LinearSegmentedColormap.from_list("custom_cmap", ["#F0496E", "#EBB839", "#0CD79F"])
    
    pivot_table = pd.pivot_table(
        df, values='Score', index=['Document Depth', 'Context Length'], 
        aggfunc='mean'
    ).reset_index()
    pivot_table = pivot_table.pivot(
        index="Document Depth", columns="Context Length", values="Score"
    )
    
    plt.figure(figsize=(17.5, 8))
    sns.heatmap(
        pivot_table,
        fmt="g",
        cmap=cmap,
        cbar_kws={'label': 'Score'}
    )

    plt.xlabel('Token Limit')
    plt.ylabel('Depth Percent')
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(f"data/heatmap_{args.max_tokens}.png")

if __name__ == "__main__":
    args = parse_config()
    main(args)