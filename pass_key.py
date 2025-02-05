
import os
import math
import fla
import torch
import re
import argparse
import random
import re
import numpy as np
from numpy import random
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns

model_path = "../v7-1B4/"

def get_gpu_memory():
    """Returns the current GPU memory usage in MB."""
    torch.cuda.synchronize()
    return torch.cuda.memory_allocated() / 1024 / 1024

def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--base_model', type=str, default="fla-hub/rwkv7-1.5B-world")
    parser.add_argument('--cache_dir', type=str, default="./cache")

    parser.add_argument('--min_tokens', type=int, default=14384, help='minimum token length to start evaluation')
    parser.add_argument('--max_tokens', type=int, default=16384, help='maximum token length for evaluation')
    parser.add_argument('--interval', type=int, default=1024, help='interval for evaluation')
    parser.add_argument('--num_tests', type=int, default=5, help='number of repeat testing for each length')
    parser.add_argument('--min_depth', type=float, default=0.2, help='minimum depth ratio to start testing')

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
    
    CHUNK_SIZE = 2048
    past_key_values = None
    chunk_input_ids = input_ids[:, :-1]
    with torch.no_grad():
        # Process all tokens in chunks
        for i in range(0, chunk_input_ids.shape[1], CHUNK_SIZE):
            chunk = chunk_input_ids[:, i:i + CHUNK_SIZE]
            outputs = model(
                chunk,
                past_key_values=past_key_values,
            )
            current_mem = torch.cuda.memory_allocated(device) / 1024**2
            max_mem = torch.cuda.max_memory_allocated(device) / 1024**2
            print(f"Memory usage before chunk {i//CHUNK_SIZE + 1}: {current_mem:.2f}MB / {max_mem:.2f}MB")

            past_key_values = outputs.past_key_values

        generation_output = model.generate(
            input_ids=input_ids[:, -1:],
            past_key_values=past_key_values,
            max_length=answer_ids.shape[-1] + 16,
            use_cache=True,
        )
        current_mem = torch.cuda.memory_allocated(device) / 1024**2
        max_mem = torch.cuda.max_memory_allocated(device) / 1024**2
        print(f"Memory usage after generate: {current_mem:.2f}MB / {max_mem:.2f}MB")
    
    model_output = tokenizer.decode(generation_output[0].cpu())
    
    # Find the number after "The pass key is"
    matches = re.findall(r"is (\d+)", model_output)
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
    torch.set_float32_matmul_precision('high')

    print("base model", args.base_model)

    # Load model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True, tmix_backend="cuda")
    model = model.to('cuda')
    tokenizer = AutoTokenizer.from_pretrained('fla-hub/rwkv7-1.5B-world', trust_remote_code=True)

    model.eval()

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

    total_tests = len(all_accuracies)
    total_passed = sum(result['Score'] for result in all_accuracies)
    total_score = (total_passed / (total_tests * args.num_tests)) * 100

    print("\nFinal Results Summary:")
    print(f"Total Tests Run: {total_tests * args.num_tests}")
    print(f"Total Tests Passed: {total_passed}")
    print(f"Overall Score: {total_score:.2f}%")

    # Print detailed breakdown
    df_summary = pd.DataFrame(all_accuracies)
    print("\nDetailed Results by Context Length and Depth:")
    print(df_summary.groupby(['Context Length', 'Document Depth'])['Score'].mean().to_string())

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
    plt.savefig(f"data/heatmap_{args.max_tokens}_rwkv7_1b5_64k.png")

if __name__ == "__main__":
    args = parse_config()
    main(args)