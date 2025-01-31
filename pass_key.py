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
    try:
        prompt, answer = generate_prompt_landmark(n_garbage, seed, n_garbage_prefix)
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids
        
        if not torch.is_tensor(input_ids):
            raise ValueError("Tokenization failed to produce a valid tensor")
            
        input_ids = input_ids.to(device)
        len_token = input_ids.shape[-1]

        # Validate input dimensions
        if len_token < 1:
            raise ValueError(f"Input sequence length {len_token} is too short")
            
        print(f"VRAM usage before generation: {get_gpu_memory():.2f} MB")
        answer_ids = tokenizer(answer, return_tensors="pt").input_ids

        CHUNK_SIZE = 1024
        prefill_ids = input_ids[:, :-1]  # all tokens except last
        next_token = input_ids[:, -1:]   # last token
        
        if prefill_ids.shape[1] == 0:
            raise ValueError("No tokens available for prefill after splitting")
            
        past_key_values = None
        last_chunk_start = 0
        
        # Track if we've processed any chunks successfully
        chunks_processed = 0
        
        # Process chunks sequentially
        with torch.no_grad():
            for i in range(0, prefill_ids.shape[1], CHUNK_SIZE):
                try:
                    chunk = prefill_ids[:, i:i + CHUNK_SIZE]
                    
                    if chunk.shape[1] == 0:
                        raise ValueError(f"Empty chunk encountered at position {i}")
                    
                    # Log memory usage for monitoring
                    current_mem = torch.cuda.memory_allocated(device) / 1024**2
                    max_mem = torch.cuda.max_memory_allocated(device) / 1024**2
                    print(f"Memory usage before chunk {i//CHUNK_SIZE + 1}: {current_mem:.2f}MB / {max_mem:.2f}MB")
                    
                    # Generate with the current chunk
                    outputs = model(
                        input_ids=chunk,
                        past_key_values=past_key_values,
                        use_cache=True,
                        return_dict=True
                    )
                    
                    if not hasattr(outputs, 'past_key_values') or outputs.past_key_values is None:
                        raise ValueError("Model failed to return past key values")
                    
                    # Update past_key_values for next iteration
                    past_key_values = outputs.past_key_values
                    last_chunk_start = i + chunk.shape[1]
                    chunks_processed += 1
                    
                except RuntimeError as e:
                    if "out of memory" in str(e):
                        torch.cuda.empty_cache()
                        raise RuntimeError(f"GPU OOM at chunk {i//CHUNK_SIZE + 1}.")
                    raise e
                    
            if chunks_processed == 0:
                raise RuntimeError("No chunks were successfully processed")

            try:
                generation_output = model.generate(
                    input_ids=next_token,
                    past_key_values=past_key_values,
                    max_length=answer_ids.shape[-1],
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
                
                if generation_output.shape[0] == 0:
                    raise ValueError("Generation produced empty output")
                    
                model_output = tokenizer.decode(generation_output[0].cpu())
                
                # Find the number after "The pass key is"
                matches = re.findall(r"What is the pass key\? The pass key is (\d+)", model_output)
                if matches:
                    model_answer = matches[0]  # Take the first match
                else:
                    model_answer = ""
                    print("Warning: Could not find pass key in model output")
                
                is_correct = (model_answer == answer)
                print(f"Found answer: {model_answer}")
                print(f"Correct answer: {answer}")
                print(f"Is correct: {is_correct}\n")
                
                return is_correct, len_token
                
            except Exception as e:
                print(f"Error during generation or post-processing: {str(e)}")
                raise
                
    except Exception as e:
        print(f"Fatal error in passkey retrieval test: {str(e)}")
        # Return a failed test result rather than crashing
        return False, 0

def main(args):
    device = "cuda:0"
    torch.cuda.set_device(device)

    print("base model", args.base_model)

    # Load model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)
    model = model.to('cuda')
    model.eval()  # Added model.eval()
    tokenizer = AutoTokenizer.from_pretrained('fla-hub/rwkv7-1.5B-world', trust_remote_code=True)

    # Calculate number of test points starting from min_tokens
    total_test_points = (args.max_tokens - args.min_tokens) // args.interval + 1
    all_accuracies = []
    
    with torch.no_grad():  # Added no_grad context for the entire evaluation loop
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