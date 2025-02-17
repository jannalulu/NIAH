import os
import math
from transformers import GenerationConfig
import torch
import argparse
import json
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

def get_gpu_memory():
    """Returns the current GPU memory usage in MB."""
    torch.cuda.synchronize()
    return torch.cuda.memory_allocated() / 1024 / 1024

def parse_config():
    """Parse command line arguments.
    
    Returns:
        argparse.Namespace: Parsed arguments including:
            - Standard evaluation parameters
            - HF model path and cache directory
            - Optional HF model arguments as JSON string
    """
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('hf_model', type=str)
    parser.add_argument('--cache_dir', type=str, default="./cache")
    parser.add_argument('--min_tokens', type=int, default=8192, help='minimum token length to start evaluation')
    parser.add_argument('--max_tokens', type=int, default=32768, help='maximum token length for evaluation')
    parser.add_argument('--interval', type=int, default=2048, help='interval for evaluation')
    parser.add_argument('--num_tests', type=int, default=5, help='number of repeat testing for each length')
    parser.add_argument('--max_depth', type=float, default=1.0, help='max depth ratio to test')
    parser.add_argument('--device', type=str, default='cuda:0', help='device to use for computation')
    parser.add_argument('--disable_hf_kv_cache', action='store_true', help='Use HuggingFace cache directory')
    parser.add_argument('--hf_model_args', type=str, default='{}',
                      help='Additional HuggingFace model arguments as JSON string')

    args = parser.parse_args()
    return args


def generate_prompt_landmark(tokenizer, pass_key, context_length, depth, final_context_length_buffer=250):
    needle = f"The pass key is {pass_key}. Remember it. {pass_key} is the pass key.\n"
    task_description = "There is an important info hidden inside a lot of irrelevant text. Find it and memorize them. I will quiz you about the important information there.\n"
    garbage = "The grass is green. The sky is blue. The sun is yellow. Here we go. There and back again.\n"
    question = "\n\nWhat is the pass key? The pass key is"
    
    tokens_in_garbage = len(tokenizer.encode(garbage))
    multiplier = math.ceil((context_length - len(tokenizer.encode(task_description)) - 25) / tokens_in_garbage)
    context = garbage * multiplier
    
    tokens_task = tokenizer.encode(task_description)
    tokens_needle = tokenizer.encode(needle)
    tokens_context = tokenizer.encode(context)
    tokens_question = tokenizer.encode(question)
    token_newline = tokenizer.encode("\n")
    
    # Reduce context length by buffer
    context_length = context_length - final_context_length_buffer - len(tokens_task) - len(tokens_question)
    
    # Truncate context if needed
    if len(tokens_context) + len(tokens_task) + len(tokens_needle) + len(question) > context_length:
        tokens_context = tokens_context[:context_length - len(tokens_needle)]
    
    if depth >= 1:
        tokens_new_context = tokens_task + tokens_context + token_newline + tokens_needle + token_newline + tokens_question

    elif depth == 0: 
        tokens_new_context = tokens_task + tokens_needle + token_newline + tokens_context + token_newline + tokens_question

    else:
        insertion_point = int(len(tokens_context) * depth)
        tokens_new_context = tokens_context[:insertion_point]
        
        # Find sentence break
        period_tokens = tokenizer.encode('.')
        while tokens_new_context and tokens_new_context[-1] not in period_tokens:
            insertion_point -= 1
            tokens_new_context = tokens_context[:insertion_point]
        
        tokens_new_context = tokens_task + tokens_new_context + token_newline + tokens_needle + token_newline + tokens_context[insertion_point:] + tokens_question
    
    print("Total Tokens in Context: ", len(tokens_new_context))
    new_context = tokenizer.decode(tokens_new_context)
    return new_context

def passkey_retrieval_test(model, tokenizer, device, context_length, depth, seed=666):
    rnd_state = random.get_state()
    random.seed(seed)
    pass_key = random.randint(1, 50000)
    random.set_state(rnd_state)
    
    prompt = generate_prompt_landmark(tokenizer, pass_key, context_length=context_length, depth=depth)
    answer = str(pass_key)
    
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    input_ids = input_ids.to(device)
    len_token = input_ids.shape[-1]

    print(f"VRAM usage before generation: {get_gpu_memory():.2f} MB")
    # answer_ids = tokenizer(answer, return_tensors="pt").input_ids
    
    with torch.inference_mode():
        # Disable chunking, as its implemented native in the model (for rwkv)
        # ---
        # CHUNK_SIZE = 2048
        # chunk_input_ids = input_ids[:, :-1]
        # past_key_values = None
        
        # # Process all tokens in chunks
        # for i in range(0, chunk_input_ids.shape[1], CHUNK_SIZE):
        #     chunk = chunk_input_ids[:, i:i + CHUNK_SIZE]
        #     outputs = model(
        #         chunk,
        #         past_key_values=past_key_values,
        #     )
        #     current_mem = torch.cuda.memory_allocated(device) / 1024**2
        #     max_mem = torch.cuda.max_memory_allocated(device) / 1024**2
        #     print(f"Memory usage before chunk {i//CHUNK_SIZE + 1}: {current_mem:.2f}MB / {max_mem:.2f}MB")
        #     past_key_values = outputs.past_key_values

        enable_kv_caching = False if args.disable_hf_kv_cache == True else True

        generation_output = model.generate(
            input_ids=input_ids[:,:],
            # past_key_values=past_key_values,
            max_new_tokens=10,
            use_cache=enable_kv_caching,
            # generation_config=GenerationConfig(do_sample=False, use_cache=enable_kv_caching),
        )
        current_mem = torch.cuda.memory_allocated(device) / 1024**2
        max_mem = torch.cuda.max_memory_allocated(device) / 1024**2
        print(f"Memory usage after generate: {current_mem:.2f}MB / {max_mem:.2f}MB")
    
        # Get the last 16 tokens from the generation output
        model_output = tokenizer.decode(generation_output[0][-14:].cpu())
    
        # Find the number after "The pass key is"
        matches = re.findall(r"is.*(\d+)", model_output)
        if matches:
            model_answer = matches[0]  # Take the first match
        else:
            model_answer = ""
        
        is_correct = (model_answer == answer)
        print(f"Model's output: ... {model_output}")
        print(f"Found answer: {model_answer}")
        print(f"Correct answer: {answer}")
        print(f"Is correct: {is_correct}\n")
        
    return is_correct, len_token

def main(args):
    device = args.device
    torch.cuda.set_device(device)
    torch.set_float32_matmul_precision('high')

    print("HF Model", args.hf_model)

    # Parse additional HF model arguments
    hf_model_args = json.loads(args.hf_model_args)
    
    # Load model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(
        args.hf_model,
        trust_remote_code=True,
        **hf_model_args
    ).bfloat16().to(device)
    tokenizer = AutoTokenizer.from_pretrained(args.hf_model, trust_remote_code=True)
    model.eval()

    # Calculate number of test points starting from min_tokens
    total_test_points = (args.max_tokens - args.min_tokens) // args.interval + 1
    all_accuracies = []
    
    for i in range(total_test_points):
        # Calculate context length starting from min_tokens
        current_tokens = args.min_tokens + (i * args.interval)
        
        # Calculate depth steps to max_depth
        depth_steps = np.linspace(0, args.max_depth, 10) # 10 steps from 0 to max_depth
        
        for depth in depth_steps:
            passed_tests = 0
            total_tokens = 0
            
            for k in range(args.num_tests):
                is_correct, len_tokens = passkey_retrieval_test(
                    model, tokenizer, device, 
                    context_length=current_tokens,
                    depth=depth,
                    seed=k
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
    # Extract last 2 path components and create sanitized filename
    model_path_parts = args.hf_model.split('/')
    sanitized_model_name = '_'.join(model_path_parts[-2:] if len(model_path_parts) > 1 else model_path_parts[-1:])
    
    plt.savefig(f"data/heatmap_tokenized_{args.max_tokens}_{sanitized_model_name}_linebreaks.png")

if __name__ == "__main__":
    args = parse_config()
    main(args)
