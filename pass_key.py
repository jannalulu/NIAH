import os
import math
import fla
from transformers import GenerationConfig
import torch
import json
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
import types
from functools import wraps

model_path = "/workspace/RWKV-block/test/v7_goose/.hf_build/v7-1B5-world/"

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
    parser.add_argument('--min_tokens', type=int, default=65536, help='minimum token length to start evaluation')
    parser.add_argument('--max_tokens', type=int, default=65536, help='maximum token length for evaluation')
    parser.add_argument('--interval', type=int, default=1024, help='interval for evaluation')
    parser.add_argument('--num_tests', type=int, default=1, help='number of repeat testing for each length')
    parser.add_argument('--max_depth', type=float, default=1.0, help='max depth ratio to test')
    parser.add_argument('--device', type=str, default='cuda:0', help='device to use for computation')
    parser.add_argument('--hf_model_args', type=str, default='{}',
                      help='Additional HuggingFace model arguments as JSON string')
    args = parser.parse_args()
    return args


def generate_prompt_landmark(tokenizer, pass_key, context_length, depth, final_context_length_buffer=250):
    needle = f"The pass key is {pass_key}. Remember it. {pass_key} is the pass key. "
    task_description = "There is an important info hidden inside a lot of irrelevant text. Find it and memorize them. I will quiz you about the important information there. "
    garbage = "The grass is green. The sky is blue. The sun is yellow. Here we go. There and back again. "
    question = "What is the pass key? The pass key is"
    
    tokens_in_garbage = len(tokenizer.encode(garbage))
    multiplier = math.ceil((context_length - len(tokenizer.encode(task_description)) - 25) / tokens_in_garbage)
    context = garbage * multiplier
    
    tokens_task = tokenizer.encode(task_description)
    tokens_needle = tokenizer.encode(needle)
    tokens_context = tokenizer.encode(context)
    tokens_question = tokenizer.encode(question)
    tokens_newline = tokenizer.encode("\n")
    
    # Reduce context length by buffer
    context_length = context_length - final_context_length_buffer - len(tokens_task) - len(tokens_question)
    
    # Truncate context if needed
    if len(tokens_context) + len(tokens_task) + len(tokens_needle) + len(tokens_question) > context_length:
        tokens_context = tokens_context[:context_length - len(tokens_needle)]
    
    if depth >= 1:
        tokens_new_context = tokens_task + tokens_context + tokens_newline + tokens_needle + tokens_newline + tokens_question

    elif depth == 0:
        tokens_new_context = tokens_task + tokens_needle + tokens_newline + tokens_context + tokens_newline + tokens_question

    else:
        insertion_point = int(len(tokens_context) * depth)
        tokens_new_context = tokens_context[:insertion_point]
        
        # Find sentence break
        period_tokens = tokenizer.encode('.')
        while tokens_new_context and tokens_new_context[-1] not in period_tokens:
            insertion_point -= 1
            tokens_new_context = tokens_context[:insertion_point]
        
        tokens_new_context = tokens_task + tokens_new_context + tokens_newline + tokens_needle + tokens_newline + tokens_context[insertion_point:] + tokens_question
    
    print("Total Tokens in Context: ", len(tokens_new_context))
    new_context = tokenizer.decode(tokens_new_context)
    return new_context

def passkey_retrieval_test(model, tokenizer, device, context_length, depth, seed=666, debug=False):
    # Generate random pass key
    rnd_state = random.get_state()
    random.seed(seed)
    pass_key = random.randint(1, 50000)
    random.set_state(rnd_state)
    
    prompt = generate_prompt_landmark(tokenizer, pass_key, context_length=context_length, depth=depth)
    answer = str(pass_key)

    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    len_token = input_ids.shape[-1]
    answer_ids = tokenizer(answer, return_tensors="pt").input_ids.to(device)

    print("VRAM usage before generation:", get_gpu_memory(), "MB")

    # Chunk all but the last token
    CHUNK_SIZE = 2048
    chunk_input_ids = input_ids[:, :-1]  # everything except the final token
    last_token = input_ids[:, -1:]      # the final token

    # If there are no tokens to chunk, just run normally (no manual cache building)
    if chunk_input_ids.shape[1] == 0:
        # This means the prompt has only 1 token to feed
        # so we generate from scratch
        print("Short prompt, skipping chunking.\n")
        generation_output = model.generate(
            input_ids=input_ids,
            max_length=answer_ids.size(-1) + 16,
            use_cache=True,
        )
    else:
        # Otherwise, process the chunk in smaller pieces
        past_key_values = None
        with torch.no_grad():
            # Feed chunk in slices
            for i in range(0, chunk_input_ids.shape[1], CHUNK_SIZE):
                subchunk = chunk_input_ids[:, i : i + CHUNK_SIZE]
                outputs = model(subchunk, past_key_values=past_key_values)
                past_key_values = outputs.past_key_values

        # Provide non-empty cache_position to align the final token
        # so that model.generate knows how many tokens are in the cache
        position_offset = chunk_input_ids.shape[1]
        
        # Add debug instrumentation if requested
        if debug:
            # Save original method
            original_prepare_inputs = model.prepare_inputs_for_generation
            debug_info = {"calls": 0, "failures": 0}
            
            # Create a direct replacement for prepare_inputs_for_generation
            # Instead of debugging the original, we'll fully replace it with our fixed version
            def fixed_prepare_inputs_for_generation(self, input_ids, past_key_values=None, inputs_embeds=None, 
                                                   cache_position=None, attention_mask=None, **kwargs):
                debug_info["calls"] += 1
                call_num = debug_info["calls"]
                
                print(f"\nDEBUG Call #{call_num} to prepare_inputs_for_generation")
                print(f"  input_ids.shape: {input_ids.shape}")
                
                # Get model inputs dict
                model_inputs = self.prepare_inputs_for_decoder_forward(
                    input_ids, past_key_values, attention_mask, inputs_embeds, **kwargs
                )
                
                # Check past_key_values
                if past_key_values is not None:
                    try:
                        past_length = past_key_values[0][0].shape[2]
                        print(f"  past_key_values shape: {past_key_values[0][0].shape}")
                        print(f"  past_length from past_key_values: {past_length}")
                    except (IndexError, AttributeError, TypeError) as e:
                        print(f"  Error getting past_length: {e}")
                        past_length = 0
                else:
                    past_length = 0
                
                # THIS IS THE KEY FIX: Always create a valid cache_position
                # We use the past_length from past_key_values to create appropriate cache_position
                print(f"  Original cache_position: {cache_position}")
                
                # Create a valid cache_position regardless of what was passed in
                # This is the critical fix to avoid the IndexError
                if cache_position is None or len(cache_position) == 0:
                    print("  Creating new cache_position")
                    cache_position = torch.arange(past_length, input_ids.shape[1], dtype=torch.long, device=input_ids.device)
                    print(f"  New cache_position: {cache_position}")
                
                # Make sure it's never empty
                if len(cache_position) == 0:
                    print("  cache_position is empty, fixing it")
                    cache_position = torch.zeros(1, dtype=torch.long, device=input_ids.device)
                    print(f"  After fix: {cache_position}")
                
                # Add cache_position to model inputs
                model_inputs["cache_position"] = cache_position
                print(f"  Final cache_position in model_inputs: {model_inputs['cache_position']}")
                
                # Create proper attention_mask if not provided
                if attention_mask is None and past_length > 0:
                    print("  Creating attention_mask")
                    attention_mask = torch.ones(1, past_length + input_ids.shape[1], dtype=torch.long, device=input_ids.device)
                    model_inputs["attention_mask"] = attention_mask
                
                # Handle use_cache
                model_inputs["use_cache"] = kwargs.get("use_cache", True)
                
                print(f"  Call #{call_num} using custom fixed method")
                return model_inputs
            
            # Replace the prepare_inputs_for_generation method with our fixed version
            model.prepare_inputs_for_generation = types.MethodType(fixed_prepare_inputs_for_generation, model)
        
        # Now that we've replaced prepare_inputs_for_generation with our fixed version,
        # generate from the last token with the entire cached prompt
        
        # Create attention mask for the full sequence
        attention_mask = torch.ones(1, position_offset + 1, device=device, dtype=torch.long)
        
        # Use a valid cache_position pointing to the end of our processed sequence
        cache_position = torch.tensor([position_offset], device=device, dtype=torch.long)
        
        print("\nGenerating with fixed method and proper attention mask")
        print(f"  last_token.shape: {last_token.shape}")
        print(f"  attention_mask.shape: {attention_mask.shape}")
        print(f"  cache_position: {cache_position}")
        
        # Generate with our patched model
        generation_output = model.generate(
            input_ids=last_token,
            past_key_values=past_key_values,
            cache_position=cache_position,
            attention_mask=attention_mask,
            use_cache=True,
            max_length=answer_ids.size(-1) + 16,
        )
        
        # Restore original method if we patched it
        if debug:
            model.prepare_inputs_for_generation = original_prepare_inputs

    # Decode and compare
    model_output = tokenizer.decode(generation_output[0], skip_special_tokens=False)
    matches = re.findall(r"is[\D]*(\d+)", model_output)
    model_answer = matches[0] if matches else ""
    is_correct = (model_answer == answer)

    print("Model's output:", model_output)
    print("Found answer:", model_answer)
    print("Correct answer:", answer)
    print("Is correct:", is_correct, "\n")

    return is_correct, len_token

def main(args):
    device = "cuda:0"
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
                # Enable debug for the first run of each configuration
                is_debug = (i == 0 and depth == 0 and k == 0)
                if is_debug:
                    print(f"\n{'='*50}\nRUNNING WITH DEBUG ENABLED\n{'='*50}")
                    
                is_correct, len_tokens = passkey_retrieval_test(
                    model, tokenizer, device, 
                    context_length=current_tokens,
                    depth=depth,
                    seed=k,
                    debug=is_debug
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
   
    plt.savefig(f"data/heatmap_tokenized_{args.max_tokens}_{sanitized_model_name}.png")
    df_summary.to_csv(f"data/results_tokenized_{args.max_tokens}_{sanitized_model_name}.csv", index=False)

if __name__ == "__main__":
    args = parse_config()
    main(args)