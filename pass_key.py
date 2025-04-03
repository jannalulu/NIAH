import os
import math
import fla
from transformers import GenerationConfig, TextStreamer
import torch
import json
import argparse
import random
import re
import numpy as np
from numpy import random
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
import transformers
import transformers.generation.utils
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
from typing import Optional, Union, List, Dict, Any
import threading
from queue import Queue

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
    parser.add_argument('--min_tokens', type=int, default=1024, help='minimum token length to start evaluation')
    parser.add_argument('--max_tokens', type=int, default=65536, help='maximum token length for evaluation')
    parser.add_argument('--interval', type=int, default=1024, help='interval for evaluation')
    parser.add_argument('--num_tests', type=int, default=5, help='number of repeat testing for each length')
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


def passkey_retrieval_test(model, tokenizer, device, context_length, depth, seed=666):
    """
    Test passkey retrieval using TextIteratorStreamer for efficient KV cache management.
    This approach uses HuggingFace's streaming functionality which handles KV caching better.
    """
    rnd_state = random.get_state()
    random.seed(seed)
    pass_key = random.randint(1, 50000)
    random.set_state(rnd_state)

    prompt = generate_prompt_landmark(tokenizer, pass_key, context_length=context_length, depth=depth)
    answer = str(pass_key)
    input_token_ids = tokenizer(prompt, return_tensors=None).input_ids
    input_ids = torch.tensor([input_token_ids], device=device)
    seq_len = input_ids.shape[-1] # Full sequence length

    answer_ids = tokenizer(answer).input_ids
    max_new_tokens_to_generate = len(answer_ids) + 20
    
    print(f"Prompt length: {seq_len} tokens")
    print(f"Using TextIteratorStreamer for chunked generation with proper KV caching")

    # Setup streamer for token-by-token generation (better KV cache handling)
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    
    # Configure greedy generation
    generation_config = GenerationConfig(
        max_new_tokens=max_new_tokens_to_generate,
        use_cache=True,
        do_sample=False,
    )
    
    model_output = ""
    
    try:
        # First approach: Chunked generation with TextIteratorStreamer
        # The streamer handles token-by-token generation with proper KV cache management
        
        generation_kwargs = {
            "input_ids": input_ids,
            "generation_config": generation_config,
            "streamer": streamer,
            "return_dict_in_generate": True
        }
        
        # Run generation in a separate thread
        thread = threading.Thread(target=model.generate, kwargs=generation_kwargs)
        thread.start()
        
        # Collect generated tokens from streamer
        for token in streamer:
            model_output += token
        
        thread.join()
        
        print(f"Streamer generation successful")
        
    except Exception as e:
        print(f"Error with streamer approach: {e}")
        print("Trying fallback method...")
        
        try:
            # Second approach: Full context at once (may OOM for very long contexts)
            generation_output = model.generate(
                input_ids=input_ids,
                generation_config=generation_config
            )
            
            # Extract generated tokens (excluding prompt)
            if generation_output.shape[1] > input_ids.shape[1]:
                model_output_ids = generation_output[0, input_ids.shape[1]:]
                model_output = tokenizer.decode(model_output_ids.cpu(), skip_special_tokens=True)
            else:
                print("Warning: No new tokens generated")
                model_output = ""
                
        except Exception as e2:
            # Third approach: Try the original chunked approach as last resort
            print(f"Error with full sequence approach: {e2}")
            print("Trying original chunked approach...")
            
            try:
                past_key_values = None
                processed_len = 0
                prefill_ids = input_ids[:, :-1]
                prefill_len = prefill_ids.shape[1]
                chunk_size = 2048

                with torch.no_grad():
                    # Process in chunks to build KV cache
                    for i in range(0, prefill_len, chunk_size):
                        chunk = prefill_ids[:, i : min(i + chunk_size, prefill_len)]
                        chunk_len = chunk.shape[1]

                        outputs = model(
                            input_ids=chunk,
                            past_key_values=past_key_values,
                            use_cache=True,
                        )
                        past_key_values = outputs.past_key_values
                        processed_len += chunk_len
                        
                        # Cleanup inside loop
                        del outputs, chunk
                        if i % (chunk_size * 5) == 0:
                            torch.cuda.empty_cache()

                    # Generate with KV cache
                    last_token_input_ids = input_ids[:, -1:]
                    final_attention_mask = torch.ones(1, processed_len + 1, dtype=torch.long, device=device)
                    
                    generation_output = model.generate(
                        input_ids=last_token_input_ids,
                        past_key_values=past_key_values,
                        attention_mask=final_attention_mask,
                        generation_config=generation_config
                    )
                    
                    # Extract generated tokens
                    if generation_output.shape[1] > last_token_input_ids.shape[1]:
                        model_output_ids = generation_output[0, last_token_input_ids.shape[1]:]
                        model_output = tokenizer.decode(model_output_ids.cpu(), skip_special_tokens=True)
                    else:
                        model_output = ""
                        
                    # Cleanup
                    del past_key_values, input_ids, generation_output
                    if 'model_output_ids' in locals():
                        del model_output_ids
                    torch.cuda.empty_cache()
                    
            except Exception as e3:
                print(f"All generation approaches failed: {e3}")
                import traceback
                traceback.print_exc()
                return False, seq_len
    
    # Extract and evaluate the numeric answer
    matches = re.findall(r"(\d+)", model_output)
    model_answer = matches[0] if matches else ""
    is_correct = (model_answer == answer)

    print(f"Generated Text: '{model_output}'")
    print(f"Extracted Answer: {model_answer}")
    print(f"Correct Answer: {answer}")
    print(f"Result: {'CORRECT' if is_correct else 'INCORRECT'}")

    # Ensure cleanup
    torch.cuda.empty_cache()
    
    return is_correct, seq_len

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
   
    plt.savefig(f"data/heatmap_tokenized_{args.max_tokens}_{sanitized_model_name}.png")
    df_summary.to_csv(f"data/results_tokenized_{args.max_tokens}_{sanitized_model_name}.csv", index=False)

if __name__ == "__main__":
    args = parse_config()
    main(args)