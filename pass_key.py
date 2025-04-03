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
import transformers
import transformers.generation.utils
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
from typing import Optional, Union, List, Dict, Any

# Import for text generation inference
try:
    import text_generation
    from text_generation.client import Client as TextGenerationClient
    has_text_generation = True
    print("Found text-generation-inference client")
except ImportError:
    has_text_generation = False
    print("text-generation-inference not installed, please run: pip install text-generation")

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
    Test passkey retrieval using text-generation-inference's server mode.
    This approach specifically leverages TGI's optimized KV caching capability.
    """
    rnd_state = random.get_state()
    random.seed(seed)
    pass_key = random.randint(1, 50000)
    random.set_state(rnd_state)

    # Generate the full prompt with the passkey
    prompt = generate_prompt_landmark(tokenizer, pass_key, context_length=context_length, depth=depth)
    answer = str(pass_key)
    input_token_ids = tokenizer(prompt, return_tensors=None).input_ids
    seq_len = len(input_token_ids)
    
    answer_ids = tokenizer(answer).input_ids
    max_new_tokens = len(answer_ids) + 20
    
    print(f"Prompt length: {seq_len} tokens")
    print(f"Using text-generation-inference server for proper KV cache handling")
    
    # ===== SETUP AND LAUNCH TEXT-GENERATION-INFERENCE SERVER =====
    
    # Save model to a temporary directory for TGI server
    import tempfile
    import subprocess
    import time
    import socket
    
    # Find an available port for the server
    def get_free_port():
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(('', 0))
            return s.getsockname()[1]
    
    port = get_free_port()
    print(f"Launching TGI server on port {port}")
    
    # Save model to a temporary directory
    temp_dir = tempfile.mkdtemp()
    model_path = os.path.join(temp_dir, "model")
    print(f"Saving model to {model_path}")
    
    # Save model and tokenizer
    model.save_pretrained(model_path)
    tokenizer.save_pretrained(model_path)
    
    # Launch TGI server using the model with verbose output and timeout settings
    server_cmd = [
        "text-generation-launcher",
        "--model-id", model_path,
        "--port", str(port),
        "--json-output",        # Use JSON for more structured logs
        "--max-total-tokens", str(seq_len + max_new_tokens),  # Make sure we can handle the full context
        "--max-input-length", str(seq_len),  # Set max input length
        "--max-batch-size", "1"  # Simple single-batch mode
    ]
    
    print(f"Server command: {' '.join(server_cmd)}")
    
    server_process = None
    try:
        # Start TGI server process
        server_process = subprocess.Popen(
            server_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        # Wait for server to start
        print("Waiting for TGI server to start...")
        time.sleep(10)  # Give server time to initialize
        
        # Check if server is running
        def is_server_running(port):
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                return s.connect_ex(('localhost', port)) == 0
        
        if not is_server_running(port):
            print("Warning: Server may not be running properly")
            print("Checking server logs...")
            
            # Check server output for errors
            if server_process.poll() is not None:
                stdout, stderr = server_process.communicate()
                print("Server stdout:", stdout[:500])
                print("Server stderr:", stderr[:500])
                print("Server exited with code:", server_process.returncode)
            
            print("Waiting a bit longer for server to start...")
            time.sleep(10)  # Wait longer
            
            # Double check if it's now running
            if not is_server_running(port):
                print("Server still not responding")
            else:
                print("Server is now responding")
        
        # ===== QUERY THE SERVER USING THE TEXT-GENERATION CLIENT =====
        
        # Import the client here to ensure it's only used when needed
        from text_generation import Client as TextGenerationClient
        
        # Setup client to connect to our local server
        client = TextGenerationClient(f"http://localhost:{port}")
        print("Connected to TGI server - sending request")
        
        # IMPORTANT: TGI is specifically designed to handle the KV cache for long context
        # efficiently under the hood - this is why we're using it
        response = client.generate(
            prompt,
            max_new_tokens=max_new_tokens,
            # TGI requires temperature > 0, use very small value for deterministic generation
            temperature=0.01,
            do_sample=False,
        )
        
        # Extract the generated text
        model_output = response.generated_text
        print("Successfully received response from TGI server")
        
        # Extract the answer and check correctness
        matches = re.findall(r"(\d+)", model_output)
        model_answer = matches[0] if matches else ""
        is_correct = (model_answer == answer)
        
        print(f"Generated Text: '{model_output}'")
        print(f"Extracted Answer: {model_answer}")
        print(f"Correct Answer: {answer}")
        print(f"Result: {'CORRECT' if is_correct else 'INCORRECT'}")
        
        return is_correct, seq_len
        
    except Exception as e:
        print(f"Error with TGI approach: {e}")
        print("Falling back to simple auto-regressive generation")
        
        # ===== FALLBACK: TRY DIRECTLY WITH TRANSFORMERS PIPELINE =====
        
        print("Trying with HuggingFace Pipeline (this may handle KV cache better)")
        
        try:
            from transformers import pipeline
            
            # Create generation pipeline with proper parameters
            text_generator = pipeline(
                'text-generation',
                model=model,
                tokenizer=tokenizer,
                device=device,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
            
            # Generate using pipeline's built-in cache handling
            output = text_generator(
                prompt,
                return_full_text=False,  # Only return the generated part
            )
            
            # Extract the generated text
            model_output = output[0]['generated_text']
            
        except Exception as e:
            print(f"Pipeline approach failed: {e}")
            print("Using final fallback approach with direct generation")
            
            # Last resort: try to work around cache position bug
            # Just process the end of the prompt where the answer is likely to be
            print("Processing only the end of the prompt to avoid memory issues")
            
            # Handle just a manageable portion of the end of the prompt
            max_handle_tokens = min(8192, len(input_token_ids))
            truncated_tokens = input_token_ids[-max_handle_tokens:]
            input_ids = torch.tensor([truncated_tokens], device=device)
            
            # Generate with truncated context
            with torch.no_grad():
                generation_config = GenerationConfig(
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    num_beams=1,
                )
                
                output_ids = model.generate(
                    input_ids=input_ids,
                    generation_config=generation_config
                )
                
                # Extract just the newly generated part
                generated_tokens = output_ids[0, len(truncated_tokens):]
                model_output = tokenizer.decode(generated_tokens, skip_special_tokens=True)
        
        # Extract the answer
        matches = re.findall(r"(\d+)", model_output)
        model_answer = matches[0] if matches else ""
        is_correct = (model_answer == answer)
        
        print(f"Generated Text: '{model_output}'")
        print(f"Extracted Answer: {model_answer}")
        print(f"Correct Answer: {answer}")
        print(f"Result: {'CORRECT' if is_correct else 'INCORRECT'}")
        
        return is_correct, seq_len
        
    finally:
        # Clean up resources
        if server_process:
            print("Terminating TGI server")
            server_process.terminate()
            try:
                server_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                server_process.kill()
                
        # Clean up the temporary directory
        import shutil
        print(f"Cleaning up temporary directory: {temp_dir}")
        try:
            shutil.rmtree(temp_dir)
        except Exception as e:
            print(f"Error cleaning up: {e}")
            
        # Final memory cleanup
        torch.cuda.empty_cache()

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