import threading
import json
import time
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import pickle
from tqdm import tqdm
import argparse
import pdb
import anthropic
import google.generativeai as genai

from models.openai_related import gpt_generation, gpt_generation_with_image, gpt_generation_multi_turn
from models.anthropic_related import generate_claude_response, generate_claude_response_with_image, generate_claude_response_multi_turn
from models.google_related import generate_gemini_response, generate_gemini_response_with_image, generate_gemini_response_multi_turn
from models.togetherai_related import togetherai_generation, togetherai_generation_multi_turn

# Global Variables:# Global variable
cumulative_input_tokens = 0
cumulative_output_tokens = 0
cumulative_count = 0
global_sample_num = 10
prompt_dict_length = 0

# Lock for synchronizing access to the global variable
lock = threading.Lock()

def update_global_counter(input_tokens, output_tokens):
    global cumulative_input_tokens, cumulative_output_tokens, cumulative_count, prompt_dict_length  
    
    with lock:
        cumulative_input_tokens += input_tokens
        cumulative_output_tokens += output_tokens
        cumulative_count += 1

def call_until_timeout(func, timeout_seconds,delay=5,**kwargs):
    """
    Calls the specified function until it succeeds or the timeout is reached.

    Args:
    - func: The function to call.
    - timeout_seconds: Total time allowed for retries in seconds.
    - delay: Delay between retries in seconds.
    """
    model = kwargs['model']
    start_time = time.time()  # Record the start time
    while True:
        elapsed_time = time.time() - start_time  # Calculate elapsed time
        if elapsed_time > timeout_seconds:
            print("Timeout reached, stopping attempts.")
            return "**********NO OUTPUT**********"
        try:
            # Try to call the function
            result, tok_related = func(**kwargs)

            model = kwargs['model']
            
            input_tokens = tok_related[0]
            output_tokens = tok_related[1]

            if 'gemini' not in model:
                update_global_counter(input_tokens=input_tokens, output_tokens=output_tokens)

            return result
        except Exception as e:
            print(f"Function call failed with error: {e}")
            remaining_time = timeout_seconds - elapsed_time
            if remaining_time <= 0:
                print(f"Timeout reached, stopping attempts.")
                return "**********NO OUTPUT**********"
            print(f"Retrying Immediately...")

            # Wait for either the specified delay or the remaining time, whichever is shorter
            if 'gemini' in model:
                time.sleep(50)
                print("gemini is going to sleep for 50s")
            else:
                pass

def check_and_create_directory(dir_path):
    """Check if a directory exists, and if not, create it.
    
    Args:
    - dir_path: The path of the directory to check and create if necessary.
    """
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

def call_prompts_in_parallel(prompts_dict, output_path, start_ind, end_ind, max_workers=50,func=gpt_generation,timeout_seconds=30,model="gpt-3.5-turbo-0125",temperature=0,max_tokens=1024):
    """
    Executes calls to `call_until_timeout` in parallel using threading, for each prompt in the prompts_dict.
    
    :param prompts_dict: A dictionary where keys are identifiers and values are prompts.
    :param max_workers: The maximum number of threads to use.
    """
    check_and_create_directory(f"./{output_path}")
    pbar = tqdm(total=end_ind-start_ind)
    id2key = {id: key for id, key in enumerate(prompts_dict)}
    key2id = {key: id for id, key in enumerate(prompts_dict)}
    # Define the fixed parameters for the call_until_timeout function.
    fixed_params = {
        "func":func,
        "timeout_seconds": timeout_seconds,
        "model": model,
        "temperature": temperature,
        "max_tokens": max_tokens,

    }
    
    # Use ThreadPoolExecutor to run tasks in parallel.
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_id = {}

        for key, prompt in prompts_dict.items():
            if key2id[key] < start_ind or key2id[key] >= end_ind:
                continue

            future = executor.submit(call_until_timeout, **fixed_params, prompt=prompt)
            future_to_id[future] = key

        
        # Process the results as they complete.
        for future in as_completed(future_to_id):
            key = future_to_id[future]
            try:
                result = future.result()
                store = {key: result}
                check_and_create_directory(f"./{output_path}/separate_output/")
                with open(f"./{output_path}/separate_output/{key2id[key]}.pickle","wb") as f:
                    pickle.dump(store,f)
                pbar.update(1)
            except Exception as exc:
                print(f"{key} generated an exception: {exc}")
                
    pbar.close()
    return key2id

def load_prompt_dict(prompt_path):
    with open(prompt_path,"rb") as f:
        prompts = pickle.load(f)
    return prompts

def print_args_and_wait(args):
    # Iterate over all attributes in the args object

    max_arg_length = max(len(arg) for arg in vars(args))

    print("*"*50)

    print("Configured Arguments:")
    # Iterate over all attributes in the args object
    for arg in vars(args):
        # Retrieve the value of each attribute
        value = getattr(args, arg)
        # Print each argument with its value, aligned by the argument name's length
        print(f"{arg.ljust(max_arg_length)} = {value}")

    print("*"*50)

    print("Use Ctrl-C to stop if needed")

    for i in range(5):
        print(f"T-minus: {5-i}")
        time.sleep(1)

    print("Start Generation...")
    

def collect_result(prompt_dict, key2id, output_path):
    all_content = {}
    not_generated_key = []
    for key in prompt_dict:
        try:
            with open(f"{output_path}/separate_output/{key2id[key]}.pickle", 'rb') as f:
                key, result = list(pickle.load(f).items())[0]
                if result == "**********NO OUTPUT**********":
                    print("No output for key: ", key)
                    raise ValueError("No output")
                all_content[key] = result
        except:
            not_generated_key.append(key)
    with open(f"{output_path}/collected_results.pickle", 'wb') as f:
        pickle.dump(all_content, f)

    return not_generated_key

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--prompt_path', default=None, type=str, required=True, help="path of the prompt dictionary.")
    parser.add_argument('--output_path', default=None, type=str, required=True, help='the path to the output_directory')
    parser.add_argument('--max_workers', default=50, type=int)
    parser.add_argument('--timeout_seconds', default=240, type=int)
    parser.add_argument('--generation_model', default="gpt-3.5-turbo-0125", type=str)
    parser.add_argument('--temperature', default=0, type=int)
    parser.add_argument('--max_tokens', default=1024, type=int)
    parser.add_argument('--start_ind', default=0, type=int)
    parser.add_argument('--end_ind', default=-1, type=int)
    parser.add_argument('--with_images', default="False", type=str)
    parser.add_argument('--multi_turn', default="False", type=str)

    args = parser.parse_args()
    
    with open("generation/model_price.json", 'r') as f:
        model_price = json.load(f)

    if args.with_images == "True" and args.multi_turn == "True":
        raise ValueError("The program so far do not support multi_turn_with_image for now. Sorry :< ")

    model = args.generation_model

    print_args_and_wait(args)

    if "yarn-mistral" in model.lower():

        print("Using Yarn-Mistral-7b")

        prompt_dict = load_prompt_dict(args.prompt_path)

        prompts = [prompt_dict[key] for key in prompt_dict]

        sampling_params = SamplingParams(temperature=args.temperature, top_p=1)

        llm = LLM(model="TheBloke/Yarn-Mistral-7B-128k-AWQ", quantization="awq", dtype="auto")

        outputs = llm.generate(prompts, sampling_params)

        output_dict = {}

        for ind, key in enumerate(prompt_dict):
            output_dict[key] = outputs[ind].outputs[0].text

        check_and_create_directory(f"./{args.output_path}")

        with open(f"./{args.output_path}/collected_results.pickle","wb") as f:
            pickle.dump(output_dict,f)

    elif "longchat" in model.lower():

        print("Using LongChat")

        prompt_dict = load_prompt_dict(args.prompt_path)

        generate_longchat_response(prompt_dict, args.temperature, args.max_tokens, args.output_path)


    else:

        if model not in model_price:
            print(f"Model {model} not supported, please add the model at model_price.json")
            exit(1)

        if 'gpt' in args.generation_model:
            print("*"*50)
            print(f"Going to use: {args.generation_model}")
            generation_func = gpt_generation
            if args.with_images == "True":
                print("!!! Images are included in the prompt !!!")
                generation_func = gpt_generation_with_image
            elif args.multi_turn == "True":
                print("!!! Using the Multi-Turn Mode !!!")
                generation_func = gpt_generation_multi_turn
        elif 'claude' in args.generation_model:
            print("*"*50)
            print(f"Going to use: {args.generation_model}")
            generation_func = generate_claude_response
            if args.with_images == "True":
                print("!!! Images are included in the prompt !!!")
                generation_func = generate_claude_response_with_image

            elif args.multi_turn == "True":
                print("!!! Using the Multi-Turn Mode !!!")
                generation_func = generate_claude_response_multi_turn

        elif 'gemini' in args.generation_model:
            print("*"*50)
            print(f"Going to use: {args.generation_model}")
            generation_func = generate_gemini_response
            if args.with_images == "True":
                print("!!! Images are included in the prompt !!!")
                generation_func = generate_gemini_response_with_image
            
            elif args.multi_turn == "True":
                print("!!! Using the Multi-Turn Mode !!!")
                generation_func = generate_gemini_response_multi_turn
        else:
            print("You'd better be using TogetherAI")
            print("*"*50)
            print(f"Going to use: {args.generation_model}")

            generation_func = togetherai_generation

            if args.with_images == "True":
                print("!!! Images are included in the prompt !!!")
                raise ValueError("TogetherAI is not support with_image")
            elif args.multi_turn == "True":
                print("!!! Using the Multi-Turn Mode !!!")
                generation_func = togetherai_generation_multi_turn


        
        prompt_dict = load_prompt_dict(args.prompt_path)

        start_ind = args.start_ind
        end_ind = args.end_ind
        if end_ind == -1:
            end_ind = len(prompt_dict)


        
        key2id = call_prompts_in_parallel(prompt_dict,args.output_path,  start_ind=start_ind, end_ind=end_ind,
                                max_workers=args.max_workers,func=generation_func,
                                timeout_seconds=args.timeout_seconds,model=args.generation_model,
                                temperature=args.temperature,max_tokens=args.max_tokens)
        
        input_price = model_price[model]["input"]
        output_price = model_price[model]["output"]
        
        input_cost = cumulative_input_tokens / 1000000 * input_price
        output_cost = cumulative_output_tokens / 1000000 * output_price

        print()
        print("="*50)
        print()


        print(f"Generated {cumulative_count} outputs.")
        print(f"Input Cost: ${input_cost}")
        print(f"Output Cost: ${output_cost}")
        print(f"Total Cost: ${input_cost + output_cost}")

        print()
        print("="*50)
        print()
        
        not_generated_key = collect_result(prompt_dict, output_path=args.output_path, key2id=key2id)

        print(f"{len(not_generated_key)} prompts are not generated")
        remaining_prompts = {}
        for key in not_generated_key:
            remaining_prompts[key] = prompt_dict[key]

        with open(f"./{args.output_path}/remaining_prompts.pickle","wb") as f:
            pickle.dump(remaining_prompts,f)

        print(f"Remaining prompts are saved at ./{args.output_path}/remaining_prompts.pickle")


