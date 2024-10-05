import argparse
import pickle
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
from utils import num_tokens_from_string

def check_gpt_tok_len(batch):
    results = []
    for k, prompt_text in batch:
        # Ensure the token length of the prompt is within the limit

        # assert num_tokens_from_string(prompt_text) <= 128000, f"Prompt too long: {num_tokens_from_string(prompt_text)}"
        num_tok = num_tokens_from_string(prompt_text)

        results.append(num_tok)
    return results

def chunk_data(data, num_chunks):
    """Split data into chunks for batching."""
    chunk_size = len(data) // num_chunks + (1 if len(data) % num_chunks != 0 else 0)
    it = iter(data)
    for _ in range(0, len(data), chunk_size):
        yield {k: data[k] for k in list(it)[:chunk_size]}

def process_in_batches(retriever, generator, threshold, scenario, trial, context_length, num_workers):
    # Create the directory path based on the given arguments
    directory_path = f'{retriever}*{generator}*{threshold}*{scenario}*{trial}'

    # Load the pickle file containing the prompt data
    with open(f'{directory_path}/answer_generation_prompt.pickle', 'rb') as f:
        prompt = pickle.load(f)

    # Split the prompt dictionary into batches (chunks)
    batches = list(chunk_data(prompt, num_workers))

    res = []
    # Use ProcessPoolExecutor to distribute batches across multiple CPUs
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        # Submit batch processing tasks for each chunk
        futures = [executor.submit(check_gpt_tok_len, list(batch.items())) for batch in batches]

        # Monitor progress using tqdm and retrieve results
        for future in tqdm(futures):
            res.extend(future.result())
    
    print(sum(res)*1.25/1000000)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process and tokenize text data.')
    parser.add_argument('--retriever', type=str, required=True, help='Output path to generation prompt.')
    parser.add_argument('--eval_data_path', type=str, required=True, help='Path to the evaluation data pickle file.')
    parser.add_argument('--threshold', type=float, default=10, required=False, help='Top amount of sentences to use')
    parser.add_argument('--generator', type=str, help='Model to use for evaluation')
    parser.add_argument('--scenario', type=str, help="Whether to test for ground truth or not")
    parser.add_argument('--trial', type=int, default=0, help="Trial number")
    parser.add_argument('--context_length', type=int, default=120000, required=False, help="Length of context for full context scenario")
    parser.add_argument('--num_workers', type=int, default=8, help="Number of workers to use for parallel processing")

    args = parser.parse_args()

    # Process in batches using multiple workers (CPUs)
    process_in_batches(args.retriever, args.generator, args.threshold, args.scenario, args.trial, args.context_length, args.num_workers)