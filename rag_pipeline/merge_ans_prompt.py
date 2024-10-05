import argparse
import pickle
import os
import math
from tqdm import tqdm
import random
import numpy as np
from collections import Counter
from utils import get_top_k_indices, dict_key_to_tuple, detok, merge_ans_prompt, sliding_window

def merge_qa_prompt(retriever, eval_data_path, threshold, scenario, generator, trial, context_length):
    """
    retriever options: ['generative ssm', 'BCE ssm', 'GPT', 'gritlm', 'llama']
    generator options: ['gpt4o', 'llama', 'sonnet']
    scenario options: ['retrieval', 'full_context', 'highlight']
    WARNING: if scenario is full context, the the retriever must be gpt4o, llama, or generative ssm
    """

    if '/' in generator:
        generator = generator.replace('/', '_')
    assert scenario in ['full_context', 'highlight']
    print('loading eval data...')
    with open(eval_data_path, 'rb') as f:
        eval_data = pickle.load(f)
    print('eval data loaded')
    # assert len(eval_data) == 41

    directory_path = f'{retriever}MambaRetriever{generator}MambaRetriever{threshold}MambaRetriever{scenario}MambaRetriever{trial}'

    with open(f'./{directory_path}/final_rag_{directory_path}_answer_generation/collected_results.pickle', 'rb') as f:
        answers_res = pickle.load(f)

    if isinstance(list(answers_res.keys())[0], str):
        answers_res = dict_key_to_tuple(answers_res)

    prompt_dict = {}

    for benchmark_name in eval_data:
        
        assert isinstance(eval_data[benchmark_name], dict)

        for datapoint_key in tqdm(eval_data[benchmark_name]):

            if (datapoint_key, 'slice1') not in answers_res:
                continue

            all_slices_keys = [k for k in answers_res if k[0] == datapoint_key]

            all_res_for_datapoint = ''
            for slice_num in range(len(all_slices_keys)):
                assert (datapoint_key, f'slice{slice_num}') in answers_res
                all_res_for_datapoint += f'Slice{slice_num}: \n' + answers_res[(datapoint_key, f'slice{slice_num}')] + '\n\n'

            assert isinstance(eval_data[benchmark_name][datapoint_key]['answer_type'], str)
            if eval_data[benchmark_name][datapoint_key]['answer_type'] == 'sentence':
                condition = "short one-sentence"
            else:
                assert eval_data[benchmark_name][datapoint_key]['answer_type'] == 'paragraph'
                condition = "short one-paragaph"

            prompt_dict[str((datapoint_key, 'slice0'))] = merge_ans_prompt.format(question = eval_data[benchmark_name][datapoint_key]['question'],\
                 lst_of_answers= all_res_for_datapoint, condition = condition)

    assert os.path.exists(directory_path)
    
    with open(f'../../generation/prompts/final_rag_{directory_path}_merge_answer_prompt.pickle', 'wb') as f:
        pickle.dump(prompt_dict, f)
    
    with open(f'./{directory_path}/merge_answer_prompt.pickle', 'wb') as f:
        pickle.dump(prompt_dict, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process and tokenize text data.')
    parser.add_argument('--retriever', type=str, required=True, help='output path to generation prompt.')
    parser.add_argument('--eval_data_path', type=str, required=True, help='Path to the evaluation data pickle file.')
    parser.add_argument('--threshold', type=float, default = 10.0, required=False, help='Top amount of sentences to use')
    parser.add_argument('--generator', type=str, help='Model to use for evaluation')
    parser.add_argument('--scenario', type=str, help="Whether to test for ground truth or not")
    parser.add_argument('--trial', type=str, default=0, help="Trial number")
    parser.add_argument('--context_length', type=int, default=120000, required = False, help="Length of context for full context scenario")

    args = parser.parse_args()

    merge_qa_prompt(args.retriever, args.eval_data_path, args.threshold, args.scenario, args.generator, args.trial, args.context_length)