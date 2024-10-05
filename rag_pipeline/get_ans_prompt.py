import argparse
import pickle
import os
import math
from tqdm import tqdm
import random
import numpy as np
from collections import Counter
from nltk import sent_tokenize
from utils import get_top_k_indices, dict_key_to_tuple, detok, full_context_prompt, sliding_window, qa_prompt, llama_qa_prompt, remove_consecutive_duplicates

def create_qa_prompt(retriever, eval_data_path, threshold, scenario, generator, trial, context_length):
    """
    retriever options: ['generative ssm', 'BCE_SSM', 'GPT', 'gritlm', 'llama']
    generator options: ['gpt4o', 'llama', 'sonnet']
    scenario options: ['retrieval', 'full_context', 'generative']
    WARNING: if scenario is full context, the the retriever must be gpt4o, llama, or generative ssm
    """
    if '/' in generator:
        generator = generator.replace('/', '_')
        # print(generator)
    # print(generator)


    print('loading eval data........')
    with open(eval_data_path, 'rb') as f:
        eval_data = pickle.load(f)

    print('eval data loaded')

    assert len(eval_data) == 41

    key_list = []
    for benchmark in eval_data:
        key_list.extend(list(eval_data[benchmark].keys()))
    assert len(key_list) == len(set(key_list))

    # print('')
    prompt_dict = {}

    assert scenario != 'full_context' #SHOULD ONLE RUN ONCE

    if scenario == 'full_context':

        assert retriever in ['gpt-4o-2024-08-06', 'Meta-Llama-3.1-8B-Instruct-Turbo'] #TODO: add in other models

        for benchmark_name in eval_data:

            assert isinstance(eval_data[benchmark_name], dict)

            for datapoint_key in tqdm(eval_data[benchmark_name]):
                
                datapoint_length = len(eval_data[benchmark_name][datapoint_key]['input_ids'])

                assert isinstance(eval_data[benchmark_name][datapoint_key]['answer_type'], str)
                if eval_data[benchmark_name][datapoint_key]['answer_type'] == 'sentence':
                    condition = "short one-sentence"
                else:
                    assert eval_data[benchmark_name][datapoint_key]['answer_type'] == 'paragraph'
                    condition = "short one-paragaph"

                if datapoint_length > context_length: 
                    chunks = sliding_window(eval_data[benchmark_name][datapoint_key]['full_text_sentences'], eval_data[benchmark_name][datapoint_key]['sentence_indices'], context_length)
                    for i, chunk in enumerate(chunks):
                        prompt_dict[str((datapoint_key, f'slice{i}'))] = full_context_prompt.format(question = eval_data[benchmark_name][datapoint_key]['question'],\
                             context = ''.join(chunk), condition = condition)
                else:
                    prompt_dict[str((datapoint_key, 'slice0'))] = full_context_prompt.format(question = eval_data[benchmark_name][datapoint_key]['question'],\
                         context = ''.join(eval_data[benchmark_name][datapoint_key]['full_text_sentences']), condition = condition)

    elif scenario == 'retrieval':
        with open(f'./prediction_logits/{args.prediction_logits_path}', 'rb') as f:
            prediction_logits = pickle.load(f)

        retrieved_sents = {}

        assert retriever in ['bce_ssm', 'gritlm', 'openai_embedding', 'bm25', 'contriever', 'dragon']

        for benchmark_name in eval_data:

            assert benchmark_name in prediction_logits, f"{benchmark_name} not in prediction logits."

            for datapoint_key in tqdm(eval_data[benchmark_name]):

                assert datapoint_key in prediction_logits[benchmark_name], datapoint_key

                logits = prediction_logits[benchmark_name][datapoint_key]

                imp_logits_indices = get_top_k_indices(logits, threshold)
             
                imp_sent_all = eval_data[benchmark_name][datapoint_key]['full_text_sentences'][imp_logits_indices[0]]

                prev_index = imp_logits_indices[0]
                for myindex in imp_logits_indices[1:]:
                    assert myindex >prev_index
                    if myindex == prev_index+1:
                        imp_sent_all += eval_data[benchmark_name][datapoint_key]['full_text_sentences'][myindex]
                    else:
                        imp_sent_all += "\n" + eval_data[benchmark_name][datapoint_key]['full_text_sentences'][myindex] # THIS IS WORKING OKAY
                    prev_index = myindex
                
                retrieved_sents[datapoint_key] = [eval_data[benchmark_name][datapoint_key]['full_text_sentences'][idx] for idx in imp_logits_indices]

                if eval_data[benchmark_name][datapoint_key]['answer_type'] == 'sentence':
                    condition = "short one-sentence"
                else:
                    assert eval_data[benchmark_name][datapoint_key]['answer_type'] == 'paragraph'
                    condition = "short one-paragaph"
                if generator in ['meta-llama_Meta-Llama-3.1-8B-Instruct-Turbo', "meta-llama_Meta-Llama-3.1-70B-Instruct-Turbo"]:
                    prompt_dict[datapoint_key] = llama_qa_prompt.format(question = eval_data[benchmark_name][datapoint_key]['question'], context = imp_sent_all, condition = condition)
                else:
                    prompt_dict[datapoint_key] = qa_prompt.format(question = eval_data[benchmark_name][datapoint_key]['question'], context = imp_sent_all, condition = condition)
    elif scenario == 'random':
        assert retriever in ['random']

        retrieved_sents = {}

        for benchmark_name in eval_data:

            for datapoint_key in tqdm(eval_data[benchmark_name]):

                random_logits_indices = random.sample(range(len(eval_data[benchmark_name][datapoint_key]['full_text_sentences'])), \
                min(int(threshold), len(eval_data[benchmark_name][datapoint_key]['full_text_sentences'])))
                random_logits_indices = sorted(random_logits_indices)

                imp_sent_all = eval_data[benchmark_name][datapoint_key]['full_text_sentences'][random_logits_indices[0]]

                prev_index = random_logits_indices[0]
                for myindex in random_logits_indices[1:]:
                    assert myindex >prev_index
                    if myindex == prev_index+1:
                        imp_sent_all += eval_data[benchmark_name][datapoint_key]['full_text_sentences'][myindex]
                    else:
                        imp_sent_all += "\n" + eval_data[benchmark_name][datapoint_key]['full_text_sentences'][myindex] # THIS IS WORKING OKAY
                    prev_index = myindex
                
                retrieved_sents[datapoint_key] = [eval_data[benchmark_name][datapoint_key]['full_text_sentences'][idx] for idx in random_logits_indices]

                if eval_data[benchmark_name][datapoint_key]['answer_type'] == 'sentence':
                    condition = "short one-sentence"
                else:
                    assert eval_data[benchmark_name][datapoint_key]['answer_type'] == 'paragraph'
                    condition = "short one-paragaph"

                prompt_dict[datapoint_key] = qa_prompt.format(question = eval_data[benchmark_name][datapoint_key]['question'], context = imp_sent_all, condition = condition)        
    elif scenario == 'generative':
        with open(f'./prediction_logits/{args.prediction_logits_path}', 'rb') as f:
            prediction_logits = pickle.load(f)
        for benchmark_name in eval_data:

            assert benchmark_name in prediction_logits, f"{benchmark_name} not in prediction logits."

            for datapoint_key in tqdm(eval_data[benchmark_name]):

                assert datapoint_key in prediction_logits[benchmark_name], datapoint_key



                imp_sent_all = sent_tokenize(prediction_logits[benchmark_name][datapoint_key])
                imp_sent_all = remove_consecutive_duplicates(imp_sent_all)
                imp_sent_all = '\n'.join(imp_sent_all)

                if eval_data[benchmark_name][datapoint_key]['answer_type'] == 'sentence':
                    condition = "short one-sentence"
                else:
                    assert eval_data[benchmark_name][datapoint_key]['answer_type'] == 'paragraph'
                    condition = "short one-paragaph"
                if generator in ['meta-llama_Meta-Llama-3.1-8B-Instruct-Turbo', "meta-llama_Meta-Llama-3.1-70B-Instruct-Turbo"]:
                    prompt_dict[datapoint_key] = llama_qa_prompt.format(question = eval_data[benchmark_name][datapoint_key]['question'], context = imp_sent_all, condition = condition)
                else:
                    prompt_dict[datapoint_key] = qa_prompt.format(question = eval_data[benchmark_name][datapoint_key]['question'], context = imp_sent_all, condition = condition)
    else:
        raise ValueError(f"Scenario {scenario} not recognized.")
    
    directory_path = f'{retriever}MambaRetriever{generator}MambaRetriever{threshold}MambaRetriever{scenario}MambaRetriever{trial}'

    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        print(f"Directory '{directory_path}' created.")
    else:
        raise ValueError(f"Directory '{directory_path}' already exists.")
    
    if scenario in ['retrieval' ,'random']:
        with open(f'./{directory_path}/retrieved_sents.pickle', 'wb') as f:
            pickle.dump(retrieved_sents, f)
    
    with open(f'../../generation/prompts/final_rag_{directory_path}_answer_generation_prompt.pickle', 'wb') as f:
        pickle.dump(prompt_dict, f)
    
    with open(f'./{directory_path}/answer_generation_prompt.pickle', 'wb') as f:
        pickle.dump(prompt_dict, f)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process and tokenize text data.')
    parser.add_argument('--retriever', type=str, required=True, help='output path to generation prompt.')
    parser.add_argument('--eval_data_path', type=str, required=True, help='Path to the evaluation data pickle file.')
    parser.add_argument('--threshold', type=float, default = 10, required=False, help='Top amount of sentences to use')
    parser.add_argument('--generator', type=str, help='Model to use for evaluation')
    parser.add_argument('--scenario', type=str, help="Whether to test for ground truth or not")
    parser.add_argument('--trial', type=str, default=0, help="Trial number")
    parser.add_argument('--context_length', type=int, default=120000, required = False, help="Length of context for full context scenario")
    parser.add_argument('--prediction_logits_path', type=str, required=False, help='Path to the prediction logits pickle file.')

    args = parser.parse_args()
    random.seed(7906)
    create_qa_prompt(args.retriever, args.eval_data_path, args.threshold, args.scenario, args.generator, args.trial, args.context_length)