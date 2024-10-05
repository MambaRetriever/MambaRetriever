import os
import pickle
from tqdm import tqdm
import ast
import numpy as np
import random
import copy
from transformers import AutoTokenizer, TrainingArguments
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
from string import punctuation
import re
from math import *

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--updated_data_path", type=str)
    parser.add_argument("--num_cpus", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tokenizer", type=str, default="EleutherAI/gpt-neox-20b")
    parser.add_argument("--impsent_collected_results_path", type=str)
    parser.add_argument("--key2question_path", type=str)
    parser.add_argument("--train_data_output_path", type=str)
    args = parser.parse_args()
    
    random.seed(args.seed)
    
    print('Loading dataset')

    with open(args.updated_data_path, 'rb') as f:
        synthetic_data = pickle.load(f)
        
    print('dataset loaded')
    
    with open(args.impsent_collected_results_path, 'rb') as f:
        collected_results = pickle.load(f)
        
    with open(args.key2question_path, 'rb') as f:
        key2question = pickle.load(f)
        
    key2parts = {}
    for k in tqdm(synthetic_data):
        datapoint = synthetic_data[k]
        idx2range = {}
        text_range = datapoint['text_chunk_indices']
        index2connection = datapoint['index2connection']
        assert len(index2connection)>0
        for connect_idx in index2connection:
            
            idx2range[connect_idx] = max(text_range[0]-connect_idx,connect_idx - text_range[-1])

        max_selected_range = sorted(idx2range.items(), key = lambda x: x[1], reverse = True)[0][1]
            
        assert sorted(idx2range.items(), key = lambda x: x[1], reverse = True)[0][1] == max(idx2range.values())

        chunk1 = []
        chunk1_indices = []
        for i in range(max(text_range[0]-2,0), min(text_range[-1]+1+2,len(datapoint['context']))):
            chunk1.append(remove_special_characters(datapoint['context'][i]))
            chunk1_indices.append(i)

        for selected_idx, selected_range in idx2range.items():
            if selected_range < max_selected_range:
                continue
            if selected_range <6:
                continue

            if "gutenberg" in index2connection[selected_idx].lower() or "copyright" in index2connection[selected_idx].lower():
                continue
            
            chunk2 = []
            chunk2_indices = []
            for i in range(max(0,selected_idx-2-1), min(len(datapoint['context']),selected_idx+2+1+1)):
                assert i not in chunk1_indices
                chunk2.append(remove_special_characters(datapoint['context'][i]))
                chunk2_indices.append(i)
            if (k, selected_idx) not in key2question:
                continue

            if chunk2_indices[-1]>chunk1_indices[0]:
                combined_chunk_indices = chunk1_indices + chunk2_indices
                assert len(combined_chunk_indices) <20
                combined_chunk = chunk1 + chunk2
            elif chunk2_indices[-1] < chunk1_indices[0]:
                combined_chunk_indices =  chunk2_indices + chunk1_indices
                assert len(combined_chunk_indices) <20
                combined_chunk =  chunk2 + chunk1
            else:
                raise ValueError("Chunks are overlapping.")

            key2parts[(k, selected_idx, tuple(combined_chunk_indices))] = (chunk1_indices, chunk2_indices)
        
    updated_synthetic_data = []
    for k in tqdm(collected_results):
        datapoint_id, selected_idx, combined_chunk_indices = ast.literal_eval(k)

        output = collected_results[k]
        if "LIST:" not in output:
            continue
        if '**LIST:**' not in output:
            output = output.split('LIST:')[-1].strip()
        else:
            output = output.split('**LIST:**')[-1].strip()
        
        assert output.count('[') == 1 and output.count(']') == 1
        if '.' in output:
            continue
        try:
            output = ast.literal_eval(output)
        except:
            continue
        tmp_imp_sent_indices = [int(x) for x in output]
    

        if len(tmp_imp_sent_indices) == 0:
            continue
        
        if max(tmp_imp_sent_indices) > (len(combined_chunk_indices) - 1):
            continue
        assert min(tmp_imp_sent_indices) >= 0

        imp_sent_indices = [combined_chunk_indices[idx] for idx in tmp_imp_sent_indices]

        first_chunk, second_chunk = key2parts[ast.literal_eval(k)]
        first_chunk_flag, second_chunk_flag = False, False

        for imp_idx in imp_sent_indices:
            if imp_idx in first_chunk:
                first_chunk_flag= True
            elif imp_idx in second_chunk:
                second_chunk_flag = True
            else:
                raise ValueError("Important sentence not in any chunk.")

        assert first_chunk_flag != False or second_chunk_flag != False
        if not (first_chunk_flag and second_chunk_flag):
            continue

        question = key2question[(datapoint_id, selected_idx)].strip() + '\n'
        tokenized_question = tokenizer.encode(question)
        datapoint = copy.deepcopy(synthetic_data[datapoint_id])
        datapoint['input_ids'] = tokenized_question + datapoint['all_input_ids']
        datapoint['sentence_indices'] = [x[1]+len(tokenized_question) for x in datapoint['tokenized_sents']]

        assert len(datapoint['sentence_indices']) == len(datapoint['context'])
        labels = []
        for sent_id in range(len(datapoint['context'])):
            if sent_id in imp_sent_indices:
                labels.append(1)
            else:
                labels.append(0)

        assert len(labels) == len(datapoint['sentence_indices'])
        
        for imp_sent_idx in imp_sent_indices:
            assert labels[imp_sent_idx] == 1
        
        datapoint['sentence_labels'] = labels
        del datapoint['all_input_ids']
        del datapoint['tokenized_sents']

        text_range = datapoint['text_chunk_indices']
        datapoint['connection_range'] = max(text_range[0]-selected_idx,selected_idx - text_range[-1])
        datapoint['selected_idx'] = selected_idx
        datapoint['important_sentence_indices'] = imp_sent_indices
        
        updated_synthetic_data.append(datapoint)
        
    random.shuffle(updated_synthetic_data)
    
    data_without_duplication = []
    included_ids = set()
    for d in tqdm(updated_synthetic_data):
        max_range = max(key2range[d['datapoint_id']])
        assert d['connection_range'] <= max_range
        if d['connection_range']< max_range:
            continue
        if d['datapoint_id'] in included_ids:
            continue
        data_without_duplication.append(d)
        included_ids.add(d['datapoint_id'])

    print(f"Length of data_without_duplication: {len(data_without_duplication)}")
    print(f"Writing data_without_duplication to {args.train_data_output_path}")
    with open(args.train_data_output_path, 'wb') as f:
        pickle.dump(data_without_duplication, f)