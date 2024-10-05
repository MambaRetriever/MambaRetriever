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
    parser.add_argument("--data_path", type=str)
    parser.add_argument("--prompt_output_path", type=str)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tokenizer", type=str, default="EleutherAI/gpt-neox-20b")
    parser.add_argument("--connection_collected_results_path", type=str)
    parser.add_argument("--updated_data_output_path", type=str)
    args = parser.parse_args()
    
    random.seed(args.seed)


    print('Loading dataset')

    with open(args.data_path, 'rb') as f:
        train_data = pickle.load(f)
        
    print('dataset loaded')
    
    
    print(f'Length of dataset loaded: {len(train_data)}')
    

    print('Obtain context')
    for k in tqdm(train_data):
        context = [item[0] for item in train_data[k]['tokenized_sents']]
        train_data[k]['context'] = context
        
    with open(args.connection_collected_results_path, 'rb') as f:
        collected_results = pickle.load(f)

    updated_synthetic_data = {}
    for k in tqdm(collected_results):
        datapoint_id, box_range, text_range = ast.literal_eval(k)

        output = collected_results[k]

        connect_index_pattern = r'\*\*Index ([\d]+)[\*\*]*'
        connect_sent_pattern = r'\*\*Index [\d]+[\*\*]*: ["]*(.*?)["]*\n'
        connect_sent_reason_pattern = r'[- ]*[\*\*]*Reason[\*\*]*:[\*\*]* (.*?)\n'
        connect_sent_reason_pattern_2 = r'- (.*?)\n'
        
        connect_indices = re.findall(connect_index_pattern, output)
        connect_sents = re.findall(connect_sent_pattern, output)
        connect_sent_reasons = re.findall(connect_sent_reason_pattern, output)

        if len(connect_sent_reasons) == 0:
            connect_sent_reasons = re.findall(connect_sent_reason_pattern_2, output)
            
        if (len(set(connect_indices)) != len(connect_indices)) and "### Reasons for Connection:" in output:
            index_response = output.split("### Reasons for Connection:")[0]
            reason_response = output.split("### Reasons for Connection:")[1]
            
            connect_indices = re.findall(connect_index_pattern, index_response)
            connect_sents = re.findall(connect_sent_pattern, index_response)
            connect_sent_reasons = re.findall(connect_sent_pattern, reason_response)

            if len(connect_sent_reasons) == 0:
                connect_sent_reasons = re.findall(connect_sent_reason_pattern_2, output)

        if len(connect_indices) == 0:
            continue

        if not ((len(connect_indices) == len(connect_sents)) and (len(connect_sents) == len(connect_sent_reasons))):
            continue

        box_indices = list(range(box_range[0], box_range[1]+1))
        text_chunk_indices = list(range(text_range[0], text_range[1]+1))
        assert len(box_indices) == (box_range[1] - box_range[0]) + 1, (len(box_indices), (box_range[1] - box_range[0]) + 1)
        assert len(text_chunk_indices) == (text_range[1] - text_range[0]) + 1, (len(text_chunk_indices), (text_range[1] - text_range[0]) + 1)

        shiftedidx2originalidx = {}
        counter = 0
        for _, b_id in enumerate(box_indices):
            if b_id in text_chunk_indices:
                continue
            shiftedidx2originalidx[counter] = b_id

        connect_indices = [int(index) for index in connect_indices]

        outofrange_flag = False
        for idx in connect_indices:
            if idx not in shiftedidx2originalidx:
                outofrange_flag = True
                break

        if outofrange_flag:
            continue

        context = synthetic_data[datapoint_id]['context']
        if len(context) <= max(list(shiftedidx2originalidx.values())):
            continue
        if not((min(connect_indices) > -1) and (max(connect_indices) < 195)):
            continue

        connect_indices = [shiftedidx2originalidx[idx] for idx in connect_indices]
        index2connection = {}
        
        for connect_idx_id, connect_idx in enumerate(connect_indices):
            index2connection[connect_idx] = connect_sent_reasons[connect_idx_id]

        datapoint = synthetic_data[datapoint_id]
        datapoint['index2connection'] = index2connection
        datapoint['text_chunk_indices'] = text_range
        datapoint['box_indices'] = box_range

        updated_synthetic_data[datapoint_id] = datapoint
        
    question_prompt = """Two chunks of text  in a document are connected with the following connection. Use this connection to build a question. Step by step explain how you would take advantage of this connection, and build a short, concise, one-sentence, concrete, non-conceptual, non-ambiguous question. IMPORTANTLY, you must use exact words from the connection given to you, but you must never refer to the chunks, never mention words such as "connection", "alignment", "relationship" between chunks. The question must be self-contained, and cannot not refer to the chunks, and must standalone makes sense.

Output your reasoning, especially how you would take advantage of this connection, and your verification, especially why the question is non-conceptual, and concrete, and self-contained and standalone makes sense, after the keyword "REASON:"
Based on your step-by-step reasoning and verification, then output the question after the keyword "QUESTION:"

Connection:
{connection}
"""

    prompt_dict = {}
    for k in tqdm(updated_synthetic_data):
        datapoint = updated_synthetic_data[k]
        idx2range = {}
        text_range = datapoint['text_chunk_indices']
        index2connection = datapoint['index2connection']
        assert len(index2connection)>0
        for connect_idx in index2connection:
            idx2range[connect_idx] = max(abs(connect_idx-text_range[0]),abs(connect_idx - text_range[-1]))
            
        assert sorted(idx2range.items(), key = lambda x: x[1], reverse = True)[0][1] == max(idx2range.values())

        max_selected_range = sorted(idx2range.items(), key = lambda x: x[1], reverse = True)[0][1]
        
        for selected_idx, selected_range in idx2range.items():
            if selected_range < max_selected_range:
                continue
            if selected_range <10:
                continue

            myprompt = question_prompt.format(connection = index2connection[selected_idx])
            if "gutenberg" in myprompt.lower() or "copyright" in myprompt.lower():
                continue
            prompt_dict[str((k, selected_idx))] = myprompt
            
    print(f'Length of prompt_dict: {len(prompt_dict)}')
    print(f'Saving prompt_dict to {output_path}')
    with open(args.prompt_output_path, 'wb') as f:
        pickle.dump(prompt_dict, f)          
        
    print(f"Saving updated_synthetic_data to {args.updated_data_output_path}")
    with open(args.updated_data_output_path, 'wb') as f:
        pickle.dump(updated_synthetic_data, f)
