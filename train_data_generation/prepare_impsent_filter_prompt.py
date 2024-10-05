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


def remove_special_characters(text):
    cleaned_text = ""
    for char in text:
        if char.isalnum() or char.isspace() or char in punctuation:
            cleaned_text += char
        else:
            cleaned_text += " "
    cleaned_text = re.sub(r'[^\S ]+', ' ', cleaned_text)
    cleaned_text = re.sub(r'[ ]+', ' ', cleaned_text)
    return cleaned_text.strip()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--updated_data_path", type=str)
    parser.add_argument("--prompt_output_path", type=str)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tokenizer", type=str, default="EleutherAI/gpt-neox-20b")
    parseradd_argument("--question_collected_results_path", type=str)
    parser.add_argument("--key2question_output_path", type=str)
    args = parser.parse_args()
    
    random.seed(args.seed)


    print('Loading dataset')

    with open(args.updated_data_path, 'rb') as f:
        synthetic_data = pickle.load(f)
        
    print('dataset loaded')
    
    print(f'Length of dataset loaded: {len(synthetic_data)}')
        
    with open(args.question_collected_results_path, 'rb') as f:
        collected_results = pickle.load(f)
        
    key2question = {}
    for k in tqdm(collected_results):
        datapoint_id, selected_index = ast.literal_eval(k)

        output = collected_results[k]
        if 'QUESTION:' not in output:
            continue
        if 'QUESTION:' not in '\n'.join(output.split('\n')[-2:]):
            continue
        question = output.split('QUESTION:')[-1].strip()
        key2question[(datapoint_id, selected_index)] = question

    find_sentence_prompt = """Given a question, and a list of indexed text elements. Select the indices of all relevant text element(s) that would be helpful to answer the question.  

Question:
{question}

list of text elements:
{list_of_text_elements}

Recall, your task is to select indices for all relevant text elements that can help you answer the question. Provide a step-by-step explanation after the keyword 'REASON:'
Based on your explanation, output a list of indices for text elements that are relevant and helpful to answer the question, in this format, [index1, index2, ...] ,after the keyword "LIST:" 
If no text element is helpful and relevant to answer the question, output an empty list [] after the keyword "LIST:"
"""


    max_range_list = []
    important_sent_prompt_dict = {}
    for k in tqdm(synthetic_data):
        datapoint = synthetic_data[k]
        idx2range = {}
        text_range = datapoint['text_chunk_indices']
        index2connection = datapoint['index2connection']
        assert len(index2connection)>0
        for connect_idx in index2connection:
            idx2range[connect_idx] = max(text_range[0]-connect_idx,connect_idx - text_range[-1])
            
        max_range_list.append(max(idx2range.values()))
        assert sorted(idx2range.items(), key = lambda x: x[1], reverse = True)[0][1] == max(idx2range.values())

        max_selected_range = sorted(idx2range.items(), key = lambda x: x[1], reverse = True)[0][1]

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
                raise ValueError("Chunks are overlapping")
                
            list_of_text_elements = ""
            for s_id, s in enumerate(combined_chunk):
                list_of_text_elements += f"Index {s_id}: {s}\n"
            gpt_question = key2question[(k, selected_idx)]
            myprompt = find_sentence_prompt.format(question = gpt_question, list_of_text_elements = list_of_text_elements)
            important_sent_prompt_dict[str((k, selected_idx, tuple(combined_chunk_indices)))] = myprompt
         
    print(f"Length of important_sent_prompt_dict: {len(important_sent_prompt_dict)}")
    print(f"Saving prompt_dict to {args.prompt_output_path}")   
    with open(args.prompt_output_path, 'wb') as f:
        pickle.dump(important_sent_prompt_dict, f)
        
    print(f"Saving key2question to {args.key2question_output_path}")
    with open(args.key2question_output_path, 'wb') as f:
        pickle.dump(key2question, f)