import math 
from transformers import AutoTokenizer
import numpy as np
import bisect
import tiktoken

def get_top_k_indices(logits, k):
    logits = logits
    index2logit = {ind:logit for ind, logit in enumerate(logits)}
    
    sorted_index2logit = sorted(index2logit.items(), key = lambda x: x[1], reverse = True)

    if k < 1:
        percentage = math.ceil(len(logits)*k)
        k = max(10, percentage)
    else:
        if isinstance(k, float):
            k = int(k)
            
    indices = [x[0] for x in sorted_index2logit[:k]]
    
    return sorted(indices)

def list_to_dict(lst):
    return {lst[i]['datapoint_id']: lst[i] for i in range(len(lst))}

def dict_key_to_tuple(d):
    return {eval(k):v for k, v in d.items()}

def detok(tokenizer, dictonary):
    out = {}
    for k in dictonary:
        out[k] = {tokenizer.decode(i):dictonary[k][i] for i in dictonary[k]}
    return out

def num_tokens_from_string(string, encoding_name='gpt-4-turbo'):
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.encoding_for_model("gpt-4-turbo")
    num_tokens = len(encoding.encode(string))
    return num_tokens

def qa_parsing(string):
    '''
    Parsing function for qa tasks
    '''
    if '**ANSWER**' in string:
        string = string.replace('**ANSWER**', 'ANSWER')
    if 'ANSWER' not in string:
        return string[:2000]

    assert 'ANSWER:' in string, f"ANSWER: not found in {string}"
    dec = string.split("ANSWER:")[-1].strip()
    if '\n' in dec:
        dec = dec.split('\n')[0].strip()

    return dec


def judge_ans_parse(string):
    '''
    Parsing function for judge answer task
    '''
    if '**' in string:
        string = string.replace('**', '')
    if '::' in string:
        string = string.replace('::', ':')
    if 'Decision: ' in string:
        string = string.replace('Decision: ', 'DECISION:')
    assert 'DECISION:' in string, f"DECISION: not found in {string}"

    
    dec = string.split('DECISION:')[-1].split('\n')[0].strip().split(" ")[0].strip().lower()

    if dec in ['yes', 'yes,', 'yes.']:
        return 'yes'
    elif dec in ['no', 'no,', 'no.']:
        return 'no'
    else:
        raise ValueError(f"Invalid decision: {dec} and {string}")



full_context_prompt = '''You are asked a question and you are given some context where the answer to the question can be found within.

Question:
{question}

Context:
{context}

Recall, the question is:
{question}

Think step by step and use the above context to answer the question. Based on your reasoning, output a {condition} answer after the keyword "ANSWER:".
'''

merge_ans_prompt = '''You are asked a question, the answer to which comes from a very long document. You have previously seen several slices of the document, and for each slice, you have generated a partial answer to the question along with some reasoning.
Review the list of partial answers and their reasoning. Based on them, generate a final answer to the question.

Question:
{question}

List of Partial Answers and Reasoning:
{lst_of_answers}

Think step by step, and use the list of partial answers and their reasoning to generate a {condition} final answer. Output the {condition} final answer after the keyword "ANSWER:".
'''

qa_prompt = '''You are asked a question and you are given a few important sentences as context where the answer to the question can be found within.

Question:
{question}

Context:
{context}

Connect the dots in the context, use some imagination and answer the question by thinking step by step. Output your reasoning after the keyword "REASON:"

Based on your reasoning, output a {condition} answer after the keyword "ANSWER:".
'''

llama_qa_prompt = '''You are asked a question and you are given a few sentences as context. Answer the question.

Question:
{question}

Context:
{context}

Output a very short one-sentence answer after the keyword "ANSWER:".
'''

def sliding_window(context, end_sentence_indices, window_size):
    window_size = window_size 
    window_size = max(window_size, 2)
    all_slices = []
    end_sentence_indices = [ele for ele in end_sentence_indices]
    start_sentence_indices = [0] + [ele+1 for ele in end_sentence_indices[:-1]]

    start_index = 0
    while start_index <= start_sentence_indices[-1]:
        end_index_approximation = start_index + window_size
        end_idx = bisect.bisect_left(end_sentence_indices, end_index_approximation) - 1
        if end_idx == -1:
            end_idx = 0
        end_index = end_sentence_indices[end_idx]
        
        if end_idx == len(end_sentence_indices)-1:
            start_index_approximation = end_index - window_size
            start_idx = bisect.bisect_left(start_sentence_indices, start_index_approximation)
            
            start_index = start_sentence_indices[start_idx]
            
            input_slice = context[start_idx:end_idx+1]
            all_slices.append(input_slice)
            # print("break")s
            break
        else:
            start_idx = start_sentence_indices.index(start_index)
            input_slice = context[start_idx:end_idx+1]
            start_index_approximation = start_index + window_size // 2
            start_idx = bisect.bisect_left(start_sentence_indices, start_index_approximation) - 1
            if start_idx == -1:
                start_idx = 0
            if start_sentence_indices[start_idx] == start_index:
                raise Exception("window size too small")
            start_index = start_sentence_indices[start_idx]
            all_slices.append(input_slice)
    return all_slices

def remove_consecutive_duplicates(input_list):
    include_list = []
    for ele in input_list:
        if ele in include_list:
            continue
        else:
            include_list.append(ele)
    
    return include_list