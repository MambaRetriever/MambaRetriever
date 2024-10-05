import argparse
import pickle
from tqdm import tqdm
import re
from transformers import AutoTokenizer
import multiprocessing
import concurrent.futures
from tqdm import tqdm
import random
import os
from transformers import AutoTokenizer, TrainingArguments
from string import punctuation
import re

tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")

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


def helper(train_data_chunk_keys):
    connect_prompt_dict = {}

    for k in tqdm(train_data_chunk_keys):
        d = train_data[k]
    
        init = random.choice(list(range(len(d['context'])-4)))
        text_chunk_indices = list(range(init, init+5))
        assert len(text_chunk_indices) ==5
        left_context_indices = list(range(max(0,init-10),min(len(d['context']),init+190)) )
        right_context_indices = list(range(max(0,init-185),min(len(d['context']),init+15)) )
        assert len(left_context_indices)<=200
        assert len(right_context_indices)<=200
        box_indices = None
        if len(left_context_indices)<len(right_context_indices):
            box_indices = right_context_indices
        elif len(left_context_indices)>len(right_context_indices):
            box_indices = left_context_indices
        else:
            if random.choice([True, False]):
                box_indices = left_context_indices
            else:
                box_indices = right_context_indices
        assert box_indices != None
    
        box = ''
        text = ''
        counter = 0
        for _, b_id in enumerate(box_indices):
            if b_id in text_chunk_indices:
                text += f"{remove_special_characters(d['context'][b_id])}"+" "
                continue
            
            box += f"Index {counter}: {remove_special_characters(d['context'][b_id])}"+"\n"
            
            counter+=1
    
        myprompt = connection_prompt.format(text = text, box = box)
        mykey = (k, (box_indices[0], box_indices[-1]), (text_chunk_indices[0], text_chunk_indices[-1]))
        connect_prompt_dict[str(mykey)] = myprompt

    return connect_prompt_dict


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str)
    parser.add_argument("--prompt_output_path", type=str)
    parser.add_argument("--num_cpus", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tokenizer", type=str, default="EleutherAI/gpt-neox-20b")
    args = parser.parse_args()
    
    random.seed(args.seed)


    connection_prompt = """Given a chunk of sentences
Chunk:
{text}
what other sentences in the document are highly connected to this chunk? Output each sentence index that is highly connected to the chunk, and explain the reason.
Document:
{box}
"""
    
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    print('Loading dataset')

    with open(args.data_path, 'rb') as f:
        train_data = pickle.load(f)
        
    print('dataset loaded')

    print(f'Length of dataset loaded: {len(train_data)}')
    

    print('Obtain context')
    for k in tqdm(train_data):
        context = [item[0] for item in train_data[k]['tokenized_sents']]
        train_data[k]['context'] = context
        

    num_cpus = args.num_cpus
    train_data_keys = list(train_data.keys())
    train_data_chunk_keys_list = [train_data_keys[i* len(train_data_keys)//num_cpus: (i+1)* len(train_data_keys)//num_cpus ] for i in range(num_cpus)]
    assert sum([len(c) for c in train_data_chunk_keys_list]) == len(train_data_keys)    
        
    final_data_dict = {}
    with tqdm(total = len(train_data_chunk_keys_list), desc = "Total Progress", unit = 'key') as total_pbar:
        with concurrent.futures.ProcessPoolExecutor(max_workers= num_cpus) as executor:
            futures = {executor.submit(helper, train_data_chunk_keys) for train_data_chunk_keys in train_data_chunk_keys_list}
            for future in concurrent.futures.as_completed(futures):
                final_data_dict.update(future.result())
                total_pbar.update(1)
    print(len(final_data_dict))
    print("start saving data")
    
    with open(args.prompt_output_path, 'wb') as f:
        pickle.dump(final_data_dict, f)