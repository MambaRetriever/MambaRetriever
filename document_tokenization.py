import argparse
import pickle
from tqdm import tqdm
import copy
import math
import re
import torch.nn.functional as F
from transformers import AutoTokenizer
from nltk.tokenize import sent_tokenize
import concurrent.futures
from tqdm import tqdm
import json


def split_with_delimiters(s: str) -> list:
    """
    split a string with delimiters
    input:
        s (str) - the input string
    output:
        ret (list) - a list of substrings
    """
    # Use a regular expression to split the string while keeping the delimiters
    ret = re.split(r'(?<=,|;)', s)
    for i in range(len(ret)-1):
        assert ret[i][-1] in [',',";"]
    return ret

def split_with_newlines(s: str) -> list:
    """
    use a regular expression to split a string with newlines
    input:
        s (str) - the input string
    output:
        ret (list) - a list of substrings
    """
    # Use a regular expression to split the string while keeping the newline characters
    ret = re.split(r'(?<=\n)', s)
    for i in range(len(ret)-1):
        assert ret[i][-1] in ["\n"]
    return ret

def divide_string(input_string, max_words):
    """
    Divide a string into substrings with a maximum number of words.
    input:
        input_string (str) - the input string
        max_words (int) - the maximum number of words per substring
    output:
        substrings (list) - a list of substrings
    """
    # Split the input string into a list of words
    words = input_string.split()
    
    # Initialize a list to store the substrings
    substrings = []
    
    # Initialize a temporary list to collect words for the current substring
    current_substring = []
    
    # Iterate over the words and add them to the current substring
    for word in words:
        current_substring.append(word)
        
        # Check if the current substring exceeds the maximum word count
        if len(current_substring) >= max_words:
            # Join the current substring into a single string and add it to the list
            substrings.append(' '.join(current_substring))
            
            # Reset the current substring
            current_substring = []
    
    # Add any remaining words as the last substring
    if current_substring:
        substrings.append(' '.join(current_substring))
    
    return substrings

def split_document(document,  tol) -> list:
    """
    Split a document into chunks of text with a maximum number of words.
    input: 
        document (str) - the document text
        tol (int) - the maximum number of words per chunk
    output:
        return_list (list) - a list of text chunks
    """
    
    #tokenize the document into sentences
    step1_sents = sent_tokenize(document)
    step2_sents = []
    for s in step1_sents:
        tmp_list = split_with_delimiters(s)
        tmp_s3 = []
        for ss in tmp_list:
            if len(ss.split())>tol:
               tmp_s3 += split_with_newlines(ss)
            else:
                tmp_s3.append(ss)
        tmp_s4 = []
        for sss in tmp_s3:
            if len(sss.split())>tol:
                sss_list = divide_string(sss, tol)
                # print(sss_list , len(sss_list))
                tmp_s4 += sss_list
            else:
                tmp_s4.append(sss)
        step2_sents += tmp_s4
    return_list = []
    for ele in step2_sents:
        # sentences that are very long withoutu spaces cannot be used such as html tags
        if len(ele)<1000:
            return_list.append(ele)

    return return_list

def divide_into_sublists(list_of_lists, train_token_num ):
    """
    Divide a list of lists into sublists with a maximum number of tokens.
    input:
        list_of_lists (list) - a list of lists
        train_token_num (int) - the maximum number of tokens per sublist
    output:
        result (list) - a list of sublists
    """
    result = []
    current_sublist = []
    current_length = 0

    for lst, s in list_of_lists:
        lst_length = len(lst)
        try:
            assert lst_length < train_token_num, (f"length assertion: ",lst,s)
        except:
            return result

        # Check if adding this list would exceed the limit
        if current_length + lst_length < train_token_num:
            current_sublist.append([lst,s])
            current_length += lst_length
        else:
            # Append the current sublist to result and start a new one
            result.append(current_sublist)
            current_sublist = [[lst,s]]
            current_length = lst_length

    # Append the last sublist if not empty
    if current_sublist:
        result.append(current_sublist)

    return result

def preprocess_string(input_string: str) -> str:
    
    return input_string.replace(" ", "")


def trim_general(chunk):
    """
    a general trim function that trims the beginning of a chunk
    input:
        chunk (list) - a list of sentences
    output:
        chunk (list) - a list of sentences with the beginning trimmed to avoid bad tokens that can be combined with \n
    """
    
    for i in range(len(chunk)):
        text = chunk[i][1]
        new_text = trim_beginning(text)
        if new_text != "":
            chunk[i][1] = new_text
            return copy.deepcopy(chunk[i:])
        else:
            continue
    return None
            

def trim_beginning(text):
    """
    trim the beginning of a text to avoid bad tokens that can be combined with \n
    input:
        text (str) - the input text
    output:
        text (str) - the trimmed text
    """
    ind = 0
    while text[ind] in [' ', '\n', '\xa0', '\r', '\t', '\x0c', '\u2003']:
        ind +=1
        if ind >= len(text):
            return ""
    return text[ind:]

def process_one_chunk(chunk, tokenizer, tol=20, lookback=5):
    """
    tokenize a chunk and find end of sentence tokens
    may be off by a few tokens since the ending of this current sentence
    maybe tokenized with the beginning of the next sentence
    input:
        chunk (list) - a list of sentences
        tokenizer (PreTrainedTokenizer) - the tokenizer to use
        tol (int) - the maximum number of tokens per chunk
        lookback (int) - the maximum number of tokens to look back
    output:
        final_chunk (list) - a list of processed chunks
        all_sent_ids (list) - a list of tokenized sentence IDs
    """
    all_sent = ''
    # remove bad tokens
    chunk = trim_general(chunk)
    
    if chunk == None:
        return None
    
    for ele in chunk:
        all_sent += ele[1]
    all_sent_ids = tokenizer(all_sent)['input_ids']
    # print(all_sent_ids)
    tmp_chunk = []
    # combine shorter sentences together
    for ind, data in enumerate(chunk):
        input_ids, sent = data
        if ind ==0:
            tmp_chunk.append(copy.deepcopy(data))
        elif len(tmp_chunk[-1][0])< tol:
            tmp_chunk[-1][0] += copy.deepcopy(input_ids)
            tmp_chunk[-1][1] += copy.deepcopy(sent)
        else:
            tmp_chunk.append(copy.deepcopy(data))
    
    # assert no information is lost
    assert sum([len(i[0]) for i in chunk]) == sum([len(i[0]) for i in tmp_chunk]), (sum([len(i[0]) for i in chunk]), sum([len(i[0]) for i in tmp_chunk]))
    
    # append the last sentence to the second to last sentence if it is too short
    if len(tmp_chunk[-1][0])<tol:
        # pop the last sentence
        # tmp_chunk.pop()
        tmp_chunk[-2][0] += copy.deepcopy(tmp_chunk[-1][0])
        tmp_chunk[-2][1] += copy.deepcopy(tmp_chunk[-1][1])
        tmp_chunk = tmp_chunk[:-1]
    
    final_chunk = []
    
    start = 0
    end = 0
    end_to_sent_dict = {}
    
    for ind,data in enumerate(tmp_chunk):
        input_ids, sent = data
        length_sent = len(input_ids)
        
        assert length_sent>=tol, (length_sent,tol)
        # start is moved to the approximate position of the next sentence, lookback is to go back by how many tokens
        start = start + length_sent - max(math.ceil(length_sent/2),lookback)
        
        # ensure we don't start in the middle of a multi-byte character
        while tokenizer.decode(weird_token[0]) in tokenizer.decode(all_sent_ids[start]):
            start -=1
            
        assert all_sent_ids[start] not in weird_token
        end = start +1
        # ensure we are still in the current sentence
        assert preprocess_string(tokenizer.decode(all_sent_ids[start:end])) in preprocess_string(sent), f"{tokenizer.decode(all_sent_ids[start:end])} dividing {sent}"
        condition = 0
        
        # move the end forward until we are at the end of the sentence, give 3 tries for multi-byte characters
        while condition <3 and end <= len(all_sent_ids):
            if not preprocess_string(tokenizer.decode(all_sent_ids[start:end])) in preprocess_string(sent):
                condition +=1
            else:
                condition =0
            end = end + 1
    
    
        
        start = end - condition-1
        k = end-condition -2 # k is the end of the sentence
            
        end_to_sent_dict[k] = sent
        k_start = k-10
        # ensure we don't start in the middle of a multi-byte character
        while all_sent_ids[k_start] in weird_token:
            k_start-=1
            
        if len(end_to_sent_dict)>=2:
            previous_k = list(end_to_sent_dict.keys())[-2]
            text_so_far = end_to_sent_dict[previous_k]+end_to_sent_dict[k]
        else:
            text_so_far = end_to_sent_dict[k]
        
        assert preprocess_string(tokenizer.decode(all_sent_ids[k_start:k+1])) in preprocess_string(text_so_far), (tokenizer.decode(all_sent_ids[k_start:k+1]), text_so_far)

        if k == len(all_sent_ids)-1:
            # print('enter')
            assert preprocess_string(tokenizer.decode(all_sent_ids[k-10:k+1]))[-2:] == preprocess_string(end_to_sent_dict[k][-500:])[-2:], f"{tokenizer.decode(all_sent_ids[k-10:k+1])}  dividing  {end_to_sent_dict[k][-500:]}"

    
        final_chunk_piece = [copy.deepcopy(sent)]+[k]
        final_chunk.append(final_chunk_piece)
        
    assert len(final_chunk)==len(tmp_chunk)
    if len(final_chunk)==0:
        return None
    return (final_chunk, all_sent_ids[:k+1])


def process_list_of_documents(id, ret, tokenizer, tol = 20, lookback = 5):
    """
    process a list of documents
    input:
        id (str) - the document ID
        ret (list) - a list of chunks
        tokenizer (PreTrainedTokenizer) - the tokenizer to use
        tol (int) - the maximum number of tokens per chunk
        lookback (int) - the maximum number of tokens to look back
    output:
        final_chunk_list (list) - a list of processed chunks
        num_fail (int) - the number of failed chunks
    """
    final_chunk_list = []
    num_fail = 0
    pc = 0
    for chunk in ret:
        try:
            processed_output = process_one_chunk(chunk, tokenizer, tol, lookback)
            if processed_output == None:
                continue
            final_chunk_list.append(processed_output)
        except:
            num_fail +=1
            # print(id, pc)
        pc+=1
    return final_chunk_list, num_fail


def transform_to_dict(dataset):
    """
    Transform the dataset into a dictionary format.
    input:
        dataset (dict) - the dataset
    output:
        new_dict (dict) - the transformed dataset
    """
    new_dict = {}
    for doc_id in dataset:
        new_dict[doc_id] = {}
        for ind in range(len(dataset[doc_id])):        
            new_dict[doc_id][f'chunk_{ind}'] = {}
            new_dict[doc_id][f'chunk_{ind}']['tokenized_sents'] = dataset[doc_id][ind][0]
            new_dict[doc_id][f'chunk_{ind}']['all_input_ids'] = dataset[doc_id][ind][1]

    return new_dict


def process_document(id):
    """
    Process a single document from the dataset.
    input: id (str) - the document ID
    output: 
        id (str) - the document ID
        final_chunk_list (list) - the list of processed chunks
        num_fail (int) - the number of failed chunks
        len(ret) (int) - the total number of chunks
    """
    # split the document into sentences
    step2_sents = split_document(document=dataset[id], tol=50)
    # put together the input ids for sentences and the sentences themselves
    step2_ids = [[tokenizer(s)['input_ids'], s] for s in step2_sents]
    # combine the sentences into chunks
    ret = divide_into_sublists(step2_ids, train_token_num = args.chunk_length) # a super large number ensures whole document in ONE chunk
    final_chunk_list, num_fail = process_list_of_documents(id, ret, tokenizer, tol=20, lookback=5)
    
    return id, final_chunk_list, num_fail, len(ret)

def main():
    mydict = {}
    total_fail = 0
    total = 0

    # Determine the number of available CPUs
    num_cpus = 100

    # Create a tqdm progress bar with the total number of keys in the dataset
    with tqdm(total=len(dataset), desc='Total Progress', unit='key') as total_pbar:
        # Use ProcessPoolExecutor for parallel processing
        with concurrent.futures.ProcessPoolExecutor(max_workers=num_cpus) as executor:
            # Submit tasks to the executor
            futures = {executor.submit(process_document, id): id for id in dataset}

            # Process each completed future
            for future in concurrent.futures.as_completed(futures):
                id, final_chunk_list, num_fail, ret_len = future.result()

                # Update the results and statistics
                mydict[id] = final_chunk_list
                total_fail += num_fail
                total += ret_len

                # Update the progress bar
                total_pbar.update(1)

    print(f"Total fail {total_fail} out of {total}")
    newdict = transform_to_dict(mydict)
    return newdict


if __name__ == "__main__":
    with open('weird_token.json', 'r') as json_file:
        weird_token = json.load(json_file)
    parser = argparse.ArgumentParser(description='Process and tokenize text data.')
    parser.add_argument('--input_path', type=str, required=True, help='Path to the input json file.')
    parser.add_argument('--output_path', type=str, required=True, help='Path to save the tokenized json file.')
    parser.add_argument('--chunk_length', type=int, required=True, help='length of chunk')
    

    args = parser.parse_args()
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
    assert ".json" in args.input_path, "input file should be a json file"
    assert ".json" in args.output_path, "output file should be a json file"
    with open(args.input_path,'r') as f:
        dataset = json.load(f)
    print('dataset loaded')

    mydict = main()

    with open(args.output_path,'w') as f:
        pickle.dump(mydict,f)