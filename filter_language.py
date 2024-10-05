import pickle
import argparse
from tqdm import tqdm
from langdetect import detect_langs
from bs4 import BeautifulSoup
import re
from concurrent.futures import ProcessPoolExecutor
import math
import json

def percentage_of_english_letters(input_string):
    total_chars = len(input_string)
    if total_chars == 0:
        return 0.0
    
    english_letters_count = sum(1 for char in input_string if char.isalpha() and char.isascii())
    
    percentage = (english_letters_count / total_chars) 
    return percentage

def clean_text(text):
    # Remove HTML tags
    soup = BeautifulSoup(text, "html.parser")
    cleaned_text = soup.get_text()

    # Remove code-like snippets (text between backticks, for example)
    cleaned_text = re.sub(r'`[^`]*`', '', cleaned_text)  # Remove inline code
    cleaned_text = re.sub(r'```[^```]*```', '', cleaned_text, flags=re.DOTALL)  # Remove block code

    return cleaned_text

def check_language(text, threshold=0.8):
    cleaned_text = clean_text(text)

    try:
        # detect_langs returns a list of detected languages with probabilities
        detected_languages = detect_langs(cleaned_text)
        
        # Find English probability
        english_prob = next((lang.prob for lang in detected_languages if lang.lang == 'en'), 0)
        
        # Return True if English probability is greater than or equal to the threshold
        return english_prob >= threshold
    except:
        return False  # If language detection fails

def process_chunk(chunk):
    result = {}
    for k, v in chunk.items():
        # print(v.keys())
        cont = [i[0] for i in v['tokenized_sents']]
        cont = ''.join(cont)
        if check_language(cont):
            result[k] = v
    return result

def split_dict_into_chunks(d, n):
    items = list(d.items())
    if n > len(items):
        n = len(items)
    chunk_size = int(math.ceil(len(items) / n))
    return [dict(items[i * chunk_size:(i + 1) * chunk_size]) for i in range(n)]

def filter_language_parallel(test, num_processes=200):
    chunks = split_dict_into_chunks(test, num_processes)
    with ProcessPoolExecutor(max_workers=len(chunks)) as executor:
        results = list(tqdm(executor.map(process_chunk, chunks), total=len(chunks)))
    # Merge the results
    to_keep = {}
    for partial_result in results:
        to_keep.update(partial_result)
    return to_keep

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', type=str, required=True, help='Path to dataset')
    parser.add_argument('--output_path', type=str, required=True, help='Path to save filtered dataset')
    parser.add_argument('--dataset', type=str, required=True, help='train_2k/train_5k/train_10k')
    parser.add_argument('--length_limit', type=int, default=100, help='Minimum length of the text')
    parser.add_argument('--top_limit', type=int, default=1000, help='Maximum length of the text')


    args = parser.parse_args()
    # Number of CPUs to use
    num_cpus = 200
    with open(args.input_path, 'r') as f:
        test = json.load(f)
    test = {f'{k}_{chunk}':val for k in test for chunk, val in test[k].items()}
    print('data loaded')

    print('original len: ', len(test))
    test_filter = filter_language_parallel(test, num_cpus)
    print('filtered len: ', len(test_filter))

    to_keep = {}
    very_long_count = 0

    for k in tqdm(test_filter):
        cont = [i[0] for i in test_filter[k]['tokenized_sents']]
        if test_filter[k]['tokenized_sents'][-1][1] > args.top_limit:
            very_long_count += 1
            continue
        # percentage_res= percentage_of_english_letters(''.join(cont))
        # if  percentage_res > 0.7 and test_filter[k]['tokenized_sents'][-1][1] >= args.length_limit:
        if test_filter[k]['tokenized_sents'][-1][1] >= args.length_limit:
            to_keep[k] = test_filter[k]
            to_keep[k]['datapoint_id'] = k
            to_keep[k]['dataset'] = args.dataset
            
    print('very long count: ', very_long_count)
    print('final len: ', len(to_keep))
    with open(args.output_path, 'wb') as f:
        pickle.dump(to_keep, f)
