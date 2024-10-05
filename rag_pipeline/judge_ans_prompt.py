import argparse
import pickle
import os
from tqdm import tqdm
import tiktoken
from utils import list_to_dict, dict_key_to_tuple, detok, num_tokens_from_string, qa_parsing

qa_judge_prompt ='''Given a question, a groundtruth answer, and an attempted answer, use the following criteria to determine if the attempted answer accurately reflects the groundtruth answer.
Criteria:
- The majority of the information in the attempted answer should overlap with the groundtruth answer. Note that the attempted answer may include additional information derived from the question.
- The attempted answer may extend the groundtruth answer while covering all its aspects, however, the attempted answer should not be contradicting the groundtruth answer.
- If the groundtruth contains numbers, the attempted answer must match when rounded to the same precision as the groundtruth.
Example 1:
Groundtruth Answer: 1983
Attempted Answer: 1.983 million
Reason: The groundtruth answer 1983 is a whole number without any units. The attempted answer uses 1.983, which is different from the whole number 1983 and thus should be considered incorrect.
Decision: NO
Example 2:
Groundtruth Answer: 93
Attempted Answer: 93 million
Reason: The attempted answer is 93 million, which uses the same digits as the attempted answer and thus should be considered correct.
Decision: YES
Question:
{question}
Groundtruth Answer:
{gt_answer}
Attempted Answer:
{answer}
Think step by step when you compare these two answers. Based on the reasoning, output a YES/NO decision after the keyword "DECISION:".
'''

mqa_judge_prompt ='''Given a multiple-choice question, a ground truth answer, and an attempted answer, the attempted answer should be the same option as the ground truth answer. It should not include any other options beyond those in the ground truth answers. Some parts of the attempted answer may overlap with information from the question.

Question:
{question}
Ground Truth Answer:
{gt_answer}
Attempted Answer:
{answer}
Think step by step when you compare these two answers. Based on the reasoning, output a YES/NO decision after the keyword "DECISION:".'''

mqa = ['longbook_choice_eng', 'meetingqa_4k', 'meetingqa_16k', 'paperqa_4k', 'paperqa_16k', 'tpo', 'quality', 'coursera', 'muld_CAC']


def judge_answer(retriever, eval_data_path, threshold, scenario, generator, trial, context_length):
    if '/' in generator:
        generator = generator.replace('/', '_')

    print('loading eval data........')
    with open(eval_data_path, 'rb') as f:
        eval_data = pickle.load(f)
    print('eval data loaded')

    # assert len(eval_data) == 41

    directory_path = f'{retriever}MambaRetriever{generator}MambaRetriever{threshold}MambaRetriever{scenario}MambaRetriever{trial}'

    with open(f'./{directory_path}/final_rag_{directory_path}_answer_generation/collected_results.pickle', 'rb') as f:
        answer_res = pickle.load(f)

    if isinstance(list(answer_res.keys())[0], str):
        answer_res = dict_key_to_tuple(answer_res)

    prompt_dict = {}
    count = 0 
    if scenario == 'full_context':
        with open(f'./{directory_path}/final_rag_{directory_path}_merge_answer/collected_results.pickle', 'rb') as f:
            merged_answer_res = pickle.load(f)

        if isinstance(list(merged_answer_res.keys())[0], str):
            merged_answer_res = dict_key_to_tuple(merged_answer_res)        
            
        for benchmark_name in eval_data:
            for datapoint_key in tqdm(eval_data[benchmark_name]):
                if (datapoint_key, 'slice0') in merged_answer_res:
                    ans = qa_parsing(merged_answer_res[(datapoint_key, 'slice0')])                   
                else:
                    assert (datapoint_key, 'slice1') not in answer_res, datapoint_key
                    ans = qa_parsing(answer_res[(datapoint_key, 'slice0')])
                if ans == 'no answer':
                    count+=1
                assert isinstance(eval_data[benchmark_name][datapoint_key]['answer'], list)
                for idx, gt_ans in enumerate(eval_data[benchmark_name][datapoint_key]['answer']):

                    if benchmark_name in mqa:
                        prompt_dict[(datapoint_key, f'ans{idx}')] = mqa_judge_prompt.format(question = eval_data[benchmark_name][datapoint_key]['question'], \
                            gt_answer = gt_ans, answer = ans)
                    else:

                        prompt_dict[(datapoint_key, f'ans{idx}')] = qa_judge_prompt.format(question = eval_data[benchmark_name][datapoint_key]['question'], \
                            gt_answer = gt_ans, answer = ans)

    elif scenario in ['retrieval', 'random', 'generative']:
        for benchmark_name in eval_data:

            for datapoint_key in tqdm(eval_data[benchmark_name]):

                ans = qa_parsing(answer_res[datapoint_key])
                    
                assert isinstance(eval_data[benchmark_name][datapoint_key]['answer'], list)

                for idx, gt_ans in enumerate(eval_data[benchmark_name][datapoint_key]['answer']):
                    if benchmark_name in mqa:
                        prompt_dict[(datapoint_key, f'ans{idx}')] = mqa_judge_prompt.format(question = eval_data[benchmark_name][datapoint_key]['question'], \
                            gt_answer = gt_ans, answer = ans)
                    else:

                        prompt_dict[(datapoint_key, f'ans{idx}')] = qa_judge_prompt.format(question = eval_data[benchmark_name][datapoint_key]['question'], \
                            gt_answer = gt_ans, answer = ans)
    else:
        raise ValueError(f"Scenario {scenario} not recognized.")

    assert os.path.exists(directory_path)

    print(f"Number of no answers: {count}")

    with open(f'../../generation/prompts/final_rag_{directory_path}_judge_answer_prompt.pickle', 'wb') as f:
        pickle.dump(prompt_dict, f)
    
    with open(f'./{directory_path}/judge_answer_prompt.pickle', 'wb') as f:
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

    args = parser.parse_args()

    judge_answer(args.retriever, args.eval_data_path, args.threshold, args.scenario, args.generator, args.trial, args.context_length)