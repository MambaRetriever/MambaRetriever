import argparse
import pickle
from tqdm import tqdm
import os
import pdb
import numpy as np
import csv
from collections import Counter
from utils import list_to_dict, detok, judge_ans_parse, dict_key_to_tuple, qa_parsing
from metrics import qa_f1_score

# qa_header = ['benchmark_name', 'accuracy', 'number_of_correct', 'total_number_of_points']
qa_header = ['benchmark_name', 'accuracy', 'f1_score', 'number_of_correct', 'total_number_of_points']
mqa = ['longbook_choice_eng', 'meetingqa_4k', 'meetingqa_16k', 'paperqa_4k', 'paperqa_16k', 'tpo', 'quality', 'coursera', 'muld_CAC']


def get_accuracy(retriever, eval_data_path, threshold, scenario, generator, trial, context_length, result_name):

    if '/' in generator:
        generator = generator.replace('/', '_')

    directory_path = f'{retriever}MambaRetriever{generator}MambaRetriever{threshold}MambaRetriever{scenario}MambaRetriever{trial}'

    print('loading eval data........')
    with open(eval_data_path, 'rb') as f:
        eval_data = pickle.load(f)
    print('eval data loaded')

    # assert len(eval_data) == 41
    with open(f'./{directory_path}/final_rag_{directory_path}_answer_generation/collected_results.pickle', 'rb') as f:
        ans_res = pickle.load(f)

    if isinstance(list(ans_res.keys())[0], str):
        ans_res = dict_key_to_tuple(ans_res)

    if len(list(ans_res.keys())[0][0]) == 2:
        ans_res = {k[0]:v for k,v in ans_res.items()}

    with open(f'./{directory_path}/final_rag_{directory_path}_judge_answer/collected_results.pickle', 'rb') as f:
        judge_res = pickle.load(f)
    
    if isinstance(list(judge_res.keys())[0], str):
        judge_res = dict_key_to_tuple(judge_res)
    
    result = [qa_header]
    avg = []
    f1_avg = []

    mcq_avg=[]
    open_ended_avg = []
    skip = 0
    total_num = 0
    for benchmark_name in eval_data:
        yes_count = 0
        benchmark_res = [benchmark_name]
        benchmark_skipped = 0

        current_benchmark_f1 = []
        for datapoint_key in tqdm(eval_data[benchmark_name]):

            if (datapoint_key, 'ans0') not in judge_res:
                import pdb; pdb.set_trace()
                benchmark_skipped+=1
                continue
            if (datapoint_key, 'ans1') in judge_res:

                all_ans_keys = [k for k in judge_res if k[0] == datapoint_key]
                
                datapoint_res = []
                datapoint_f1 = []
                for i in range(len(all_ans_keys)):
                    ans = qa_parsing(ans_res[datapoint_key])
                    datapoint_f1.append(qa_f1_score(ans, eval_data[benchmark_name][datapoint_key]['answer'][i]))


                    dec = judge_ans_parse(judge_res[(datapoint_key, f'ans{i}')])
                    if dec == 'yes':
                        datapoint_res.append(True)
                    elif dec=='no':
                        datapoint_res.append(False)
                    else:
                        raise ValueError(f"Decision {dec} not recognized.")
                if any(datapoint_res):
                    yes_count+=1
                
                assert max(datapoint_f1) != np.nan
                
                current_benchmark_f1.append(max(datapoint_f1))

            else:
                assert (datapoint_key, 'ans1') not in judge_res
                try:
                    dec = judge_ans_parse(judge_res[(datapoint_key, 'ans0')])
                except:
                    print(datapoint_key)
                ans = qa_parsing(ans_res[datapoint_key])
                datapoint_f1 = qa_f1_score(ans, eval_data[benchmark_name][datapoint_key]['answer'][0])
                assert datapoint_f1 != np.nan
                current_benchmark_f1.append(datapoint_f1)
                if dec == 'yes':
                    yes_count+=1

        print(f'number of points in {benchmark_name} skipped: ', benchmark_skipped)


        avg.append(yes_count/(len(eval_data[benchmark_name])-benchmark_skipped))
        if benchmark_name in mqa: 
            mcq_avg.append(yes_count/(len(eval_data[benchmark_name])-benchmark_skipped))
            benchmark_res.extend([yes_count/(len(eval_data[benchmark_name])-benchmark_skipped), "f1 skipped for MQA", yes_count, len(eval_data[benchmark_name])-benchmark_skipped])

        else:
            open_ended_avg.append(yes_count/(len(eval_data[benchmark_name])-benchmark_skipped))
            f1_avg.append(np.mean(current_benchmark_f1))
            benchmark_res.extend([yes_count/(len(eval_data[benchmark_name])-benchmark_skipped), np.mean(current_benchmark_f1), yes_count, len(eval_data[benchmark_name])-benchmark_skipped])


        
        result.append(benchmark_res)
        skip+=benchmark_skipped
        total_num+=len(eval_data[benchmark_name])-benchmark_skipped

    result.append([directory_path, np.mean(avg), np.mean(f1_avg), sum([row[3] for row in result[1:]]), total_num])
    # print(result)
    with open(f'./{directory_path}/{result_name}.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        
        # Write each row from the data into the CSV file
        writer.writerows(result)

    print(f'total number of points skipped: {skip}')
    for row in result[1:]:
        print(f'For datapoint in {row[0]}')
        if scenario != 'full_context':
            print(f'Under the threshold of {threshold}')
        if row[0] == directory_path:
            if len(mcq_avg) > 0:
                print(f'Total Average MCQ: {np.mean(mcq_avg)}')
            if len(open_ended_avg) > 0:
                print(f'Total Average Open ended: {np.mean(open_ended_avg)}')   
        print(f'Total number of correct datapoints: {row[3]}/{row[4]}')
        print(f'Accuracy: {row[1]}')
        print(f'f1 score: {row[2]}')
        print('-'*50)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process and tokenize text data.')
    parser.add_argument('--retriever', type=str, required=True, help='output path to generation prompt.')
    parser.add_argument('--eval_data_path', type=str, required=True, help='Path to the evaluation data pickle file.')
    parser.add_argument('--threshold', type=float, default = 10, required=False, help='Top amount of sentences to use')
    parser.add_argument('--generator', type=str, help='Model to use for evaluation')
    parser.add_argument('--scenario', type=str, help="Whether to test for ground truth or not")
    parser.add_argument('--trial', type=str, default=0, help="Trial number")
    parser.add_argument('--context_length', type=int, default=120000, required = False, help="Length of context for full context scenario")
    parser.add_argument('--result_name', type=str, default='final_rag_results', required = False, help="Name of the result file")
    args = parser.parse_args()


    get_accuracy(args.retriever, args.eval_data_path, args.threshold, args.scenario, args.generator, args.trial, args.context_length, args.result_name)