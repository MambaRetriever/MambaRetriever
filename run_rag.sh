#!/bin/bash

retriever=$1
eval_data_path=$2
threshold=$3
generator=$4
scenario=$5
exp_name=$6
prediction_logits_path=$7
# Assign arguments to variables
# retriever in ['bce_ssm', 'gritlm', 'openai_embedding', 'bm25', 'contriever', 'dragon']
# generator=["gpt-4o-2024-08-06","meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo","meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo"]
#scenario in ['retrieval', 'full_context','random']
save_path=$generator

if [[ $generator == *"Llama"* ]]; then
  save_path="${generator//\//_}"
else
  save_path=$generator
fi



echo "--------------------"
echo "Run settings..."
echo "retriever: ${retriever}"
echo "eval_data: ${eval_data_path}"
echo "threshold: ${threshold}"
echo "generator: ${generator}"
echo "scenario: ${scenario}"
echo "--------------------"
echo "Evaluating under..."

directory_path="${retriever}MambaRetriever${save_path}MambaRetriever${threshold}MambaRetriever${scenario}MambaRetriever${exp_name}"

echo $directory_path
echo "--------------------"
echo "Create answer prompt from retrieved sentences..."

python rag_pipeline/get_ans_prompt.py --retriever $retriever --eval_data_path $eval_data_path --generator $generator --scenario $scenario --trial $exp_name --prediction_logits_path $prediction_logits_path --threshold $threshold




echo "--------------------"
echo "Run answer generation..."
cd ..

python generation/generation.py --prompt_path generation/prompts/final_rag_${directory_path}_answer_generation_prompt.pickle --output_path final_evaluation/${directory_path}/final_rag_${directory_path}_answer_generation --max_workers 40 --timeout_seconds 240 --generation_model $generator --temperature 0 --max_tokens 2000

cd -

echo "--------------------"
echo "Create eval answer prompt..."

python rag_pipeline/judge_ans_prompt.py --retriever $retriever --eval_data_path $eval_data_path --generator $generator --scenario $scenario --trial $exp_name --threshold $threshold

echo "--------------------"
echo "Run eval answer generation..."
cd ..

python3 generation/generation.py --prompt_path generation/prompts/final_rag_${directory_path}_judge_answer_prompt.pickle --output_path final_evaluation/${directory_path}/final_rag_${directory_path}_judge_answer --max_workers 100 --timeout_seconds 240 --generation_model gpt-4o-2024-08-06 --temperature 0 --max_tokens 2000


echo "--------------------"
cd -
echo "evaluate answer"

python rag_pipeline/evaluation.py --retriever $retriever --eval_data_path $eval_data_path --generator $generator --scenario $scenario --trial $exp_name --threshold $threshold

