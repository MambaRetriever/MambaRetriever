
<div align="center">
<h1>Single-Pass Document Scanning for Question Answering (COLM 2025)</h1>

<p class="authors" style="font-family: serif; font-size: 16px; line-height: 1.4;">
  <b>
    Weili Cao<sup>*</sup>, Jianyou Wang<sup>*</sup>, Youze Zheng<sup>*</sup>, Longtian Bao<sup>*</sup>, Qirui Zheng,<br>
    Taylor Berg-kirkpatrick, Ramamohan Paturi<sup>+</sup>, Leon Bergen<sup>+</sup>
  </b>
</p>

Laboratory for Emerging Intelligence (LEI)

CSE Department, University of California, San Diego

La Jolla, CA 92093

[![COLM 2025](https://img.shields.io/badge/COLM-2025-purple.svg)](https://colmweb.org/)&nbsp; [![arXiv](https://img.shields.io/badge/arXiv-2504.03101-<COLOR>.svg)](https://arxiv.org/abs/2504.03101)

</div>

## Table of Contents
* [Abstract](#abstract)
* [Dataset](#datasets)
* [Setup](#setup)
* [Model Checkpoints](#model-checkpoints)
* [Evaluation](#evaluation)
  * [Evaluation Data Structure](#evaluation-data-strucutre)
* [Training](#training)
* [Synthetic Data Generation](#synthetic-data-generation)
* [License](#license)

## Abstract
Handling extremely large documents for question answering is challenging: chunk-based embedding methods often lose track of important global context, while full-context transformers can be prohibitively expensive for hundreds of thousands of tokens. We propose a single-pass document scanning approach that processes the entire text in linear time, preserving global coherence while deciding which sentences are most relevant to the query. On 41 QA benchmarks, our single-pass scanner consistently outperforms chunk-based embedding methods and competes with large language models at a fraction of the computational cost. By conditioning on the entire preceding context without chunk breaks, the method preserves global coherence, which is especially important for long documents. Overall, single-pass document scanning offers a simple solution for question answering over massive text.

## Datasets
Our training and evaluation dataset is freely and publicly available at [zenodo](https://zenodo.org/records/13900121?token=eyJhbGciOiJIUzUxMiJ9.eyJpZCI6IjBjN2I2MGNlLTRkYzgtNDJmNS1iYTQ1LWVjNjUyMjFlMzhjMCIsImRhdGEiOnt9LCJyYW5kb20iOiI5MGNhMTViMDMyNTRjY2U2ZTBlNjVlNDJmODcxM2JlYyJ9.6Nwi0FdA35kHBYiAndany3O47vDLGBbvj7M3SmASbmE_rKtJgzPyk01glzBgb-8xxwEdX5usgn4HQB6F4AIPzQ) and [huggingface](https://huggingface.co/datasets/MambaRetriever/MambaRetriever).

Our train set is `mambaretriever_train.jsonl`, our test set by categories is `mambaretriever_test_per_category.json`, and out test set by dataset is `mambaretriever_test.json`

## Setup
We highly recommend creating a new conda environment first:
```
conda create -n mamba_retriever python=3.10.14
conda activate mamba_retriever
```

Then, run the following in your terminal:
```
git clone https://github.com/state-spaces/mamba.git
conda install cudatoolkit==11.8 -c nvidia
pip install -r requirements.txt
pip3 install torch==2.1.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install accelerate -U
cd mamba
pip install --no-build-isolation git+https://github.com/state-spaces/mamba.git
```

Next, download and install the following two files from https://github.com/state-spaces/mamba/releases and https://github.com/Dao-AILab/causal-conv1d/releases:
```
mamba_ssm-2.2.2+cu118torch2.1cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
causal_conv1d-1.4.0+cu118torch2.1cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
```
```
wget https://github.com/state-spaces/mamba/releases/download/v2.2.2/mamba_ssm-2.2.2+cu118torch2.1cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
wget https://github.com/Dao-AILab/causal-conv1d/releases/download/v1.4.0/causal_conv1d-1.4.0+cu118torch2.1cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
```

You can install them using
```
pip install mamba_ssm-2.2.2+cu118torch2.1cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
pip install causal_conv1d-1.4.0+cu118torch2.1cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
```

## Model Checkpoints
Our model checkpoints are available at [Hugging Face](https://huggingface.co/MambaRetriever): `MambaRetriever/SPScanner-130m`, `MambaRetriever/SPScanner-1.3b`.

Our evaluation script will automatically load these checkpoints.

## Evaluation
To replicate our result and run evaluation of our model checkpoints, use
```
bash run_evaluation.sh
```
You can change variables in the script for your own needs:

Set `MODEL` to either `mamba2-130m` or `mamba2-1.3b` depending on which checkpoint to evaluate on.

Set `EXP_NAME` to the experiment name.

Running this evaluation script will save a prediction logit file under folder `rag_pipeline/prediction_logits`, with file name `EXP_NAME`.

Next, to evaluate the model performance on our benchmarks, use

```
bash run_rag.sh [retriever] [eval_data_path] [threshold] [generator] [scenario] [exp_name] [prediction_logits_path]
```

We explain the arguments as follows:

* `retriever`: The retriever model to evaluate on. Choose from:
  * `bce_ssm`: Evaluate results for mamba
  * `gritlm`,`openai_embedding`, `bm25`, `contriever`, `dragon`: Evaluate results for corresponding models
* `eval_data_path`: The path to the evaluation dataset
* `threshold`: The number of sentences or chunks to retrieve
* `generator`: The model used to generate the answer, e.g. `gpt-4o-2024-08-06`,`meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo`.
* `scenario`: Choose from:
  * `retrieval`: Used for RAG
  * `full_context`: Used to evaluate LLMs given full context
  * `random`: Random baseline
* `exp_name`: The experiment name
* `prediction_logits_path`: The datapath of the prediction logits.
  * Note: The previous step would save the prediction logits under path `rag_pipeline/prediction_logits/{EXP_NAME}`. You should enter the path here corresponding to your `EXP_NAME`.
 
### Evaluation Data Strucutre
```
{
    "benchmark_name": {
        "datapoint_id": {
            input_ids: list: tokenized full context,
            full_text_sentences: list: sentence level,
            question: string,
            answer: list: element containing answer,
            answer_type: either paragraph/sentence,
            sentence_indices: list: index end of sentence indices for input_ids
        },
        ...,
        "datapoint_id": {
            ...
        },
    }
}
```

## Training
Our model architecture is built upon [mamba](https://github.com/state-spaces/mamba).

To run training, use
```
bash run_training.sh
```

You can change variables in the script for your own needs:

Set `MODEL` to be any pretrained checkpoints: `mamba2-130m`, `mamba2-370m`, `mamba2-780m`, `mamba2-1.3b`, `mamba2-2.7b`.

Set `EXP_NAME` to the experiment name.

Set `DATA_PATH` to the training data path.

The model checkpoint will be automatically saved under folder `output`.

## Synthetic Data Generation

You should use the following script to tokenize raw data first.

```
python document_tokenization.py --input_path [input_path] --output_path [output_path] --chunk_length [chunk_length]
```

* `input_path`: The path for raw data, it should be a key to text dictionary.
* `output_path`: The path for output file
* `chunk_length`: The token length of the chunk after tokenization. Choose from 2000,5000,10000 to replicate the results from the paper.

You should then filter out non-English documents using the following command:

```
python filter_language.py --input_path [input_path] --output_path [output_path] --dataset [dataset] --length_limit [length_limit] --top_limit [top_limit]
```

* `input_path`: The path for tokenized data.
* `output_path`: The path for the output file.
* `dataset`: The type of the data e.g train_2k, train_5k, train_10k
* `length_limit`: The length limit to filter out chunks less than this length limit.
* `top_limit`: The length limit to filter out chunks greater than this length limit.

The data is ready to go through the synthetic data generation steps. 

Synthetic data generation involves three stages, each requiring prompt preparation and LLM generation. 

Firstly, use the following to generate a prompt dict for finding connections:
```
python train_data_generation/prepare_connection_prompt_dict.py --data_path [data_path] --prompt_output_path [prompt_output_path]
```
* `data_path`: The path for the tokenized raw data.
* `prompt_output_path`: The path to save the connection prompt dictionary.

After LLM generation result is obtained, use the following to generate the second prompt dict for generating question from connection:
```
python train_data_generation/prepare_question_prompt.py --data_path [data_path] --prompt_output_path [prompt_output_path] --connection_collected_results_path [connection_collected_results_path] --updated_data_output_path [updated_data_output_path]
```
* `data_path`: The path for the tokenized raw data.
* `prompt_output_path`: The path to save the question formulation prompt dictionary.
* `connection_collected_results_path`: The path to the LLM generation result for connection generation.
* `updated_data_output_path`: The path to save updated raw data, where additional keys are added for further processing.

Afterwards, use the following to generate the third prompt dict for finding important sentences from question:
```
python train_data_generation/prepare_impsent_filter_prompt.py --updated_data_path [updated_data_path] --prompt_output_path [prompt_output_path] --question_collected_results_path [question_collected_results_path] --key2question_output_path [key2question_output_path]
```
* `updated_data_output_path`: The path to save updated raw data, where additional keys are added for further processing.
* `prompt_output_path`: The path to save important sentence filter prompt dictionary.
* `question_collected_results_path`: The path to the LLM generation result for question formulation.
* `key2question_output_path`: The path to save question results for each datapoint.

Finally, run the following to collect and obtain training data
```
python train_data_generation/prepare_train_data.py --updated_data_output_path [updated_data_output_path] --impsent_collected_results_path [impsent_collected_results_path] --key2question_path [key2question_path] -- train_data_output_path [train_data_output_path]
```
* `updated_data_output_path`: The path to save updated raw data.
* `impsent_collected_results_path`: The path to the LLM generation result for important sentence filtering.
* `key2question_path`: The path to the saved question results for each datapoint.
* `train_data_output_path`: The path to save the training data.

Note, at each step, run the following for LLM generation:
```
python generation/generation.py --prompt_path [prompt_path] --output_path [output_path] --generation_model [generation_model] --max_tokens [max_tokens]
```
* `prompt_path`: The path to the prompt dictionary for generation.
* `output_path`: The path to save the LLM generation result.
* `generation_model`: The model used for generation.
* `max_tokens`: The maximum output token for the LLM generation.

## License

The code in this project is licensed under the MIT license.

The dataset is under a CC-BY-NC license.
