# TURN YOUR STATE SPACE MODEL INTO A RETRIEVER

## Paper abstract
We present a novel method for long document understanding, leveraging the
Mamba architecture’s linear complexity processing capabilities. Our model, finetuned
from a Mamba language model checkpoint, processes queries in the full
document context, enabling more accurate retrieval. To address the scarcity
of long-context retrieval data, we explore synthetic data generation techniques,
finding link-based generation most effective. Our 130M model, paired with an
LLM generator, outperforms the best open-source embedding-based retriever,
which is more than 50 times larger. On documents with more than 256k tokens,
the 1.3B model demonstrates comparable performance to GPT-4o. These
results, evaluated on 41 QA benchmarks drawn from financial reports, government
documents, and creative works, demonstrate our model’s potential for improving
long document understanding in resource-constrained environments. Our
approach paves the way for more efficient processing of complex documents
across various fields.

## Datasets
Our dataset is freely and publicly available at [zenodo](https://doi.org/10.5281/zenodo.13892030).

Our train set is `mambaretriever_synthetic_data.pickle`, our test set by categories is `mamba_retriever_testset_by_categories.pickle`, and out test set by dataset is `mamba_retriever_41_testsets.pickle`

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
pip install .
```

Next, download and install the following two files from https://github.com/state-spaces/mamba/releases and https://github.com/Dao-AILab/causal-conv1d/releases:
```
mamba_ssm-2.2.2+cu118torch2.1cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
causal_conv1d-1.4.0+cu118torch2.1cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
```

You can install them using
```
pip install mamba_ssm-2.2.2+cu118torch2.1cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
pip install causal_conv1d-1.4.0+cu118torch2.1cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
```

## Synthetic Data Generation

Synthetic data generation involves three stages, each requiring prompt preparation and LLM generation. 

Firstly, use
```
python train_data_generation/prepare_connection_prompt_dict.py --data_path [data_path] --prompt_output_path [prompt_output_path]
```
* `data_path`: The path for the tokenized raw data.
* `prompt_output_path`: The path to save the connection prompt dictionary.

After LLM generation result is obtained, use
```
python train_data_generation/prepare_question_prompt.py --data_path [data_path] --prompt_output_path [prompt_output_path] --connection_collected_results_path [connection_collected_results_path] --updated_data_output_path [updated_data_output_path]
```
* `data_path`: The path for the tokenized raw data.
* `prompt_output_path`: The path to save the question formulation prompt dictionary.
* `connection_collected_results_path`: The path to the LLM generation result for connection generation.
* `updated_data_output_path`: The path to save updated raw data, where additional keys are added for further processing.

Afterwards, use
```
python train_data_generation/prepare_impsent_filter_prompt.py --updated_data_path [updated_data_path] --prompt_output_path [prompt_output_path] --question_collected_results_path [question_collected_results_path] --key2question_output_path [key2question_output_path]
```
* `updated_data_output_path`: The path to save updated raw data, where additional keys are added for further processing.
* `prompt_output_path`: The path to save important sentence filter prompt dictionary.
* `question_collected_results_path`: The path to the LLM generation result for question formulation.
* `key2question_output_path`: The path to save question results for each datapoint.

Finally, run the following to obtain train data
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


## Model Checkpoints
Our finetuned model checkpoints are uploaded to [Hugging Face](https://huggingface.co/MambaRetriever): `MambaRetriever/mambaretriever-130m`, `MambaRetriever/mambaretriever-1.3b`.

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
* `scenario`: Choose from:
  * `retrieval`: Used for RAG
  * `full_context`: Used to evaluate LLMs given full context
  * `random`: Random baseline
* `exp_name`: The experiment name
* `prediction_logits_path`: The datapath of the prediction logits.
  * Note: The previous step would save the prediction logits under path `rag_pipeline/prediction_logits/{EXP_NAME}`. You should enter the path here corresponding to your `EXP_NAME`.
  
## Training
Our model architecture is base on code from [mamba](https://github.com/state-spaces/mamba).

To run training, use
```
bash run_training.sh
```

You can change variables in the script for your own needs:

Set `MODEL` to be any pretrained checkpoints: `mamba2-130m`, `mamba2-370m`, `mamba2-780m`, `mamba2-1.3b`, `mamba2-2.7b`.

Set `EXP_NAME` to the experiment name.

Set `DATA_PATH` to the training data path.

The model checkpoint will be automatically saved under folder `output`.
