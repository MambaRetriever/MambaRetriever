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
across various fields. All code, datasets and model checkpoints are available at
https://github.com/MambaRetriever/MambaRetriever

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

## Model Checkpoints
Our finetuned models are uploaded to [Hugging Face](https://huggingface.co/MambaRetriever): `MambaRetriever/mambaretriever-130m`, `MambaRetriever/mambaretriever-1.3b`.

Our evaluation script will automatically load these checkpoints.

## Evaluation
To replicate our result and run evaluation of our model checkpoints, use
```
bash run_evaluation.sh
```
You can change variables in the script for your own needs:

Set `MODEL` to either `mamba2-130m` or `mamba2-1.3b` depending on which checkpoint to evaluate on.

Set `EXP_NAME` to the experiment name.

Running this evaluation script will save a prediction logit file under folder `prediction_logits`, with file name `EXP_NAME`.

## Training
To run training of mamba model, use
```
bash run_training.sh
```

You can change variables in the script for your own needs:

Set `MODEL` to be your chosen pretrained mamba2 checkpoints from one of `mamba2-130m`, `mamba2-370m`, `mamba2-780m`, `mamba2-1.3b`, `mamba2-2.7b`.

Set `EXP_NAME` to the experiment name.

Set `DATA_PATH` to the training data path.

The model checkpoint will be automatically saved under folder `output`.
