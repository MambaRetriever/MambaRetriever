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
