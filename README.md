# EQAS Reproducibility Package

This package contains the complete reproducibility code for the EQAS workflow:
dataset loading and validation, all standalone QA model notebooks, the structured
knowledge graph, GPT prompt experiments, LangChain/Chroma retrieval, confidence-based
routing, component ablations, hallucination evaluation, latency measurement, fallback
rate calculation, human evaluation aggregation, and same-test-set comparison.

The Home Appliances dataset is not duplicated inside this archive. All dataset loaders
use the public repository directly:

https://github.com/Gokcimen/Home_Appliance_Dataset

Expected final split:
- train: 8,000 QA pairs
- validation: 1,000 QA pairs
- test: 1,000 QA pairs
- total: 10,000 QA pairs
- unique primary product/entity titles: 1,111
- cross-split QA-ID overlap: 0
- cross-split exact product-title overlap: 0

## Software environments

The software environments are intentionally separated.

### Original ten transformer QA models
DeBERTa, ConvBERT, ELECTRA, CamemBERT, DistilBERT, BERT-base, RoBERTa,
BigBird, BART, and ALBERT use the original QA benchmark environment:

- Python 3.10.12
- PyTorch 2.0.1
- Transformers 4.33.1

### Contemporary instruction-tuned models
Gemma-4-E4B-it and Llama-3.3-70B-Instruct use the updated compatible environment:

- Transformers 5.16.1
- PyTorch 2.13.0
- PEFT 0.20.0
- TRL 1.12.0
- bitsandbytes 0.50.2

The software version is therefore not treated as an experimental variable shared across
models released in different years. Comparability is maintained through the same QA task,
the same product-disjoint data partition, and the same EM/token-F1 evaluation definitions.

## Knowledge graph

Structural graph counts expected by the validation code:
- 1,111 primary product/entity nodes
- 10,000 question nodes
- 10,000 answer nodes
- 21,111 total nodes
- 10,000 product/entity-to-question edges
- 10,000 question-to-answer edges
- 20,000 total edges

The graph is implemented as a Neo4j property graph. Controlled held-out evaluation must
not use the ground-truth test answer node as an answer lookup.

## Table-7 metrics

The package implements all four Table-7 metric families directly in code:

1. response-level hallucination rate;
2. mean end-to-end latency;
3. fallback rate;
4. overall human-evaluation score.

Atomic-claim-level hallucination rate is also implemented as a supplementary metric.

See:
- `src/eqas/table7_metrics.py`
- `src/eqas/hallucination.py`
- `Colab/06_Evaluation/Table7_Metrics.ipynb`

