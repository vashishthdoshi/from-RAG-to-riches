# RAG System Implementation - Assignment 2

A complete Retrieval-Augmented Generation (RAG) system built with open-source models, featuring a naive baseline and enhanced implementations with comprehensive evaluation.

---

## 📝 Project Overview

This project implements and evaluates a RAG pipeline for question-answering tasks using the `rag-datasets/rag-mini-wikipedia` dataset. The system progresses from a naive baseline, through systematic parameter exploration, to an enhanced version incorporating advanced features like query re-writing and re-ranking. The entire pipeline was developed and tested on a local machine to ensure reproducibility and avoid cloud-based limitations.

---

## 🏗️ System Architecture

The system is built with a modular pipeline structure, allowing for easy testing and scaling of individual components.

### Naive RAG Pipeline:
* **Embedding**: `sentence-transformers/all-mpnet-base-v2` (768 dimensions)
* **Vector Database**: **FAISS** with cosine similarity
* **Generation**: Google's **`flan-t5-base`**
* **Retrieval**: Configurable `top-k` semantic search

### Enhanced RAG Pipeline:
* **Re-ranking Layer**: An additional re-ranking step using a cross-encoder (`ms-marco-MiniLM-L-6-v2`).
* **Process**: Retrieves top-10 documents, then re-ranks to select the optimal top-3 for the generator.
* **Query Rewriting**: An optional capability to refine user prompts for expanded coverage.

---

## 📂 Repository Structure

```

from-RAG-to-riches/
├── src/
│   ├── naive_rag.py          \# Core RAG implementation
│   ├── enhanced_rag.py       \# Enhanced RAG with re-ranking
│   ├── utils.py              \# Helper functions (F1, EM, prompting)
│   ├── config.py             \# Configuration parameters
│   ├── preparing_ragas_data.py \# Generate RAGAs evaluation data
├── notebooks/
│   ├── 01_dataset_exploration.ipynb
│   ├── 03_evaluation.ipynb   \# Step 3: Prompting strategies
│   ├── 04_parameter_experimentation.ipynb \# Step 4: Embeddings & top-k
│   ├── 05_enhanced_evaluation.ipynb \# Step 5: Enhancement testing
│   └── 06_ragas_evals.ipynb \# Step 6: RAGAs metrics
├── results/                  \# Evaluation outputs (CSV, JSON)
├── data/
│   ├── 03_prompting_comparison.csv \# Prompting Strategy Evaluations
│   ├── 04_parater_experiments.csv   \# Experimentation Evaluations
│   ├── 05_enhancement_eval.csv \# Enhancement Evaluations
│   ├── 06_ragas_comparison.csv \# RAGAs evaluation of naive and enhanced implementation
│   ├── ragas_enhanced_data.json \# Dataset for RAGAs Eval (Enhanced)
│   ├── ragas_naive_data.json \# Dataset for RAGAs Eval (Naive)
│   ├── step3_detailed_reults.json
│   └── processed/           \# Generated embeddings, stats
├── requirements.txt
└── README.md

````

---

## ⚙️ Requirements & Installation

### Hardware:
* **16GB RAM** minimum
* ~5GB storage (for models, data, and code)

### Software:
* Python 3.9+
* Anaconda (recommended)

### Installation Steps

**1. Create Main Environment**
```bash
# Clone repository
git clone [https://github.com/vashishthdoshi/from-RAG-to-riches]
cd from-RAG-to-riches

# Create conda environment
conda create -n rag-exercise python=3.9 -y
conda activate rag-exercise

# Install dependencies
pip install -r requirements.txt
````

**2. Create RAGAs Environment (Optional, for Step 6)**

```bash
conda create -n ragas-eval python=3.10 -y
conda activate ragas-eval
pip install ragas openai langchain langchain-openai datasets pandas
```

-----

## 🚀 Usage

### Quick Start

```python
from src.naive_rag import NaiveRAG

# Initialize system
rag = NaiveRAG(embedding_model_name="all-mpnet-base-v2")

# Load data
documents, qa_pairs = rag.load_dataset()

# Create embeddings
rag.create_embeddings(batch_size=32)

# Build vector database
rag.build_vector_db()

# Load generator
rag.load_generator()

# Query the system
result = rag.query("What is the capital of France?", top_k=1)
print(result['answer'])
```

### Running Evaluations

  * **Step 3: Prompting Strategies**
    `jupyter notebook notebooks/03_evaluation.ipynb`
  * **Step 4: Parameter Experiments**
    `jupyter notebook notebooks/04_parameter_experimentation.ipynb`
  * **Step 5: Enhanced RAG**
    `jupyter notebook notebooks/05_enhanced_evaluation.ipynb`
  * **Step 6: RAGAs Evaluation**
    ```bash
    # Prepare data in main environment
    python src/preparing_ragas_data.py

    # Switch to RAGAs environment
    conda activate ragas-eval

    # Set your OpenAI API key
    export OPENAI_API_KEY='your-key-here'

    # Run evaluation notebook
    jupyter notebook notebooks/06_ragas_evals.ipynb
    ```

-----

## 📊 Experiments and Results

### Key Results Summary

| Configuration           | F1 Score | Exact Match % |
| :---------------------- | :------: | :-----------: |
| Naive (best, `top-k=1`) | 0.632    | 56%           |
| Enhanced (re-ranking)   | 0.615    | 56%           |

  * **Best Prompting Strategy**: Basic (F1=0.444, EM=39%)

#### RAGAs Metrics (Naive vs. Enhanced)

The enhanced RAG shows significant improvements across all RAGAs metrics, especially in retriever performance (`Context Precision`).

| Metric            | Naive | Enhanced | Improvement |
| :---------------- | :---: | :------: | :---------: |
| **Faithfulness** | 0.693 | **0.775** | +0.081      |
| **Answer Relevancy** | 0.727 | **0.719** | -0.008       |
| **Context Precision** | 0.729 | **0.909** | +0.179      |
| **Context Recall** | 0.617 | **0.666** | 0.049      |

### Detailed Experimental Findings

#### Experiment 1: Prompting Strategies

Four prompting strategies were tested. The **"basic" prompt** (the question as-is) yielded the best balance of performance and efficiency. Verbose strategies like "Chain of Thought" performed poorly on strict metrics like F1 and Exact Match.

| Strategy              | Avg F1 Score | Exact Match % | Time (min) |
| :-------------------- | :----------: | :-----------: | :--------: |
| **Basic** | **0.444** | **39.00%** | **1.8** |
| Chain of Thought (CoT)| 0.064        | 0.00%         | 8.3        |
| Persona               | 0.403        | 35.00%        | 1.5        |
| Instruction           | 0.267        | 20.00%        | 1.3        |

#### Experiment 2: Embedding Dimensions and Retrieval (`top_k`)

Experiments were run to find the optimal embedding model and `top_k` value. The `all-mpnet-base-v2` model with `top_k=1` achieved the highest F1 score. For enhancement testing, `top_k=3` was selected as the baseline to provide a richer context for the re-ranking step.

| Embedding Model           | Top-K | Avg F1 Score | Exact Match % |
| :------------------------ | :---: | :----------: | :-----------: |
| `all-MiniLM-L6-v2`        | 3     | 0.620        | 58.00%        |
| `all-mpnet-base-v2`       | 1     | **0.632** | **56.00%** |
| **`all-mpnet-base-v2`** | **3** | **0.556** | **50.00%** |

#### Experiment 3: Enhancing the RAG Pipeline

Re-ranking and Query Rewriting were added to the baseline (`top_k=3`). **Re-ranking provided the most significant performance boost**, improving the F1 score from 0.556 to 0.615.

| Configuration              | F1 Score | Exact Match % | Re-ranking | Query Rewriting |
| :------------------------- | :------: | :-----------: | :--------: | :-------------: |
| Baseline (no enhancements) | 0.556    | 50.00%        | FALSE      | FALSE           |
| **Re-ranking only** | **0.615**| **56.00%** | **TRUE** | **FALSE** |
| Query rewriting only       | 0.576    | 52.00%        | FALSE      | TRUE            |
| Both enhancements          | 0.615    | 56.00%        | TRUE       | TRUE            |

-----

## ⚠️ Known Issues & Solutions

  * **Storage Constraints**: `flan-t5-base` requires \~1GB. Use the `cache_dir` parameter to control its location. Clear the conda cache if needed: `conda clean --all -y`.
  * **Dataset Configuration**: When loading the dataset, use the `'text-corpus'` configuration with the `'passages'` split. The relevant column name is `'passage'`.
  * **RAGAs API Timeouts**: Expect a high timeout rate (40-60%) with the OpenAI API. Use `np.nanmean()` to aggregate partial results. Consider the free Gemini API as an alternative.

-----

## 📚 Dataset

  * **Source**: [`rag-datasets/rag-mini-wikipedia`](https://huggingface.co/datasets/rag-datasets/rag-mini-wikipedia)
  * **Structure**:
      * **Text Corpus**: 3,200 Wikipedia passages.
      * **Q\&A Pairs**: 918 test questions with ground truth answers.

-----

## 🔄 Reproducibility

All experiments use fixed test sets (the first 50-100 questions) for consistency. All key configuration parameters are centralized in `src/config.py`. Random seeds were not set, as retrieval is deterministic.

-----

## 📜 License

This project is licensed under the **MIT License**.

-----
  * This was an assignment based on course work from Carnegie Mellon University.
