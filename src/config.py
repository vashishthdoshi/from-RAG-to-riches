"""
Configuration settings for the RAG system
"""

#Model configurations
EMBEDDING_MODEL = "all-MiniLM-L6-v2", "all-mpnet-base-v2"
GENERATION_MODEL = "google/flan-t5-base" #Will use this for generation

#Dataset configuration
DATASET_NAME = "rag-datasets/rag-mini-wikipedia"
CORPUS_CONFIG = "text-corpus" #For documents to search
QA_CONFIG = "question-answer" # For evaluation AQ pairs
TEST_SIZE = 100
TEST_SIZE = 50 #for enhancement evaluation and experimentation #Number of questions to evaluate on

#Vector database configuration
FAISS_INDEX_TYPE = "IndexFlatIP" #Inner product (cosine similarity)
TOP_K_RETRIEVAL = 1 #Number of documents to retrieve

#Evaluation confirguration
PROMPTING_STRATEGIES = ["basic", "cot", "persona", "instruction"]
METRICS - ["f1_score", "exact_match"]

#File paths
DATA_DIR = "../data"
RESULTS_DIR = "../results"
EMBEDDINGS_PATH = "../data/processed/embeddings.pkl"
INDEX_PATH = "../data/processed/faiss_index"

#Logging confirguration
LOG_LEVEL = "INFO"

print("Yes, Configuration loaded successfully!")
