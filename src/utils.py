"""
Utility functions for RAG system
"""

import json
import pickle
import os
from typing import List, Dict, Any

def save_results(results: Dict[Any, Any], filepath: str):
    """Save results to JSON file"""
    with open(filepath, 'w') as f:
        json.dump(results, f, indent=2)

def load_results(filepath: str) -> Dict[Any, Any]:
    """Load results from JSON file"""
    with open(filepath, 'r') as f:
        return json.load(f)

def save_embeddings(embeddings, filepath: str):
    """Save embeddings using pickle"""
    with open(filepath, 'wb') as f:
        pickle.dump(embeddings, f)

def load_embeddings(filepath: str):
    """Load embeddings from pickle file"""
    with open(filepath, 'rb') as f:
        return pickle.load(f)

def chunk_text(text: str, chunk_size: int = 512, overlap: int = 50) -> List[str]:
    """Split text into overlapping chunks"""
    words = text.split()
    chunks = []
    
    for i in range(0, len(words), chunk_size - overlap):
        chunk = ' '.join(words[i:i + chunk_size])
        chunks.append(chunk)
        
    return chunks

def format_prompt(question: str, context: str, strategy: str = "basic") -> str:
    """Format prompts for different strategies"""
    
    prompts = {
        "basic": f"Context: {context}\n\nQuestion: {question}\n\nAnswer:",
        
        "cot": f"Context: {context}\n\nQuestion: {question}\n\nLet's think through to the answer in a step by step manner.\n\nAnswer:",
        
        "persona": f"You are an expert researcher with great depth in your knowledge about this topic.\n\nContext: {context}\n\nQuestion: {question}\n\nAs an expert, provide a comprehensive answer:\n\nAnswer:",
        
        "instruction": f"Instructions: Based on the provided context, answer the question accurately and concisely. If the answer is not in the context, say so. Do not answer for the sake of answering, say no if the information is unavailable.\n\nContext: {context}\n\nQuestion: {question}\n\nAnswer:"
    }
    
    return prompts.get(strategy, prompts["basic"])

def calculate_f1_score(predicted: str, actual: str) -> float:
    """Calculate F1 score between predicted and actual answers"""
    # Simple word-based F1 calculation
    pred_words = set(predicted.lower().split())
    actual_words = set(actual.lower().split())
    
    if len(pred_words) == 0 and len(actual_words) == 0:
        return 1.0
    if len(pred_words) == 0 or len(actual_words) == 0:
        return 0.0
    
    intersection = pred_words.intersection(actual_words)
    precision = len(intersection) / len(pred_words)
    recall = len(intersection) / len(actual_words)
    
    if precision + recall == 0:
        return 0.0
    
    f1 = 2 * (precision * recall) / (precision + recall)
    return f1

def exact_match(predicted: str, actual: str) -> bool:
    """Calculate exact match between predicted and actual answers"""
    return predicted.lower().strip() == actual.lower().strip()

print("Utils module loaded successfully!")
