"""
Naive RAG Implementation
Assignment 2 - Applications of NL(X) and LLMs - F25

Vashishth Doshi

This module implements a basic Retrieval-Augmented Generation system using:
- Sentence-transformers for document embeddings
- FAISS for vector similarity search
- Transformers for answer generation
"""
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import pipeline
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pandas as pd
from datasets import load_dataset
import logging
from typing import List, Dict, Any, Tuple
import sys
import os

# Adding src directory to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
	sys.path.insert(0, current_dir)
from utils import format_prompt

#Setup logging
logging.basicConfig(
	level=logging.INFO,
	format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class NaiveRAG:
	"""
	Naive RAG system implementation.
	
	This class implements a basic retrieval-augmented generation pipeline:
	1. Embed documents using sentence-transformers
	2. Store embeddings in FAISS vector database
	3. Retrieve relevant documents for queries
	4. Generate answers using retrieved context
	
	Attributes:
		embedding_model (SentenceTransformer): Model for creating embeddings
		vector_db (faiss.Index): FAISS index for similarity search
		documents (List[str]): List of document texts
		qa_pairs (Dataset): Question-answer pairs for evaluation
		embeddings (np.ndarray): Document embeddings
		generator: Text generation model
		tokenizer: Tokenizer for generation model
	"""
	
	def __init__(self, embedding_model_name="all-MiniLM-L6-v2"):
		"""
		Initialize the RAG system with embedding model

		Args:
			embedding_model_name: Name of sentence-transformer model to use
		
		Raises:
			RuntimeError: If model initialization fails
		"""
		
		try:
			logger.info(f"Initializing RAG with embedding model: {embedding_model_name}")
			self.embedding_model = SentenceTransformer(embedding_model_name)
			self.vector_db = None
			self.documents = None
			self.qa_pairs = None
			self.embeddings = None
			self.generator = None
			self.tokenizer = None
			logger.info("RAG system initialized successfully")
		except Exception as e:
			logger.error(f"Failed to initialize RAG system: {e}")
			raise RuntimeError(f"RAG initialization failed: {e}")

	def load_dataset(self):
		"""
		Load and prepare the RAG Mini Wikipedia dataset

		Returns:
			Tuple of (documents, qa_pairs)
		
		Raises:
			RuntimeError: If dataset loading fails
		"""

		try:
			logger.info("Loading RAG Mini Wikipedia dataset...")
			
			# Load text corpus for retrieval
			logger.info("Loading text corpus...")
			corpus_dataset = load_dataset("rag-datasets/rag-mini-wikipedia", "text-corpus")
			
			# Load Q&A pairs for evaluation
			logger.info("Loading Q&A pairs...")
			qa_dataset = load_dataset("rag-datasets/rag-mini-wikipedia", "question-answer")
			
			# Store documents for retrieval (using 'passages' split)
			self.documents = corpus_dataset['passages']['passage']
			
			# Store Q&A pairs for evaluation (using 'test' split)
			self.qa_pairs = qa_dataset['test']
			
			logger.info(f"Loaded {len(self.documents)} documents and {len(self.qa_pairs)} Q&A pairs")
			return self.documents, self.qa_pairs

		except Exception as e:
			logger.error(f"Failed to load dataset: {e}")
			raise RuntimeError(f"Dataset loading failed: {e}")

	def create_embeddings(self, batch_size: int = 32) -> np.ndarray:
		"""
		Create embeddings for documents

		Args:
			batch_size: Number of documents to process at once

		Returns:
			numpy array of document embeddings (shape: [num_docs, embedding_dim])

		Raises:
			ValueError: If document not loaded
			RuntimeError: If embedding creation fails
		"""

		if self.documents is None:
			raise ValueError("Documents not loaded. Call load_dataset() first.")

		try:
			logger.info(f"Creating embeddings for {len(self.documents)} documents...")
			
			# Create embeddings in batches to manage memory
			self.embeddings = self.embedding_model.encode(
				self.documents,
				batch_size=batch_size,
				show_progress_bar=True,
				convert_to_numpy=True
			)
			
			logger.info(f"Created embeddings with shape: {self.embeddings.shape}")
			return self.embeddings

		except Exception as e:
			logger.error(f"Failed to create embeddings: {e}")
			raise RuntimeError(f"Embedding creation failed: {e}")

	def build_vector_db(self) -> faiss.Index:
		"""
		Build FAISS vector database from document embeddings.
		
		Returns:
			FAISS index for similarity search
		
		Raises:
			ValueError: If embeddings not created
			RuntimeError: If index building fails
		"""
		
		if self.embeddings is None:
			raise ValueError("Embeddings not created. Call create_embeddings() first.")
		
		try:
			logger.info("Building FAISS vector database...")
			
			# Get embedding dimension
			embedding_dim = self.embeddings.shape[1]
			
			# Creating FAISS index (using inner product for cosine similarity)
			# Normalize embeddings for cosine similarity
			faiss.normalize_L2(self.embeddings)
			
			# Creating index
			self.vector_db = faiss.IndexFlatIP(embedding_dim)
			
			# Adding embeddings to index
			self.vector_db.add(self.embeddings)
			
			logger.info(f"FAISS index built with {self.vector_db.ntotal} vectors")
			return self.vector_db

		except Exception as e:
			logger.error(f"Failed to build vector database: {e}")
			raise RuntimeError(f"Vector database creation failed: {e}")

	def retrieve(self, query: str, top_k: int = 1) -> List[Tuple[str, float]]:
		"""
		Retrieve relevant documents for a query.
		
		Args:
			query: Question or search query
			top_k: Number of documents to retrieve
		
		Returns:
			List of (document_text, similarity_score) tuples
		
		Raises:
			ValueError: If vector database not built
			RuntimeError: If retrieval fails
		"""
		
		if self.vector_db is None:
			raise ValueError("Vector database not built. Call build_vector_db() first.")
		
		try:
			# Encode query
			query_embedding = self.embedding_model.encode([query], convert_to_numpy=True)
			
			# Normalize for cosine similarity
			faiss.normalize_L2(query_embedding)
			
			# Search
			similarities, indices = self.vector_db.search(query_embedding, top_k)
			
			# Return documents with scores
			results = [
				(self.documents[idx], float(sim))
				for sim, idx in zip(similarities[0], indices[0])
			]
			
			return results
			
		except Exception as e:
			logger.error(f"Retrieval failed: {e}")
			raise RuntimeError(f"Retrieval failed: {e}")

	def load_generator(self, model_name: str = "google/flan-t5-base"):
		"""
		Load text generation model.
		
		Args:
			model_name: HuggingFace model identifier
		
		Raises:
			RuntimeError: If model loading fails
		"""
		
		try:
			logger.info(f"Loading generation model: {model_name}")

			cache_dir = "Z:/hf_cache"
			
			self.tokenizer = AutoTokenizer.from_pretrained(model_name)
			self.generator = AutoModelForSeq2SeqLM.from_pretrained(model_name)
			logger.info("Generation model loaded successfully")
			
		except Exception as e:
			logger.error(f"Failed to load generator: {e}")
			raise RuntimeError(f"Generator loading failed: {e}")

	def generate_answer(
		self,
		query: str,
		context: str,
		prompt_strategy: str = "basic",
		max_length: int = 100
	) -> str:
		"""
		Generate answer based on query and context.
		
		Args:
			query: The question
			context: Retrieved context
			prompt_strategy: Prompting approach ("basic", "cot", "persona", "instruction")
			max_length: Maximum answer length
		
		Returns:
			Generated answer text
		
		Raises:
			ValueError: If generator not loaded
			RuntimeError: If generation fails
		"""
		if self.generator is None or self.tokenizer is None:
			raise ValueError("Generator not loaded. Call load_generator() first.")
		
		try:
			# Format prompt based on strategy
                        prompt = format_prompt(query, context, prompt_strategy)
			
			# Tokenize
                        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
			
			# Generate
                        outputs = self.generator.generate(
				**inputs,
				max_length=max_length,
				num_beams=4,
				early_stopping=True
			)
			
			# Decode
                        answer = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
			
                        return answer
			
		except Exception as e:
			logger.error(f"Answer generation failed: {e}")
			raise RuntimeError(f"Generation failed: {e}")

	def query(
		self,
		question: str,
		top_k: int = 1,
		prompt_strategy: str = "basic"
	) -> Dict[str, Any]:
		"""
		Complete RAG pipeline: retrieve and generate.
		
		Args:
			question: The question to answer
			top_k: Number of documents to retrieve
			prompt_strategy: Prompting approach
		
		Returns:
			Dictionary with question, answer, context, and score
			"""
		
		try:
			# Retrieve relevant documents
                        retrieved = self.retrieve(question, top_k=top_k)
		
                        # Use top document = 1 for naive implementation, and then also 3 and 5:
                        if top_k == 1:
                                context, score = retrieved[0]
                        else:
			# Concatenate multiple retrieved documents
                                contexts = [doc for doc, _ in retrieved]
                                context = "\n\n---\n\n".join(contexts)
                                score = retrieved[0][1]
			
			# Generate answer
                        answer = self.generate_answer(question, context, prompt_strategy)
			
                        return {
				'question': question,
				'answer': answer,
				'context': context,
				'retrieval_score': score,
				'prompt_strategy': prompt_strategy
			}
			
		except Exception as e:
			logger.error(f"Query failed: {e}")
			return {
				'question': question,
				'answer': f"Error: {str(e)}",
				'context': None,
				'retrieval_score': 0.0,
				'prompt_strategy': prompt_strategy
			}

if __name__ == "__main__":
	#Test the system
	rag = NaiveRAG()
	print("Yes, Naive RAG system initialized successfully!")
