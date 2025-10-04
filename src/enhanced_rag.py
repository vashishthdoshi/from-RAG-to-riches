"""
Enhanced RAG Implementation with Re-ranking and Query Rewriting
Assignment 2 - Step 5

Vashishth
"""
from naive_rag import NaiveRAG
from sentence_transformers import CrossEncoder
from typing import List, Tuple, Dict, Any
import logging

logger = logging.getLogger(__name__)

class EnhancedRAG(NaiveRAG):
	"""
	RAG, with two enhancements:
	1. Re-ranking using cross-encoder
	2. Query rewriting for better retrieval
	"""
	
	def __init__(self, embedding_model_name="all-mpnet-base-v2"):
		"""Initializes enhanced RAG with cross-encoder for re-ranking"""
		super().__init__(embedding_model_name)
		
		# Load cross-encoder for re-ranking
		logger.info("Loading cross-encoder for re-ranking...")
		self.reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
		logger.info("Cross-encoder loaded")
	
	def rewrite_query(self, query: str) -> List[str]:
		"""
		Generate query variations for better retrieval coverage.
		
		Args:
			query: Original query
			
		Returns:
			List of query variations including original
		"""
		# Use generator to create variations
		if self.generator is None:
			return [query]  # Fallback to original if no generator
		
		try:
			# Generate a paraphrased version
			prompt = f"Re-write this question to keep the samemeaning, but be more deliberate and explicit in the ask:\n\nQuestion: {query}\n\nRewritten question:"
			
			inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=256)
			outputs = self.generator.generate(**inputs, max_length=100, num_beams=2)
			rewritten = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
			
			# Return original + rewritten (remove duplicates)
			variations = [query, rewritten]
			return list(set(v.strip() for v in variations if v.strip()))
			
		except Exception as e:
			logger.warning(f"Query rewriting failed: {e}")
			return [query]
	
	def retrieve_and_rerank(
		self,
		query: str,
		initial_k: int = 10,
		final_k: int = 3
	) -> List[Tuple[str, float]]:
		"""
		Retrieves top_k documents and re-ranks to top_k' using cross-encoder.
		
		Args:
			query: Search query
			initial_k: Number of candidates to retrieve
			final_k: Number to return after re-ranking
			
		Returns:
			List of (document, score) tuples after re-ranking
		"""
		# Get initial candidates
		candidates = self.retrieve(query, top_k=initial_k)
		
		# Re-rank using cross-encoder
		pairs = [[query, doc] for doc, _ in candidates]
		rerank_scores = self.reranker.predict(pairs)
		
		# Combine and sort by re-ranking scores
		reranked = list(zip([doc for doc, _ in candidates], rerank_scores))
		reranked.sort(key=lambda x: x[1], reverse=True)
		
		return reranked[:final_k]
	
	def query(
		self,
		question: str,
		top_k: int = 3,
		prompt_strategy: str = "basic",
		use_reranking: bool = True,
		use_rewriting: bool = True
	) -> Dict[str, Any]:
		"""
		Enhanced RAG pipeline with re-ranking and query rewriting.
		
		Args:
			question: Question to answer
			top_k: Number of documents to use for generation
			prompt_strategy: Prompting approach
			use_reranking: Whether to use re-ranking
			use_rewriting: Whether to use query rewriting
			
		Returns:
			Dictionary with results
		"""
		try:
			retrieved_docs = []
			
			if use_rewriting:
				# Generate query variations
				query_variations = self.rewrite_query(question)
				logger.info(f"Generated {len(query_variations)} query variations")
				
				# Retrieve for each variation
				for q_var in query_variations:
					if use_reranking:
						docs = self.retrieve_and_rerank(q_var, initial_k=10, final_k=top_k)
					else:
						docs = self.retrieve(q_var, top_k=top_k)
					retrieved_docs.extend(docs)
				
				# Remove duplicates and take top-k by score
				seen_docs = set()
				unique_docs = []
				for doc, score in sorted(retrieved_docs, key=lambda x: x[1], reverse=True):
					if doc not in seen_docs:
						unique_docs.append((doc, score))
						seen_docs.add(doc)
					if len(unique_docs) >= top_k:
						break
				retrieved_docs = unique_docs
				
			else:
				# Standard retrieval
				if use_reranking:
					retrieved_docs = self.retrieve_and_rerank(question, initial_k=10, final_k=top_k)
				else:
					retrieved_docs = self.retrieve(question, top_k=top_k)
			
			# Combine contexts
			if len(retrieved_docs) == 1:
				context = retrieved_docs[0][0]
				score = retrieved_docs[0][1]
			else:
				contexts = [doc for doc, _ in retrieved_docs]
				context = "\n\n---\n\n".join(contexts)
				score = retrieved_docs[0][1]
			
			# Generate answer
			answer = self.generate_answer(question, context, prompt_strategy)
			
			return {
				'question': question,
				'answer': answer,
				'context': context,
				'retrieval_score': score,
				'prompt_strategy': prompt_strategy,
				'num_query_variations': len(self.rewrite_query(question)) if use_rewriting else 1,
				'used_reranking': use_reranking,
				'used_rewriting': use_rewriting
			}
			
		except Exception as e:
			logger.error(f"Enhanced query failed: {e}")
			return {
				'question': question,
				'answer': f"Error: {str(e)}",
				'context': None,
				'retrieval_score': 0.0,
				'prompt_strategy': prompt_strategy
			}

if __name__ == "__main__":
	print("Enhanced RAG module loaded")
