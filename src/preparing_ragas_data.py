"""
Prepare RAGAs evaluation data for both Naive and Enhanced RAG
"""
from naive_rag import NaiveRAG
from enhanced_rag import EnhancedRAG
import json
import os

def prepare_ragas_data():
	"""Generates predictions from both RAG systems"""
	
	test_size = 50
	
	# Prepare Naive RAG dataset for RAGAs
	print("\n")
	print("PREPARING NAIVE RAG DATA")
	print("\n")
	
	naive_rag = NaiveRAG(embedding_model_name="all-mpnet-base-v2")
	naive_rag.load_dataset()
	naive_rag.create_embeddings()
	naive_rag.build_vector_db()
	naive_rag.load_generator()
	
	naive_data = []
	print(f"Generating {test_size} predictions from Naive RAG...")
	for i in range(test_size):
		qa = naive_rag.qa_pairs[i]
		result = naive_rag.query(qa['question'], top_k=1, prompt_strategy="basic")
		
		naive_data.append({
			'question': qa['question'],
			'answer': result['answer'],
			'contexts': [result['context']],
			'ground_truth': qa['answer']
		})
	
	# Prepare Enhanced RAG dataset for RAGAs
	print("\n")
	print("PREPARING ENHANCED RAG DATA")
	print("\n")
	
	enhanced_rag = EnhancedRAG(embedding_model_name="all-mpnet-base-v2")
	enhanced_rag.load_dataset()
	enhanced_rag.create_embeddings()
	enhanced_rag.build_vector_db()
	enhanced_rag.load_generator()
	
	enhanced_data = []
	print(f"Generating {test_size} predictions from Enhanced RAG...")
	for i in range(test_size):
		qa = enhanced_rag.qa_pairs[i]
		result = enhanced_rag.query(
			qa['question'],
			top_k=1,
			prompt_strategy="basic",
			use_reranking=True,
			use_rewriting=False
		)
		
		enhanced_data.append({
			'question': qa['question'],
			'answer': result['answer'],
			'contexts': [result['context']],
			'ground_truth': qa['answer']
		})
	
	# Save both
	os.makedirs('../results', exist_ok=True)
	
	with open('../results/ragas_naive_data.json', 'w') as f:
		json.dump(naive_data, f, indent=2)
	
	with open('../results/ragas_enhanced_data.json', 'w') as f:
		json.dump(enhanced_data, f, indent=2)
	
	print(f"\nSaved naive RAG data: {len(naive_data)} predictions")
	print(f"Saved enhanced RAG data: {len(enhanced_data)} predictions")
	print("\nData ready for RAGAs evaluation")

if __name__ == "__main__":
	prepare_ragas_data()
