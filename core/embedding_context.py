"""
Embedding-based Context Compression for Roku AI

Instead of injecting raw profile JSON (~559 tokens), we:
1. Encode profile sections into dense embeddings
2. Convert embeddings to soft prompt tokens
3. Inject fewer tokens with same semantic content

Tradeoff: Slightly lossy, but much more compact.
"""

import json
import time
from pathlib import Path
from typing import Optional, Dict, List, Any
import numpy as np

from sentence_transformers import SentenceTransformer


class EmbeddingContextManager:
    """Manages user context via embedding compression."""
    
    def __init__(
        self, 
        profiles_dir: str = "data/profiles",
        model_name: str = "all-MiniLM-L6-v2",  # 80MB, fast, 384-dim
    ):
        self.profiles_dir = Path(profiles_dir)
        self.current_profile: Optional[Dict[str, Any]] = None
        self.current_user: Optional[str] = None
        
        # Load embedding model
        print(f"Loading embedding model: {model_name}...")
        self.encoder = SentenceTransformer(model_name)
        self.embedding_dim = self.encoder.get_sentence_embedding_dimension()
        print(f"Embedding dimension: {self.embedding_dim}")
        
        # Cached embeddings
        self._profile_embeddings: Optional[np.ndarray] = None
        self._profile_texts: Optional[List[str]] = None
    
    def load_profile(self, username: str) -> bool:
        """Load and encode a user profile."""
        profile_path = self.profiles_dir / f"{username}.json"
        if not profile_path.exists():
            return False
        
        with open(profile_path, 'r') as f:
            data = json.load(f)
        
        self.current_profile = data.get('profile', data)
        self.current_user = username
        
        # Encode profile into embeddings
        self._encode_profile()
        return True
    
    def _encode_profile(self) -> None:
        """Convert profile to embeddings."""
        if not self.current_profile:
            return
        
        p = self.current_profile
        
        # Create semantic chunks (each becomes an embedding)
        chunks = [
            f"User's name is {p.get('identity', {}).get('name', 'Unknown')}. {p.get('identity', {}).get('description', '')}",
            f"Work: {p.get('work', {}).get('role', '')}. Institution: {p.get('work', {}).get('company', '')}",
            f"Current project: {p.get('work', {}).get('current_projects', '')}",
            f"Skills: {p.get('work', {}).get('skills', '')}",
            f"Schedule: Wake time {p.get('schedule', {}).get('wake_time', '')}. Work hours: {p.get('schedule', {}).get('work_end', '')}",
            f"Commitments: {p.get('schedule', {}).get('regular_commitments', '')}",
            f"Ideal day: {p.get('schedule', {}).get('ideal_day', '')}",
            f"Communication preference: {p.get('preferences', {}).get('communication_style', '')}. Reminders needed: {p.get('preferences', {}).get('helpful_reminders', '')}",
            f"Short-term goal: {p.get('goals', {}).get('short_term', '')}",
            f"Long-term goal: {p.get('goals', {}).get('long_term', '')}. How to help: {p.get('goals', {}).get('how_to_help', '')}",
        ]
        
        self._profile_texts = chunks
        
        # Encode all chunks
        start = time.time()
        self._profile_embeddings = self.encoder.encode(chunks, convert_to_numpy=True)
        encode_time = time.time() - start
        
        print(f"Encoded {len(chunks)} profile chunks in {encode_time*1000:.1f}ms")
    
    def get_compressed_context(self) -> str:
        """
        Get compressed profile representation.
        
        For LLMs that don't support direct embedding injection,
        we use a retrieval approach: encode the query, find most
        relevant profile chunks, inject only those.
        """
        if self._profile_texts is None:
            return ""
        
        # For now, return all chunks as condensed text
        # In a full implementation, we'd do query-based retrieval
        condensed = " | ".join(self._profile_texts)
        return condensed
    
    def retrieve_relevant(self, query: str, top_k: int = 5) -> str:
        """Retrieve only the most relevant profile sections for a query."""
        if self._profile_embeddings is None or self._profile_texts is None:
            return ""
        
        # Encode query
        query_emb = self.encoder.encode([query], convert_to_numpy=True)[0]
        
        # Compute cosine similarities
        similarities = np.dot(self._profile_embeddings, query_emb) / (
            np.linalg.norm(self._profile_embeddings, axis=1) * np.linalg.norm(query_emb)
        )
        
        # Get top-k indices
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        # Return relevant chunks
        relevant = [self._profile_texts[i] for i in top_indices]
        return " | ".join(relevant)
    
    def get_embedding_stats(self) -> Dict[str, Any]:
        """Get stats about the embeddings."""
        if self._profile_embeddings is None:
            return {}
        
        return {
            "num_chunks": len(self._profile_texts),
            "embedding_dim": self.embedding_dim,
            "total_embedding_size_bytes": self._profile_embeddings.nbytes,
            "total_embedding_size_kb": self._profile_embeddings.nbytes / 1024,
        }


def benchmark_comparison():
    """Compare JSON injection vs embedding compression."""
    from core.context_manager import ContextManager
    
    print("="*60)
    print("BENCHMARK: JSON Injection vs Embedding Compression")
    print("="*60)
    
    # JSON approach
    print("\n[1] JSON Injection Approach")
    json_cm = ContextManager()
    
    start = time.time()
    json_cm.load_profile("Srimaan")
    json_load_time = time.time() - start
    
    json_context = json_cm.get_profile_summary()
    json_tokens = json_cm.get_context_tokens_estimate()
    json_bytes = len(json_context.encode('utf-8'))
    
    print(f"    Load time: {json_load_time*1000:.1f}ms")
    print(f"    Context size: {json_bytes} bytes")
    print(f"    Estimated tokens: {json_tokens}")
    
    # Embedding approach
    print("\n[2] Embedding Compression Approach")
    
    start = time.time()
    emb_cm = EmbeddingContextManager()
    model_load_time = time.time() - start
    
    start = time.time()
    emb_cm.load_profile("Srimaan")
    emb_load_time = time.time() - start
    
    stats = emb_cm.get_embedding_stats()
    
    # Full context (all chunks)
    full_context = emb_cm.get_compressed_context()
    full_tokens = len(full_context) // 4
    
    # Query-based retrieval (only relevant chunks)
    test_query = "What am I studying?"
    start = time.time()
    relevant_context = emb_cm.retrieve_relevant(test_query, top_k=3)
    retrieval_time = time.time() - start
    relevant_tokens = len(relevant_context) // 4
    
    print(f"    Model load time: {model_load_time*1000:.1f}ms (one-time)")
    print(f"    Profile encode time: {emb_load_time*1000:.1f}ms")
    print(f"    Retrieval time: {retrieval_time*1000:.1f}ms")
    print(f"    Embedding storage: {stats['total_embedding_size_kb']:.1f} KB")
    print(f"    Full context tokens: {full_tokens}")
    print(f"    Relevant-only tokens (top-3): {relevant_tokens}")
    
    # Comparison
    print("\n" + "="*60)
    print("COMPARISON SUMMARY")
    print("="*60)
    
    print(f"\n{'Metric':<30} {'JSON':<15} {'Embedding':<15} {'Savings':<15}")
    print("-"*75)
    print(f"{'Tokens (full)':<30} {json_tokens:<15} {full_tokens:<15} {(1-full_tokens/json_tokens)*100:.0f}%")
    print(f"{'Tokens (query-based)':<30} {json_tokens:<15} {relevant_tokens:<15} {(1-relevant_tokens/json_tokens)*100:.0f}%")
    print(f"{'Storage (bytes)':<30} {json_bytes:<15} {int(stats['total_embedding_size_kb']*1024):<15} {'N/A':<15}")
    print(f"{'Load time (ms)':<30} {json_load_time*1000:.1f}{'ms':<12} {emb_load_time*1000:.1f}{'ms':<12} {'N/A':<15}")
    
    # Phone estimate
    print("\n" + "="*60)
    print("PHONE PERFORMANCE ESTIMATE (iPhone 15 Pro)")
    print("="*60)
    
    # Assume phone is 3-5x slower for LLM, 2x slower for embeddings
    phone_slowdown_llm = 4
    phone_slowdown_emb = 2
    
    # Llama 3.2 3B prompt eval: ~275 tok/s on M1 Pro
    m1_tok_per_sec = 275
    phone_tok_per_sec = m1_tok_per_sec / phone_slowdown_llm
    
    json_first_msg_m1 = json_tokens / m1_tok_per_sec
    json_first_msg_phone = json_tokens / phone_tok_per_sec
    
    relevant_first_msg_m1 = relevant_tokens / m1_tok_per_sec
    relevant_first_msg_phone = relevant_tokens / phone_tok_per_sec
    
    print(f"\n{'Scenario':<35} {'M1 Pro':<15} {'iPhone 15 Pro':<15}")
    print("-"*65)
    print(f"{'JSON injection (first msg)':<35} {json_first_msg_m1:.2f}s{'':<10} {json_first_msg_phone:.2f}s")
    print(f"{'Embedding retrieval (first msg)':<35} {relevant_first_msg_m1:.2f}s{'':<10} {relevant_first_msg_phone:.2f}s")
    print(f"{'Time saved on phone':<35} {'':<15} {json_first_msg_phone - relevant_first_msg_phone:.2f}s")
    
    # Test accuracy
    print("\n" + "="*60)
    print("ACCURACY TEST: Query-based Retrieval")
    print("="*60)
    
    test_queries = [
        "What is my name?",
        "What am I studying?",
        "What are my goals?",
        "When do I wake up?",
    ]
    
    for q in test_queries:
        relevant = emb_cm.retrieve_relevant(q, top_k=2)
        print(f"\nQ: {q}")
        print(f"Retrieved: {relevant[:200]}...")


if __name__ == "__main__":
    benchmark_comparison()
