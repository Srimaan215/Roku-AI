"""
Reasoning Layer for Roku AI

Implements RAG-CoT (Retrieval-Augmented Generation with Chain-of-Thought):
1. Embed user query
2. Retrieve relevant context chunks
3. Format context for CoT reasoning
4. Model explicitly reasons before answering

This approach combines:
- Vector retrieval for "what's relevant"
- CoT prompting for "how to reason"
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

# Embedding model
try:
    from sentence_transformers import SentenceTransformer
    EMBEDDINGS_AVAILABLE = True
except ImportError:
    EMBEDDINGS_AVAILABLE = False


@dataclass
class ContextChunk:
    """A piece of retrievable context."""
    id: str
    text: str
    source: str  # 'profile', 'calendar', 'weather', 'health', etc.
    metadata: Dict[str, Any] = field(default_factory=dict)
    embedding: Optional[np.ndarray] = None
    
    def __repr__(self):
        return f"ContextChunk({self.id}, source={self.source}, len={len(self.text)})"


class ContextStore:
    """
    Vector store for context chunks.
    Uses simple cosine similarity for retrieval.
    """
    
    def __init__(self, embedding_model: str = "all-MiniLM-L6-v2"):
        if not EMBEDDINGS_AVAILABLE:
            raise ImportError("sentence-transformers not installed. Run: pip install sentence-transformers")
        
        self.encoder = SentenceTransformer(embedding_model)
        self.chunks: List[ContextChunk] = []
        self._embeddings_matrix: Optional[np.ndarray] = None
    
    def add_chunk(self, chunk: ContextChunk) -> None:
        """Add a context chunk and compute its embedding."""
        if chunk.embedding is None:
            chunk.embedding = self.encoder.encode(chunk.text, convert_to_numpy=True)
        self.chunks.append(chunk)
        self._embeddings_matrix = None  # Invalidate cache
    
    def add_chunks(self, chunks: List[ContextChunk]) -> None:
        """Add multiple chunks efficiently."""
        texts = [c.text for c in chunks if c.embedding is None]
        if texts:
            embeddings = self.encoder.encode(texts, convert_to_numpy=True)
            idx = 0
            for chunk in chunks:
                if chunk.embedding is None:
                    chunk.embedding = embeddings[idx]
                    idx += 1
        self.chunks.extend(chunks)
        self._embeddings_matrix = None
    
    def clear(self) -> None:
        """Clear all chunks."""
        self.chunks = []
        self._embeddings_matrix = None
    
    def _get_embeddings_matrix(self) -> np.ndarray:
        """Get or compute the embeddings matrix."""
        if self._embeddings_matrix is None:
            self._embeddings_matrix = np.vstack([c.embedding for c in self.chunks])
        return self._embeddings_matrix
    
    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        source_filter: Optional[List[str]] = None,
        threshold: float = 0.0
    ) -> List[Tuple[ContextChunk, float]]:
        """
        Retrieve most relevant chunks for a query.
        
        Args:
            query: User's question
            top_k: Number of chunks to return
            source_filter: Only return chunks from these sources
            threshold: Minimum similarity score
            
        Returns:
            List of (chunk, similarity_score) tuples
        """
        if not self.chunks:
            return []
        
        query_embedding = self.encoder.encode(query, convert_to_numpy=True)
        embeddings = self._get_embeddings_matrix()
        
        # Cosine similarity
        similarities = np.dot(embeddings, query_embedding) / (
            np.linalg.norm(embeddings, axis=1) * np.linalg.norm(query_embedding)
        )
        
        # Apply source filter
        if source_filter:
            mask = np.array([c.source in source_filter for c in self.chunks])
            similarities = np.where(mask, similarities, -1)
        
        # Get top-k
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            score = similarities[idx]
            if score >= threshold:
                results.append((self.chunks[idx], float(score)))
        
        return results


class ReasoningLayer:
    """
    RAG-CoT reasoning layer for Roku AI.
    
    Combines:
    - Context retrieval (what's relevant)
    - Chain-of-thought prompting (how to reason)
    """
    
    def __init__(self, embedding_model: str = "all-MiniLM-L6-v2"):
        self.store = ContextStore(embedding_model)
        self.last_retrieved: List[Tuple[ContextChunk, float]] = []
    
    def load_profile_chunks(self, profile: Dict[str, Any], username: str) -> None:
        """Convert user profile into retrievable chunks."""
        chunks = []
        
        # Identity
        if 'identity' in profile:
            identity = profile['identity']
            chunks.append(ContextChunk(
                id=f"{username}_identity",
                text=f"User's name is {identity.get('name', 'Unknown')}. {identity.get('description', '')}",
                source="profile",
                metadata={"section": "identity"}
            ))
        
        # Location
        if 'location' in profile:
            loc = profile['location']
            loc_text = f"User lives in {loc.get('home', 'Unknown')}. "
            if loc.get('work'):
                loc_text += f"Works at {loc['work']}. "
            if loc.get('known_places'):
                places = [f"{k}: {v}" for k, v in loc['known_places'].items()]
                loc_text += f"Known places: {', '.join(places)}."
            chunks.append(ContextChunk(
                id=f"{username}_location",
                text=loc_text,
                source="profile",
                metadata={"section": "location"}
            ))
        
        # Work/Education
        if 'work' in profile:
            work = profile['work']
            work_text = f"Role: {work.get('role', '')}. "
            work_text += f"Institution: {work.get('company', '')}. "
            if work.get('current_projects'):
                work_text += f"Current project: {work['current_projects']}. "
            if work.get('skills'):
                work_text += f"Skills: {work['skills']}."
            chunks.append(ContextChunk(
                id=f"{username}_work",
                text=work_text,
                source="profile",
                metadata={"section": "work"}
            ))
        
        # Schedule
        if 'schedule' in profile:
            sched = profile['schedule']
            sched_text = f"Wake time: {sched.get('wake_time', '')}. "
            sched_text += f"Work ends: {sched.get('work_end', '')}. "
            if sched.get('regular_commitments'):
                sched_text += f"Regular commitments: {sched['regular_commitments']}. "
            if sched.get('ideal_day'):
                sched_text += f"Ideal day: {sched['ideal_day']}."
            chunks.append(ContextChunk(
                id=f"{username}_schedule",
                text=sched_text,
                source="profile",
                metadata={"section": "schedule"}
            ))
        
        # Goals
        if 'goals' in profile:
            goals = profile['goals']
            goals_text = f"Short-term goals: {goals.get('short_term', '')}. "
            goals_text += f"Long-term goals: {goals.get('long_term', '')}. "
            if goals.get('how_to_help'):
                goals_text += f"How to help: {goals['how_to_help']}."
            chunks.append(ContextChunk(
                id=f"{username}_goals",
                text=goals_text,
                source="profile",
                metadata={"section": "goals"}
            ))
        
        # Preferences
        if 'preferences' in profile:
            prefs = profile['preferences']
            prefs_text = f"Communication style: {prefs.get('communication_style', '')}. "
            if prefs.get('helpful_reminders'):
                prefs_text += f"Helpful reminders: {prefs['helpful_reminders']}."
            chunks.append(ContextChunk(
                id=f"{username}_preferences",
                text=prefs_text,
                source="profile",
                metadata={"section": "preferences"}
            ))
        
        self.store.add_chunks(chunks)
    
    def update_calendar_context(self, calendar_text: str) -> None:
        """Update calendar context chunk."""
        # Remove old calendar chunk
        self.store.chunks = [c for c in self.store.chunks if c.source != "calendar"]
        self.store._embeddings_matrix = None
        
        # Add new
        self.store.add_chunk(ContextChunk(
            id="calendar_current",
            text=calendar_text,
            source="calendar",
            metadata={"updated": datetime.now().isoformat()}
        ))
    
    def update_weather_context(self, weather_text: str) -> None:
        """Update weather context chunk."""
        # Remove old weather chunk
        self.store.chunks = [c for c in self.store.chunks if c.source != "weather"]
        self.store._embeddings_matrix = None
        
        # Add new
        self.store.add_chunk(ContextChunk(
            id="weather_current",
            text=weather_text,
            source="weather",
            metadata={"updated": datetime.now().isoformat()}
        ))
    
    def update_time_context(self) -> None:
        """Update current time context."""
        now = datetime.now()
        is_weekend = now.weekday() >= 5
        
        # Time of day
        hour = now.hour
        if 6 <= hour < 12:
            period = "morning"
        elif 12 <= hour < 17:
            period = "afternoon"
        elif 17 <= hour < 21:
            period = "evening"
        else:
            period = "night"
        
        time_text = (
            f"Current date: {now.strftime('%A, %B %d, %Y')}. "
            f"Current time: {now.strftime('%I:%M %p')}. "
            f"It is currently {period}. "
            f"Today is a {'weekend' if is_weekend else 'weekday'}."
        )
        
        # Remove old time chunk
        self.store.chunks = [c for c in self.store.chunks if c.id != "time_current"]
        self.store._embeddings_matrix = None
        
        self.store.add_chunk(ContextChunk(
            id="time_current",
            text=time_text,
            source="time",
            metadata={"timestamp": now.isoformat()}
        ))
    
    def retrieve_context(self, query: str, top_k: int = 4) -> str:
        """
        Retrieve relevant context for a query.
        Returns formatted context string for CoT prompting.
        """
        self.last_retrieved = self.store.retrieve(query, top_k=top_k)
        
        if not self.last_retrieved:
            return "No relevant context found."
    def get_current_time_context(self) -> str:
        """Get current time context string (always included, not retrieved)."""
        now = datetime.now()
        is_weekend = now.weekday() >= 5
        
        hour = now.hour
        if 6 <= hour < 11:
            period = "morning"
            meal_status = "Before lunch. Good time for breakfast."
        elif 11 <= hour < 14:
            period = "midday"
            meal_status = "Lunch time window (11am-2pm)."
        elif 14 <= hour < 17:
            period = "afternoon"
            meal_status = "After lunch, before dinner. Lunch window has passed."
        elif 17 <= hour < 21:
            period = "evening"
            meal_status = "Dinner time (5pm-9pm). Lunch was 3+ hours ago - YES it's too late for lunch."
        else:
            period = "night"
            meal_status = "Late night. Past dinner time."
        
        return (
            f"RIGHT NOW: {now.strftime('%I:%M %p')} on {now.strftime('%A, %B %d, %Y')} "
            f"({'weekend' if is_weekend else 'weekday'})\n"
            f"TIME OF DAY: {period}\n"
            f"MEAL TIMING: {meal_status}"
        )
    
    def retrieve_context(self, query: str, top_k: int = 4) -> str:
        """
        Retrieve relevant context for a query.
        Returns formatted context string for CoT prompting.
        """
        self.last_retrieved = self.store.retrieve(query, top_k=top_k)
        
        lines = ["RETRIEVED CONTEXT:"]
        for chunk, score in self.last_retrieved:
            lines.append(f"[{chunk.source}] {chunk.text}")
        
        return "\n".join(lines)
    
    def build_cot_prompt(
        self,
        query: str,
        username: str,
        include_reasoning_hint: bool = True
    ) -> str:
        """
        Build a Chain-of-Thought prompt with retrieved context.
        """
        # Always update time context in store (for retrieval)
        self.update_time_context()
        
        # Get time context (ALWAYS included, not relying on retrieval)
        time_context = self.get_current_time_context()
        
        # Retrieve relevant context
        context = self.retrieve_context(query, top_k=4)
        
        # Build prompt - time context is ALWAYS at the top
        system = f"""You are Roku, a personal AI assistant for {username}. You are helpful, warm, and casual.

{time_context}

When answering questions:
1. First, note the current time and date
2. Examine the retrieved context
3. Reason through what information is relevant
4. Give a clear, helpful answer

If the context doesn't contain the information needed, say so honestly.

{context}"""

        if include_reasoning_hint:
            assistant_start = "Let me check the relevant context:\n"
        else:
            assistant_start = ""
        
        prompt = f"""<|start_header_id|>system<|end_header_id|>

{system}<|eot_id|><|start_header_id|>user<|end_header_id|>

{query}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

{assistant_start}"""
        
        return prompt
    
    def get_retrieved_sources(self) -> List[str]:
        """Get the sources that were retrieved for the last query."""
        return [chunk.source for chunk, _ in self.last_retrieved]


if __name__ == "__main__":
    import json
    
    print("Testing Reasoning Layer...")
    
    # Load profile
    with open("data/profiles/Srimaan.json") as f:
        profile_data = json.load(f)
    profile = profile_data.get('profile', profile_data)
    
    # Initialize reasoning layer
    layer = ReasoningLayer()
    layer.load_profile_chunks(profile, "Srimaan")
    
    # Add calendar context
    layer.update_calendar_context(
        "Calendar for today: No remaining events. You are free this afternoon, evening, and tonight."
    )
    
    # Test queries
    queries = [
        "Am I free tonight?",
        "What am I working on?",
        "Where do I go to work out?",
        "What are my goals?",
    ]
    
    for q in queries:
        print(f"\n{'='*60}")
        print(f"Query: {q}")
        print(f"{'='*60}")
        
        context = layer.retrieve_context(q)
        print(context)
        print(f"\nSources: {layer.get_retrieved_sources()}")
