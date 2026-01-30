"""
Context management and RAG (Retrieval Augmented Generation)
"""
from typing import List, Dict, Optional
from pathlib import Path
import json
from datetime import datetime

try:
    import chromadb
    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False


class ContextManager:
    """Manage user context and conversation history using RAG"""
    
    def __init__(self, data_dir: str = "~/roku-ai/data"):
        """
        Initialize context manager
        
        Args:
            data_dir: Directory for storing context data
        """
        self.data_dir = Path(data_dir).expanduser()
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize ChromaDB if available
        if CHROMADB_AVAILABLE:
            self.chroma_client = chromadb.PersistentClient(
                path=str(self.data_dir / "context")
            )
            self.collection = self.chroma_client.get_or_create_collection(
                name="user_context"
            )
            print("ChromaDB context store initialized")
        else:
            self.chroma_client = None
            self.collection = None
            print("ChromaDB not available, using simple context")
        
        # In-memory conversation history
        self.conversation_history: List[Dict] = []
        
        # User preferences (loaded from disk)
        self.preferences = self._load_preferences()
    
    def _load_preferences(self) -> Dict:
        """Load user preferences from disk"""
        pref_path = self.data_dir / "user_profile" / "preferences.json"
        if pref_path.exists():
            with open(pref_path, "r") as f:
                return json.load(f)
        return {}
    
    def save_preferences(self):
        """Save user preferences to disk"""
        pref_dir = self.data_dir / "user_profile"
        pref_dir.mkdir(parents=True, exist_ok=True)
        
        with open(pref_dir / "preferences.json", "w") as f:
            json.dump(self.preferences, f, indent=2)
    
    def add_message(self, role: str, content: str):
        """
        Add message to conversation history
        
        Args:
            role: "user" or "assistant"
            content: Message content
        """
        message = {
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat(),
        }
        self.conversation_history.append(message)
        
        # Also store in vector DB for RAG
        if self.collection is not None:
            self.collection.add(
                documents=[content],
                metadatas=[{"role": role, "timestamp": message["timestamp"]}],
                ids=[f"msg_{len(self.conversation_history)}"],
            )
    
    def add_context(self, context_type: str, content: str, metadata: Dict = None):
        """
        Add context information (preferences, facts, etc.)
        
        Args:
            context_type: Type of context (preference, fact, schedule, etc.)
            content: Context content
            metadata: Additional metadata
        """
        if self.collection is None:
            return
        
        meta = {"type": context_type, "timestamp": datetime.now().isoformat()}
        if metadata:
            meta.update(metadata)
        
        context_id = f"{context_type}_{datetime.now().timestamp()}"
        
        self.collection.add(
            documents=[content],
            metadatas=[meta],
            ids=[context_id],
        )
    
    def get_relevant_context(self, query: str, n_results: int = 5) -> List[str]:
        """
        Retrieve relevant context for a query
        
        Args:
            query: User query
            n_results: Number of results to return
            
        Returns:
            List of relevant context strings
        """
        if self.collection is None:
            return []
        
        try:
            results = self.collection.query(
                query_texts=[query],
                n_results=n_results,
            )
            return results["documents"][0] if results["documents"] else []
        except Exception as e:
            print(f"Context retrieval error: {e}")
            return []
    
    def get_recent_history(self, n_messages: int = 10) -> List[Dict]:
        """
        Get recent conversation history
        
        Args:
            n_messages: Number of recent messages
            
        Returns:
            List of recent messages
        """
        return self.conversation_history[-n_messages:]
    
    def clear_history(self):
        """Clear conversation history"""
        self.conversation_history = []
    
    def save_conversation(self, filename: Optional[str] = None):
        """
        Save current conversation to disk
        
        Args:
            filename: Optional filename (default: timestamp-based)
        """
        conv_dir = self.data_dir / "conversations"
        conv_dir.mkdir(parents=True, exist_ok=True)
        
        if filename is None:
            filename = f"conversation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        with open(conv_dir / filename, "w") as f:
            json.dump(self.conversation_history, f, indent=2)
        
        print(f"Conversation saved to {conv_dir / filename}")
    
    def build_context_prompt(self, query: str) -> str:
        """
        Build context-enhanced prompt for LLM
        
        Args:
            query: User's current query
            
        Returns:
            Context string to prepend to prompt
        """
        context_parts = []
        
        # Add relevant context from RAG
        relevant = self.get_relevant_context(query, n_results=3)
        if relevant:
            context_parts.append("Relevant context:")
            for ctx in relevant:
                context_parts.append(f"- {ctx}")
            context_parts.append("")
        
        # Add user preferences if relevant
        if self.preferences:
            context_parts.append("User preferences:")
            for key, value in self.preferences.items():
                context_parts.append(f"- {key}: {value}")
            context_parts.append("")
        
        return "\n".join(context_parts)


# Example usage
if __name__ == "__main__":
    ctx = ContextManager()
    
    # Add some context
    ctx.add_context("preference", "User prefers morning meetings")
    ctx.add_context("fact", "User's name is Alex")
    
    # Add conversation
    ctx.add_message("user", "What's a good time for a meeting?")
    ctx.add_message("assistant", "Based on your preferences, morning works best.")
    
    # Query context
    relevant = ctx.get_relevant_context("schedule a meeting")
    print("Relevant context:", relevant)
    
    # Build context prompt
    prompt = ctx.build_context_prompt("When should we meet?")
    print("Context prompt:\n", prompt)
