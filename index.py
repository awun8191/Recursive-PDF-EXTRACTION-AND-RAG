"""
Comprehensive RAG System Demo
Combining Ollama embeddings + ChromaDB retrieval + Gemini generation
"""

import os
import sys
from pathlib import Path

# Add src directory to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from typing import List, Dict, Any, Optional
import chromadb
from src.services.Ollama.ollama_service import OllamaService
from src.services.Gemini.gemini_service import GeminiService
from src.services.RAG.convert_to_embeddings import build_client
from src.utils.logging_utils import get_rag_logger

class RAGService:
    """Complete RAG service combining Ollama embeddings, ChromaDB retrieval, and Gemini generation."""
    
    def __init__(
        self,
        chroma_persist_dir: str = "./chroma_db",
        collection_name: str = "pdfs",
        ollama_model: str = "bge-m3:latest",
        top_k: int = 5,
        similarity_threshold: float = 0.1
    ):
        self.logger = get_rag_logger("RAGService")
        self.top_k = top_k
        self.similarity_threshold = similarity_threshold
        
        # Initialize Ollama service for embeddings
        try:
            self.ollama_service = OllamaService(model=ollama_model)
            self.logger.success(f"Initialized Ollama service with model: {ollama_model}")
        except Exception as e:
            self.logger.error(f"Failed to initialize Ollama service: {e}")
            raise RuntimeError("Ollama service is required for RAG") from e
        
        # Initialize Gemini service for text generation
        try:
            self.gemini_service = GeminiService()
            self.logger.success("Initialized Gemini service for text generation")
        except Exception as e:
            self.logger.error(f"Failed to initialize Gemini service: {e}")
            raise RuntimeError("Gemini service is required for RAG") from e
        
        # Initialize ChromaDB client
        try:
            self.chroma_client = build_client(chroma_persist_dir)
            self.collection = self.chroma_client.get_or_create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine"}
            )
            self.logger.success(f"Connected to ChromaDB collection: {collection_name}")
            
            # Get collection stats
            count = self.collection.count()
            self.logger.info(f"Collection contains {count:,} documents")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize ChromaDB: {e}")
            raise RuntimeError("ChromaDB is required for RAG") from e
    
    def retrieve_relevant_docs(self, query: str) -> List[Dict[str, Any]]:
        """Retrieve relevant documents using Ollama embeddings and ChromaDB similarity search."""
        try:
            # Generate query embedding using Ollama with target dimension 1024 to match stored embeddings
            self.logger.info(f"Generating query embedding for: '{query[:50]}...'")
            query_embedding = self.ollama_service.embed([query], target_dim=1024)[0]
            
            if not query_embedding:
                self.logger.warning("Empty query embedding generated")
                return []
            
            # Search ChromaDB for similar documents
            self.logger.info(f"Searching ChromaDB for top {self.top_k} similar documents")
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=self.top_k,
                include=['documents', 'metadatas', 'distances']
            )
            
            # Process and filter results
            relevant_docs = []
            if results['documents'] and results['documents'][0]:
                for i, (doc, metadata, distance) in enumerate(zip(
                    results['documents'][0],
                    results['metadatas'][0],
                    results['distances'][0]
                )):
                    # Convert distance to similarity (lower distance = higher similarity)
                    similarity = 1 - distance
                    
                    if similarity >= self.similarity_threshold:
                        relevant_docs.append({
                            'rank': i + 1,
                            'content': doc,
                            'similarity': similarity,
                            'distance': distance,
                            'metadata': metadata
                        })
                        
                        # Log document info
                        course = metadata.get('COURSE_CODE', 'Unknown')
                        dept = metadata.get('DEPARTMENT', 'Unknown')
                        self.logger.info(f"  {i+1}. {course} ({dept}) - Similarity: {similarity:.3f}")
            
            self.logger.success(f"Retrieved {len(relevant_docs)} relevant documents")
            return relevant_docs
            
        except Exception as e:
            self.logger.error(f"Document retrieval failed: {e}")
            return []
    
    def generate_response(self, query: str, relevant_docs: List[Dict[str, Any]]) -> str:
        """Generate response using Gemini based on query and retrieved documents."""
        try:
            if not relevant_docs:
                return "I couldn't find any relevant documents to answer your query. Please try a different question or ensure the document collection has been populated."
            
            # Construct context from relevant documents
            context_parts = []
            for doc in relevant_docs:
                course = doc['metadata'].get('COURSE_CODE', 'Unknown Course')
                dept = doc['metadata'].get('DEPARTMENT', 'Unknown Dept')
                content = doc['content'][:500]  # Limit context length
                
                context_parts.append(f"**{course} ({dept})**:\n{content}")
            
            context = "\n\n".join(context_parts)
            
            # Create prompt for Gemini
            prompt = f"""Based on the following course materials, please answer the user's question comprehensively and accurately.

**User Question:** {query}

**Relevant Course Materials:**
{context}

**Instructions:**
- Provide a detailed, well-structured answer based on the course materials above
- Reference specific courses when relevant
- If the materials don't fully answer the question, acknowledge this limitation
- Use clear, academic language appropriate for educational content

**Answer:**"""
            
            self.logger.info("Generating response using Gemini")
            from src.data_models.gemini_config import GeminiConfig
            config = GeminiConfig(
                temperature=0.7,
                max_output_tokens=1024,
                top_p=0.9
            )
            response = self.gemini_service.generate(
                prompt=prompt,
                generation_config=config
            )
            
            if isinstance(response, dict) and 'result' in response:
                return response['result']
            elif isinstance(response, str):
                return response
            else:
                return str(response)
                
        except Exception as e:
            self.logger.error(f"Response generation failed: {e}")
            return f"I encountered an error while generating a response: {e}"
    
    def query(self, question: str) -> Dict[str, Any]:
        """Complete RAG pipeline: retrieve + generate."""
        self.logger.info(f"Processing RAG query: '{question}'")
        
        # Step 1: Retrieve relevant documents
        relevant_docs = self.retrieve_relevant_docs(question)
        
        # Step 2: Generate response
        response = self.generate_response(question, relevant_docs)
        
        # Return complete result
        result = {
            'query': question,
            'response': response,
            'retrieved_docs': len(relevant_docs),
            'sources': [
                {
                    'course': doc['metadata'].get('COURSE_CODE', 'Unknown'),
                    'department': doc['metadata'].get('DEPARTMENT', 'Unknown'),
                    'similarity': doc['similarity'],
                    'preview': doc['content'][:100] + "..." if len(doc['content']) > 100 else doc['content']
                }
                for doc in relevant_docs
            ]
        }
        
        self.logger.success("RAG query completed successfully")
        return result

def main():
    """Demo the RAG system with sample queries."""
    logger = get_rag_logger("RAGDemo")
    
    print("🚀 Initializing RAG System...")
    print("=" * 60)
    
    try:
        # Initialize RAG service
        rag = RAGService(
            chroma_persist_dir="./chroma_db",
            collection_name="pdfs",
            ollama_model="bge-m3:latest",
            top_k=3
        )
        
        print("\n✅ RAG System Ready!")
        print(f"   📊 ChromaDB Collection: {rag.collection.count():,} documents")
        print(f"   🤖 Ollama Model: {rag.ollama_service.model}")
        print(f"   🧠 Gemini Service: Active")
        
        # Sample queries for demonstration
        sample_queries = [
            "What is machine learning?",
            "Explain the concept of algorithms",
            "What are the fundamentals of computer science?",
            "How does data structures work?",
            "What is software engineering?"
        ]
        
        print("\n" + "=" * 60)
        print("🔍 SAMPLE RAG QUERIES")
        print("=" * 60)
        
        for i, query in enumerate(sample_queries, 1):
            print(f"\n🎯 Query {i}: {query}")
            print("-" * 40)
            
            try:
                result = rag.query(query)
                
                print(f"📚 Sources: {result['retrieved_docs']} documents")
                for j, source in enumerate(result['sources'], 1):
                    print(f"   {j}. {source['course']} ({source['department']}) - {source['similarity']:.2f}")
                
                print(f"\n💬 Response:")
                print(result['response'][:300] + "..." if len(result['response']) > 300 else result['response'])
                
            except Exception as e:
                print(f"❌ Error processing query: {e}")
            
            if i < len(sample_queries):
                print("\n" + "-" * 60)
        
        print("\n" + "=" * 60)
        print("🎉 RAG System Demo Complete!")
        print("\nTo use this system:")
        print("1. Ensure Ollama is running with bge-m3:latest model")
        print("2. Have ChromaDB collection populated with documents")
        print("3. Configure Gemini API keys for generation")
        print("4. Call rag.query('your question') for responses")
        
    except Exception as e:
        print(f"\n❌ RAG System initialization failed: {e}")
        print("\nTroubleshooting:")
        print("1. Check if Ollama is running: ollama serve")
        print("2. Pull the model: ollama pull bge-m3:latest")
        print("3. Verify ChromaDB collection exists and has documents")
        print("4. Ensure Gemini API keys are properly configured")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())