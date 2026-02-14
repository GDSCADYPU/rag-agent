"""
Day 2 - Exercise 3: Vector Database Knowledge Base
==================================================

🎉 This is where the magic happens! 🎉

What is a Vector Database?
Imagine a library where books aren't organized alphabetically, but by
their MEANING. Books about similar topics sit near each other, even if
their titles are completely different. That's what vector databases do!

How it works:
1. Convert text → vectors (lists of numbers representing meaning)
2. Store vectors in a specialized database
3. When you search, convert your query → vector
4. Find the closest matching vectors = most relevant content!

This is the SECRET behind:
✅ Google Search understanding your intent
✅ ChatGPT remembering your documents
✅ Recommendation systems ("You might also like...")
✅ Semantic search engines

What you'll learn:
✓ Vector embeddings (converting text to numbers)
✓ ChromaDB (a powerful vector database)
✓ Similarity search at scale
✓ Building a production RAG system foundation
"""

import chromadb
from chromadb.utils import embedding_functions
from typing import List, Dict
import uuid
import sys
sys.path.append('.')

# Import our previous utilities
from chunking_utility import TextChunker
from pdf_processor import PDFProcessor


class KnowledgeBase:
    """
    Your intelligent knowledge vault! 🧠💾
    
    This class:
    1. Takes your documents
    2. Converts them to vector embeddings (magic numbers!)
    3. Stores them in ChromaDB
    4. Lets you search by MEANING, not just keywords
    
    Real-world comparison:
    Traditional database: "Find documents containing 'Python'"
    Vector database: "Find documents about programming languages"
                     → Returns Python, JavaScript, Java, C++, etc.
    """
    
    def __init__(self, collection_name: str = "gdg_knowledge"):
        """
        Initialize your knowledge base!
        
        Args:
            collection_name (str): Name for this knowledge collection
                                  (like a database table name)
        """
        print("🚀 Initializing Knowledge Base...")
        
        # Initialize ChromaDB client (in-memory for this workshop)
        # In production, you'd use persistent storage
        self.client = chromadb.Client()
        
        # Initialize embedding function
        # This converts text → 384-dimensional vectors!
        # "all-MiniLM-L6-v2" is a lightweight, fast model perfect for learning
        self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )
        
        print("   Loading embedding model: all-MiniLM-L6-v2")
        print("   (This creates 384-dimensional vectors)")
        
        # Create or get collection
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=self.embedding_function,
            metadata={"description": "GDG Workshop Knowledge Base"}
        )
        
        # Initialize helper utilities
        self.chunker = TextChunker(chunk_size=500, overlap=50)
        self.pdf_processor = PDFProcessor()
        
        current_count = self.collection.count()
        
        print(f"✅ Knowledge Base '{collection_name}' ready!")
        print(f"   Current documents: {current_count} chunks")
        print()
    
    def add_document(self, text: str, metadata: Dict = None) -> List[str]:
        """
        Add a document to the knowledge base.
        
        The pipeline:
        1. Chunk the text (using our TextChunker)
        2. Generate embeddings (automatically by ChromaDB)
        3. Store chunks + embeddings + metadata
        
        Args:
            text (str): The document text
            metadata (dict): Optional metadata (source, date, author, etc.)
            
        Returns:
            list: IDs of chunks that were added
            
        Example:
            >>> kb = KnowledgeBase()
            >>> ids = kb.add_document(
            ...     "GDG events are free for all students...",
            ...     metadata={'source': 'GDG FAQ', 'type': 'guidelines'}
            ... )
            >>> print(f"Added {len(ids)} chunks")
        """
        if metadata is None:
            metadata = {}
        
        print(f"📄 Processing document...")
        
        # Step 1: Chunk the document
        chunks = self.chunker.chunk_text(text, method='sentences')
        print(f"   ✂️  Created {len(chunks)} chunks")
        
        # Step 2: Prepare data for ChromaDB
        ids = []
        texts = []
        metadatas = []
        
        for chunk in chunks:
            # Generate unique ID for this chunk
            chunk_id = str(uuid.uuid4())
            ids.append(chunk_id)
            
            # The actual text
            texts.append(chunk['text'])
            
            # Combine our metadata with chunk metadata
            chunk_metadata = {
                **metadata,  # User-provided metadata
                'chunk_id': chunk['chunk_id'],
                'word_count': chunk['word_count'],
                'method': chunk.get('method', 'unknown')
            }
            metadatas.append(chunk_metadata)
        
        # Step 3: Add to ChromaDB (embeddings generated automatically!)
        print(f"   🧮 Generating embeddings...")
        self.collection.add(
            ids=ids,
            documents=texts,
            metadatas=metadatas
        )
        
        print(f"✅ Added {len(chunks)} chunks to knowledge base")
        print(f"   Total chunks in KB: {self.collection.count()}\n")
        
        return ids
    
    def add_pdf(self, pdf_path: str) -> List[str]:
        """
        Add a PDF document to the knowledge base.
        
        This combines PDF processing + chunking + embedding!
        
        Args:
            pdf_path (str): Path to PDF file
            
        Returns:
            list: IDs of added chunks
        """
        print(f"📕 Adding PDF: {pdf_path}")
        
        # Extract text and metadata from PDF
        doc = self.pdf_processor.extract_with_metadata(pdf_path)
        
        if 'error' in doc:
            print(f"❌ Failed to process PDF: {doc['error']}")
            return []
        
        # Prepare metadata
        metadata = {
            'source': doc['filename'],
            'source_type': 'pdf',
            'num_pages': doc['num_pages'],
            'title': doc['metadata'].get('title', 'Unknown')
        }
        
        # Add the full text with metadata
        return self.add_document(doc['full_text'], metadata)
    
    def add_pdf_directory(self, directory_path: str) -> Dict:
        """
        Add all PDFs from a directory to knowledge base.
        
        Perfect for: "Load all company documentation into the system"
        
        Args:
            directory_path (str): Path to directory with PDFs
            
        Returns:
            dict: Summary statistics
        """
        print(f"📁 Processing directory: {directory_path}\n")
        
        documents = self.pdf_processor.process_directory(directory_path)
        
        total_chunks = 0
        successful_docs = 0
        
        for doc in documents:
            if 'error' not in doc:
                chunk_ids = self.add_pdf(doc['path'])
                total_chunks += len(chunk_ids)
                successful_docs += 1
        
        summary = {
            'documents_processed': successful_docs,
            'total_chunks_added': total_chunks,
            'total_in_kb': self.collection.count()
        }
        
        print("\n" + "=" * 70)
        print("📊 BATCH PROCESSING SUMMARY")
        print("=" * 70)
        for key, value in summary.items():
            print(f"   {key}: {value}")
        print()
        
        return summary
    
    def query(self, query_text: str, top_k: int = 3) -> List[Dict]:
        """
        Search the knowledge base! 🔍
        
        This is where the magic happens:
        1. Convert your query to a vector
        2. Find the top_k most similar vectors in the database
        3. Return the corresponding text chunks
        
        Args:
            query_text (str): Your search query
            top_k (int): How many results to return (default: 3)
            
        Returns:
            list: Most relevant chunks with metadata
            
        Example:
            >>> results = kb.query("How do I register?", top_k=2)
            >>> for result in results:
            ...     print(result['text'])
        """
        print(f"🔍 Searching for: '{query_text}'")
        print(f"   Looking for top {top_k} results...")
        
        # Query ChromaDB (it handles embedding the query automatically!)
        results = self.collection.query(
            query_texts=[query_text],
            n_results=top_k
        )
        
        # Format results nicely
        formatted_results = []
        
        for i in range(len(results['ids'][0])):
            # Calculate similarity score (1 - distance = similarity)
            distance = results['distances'][0][i] if 'distances' in results else None
            similarity = (1 - distance) if distance is not None else None
            
            formatted_results.append({
                'id': results['ids'][0][i],
                'text': results['documents'][0][i],
                'metadata': results['metadatas'][0][i],
                'distance': distance,
                'similarity': similarity
            })
        
        print(f"✅ Found {len(formatted_results)} relevant chunks\n")
        
        return formatted_results
    
    def get_stats(self) -> Dict:
        """
        Get statistics about your knowledge base.
        
        Returns:
            dict: KB statistics
        """
        return {
            'collection_name': self.collection.name,
            'total_chunks': self.collection.count(),
            'embedding_dimension': 384,  # for all-MiniLM-L6-v2
            'embedding_model': 'all-MiniLM-L6-v2'
        }
    
    def clear(self):
        """
        Clear all documents from the knowledge base.
        
        ⚠️  Warning: This deletes everything!
        """
        print("⚠️  Clearing knowledge base...")
        self.client.delete_collection(self.collection.name)
        
        self.collection = self.client.create_collection(
            name=self.collection.name,
            embedding_function=self.embedding_function
        )
        
        print("✅ Knowledge base cleared (all documents removed)\n")


# ============================================================================
# DEMO: Let's build a knowledge base! 🚀
# ============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("KNOWLEDGE BASE DEMO - The Heart of RAG Systems!")
    print("=" * 70 + "\n")
    
    # Initialize knowledge base
    kb = KnowledgeBase(collection_name="gdg_demo")
    
    # Sample GDG documentation
    gdg_docs = """
    Google Developer Groups (GDG) are community groups for college and university 
    students interested in Google developer technologies. Students from all undergraduate 
    or graduate programs with an interest in growing as a developer are welcome. By 
    joining a GDG, students grow their knowledge in a peer-to-peer learning environment 
    and build solutions for local businesses and their community.
    
    Events and Activities:
    GDG chapters host various events including workshops, hackathons, study jams, and 
    tech talks. These events are designed to help students learn new technologies, 
    network with peers, and gain practical experience. Workshops typically run from 
    9:00 AM to 5:00 PM and cover topics like AI, Cloud Computing, Android Development, 
    and Web Technologies.
    
    How to Join:
    To join a GDG chapter, visit gdg.community.dev and find your local chapter. 
    Registration is free and open to all students. Once registered, you'll receive 
    notifications about upcoming events and gain access to exclusive resources and 
    learning materials.
    
    Leadership:
    Each GDG chapter is led by passionate student organizers who work closely with 
    Google Developer Experts and the broader developer community. Leaders organize 
    events, manage the community, and ensure members have a great learning experience.
    """
    
    # Add documentation to knowledge base
    print("=" * 70)
    print("STEP 1: Adding documents to knowledge base")
    print("=" * 70 + "\n")
    
    kb.add_document(
        gdg_docs,
        metadata={
            'source': 'GDG Guidelines',
            'type': 'official',
            'category': 'documentation'
        }
    )
    
    # Display stats
    stats = kb.get_stats()
    print("📊 Knowledge Base Stats:")
    for key, value in stats.items():
        print(f"   {key}: {value}")
    print()
    
    # Test semantic search!
    print("=" * 70)
    print("STEP 2: Testing semantic search")
    print("=" * 70 + "\n")
    
    test_queries = [
        "How do I join GDG?",
        "What time do workshops start?",
        "What kind of events does GDG organize?",
        "Who leads GDG chapters?"
    ]
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n{'─' * 70}")
        print(f"Query {i}: '{query}'")
        print('─' * 70)
        
        results = kb.query(query, top_k=2)
        
        for j, result in enumerate(results, 1):
            similarity_pct = result['similarity'] * 100 if result['similarity'] else 0
            
            print(f"\nResult {j} (Similarity: {similarity_pct:.1f}%):")
            print(f"  Source: {result['metadata'].get('source', 'Unknown')}")
            print(f"  Text: {result['text'][:200]}...")
            
            if similarity_pct > 80:
                print(f"  Quality: 🎯 Excellent match!")
            elif similarity_pct > 60:
                print(f"  Quality: ✅ Good match")
            else:
                print(f"  Quality: 🤔 Moderate match")
    
    print("\n" + "=" * 70)
    print("💡 WHAT JUST HAPPENED?")
    print("=" * 70)
    print("""
1. We added a document → automatically chunked into pieces
2. Each chunk → converted to a 384-dimensional vector
3. Vectors stored in ChromaDB with metadata
4. When we query → query becomes a vector too
5. ChromaDB finds vectors most similar to query vector
6. Returns the most relevant chunks!

This is SEMANTIC SEARCH - understanding meaning, not just keywords! 🎯
    """)
    
    print("\n" + "=" * 70)
    print("🎉 CONGRATULATIONS! You've built a vector database!")
    print("=" * 70)
    print("""
You now understand:
✅ Vector embeddings (text → numbers)
✅ Similarity search at scale
✅ Document chunking + storage
✅ Metadata tracking
✅ The foundation of RAG systems!

Tomorrow (Day 3): We'll connect this to an LLM for intelligent Q&A! 🚀
    """)