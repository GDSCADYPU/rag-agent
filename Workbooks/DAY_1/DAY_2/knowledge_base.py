"""
Day 2 - Exercise 3: Vector Database Knowledge Base - HANDS-ON WORKBOOK
=======================================================================

LEARNING OBJECTIVES:
☐ Understand vector embeddings (converting text to numbers)
☐ Work with ChromaDB (a powerful vector database)
☐ Implement semantic similarity search
☐ Build a production RAG system foundation

What is a Vector Database?
===========================
Imagine a library where books aren't organized alphabetically, but by
their MEANING. Books about similar topics sit near each other, even if
their titles are completely different. That's what vector databases do!

How it works:
1. Convert text → vectors (lists of numbers representing meaning)
2. Store vectors in a specialized database
3. When you search, convert your query → vector
4. Find the closest matching vectors = most relevant content!

This is the SECRET behind:
✓ Google Search understanding your intent
✓ ChatGPT remembering your documents
✓ Recommendation systems ("You might also like...")
✓ Semantic search engines

WORKSHOP PROGRESS CHECKLIST:
============================
☐ Step 1: Initialize ChromaDB client and collection
☐ Step 2: Set up embedding function (all-MiniLM-L6-v2)
☐ Step 3: Initialize TextChunker helper
☐ Step 4: Implement add_document() pipeline
☐ Step 5: Implement query() for semantic search
☐ Step 6: Implement get_stats()
☐ Step 7: Implement clear() method
☐ Step 8: Run the demo and verify results
☐ BONUS: Experiment with different queries and observe similarity scores
"""

import chromadb
from chromadb.utils import embedding_functions
from typing import List, Dict
import uuid
import sys
sys.path.append('.')

# Import our previous utilities
from chunking_utility import TextChunker


class KnowledgeBase:
    """
    Your intelligent knowledge vault! 
    
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
        TODO #1: Initialize your knowledge base!
        ========================================
        
        TASK: Set up ChromaDB client, embedding function, and create/get collection.
        
        Args:
            collection_name (str): Name for this knowledge collection
                                  (like a database table name)
                                  
        IMPLEMENTATION STEPS:
        1. Create ChromaDB client (in-memory mode)
        2. Initialize SentenceTransformer embedding function
        3. Create or get the collection with metadata
        4. Initialize TextChunker helper
        5. Print initialization messages
        
        HINTS:
        - Use chromadb.Client() for in-memory database
        - Model: "all-MiniLM-L6-v2" creates 384-dimensional vectors
        - Use self.client.get_or_create_collection()
        """
        print("🚀 Initializing Knowledge Base...")
        
        # TODO: Initialize ChromaDB client (in-memory for this workshop)
        # In production, you'd use persistent storage
        self.client = chromadb.Client()
        
        # TODO: Initialize embedding function
        # This converts text → 384-dimensional vectors!
        # "all-MiniLM-L6-v2" is a lightweight, fast model perfect for learning
        self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )
        
        print("   Loading embedding model: all-MiniLM-L6-v2")
        print("   (This creates 384-dimensional vectors)")
        
        # TODO: Create or get collection
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=self.embedding_function,
            metadata={"description": "GDG Workshop Knowledge Base"}
        )
        
        # TODO: Initialize helper utilities
        self.chunker = TextChunker(chunk_size=500, overlap=50)
        
        current_count = self.collection.count()
        
        print(f"✅ Knowledge Base '{collection_name}' ready!")
        print(f"   Current documents: {current_count} chunks")
        print()
    
    def add_document(self, text: str, metadata: Dict = None) -> List[str]:
        """
        TODO #2: Add a document to the knowledge base.
        ==============================================
        
        TASK: Implement the complete document ingestion pipeline.
        
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
            
        ALGORITHM:
        1. Initialize metadata if None
        2. Chunk the document using self.chunker.chunk_text()
        3. Prepare lists for: ids, texts, metadatas
        4. For each chunk:
           a. Generate unique ID (use uuid.uuid4())
           b. Extract text
           c. Combine metadata with chunk metadata
        5. Add all chunks to ChromaDB using self.collection.add()
        6. Return the list of IDs
        
        HINTS:
        - Use str(uuid.uuid4()) for unique IDs
        - Use method='sentences' for better chunking
        - ChromaDB generates embeddings automatically!
        """
        if metadata is None:
            metadata = {}
        
        print(f"📄 Processing document...")
        
        # TODO: Step 1 - Chunk the document
        chunks = self.chunker.chunk_text(text, method='sentences')
        print(f"   ✂️  Created {len(chunks)} chunks")
        
        # TODO: Step 2 - Prepare data for ChromaDB
        ids = []
        texts = []
        metadatas = []
        
        for chunk in chunks:
            # TODO: Generate unique ID for this chunk
            chunk_id = str(uuid.uuid4())
            ids.append(chunk_id)
            
            # TODO: The actual text
            texts.append(chunk['text'])
            
            # TODO: Combine our metadata with chunk metadata
            chunk_metadata = {
                **metadata,  # User-provided metadata
                'chunk_id': chunk['chunk_id'],
                'word_count': chunk['word_count'],
                'method': chunk.get('method', 'unknown')
            }
            metadatas.append(chunk_metadata)
        
        # TODO: Step 3 - Add to ChromaDB (embeddings generated automatically!)
        print(f"   🧮 Generating embeddings...")
        self.collection.add(
            ids=ids,
            documents=texts,
            metadatas=metadatas
        )
        
        print(f"✅ Added {len(chunks)} chunks to knowledge base")
        print(f"   Total chunks in KB: {self.collection.count()}\n")
        
        return ids
    
    def query(self, query_text: str, top_k: int = 3) -> List[Dict]:
        """
        TODO #3: Search the knowledge base! 🔍
        =======================================
        
        TASK: Implement semantic similarity search.
        
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
                
        ALGORITHM:
        1. Print search status
        2. Query ChromaDB using self.collection.query()
           - Pass query_texts as a list: [query_text]
           - Set n_results=top_k
        3. Format results into a list of dictionaries:
           - For each result, extract: id, text, metadata, distance
           - Calculate similarity = 1 - distance
        4. Return formatted results
        
        HINTS:
        - ChromaDB returns: ids, documents, metadatas, distances
        - Each is a list of lists: results['ids'][0][i]
        - Distance is how "far apart" vectors are (lower = more similar)
        - Similarity score = 1 - distance (higher = more similar)
        """
        print(f"🔍 Searching for: '{query_text}'")
        print(f"   Looking for top {top_k} results...")
        
        # TODO: Query ChromaDB (it handles embedding the query automatically!)
        results = self.collection.query(
            query_texts=[query_text],
            n_results=top_k
        )
        
        # TODO: Format results nicely
        formatted_results = []
        
        for i in range(len(results['ids'][0])):
            # TODO: Calculate similarity score (1 - distance = similarity)
            distance = results['distances'][0][i] if 'distances' in results else None
            similarity = (1 - distance) if distance is not None else None
            
            # TODO: Create formatted result dictionary
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
        TODO #4: Get statistics about your knowledge base.
        ===================================================
        
        TASK: Return a dictionary with KB statistics.
        
        Returns:
            dict: KB statistics
            
        HINT: Include collection_name, total_chunks, embedding_dimension, model
        """
        # TODO: Return statistics dictionary
        return {
            'collection_name': self.collection.name,
            'total_chunks': self.collection.count(),
            'embedding_dimension': 384,  # for all-MiniLM-L6-v2
            'embedding_model': 'all-MiniLM-L6-v2'
        }
    
    def clear(self):
        """
        TODO #5: Clear all documents from the knowledge base.
        ======================================================
        
        TASK: Delete the collection and recreate it empty.
        
        ⚠️  Warning: This deletes everything!
        
        ALGORITHM:
        1. Print warning message
        2. Delete the collection using self.client.delete_collection()
        3. Recreate the collection with same name and embedding function
        4. Print confirmation
        
        HINT: Use self.client.create_collection() to recreate
        """
        # TODO: Clear knowledge base
        print("⚠️  Clearing knowledge base...")
        self.client.delete_collection(self.collection.name)
        
        # TODO: Recreate empty collection
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
    
    # TODO #6: Initialize knowledge base
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
    
    # TODO #7: Add documentation to knowledge base
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
    
    # TODO #8: Display stats
    stats = kb.get_stats()
    print("📊 Knowledge Base Stats:")
    for key, value in stats.items():
        print(f"   {key}: {value}")
    print()
    
    # TODO #9: Test semantic search!
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
        
        # TODO: Execute query
        results = kb.query(query, top_k=2)
        
        # TODO: Display results with similarity scores
        for j, result in enumerate(results, 1):
            similarity_pct = result['similarity'] * 100 if result['similarity'] else 0
            
            print(f"\nResult {j} (Similarity: {similarity_pct:.1f}%):")
            print(f"  Source: {result['metadata'].get('source', 'Unknown')}")
            print(f"  Text: {result['text'][:200]}...")
            
            # TODO: Add quality indicator based on similarity
            if similarity_pct > 80:
                print(f"  Quality: 🎯 Excellent match!")
            elif similarity_pct > 60:
                print(f"  Quality: ✅ Good match")
            else:
                print(f"  Quality: 🤔 Moderate match")
    
    print("\n" + "=" * 70)
    print("CONGRATULATIONS! You've built a vector database! 🎉")
    print("=" * 70)
    print("\n📚 Key Takeaways:")
    print("   ✓ Vector embeddings capture semantic meaning")
    print("   ✓ ChromaDB enables fast similarity search")
    print("   ✓ Similarity scores tell you how relevant results are")
    print("   ✓ This is the foundation of RAG systems!")
    print("\n🚀 Next Steps:")
    print("   → Try different queries and observe results")
    print("   → Experiment with adding more documents")
    print("   → Understand how embeddings work under the hood")
    print("   → Move on to building a complete RAG system!")
    print("\n💡 Understanding the Magic:")
    print("   Each chunk is converted to a 384-dimensional vector")
    print("   Similar meanings = vectors close together in space")
    print("   Query vector is compared to all chunk vectors")
    print("   Closest vectors = most semantically similar chunks")
    print()