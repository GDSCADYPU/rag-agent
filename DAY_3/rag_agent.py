"""
Day 3 - Exercise 2: RAG Agent
=============================

🎉 THIS IS IT! The moment everything comes together! 🎉

What is RAG (Retrieval-Augmented Generation)?
==============================================

Imagine you're taking an open-book exam:
1. You read the question (query)
2. You search your textbook for relevant info (retrieval)
3. You write an answer using that info (generation)

That's exactly what RAG does!

Traditional AI: "Here's what I remember from training"
RAG AI: "Let me check my knowledge base and give you accurate, sourced answers"

The RAG Pipeline:
1. User asks a question
2. Convert question → vector
3. Search knowledge base for similar vectors
4. Retrieve top matching chunks
5. Send chunks + question to LLM
6. LLM generates answer based on retrieved context
7. Return answer with sources!

Why RAG is revolutionary:
✅ Up-to-date information (no training required!)
✅ Cites sources (no hallucinations!)
✅ Domain-specific knowledge
✅ Cost-effective (no need to fine-tune models)

What you'll learn:
✓ Building a complete RAG pipeline
✓ Combining retrieval + generation
✓ Prompt engineering for better answers
✓ Source attribution
✓ Production-ready AI applications
"""

import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, os.path.join(parent_dir, 'DAY_2'))

from gemini_wrapper import GeminiWrapper
from knowledge_base import KnowledgeBase
from typing import List, Dict


class RAGAgent:
    """
    Your intelligent RAG-powered assistant! 🧠🔍🤖
    
    This combines:
    - Vector Database (Knowledge Base)
    - Language Model (Gemini)
    - Smart prompting
    
    Result: An AI that answers questions based on YOUR documents
    with proper citations!
    
    Real-world use cases:
    - Customer support chatbots
    - Internal company Q&A systems
    - Study assistants for students
    - Legal/medical document analysis
    """
    
    def __init__(
        self,
        gemini_api_key: str,
        knowledge_base: KnowledgeBase = None,
        temperature: float = 0.3
    ):
        """
        Initialize your RAG Agent!
        
        Args:
            gemini_api_key (str): Your Gemini API key
            knowledge_base (KnowledgeBase): Pre-built knowledge base
            temperature (float): Lower = more factual (0.0-1.0)
                               For RAG, we want factual answers!
        
        Why temperature=0.3?
        We want the AI to stick to the facts from our documents,
        not be creative and potentially make things up!
        """
        print("🚀 Initializing RAG Agent...\n")
        
        # Initialize Gemini with RAG-optimized settings
        self.llm = GeminiWrapper(
            api_key=gemini_api_key,
            temperature=temperature
        )
        
        # Set a persona for accurate, sourced responses
        self.llm.set_persona(
            "You are a helpful AI assistant with access to a knowledge base. "
            "When answering questions, you ALWAYS cite the source documents you used. "
            "If you don't find relevant information in the knowledge base, you say so honestly. "
            "You are accurate, helpful, and always provide context from the documents. "
            "You never make up information - you only use what's in the provided context."
        )
        
        self.knowledge_base = knowledge_base
        
        print("✅ RAG Agent ready!")
        print("   Mode: Retrieval-Augmented Generation")
        print("   Source attribution: Enabled")
        print("   Hallucination protection: Active")
        print()
    
    def set_knowledge_base(self, knowledge_base: KnowledgeBase):
        """
        Connect a knowledge base to this agent.
        
        You can swap knowledge bases to change what the agent knows!
        
        Args:
            knowledge_base (KnowledgeBase): The knowledge base to use
        """
        self.knowledge_base = knowledge_base
        stats = knowledge_base.get_stats()
        print(f"✅ Knowledge base connected!")
        print(f"   Collection: {stats['collection_name']}")
        print(f"   Chunks available: {stats['total_chunks']}\n")
    
    def retrieve_context(self, query: str, top_k: int = 3) -> List[Dict]:
        """
        Retrieve relevant context from the knowledge base.
        
        This is the "R" in RAG - Retrieval!
        
        Args:
            query (str): User's question
            top_k (int): How many chunks to retrieve
        
        Returns:
            list: Most relevant document chunks
        """
        if not self.knowledge_base:
            print("⚠️  No knowledge base connected!")
            return []
        
        results = self.knowledge_base.query(query, top_k=top_k)
        return results
    
    def build_prompt_with_context(self, query: str, context_chunks: List[Dict]) -> str:
        """
        Build a carefully crafted prompt for the LLM.
        
        This is critical! A good prompt = good answer.
        
        The prompt structure:
        1. Provide context from knowledge base
        2. Label each source clearly
        3. State the user's question
        4. Give clear instructions on how to answer
        
        Args:
            query (str): User's question
            context_chunks (list): Retrieved document chunks
        
        Returns:
            str: Complete prompt for LLM
        """
        if not context_chunks:
            return f"""The user asked: "{query}"

You don't have any relevant information in your knowledge base to answer this question.
Please respond honestly that you don't have this information available, and suggest 
that the user might need to provide relevant documents or ask a different question."""
        
        # Build context section with clear source labeling
        context_text = "=== KNOWLEDGE BASE CONTEXT ===\n\n"
        context_text += "Here are relevant excerpts from the knowledge base:\n\n"
        
        for i, chunk in enumerate(context_chunks, 1):
            source = chunk['metadata'].get('source', 'Unknown Source')
            source_type = chunk['metadata'].get('source_type', '')
            
            context_text += f"[Source {i}: {source}]\n"
            context_text += f"{chunk['text']}\n\n"
        
        # Build complete prompt with instructions
        prompt = f"""{context_text}
=== USER QUESTION ===

{query}

=== INSTRUCTIONS ===

Please answer the user's question using ONLY the information provided in the context above.

Important guidelines:
1. Cite which source(s) you used (e.g., "According to Source 1...", "Source 2 states...")
2. If the context contains the answer, provide it clearly and concisely
3. If the context doesn't fully answer the question, say so and explain what information is available
4. DO NOT make up information or use knowledge outside the provided context
5. Be helpful and conversational while staying factual

Your answer:"""
        
        return prompt
    
    def answer(self, query: str, top_k: int = 3, verbose: bool = True) -> Dict:
        """
        Answer a question using the full RAG pipeline! 🎯
        
        This is where all the magic happens:
        1. Retrieve relevant context
        2. Build prompt with context
        3. Generate answer using LLM
        4. Return answer + sources
        
        Args:
            query (str): User's question
            top_k (int): How many document chunks to retrieve
            verbose (bool): Print progress information
        
        Returns:
            dict: Contains answer, sources, confidence, and metadata
        
        Example:
            >>> agent = RAGAgent(api_key="...")
            >>> result = agent.answer("How do I register for GDG?")
            >>> print(result['answer'])
            >>> for source in result['sources']:
            ...     print(f"Source: {source['metadata']['source']}")
        """
        if verbose:
            print(f"\n{'='*70}")
            print(f"🔍 RAG PIPELINE STARTING")
            print(f"{'='*70}\n")
            print(f"Query: '{query}'\n")
        
        # STEP 1: Retrieval
        if verbose:
            print("Step 1/3: 🔍 Retrieving relevant context...")
        
        context_chunks = self.retrieve_context(query, top_k=top_k)
        
        if context_chunks:
            if verbose:
                print(f"   ✅ Found {len(context_chunks)} relevant chunks")
                for i, chunk in enumerate(context_chunks, 1):
                    similarity = chunk.get('similarity', 0) * 100
                    source = chunk['metadata'].get('source', 'Unknown')
                    print(f"      {i}. {source} (Similarity: {similarity:.1f}%)")
        else:
            if verbose:
                print("   ⚠️  No relevant context found")
        
        # STEP 2: Prompt Building
        if verbose:
            print("\nStep 2/3: 📝 Building prompt with context...")
        
        prompt = self.build_prompt_with_context(query, context_chunks)
        
        if verbose:
            print(f"   ✅ Prompt ready ({len(prompt)} characters)")
        
        # STEP 3: Generation
        if verbose:
            print("\nStep 3/3: 🤖 Generating answer with Gemini...")
        
        answer = self.llm.generate(prompt)
        
        if verbose:
            print(f"   ✅ Answer generated ({len(answer)} characters)\n")
            print(f"{'='*70}\n")
        
        # Compile complete result
        result = {
            'query': query,
            'answer': answer,
            'sources': [
                {
                    'text': chunk['text'][:300] + '...' if len(chunk['text']) > 300 else chunk['text'],
                    'metadata': chunk['metadata'],
                    'similarity': chunk.get('similarity', 0)
                }
                for chunk in context_chunks
            ],
            'num_sources': len(context_chunks),
            'has_sources': len(context_chunks) > 0
        }
        
        return result
    
    def interactive_mode(self):
        """
        Launch interactive Q&A mode!
        
        Perfect for testing and demos.
        Type 'quit' to exit.
        """
        print("\n" + "="*70)
        print("🤖 INTERACTIVE RAG AGENT")
        print("="*70)
        print("\nAsk me anything about the knowledge base!")
        print("Type 'quit' to exit\n")
        
        while True:
            try:
                question = input("You: ").strip()
                
                if question.lower() in ['quit', 'exit', 'q']:
                    print("\n👋 Thanks for chatting! Goodbye!")
                    break
                
                if not question:
                    continue
                
                result = self.answer(question, verbose=False)
                
                print(f"\n🤖 Agent: {result['answer']}\n")
                
                if result['sources']:
                    print(f"📚 Sources ({len(result['sources'])}):")
                    for i, source in enumerate(result['sources'], 1):
                        print(f"   {i}. {source['metadata'].get('source', 'Unknown')}")
                print()
                
            except KeyboardInterrupt:
                print("\n\n👋 Interrupted. Goodbye!")
                break
            except Exception as e:
                print(f"\n❌ Error: {str(e)}\n")


# ============================================================================
# DEMO: Let's build a complete RAG system! 🚀
# ============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("RAG AGENT DEMONSTRATION - The Complete AI System!")
    print("=" * 70 + "\n")
    
    try:
        import os
        from dotenv import load_dotenv
        load_dotenv()
        
        api_key = os.getenv('GEMINI_API_KEY')
        
        if not api_key:
            print("⚠️  Please set GEMINI_API_KEY in your .env file!")
            print("\n1. Get your key: https://makersuite.google.com/app/apikey")
            print("2. Create .env file with: GEMINI_API_KEY=your_key_here")
            print("3. Run this script again!")
            sys.exit(1)
        
        # Initialize knowledge base
        print("Step 1: Setting up Knowledge Base")
        print("-" * 70 + "\n")
        
        kb = KnowledgeBase("gdg_rag_demo")
        
        # Add sample GDG documentation
        sample_docs = """
        GDG Event Registration and Participation Guide:
        
        Registration is free and open to all students. Visit gdg.community.dev to find 
        your local chapter and register for events. You'll need a Google account to sign up.
        
        Events typically run from 9:00 AM to 5:00 PM. We provide WiFi, power outlets, 
        coffee, snacks, and lunch. Please bring your laptop with a charger.
        
        Workshop Prerequisites:
        For our AI workshop, please ensure you have:
        - Python 3.8 or higher installed
        - A code editor (VS Code recommended)
        - 8GB RAM minimum
        - Enthusiasm to learn!
        
        What to Expect:
        Day 1 focuses on Python basics and NLP fundamentals.
        Day 2 covers vector databases and document processing.
        Day 3 is all about building RAG systems with Gemini AI.
        
        Certificates are provided to all participants who complete the workshop.
        """
        
        kb.add_document(
            sample_docs,
            metadata={
                'source': 'GDG Workshop Guide',
                'type': 'guidelines',
                'category': 'event-info'
            }
        )
        
        # Initialize RAG agent
        print("\nStep 2: Initializing RAG Agent")
        print("-" * 70 + "\n")
        
        agent = RAGAgent(
            gemini_api_key=api_key,
            knowledge_base=kb,
            temperature=0.3
        )
        
        # Test queries
        print("Step 3: Testing RAG System")
        print("-" * 70 + "\n")
        
        test_questions = [
            "How much does it cost to attend GDG events?",
            "What should I bring to the workshop?",
            "What are the prerequisites for the AI workshop?",
            "Will I get a certificate?"
        ]
        
        for i, question in enumerate(test_questions, 1):
            print(f"\n{'='*70}")
            print(f"QUESTION {i}")
            print(f"{'='*70}\n")
            
            result = agent.answer(question, top_k=2, verbose=True)
            
            print(f"Q: {result['query']}")
            print(f"\nA: {result['answer']}\n")
            
            if result['sources']:
                print(f"📚 Sources used: {result['num_sources']}")
                for j, source in enumerate(result['sources'], 1):
                    similarity = source['similarity'] * 100
                    print(f"   {j}. {source['metadata'].get('source')} ({similarity:.1f}% relevant)")
            
            print()
        
        print("=" * 70)
        print("💡 WHAT YOU JUST BUILT:")
        print("=" * 70)
        print("""
🎉 CONGRATULATIONS! You've built a production-ready RAG system!

Your system can:
✅ Store and index documents in a vector database
✅ Retrieve relevant information using semantic search
✅ Generate accurate answers using Gemini AI
✅ Cite sources for transparency
✅ Handle queries it can't answer honestly

This is the SAME technology used by:
- ChatGPT with document upload
- Enterprise AI assistants
- Customer support chatbots
- Legal/medical AI systems

From here, you can:
1. Add more documents (PDFs, websites, etc.)
2. Build a web interface (Streamlit - next exercise!)
3. Deploy to production
4. Scale to millions of documents

You're now an AI Engineer! 🚀
        """)
        
        # Optional: Interactive mode
        print("=" * 70)
        print("Want to try the interactive mode? (y/n)")
        choice = input("> ").strip().lower()
        
        if choice == 'y':
            agent.interactive_mode()
        
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 70)
    print("✨ RAG Agent complete!")
    print("Next up: Streamlit Web App - Share your creation with the world!")
    print("=" * 70 + "\n")