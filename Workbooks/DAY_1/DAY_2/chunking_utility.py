"""
Day 2 - Exercise 1: Text Chunking Utility - HANDS-ON WORKBOOK
==============================================================

LEARNING OBJECTIVES:
☐ Understand why chunking is critical for AI systems
☐ Implement word-based chunking with overlap
☐ Implement sentence-based chunking (better for RAG!)
☐ Build production-ready text processing tools

The Problem:
Imagine trying to remember an entire textbook at once. Impossible, right?
AI models have the same challenge! They have a "context window" - a limit
to how much text they can process at once.

The Solution: CHUNKING!
Break large documents into smaller, digestible pieces while preserving context.

Real-world example:
Instead of feeding a 100-page manual to an AI, we:
1. Break it into ~500-word chunks
2. Add overlap so context isn't lost between chunks
3. Store each chunk separately
4. Retrieve only relevant chunks when needed

WORKSHOP PROGRESS CHECKLIST:
============================
☐ Step 1: Complete the __init__ method
☐ Step 2: Implement split_into_sentences()
☐ Step 3: Implement count_words()
☐ Step 4: Implement chunk_by_words() with sliding window
☐ Step 5: Implement chunk_by_sentences() (ADVANCED!)
☐ Step 6: Complete the chunk_text() wrapper method
☐ Step 7: Implement get_chunk_stats()
☐ Step 8: Run the demo and verify output
☐ BONUS: Experiment with different chunk sizes and overlaps
"""

from typing import List, Dict
import re

class TextChunker:
    """
    An intelligent text chunking system!
    
    Think of this as a librarian who takes a huge book and divides it
    into manageable chapters, making sure each chapter makes sense on
    its own while maintaining the story flow.
    
    Why chunking matters:
    - AI models have token limits (e.g., GPT-4: ~8k tokens)
    - Smaller chunks = faster processing
    - Better chunk = better retrieval = better answers!
    """
    
    def __init__(self, chunk_size: int = 500, overlap: int = 50):
        """
        TODO #1: Initialize the chunker with smart defaults.
        ====================================================
        
        TASK: Store the chunk_size and overlap parameters, then print
              initialization information.
        
        Args:
            chunk_size (int): Target number of words per chunk
                              Default 500 = ~2-3 paragraphs
                              
            overlap (int): Words to overlap between chunks
                          Default 50 = preserves context
                          
        Example:
            Chunk 1: words 0-500
            Chunk 2: words 450-950 (50 word overlap with Chunk 1)
            Chunk 3: words 900-1400 (50 word overlap with Chunk 2)
            
        Why overlap?
        If a key concept spans chunks 1-2, the overlap ensures we
        don't lose that context!
        
        HINT: You need to:
        1. Store chunk_size as self.chunk_size
        2. Store overlap as self.overlap
        3. Print initialization messages
        """
        # TODO: Store chunk_size and overlap as instance variables
        self.chunk_size = chunk_size
        self.overlap = overlap
        
        # TODO: Print initialization information
        print(f"✨ TextChunker initialized!")
        print(f"   Chunk size: {chunk_size} words")
        print(f"   Overlap: {overlap} words")
        print(f"   Strategy: Preserve context with intelligent overlap")
    
    def split_into_sentences(self, text: str) -> List[str]:
        """
        TODO #2: Split text into sentences intelligently.
        ==================================================
        
        TASK: Use regex to split text on sentence-ending punctuation
              followed by whitespace, then clean up the results.
        
        This is trickier than it looks! We need to handle:
        - Dr. Smith (not end of sentence)
        - U.S.A. (not 3 sentences)
        - "Hello!" she said. (actual end)
        
        Our approach: Split on punctuation followed by space
        
        Args:
            text (str): Text to split
            
        Returns:
            List of sentences
            
        HINT: Use re.split() with pattern r'(?<=[.!?])\s+'
              (?<=[.!?]) means "preceded by . or ! or ?"
              \s+ means "followed by one or more whitespace"
        """
        # TODO: Split on sentence-ending punctuation followed by whitespace
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        # TODO: Clean up - remove empty strings and strip whitespace
        clean_sentences = [s.strip() for s in sentences if s.strip()]
        
        return clean_sentences
    
    def count_words(self, text: str) -> int:
        """
        TODO #3: Count words in text.
        ==============================
        
        TASK: Split the text on whitespace and count the resulting words.
        
        Simple but essential - we use this to ensure chunks
        don't exceed our size limit!
        
        HINT: Use text.split() to get a list of words, then len()
        """
        # TODO: Split text and count words
        return len(text.split())
    
    def chunk_by_words(self, text: str) -> List[Dict]:
        """
        TODO #4: Chunk text by word count with overlap.
        ================================================
        
        TASK: Implement a sliding window approach to chunk text.
        
        Strategy: Sliding window approach
        - Start at word 0
        - Take next 'chunk_size' words
        - Move forward by (chunk_size - overlap)
        - Repeat until end
        
        Pros: Precise control over chunk size
        Cons: Might split mid-sentence
        
        Returns:
            List of chunk dictionaries with metadata
            
        ALGORITHM:
        1. Split text into words
        2. Initialize: chunks=[], chunk_id=0, start=0
        3. While start < total_words:
           a. Calculate end = min(start + chunk_size, total_words)
           b. Extract chunk_words = words[start:end]
           c. Create chunk dictionary with metadata
           d. Append to chunks list
           e. Move start forward: start = end - overlap
           f. Increment chunk_id
        4. Return chunks
        
        HINT: Use a while loop and be careful with the overlap calculation!
        """
        # TODO: Split text into words
        words = text.split()
        chunks = []
        chunk_id = 0
        start = 0
        
        # TODO: Implement sliding window loop
        while start < len(words):
            # TODO: Calculate end position
            end = min(start + self.chunk_size, len(words))
            
            # TODO: Extract chunk words
            chunk_words = words[start:end]
            chunk_text = ' '.join(chunk_words)
            
            # TODO: Store chunk with metadata
            chunks.append({
                'chunk_id': chunk_id,
                'text': chunk_text,
                'start_word': start,
                'end_word': end,
                'word_count': len(chunk_words),
                'method': 'word-based'
            })
            
            chunk_id += 1
            
            # TODO: Move start position (with overlap)
            start = end - self.overlap
            
            # Prevent infinite loop
            if start <= chunks[-1]['start_word']:
                break
        
        return chunks
    
    def chunk_by_sentences(self, text: str) -> List[Dict]:
        """
        TODO #5: Chunk text by sentences, respecting chunk_size limit.
        ===============================================================
        
        TASK: Group sentences into chunks while preserving semantic meaning.
        
        Strategy: Sentence-aware chunking
        - Split into sentences
        - Group sentences until we hit chunk_size
        - Keep last 2 sentences for overlap
        - Continue until end
        
        Pros: Never splits mid-sentence (better semantic meaning!)
        Cons: Chunks may vary in size
        
        This is BETTER for RAG systems because semantic meaning is preserved!
        
        Returns:
            List of chunk dictionaries with metadata
            
        ALGORITHM:
        1. Split text into sentences using split_into_sentences()
        2. Initialize: chunks=[], current_chunk=[], current_word_count=0, chunk_id=0
        3. For each sentence:
           a. Count words in sentence
           b. If adding sentence exceeds chunk_size AND current_chunk is not empty:
              - Save current_chunk to chunks with metadata
              - Start new chunk with overlap (last 2 sentences)
              - Increment chunk_id
           c. Add sentence to current_chunk
           d. Update current_word_count
        4. Don't forget to save the last chunk!
        5. Return chunks
        
        HINT: Use self.split_into_sentences() and self.count_words()
        """
        # TODO: Split text into sentences
        sentences = self.split_into_sentences(text)
        chunks = []
        current_chunk = []
        current_word_count = 0
        chunk_id = 0
        
        # TODO: Iterate through sentences and build chunks
        for sentence in sentences:
            sentence_word_count = self.count_words(sentence)
            
            # TODO: Check if adding this sentence exceeds chunk_size
            if current_word_count + sentence_word_count > self.chunk_size and current_chunk:
                # TODO: Save current chunk
                chunks.append({
                    'chunk_id': chunk_id,
                    'text': ' '.join(current_chunk),
                    'sentence_count': len(current_chunk),
                    'word_count': current_word_count,
                    'method': 'sentence-based'
                })
                
                # TODO: Start new chunk with overlap (keep last 2 sentences)
                overlap_sentences = current_chunk[-2:] if len(current_chunk) >= 2 else current_chunk
                current_chunk = overlap_sentences
                current_word_count = sum(self.count_words(s) for s in current_chunk)
                
                chunk_id += 1
            
            # TODO: Add sentence to current chunk
            current_chunk.append(sentence)
            current_word_count += sentence_word_count
        
        # TODO: Don't forget the last chunk!
        if current_chunk:
            chunks.append({
                'chunk_id': chunk_id,
                'text': ' '.join(current_chunk),
                'sentence_count': len(current_chunk),
                'word_count': current_word_count,
                'method': 'sentence-based'
            })
        
        return chunks
    
    def chunk_text(self, text: str, method: str = 'sentences') -> List[Dict]:
        """
        TODO #6: Main chunking method - your one-stop chunking solution!
        ================================================================
        
        TASK: Route to the appropriate chunking method based on the method parameter.
        
        Args:
            text (str): Text to chunk
            method (str): 'words' or 'sentences' (default: 'sentences')
        
        Returns:
            List of chunk dictionaries
            
        Recommendation: Use 'sentences' for RAG systems!
        
        HINT: Use if/elif/else to check method and call the right function
        """
        # TODO: Implement method routing
        if method == 'words':
            return self.chunk_by_words(text)
        elif method == 'sentences':
            return self.chunk_by_sentences(text)
        else:
            raise ValueError(f"Unknown method: {method}. Use 'words' or 'sentences'")
    
    def get_chunk_stats(self, chunks: List[Dict]) -> Dict:
        """
        TODO #7: Get statistics about your chunks.
        ===========================================
        
        TASK: Calculate useful statistics from the chunks list.
        
        Useful for optimizing chunk_size and overlap!
        
        HINT: Extract word_count from each chunk, then calculate:
        - total_chunks
        - avg_words_per_chunk
        - min_words
        - max_words
        - total_words
        - method (from first chunk)
        """
        if not chunks:
            return {'error': 'No chunks provided'}
        
        # TODO: Extract word counts from all chunks
        word_counts = [chunk['word_count'] for chunk in chunks]
        
        # TODO: Calculate and return statistics
        return {
            'total_chunks': len(chunks),
            'avg_words_per_chunk': sum(word_counts) / len(word_counts),
            'min_words': min(word_counts),
            'max_words': max(word_counts),
            'total_words': sum(word_counts),
            'method': chunks[0].get('method', 'unknown')
        }


# ============================================================================
# DEMO: Let's see intelligent chunking in action! 🚀
# ============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("TEXT CHUNKING DEMONSTRATION - Building Blocks of RAG Systems!")
    print("=" * 70 + "\n")
    
    # Sample text about AI (realistic length)
    sample_text = """
    Artificial Intelligence has revolutionized the way we interact with technology. 
    Machine learning algorithms can now process vast amounts of data with incredible 
    speed and accuracy. These algorithms learn patterns from historical data and make 
    predictions on new, unseen data.
    
    Deep learning, a powerful subset of machine learning, uses neural networks with 
    multiple layers. These networks can automatically learn hierarchical representations 
    of data. Each layer learns increasingly complex features, from simple edges in 
    images to complete objects.
    
    Natural Language Processing is another crucial area of AI. It enables computers 
    to understand, interpret, and generate human language. Recent advances in NLP 
    have led to powerful language models like GPT and BERT. These models can perform 
    various tasks such as translation, summarization, and question answering.
    
    Computer vision is yet another fascinating field within AI. It allows machines 
    to interpret and understand visual information from the world. Applications 
    include facial recognition, object detection, and autonomous vehicles. Self-driving 
    cars use computer vision to navigate roads safely.
    
    The future of AI holds immense potential for transforming industries and improving 
    our daily lives. From healthcare diagnostics to personalized education, AI systems 
    are becoming increasingly sophisticated. However, we must also consider ethical 
    implications and ensure AI development benefits humanity as a whole.
    """
    
    print("Sample text loaded:")
    print(f"   Total words: {len(sample_text.split())}")
    print(f"   Total characters: {len(sample_text)}\n")
    
    print("-" * 70)
    print("EXPERIMENT 1: Sentence-based chunking (RECOMMENDED)")
    print("-" * 70 + "\n")
    
    # TODO #8: Create chunker with reasonable defaults
    chunker = TextChunker(chunk_size=50, overlap=10)
    chunks_sentences = chunker.chunk_text(sample_text, method='sentences')
    
    print(f"✅ Created {len(chunks_sentences)} chunks\n")
    
    for i, chunk in enumerate(chunks_sentences[:3], 1):  # Show first 3
        print(f"Chunk {chunk['chunk_id']} ({chunk['word_count']} words):")
        print(f"   {chunk['text'][:150]}...")
        print()
    
    # Show overlap in action
    if len(chunks_sentences) >= 2:
        print("OVERLAP VISUALIZATION:")
        print("-" * 70)
        chunk1_end = ' '.join(chunks_sentences[0]['text'].split()[-10:])
        chunk2_start = ' '.join(chunks_sentences[1]['text'].split()[:10])
        print(f"End of Chunk 0:   ...{chunk1_end}")
        print(f"Start of Chunk 1: {chunk2_start}...")
        print("✨ Notice the overlap? This preserves context!\n")
    
    # Stats
    stats = chunker.get_chunk_stats(chunks_sentences)
    print("CHUNKING STATISTICS:")
    print("-" * 70)
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"   {key}: {value:.1f}")
        else:
            print(f"   {key}: {value}")
    
    print("\n" + "-" * 70)
    print("EXPERIMENT 2: Word-based chunking (for comparison)")
    print("-" * 70 + "\n")
    
    chunks_words = chunker.chunk_text(sample_text, method='words')
    
    print(f"✅ Created {len(chunks_words)} chunks\n")
    print(f"Chunk 0 (word-based):")
    print(f"   {chunks_words[0]['text'][:150]}...")
    print("\n⚠️  Notice: Might cut off mid-sentence!")
    
    print("\n" + "=" * 70)
    print("CONGRATULATIONS! You've built a production-ready text chunker! 🎉")
    print("=" * 70)
    print("\n📚 Key Takeaways:")
    print("   ✓ Sentence-based chunking preserves semantic meaning")
    print("   ✓ Overlap prevents context loss between chunks")
    print("   ✓ This is a critical building block for RAG systems")
    print("\n🚀 Next Steps:")
    print("   → Experiment with different chunk_size values")
    print("   → Try different overlap amounts")
    print("   → Move on to Exercise 2: Vector Embeddings!")
    print()