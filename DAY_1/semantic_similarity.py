"""
Day 1 - Exercise 2: Semantic Similarity Calculator
===================================================

Welcome to the fascinating world of semantic similarity! 🧠

Imagine this: You know that "king" and "queen" are related, but "king" 
and "car" are not. How does a computer understand this? The answer: 
VECTORS and COSINE SIMILARITY!

Real-world analogy:
Think of words as arrows pointing in different directions in space.
Similar words point in similar directions. We measure how "aligned" 
two arrows are to determine similarity.

What you'll learn:
✓ Vector representations of meaning
✓ Cosine similarity mathematics
✓ Why AI uses geometry to understand language
"""

import math
from typing import List

class SemanticSimilarity:
    """
    A similarity calculator that measures how "close" two vectors are.
    
    In real AI systems, words are converted to vectors (lists of numbers).
    Words with similar meanings have similar vectors!
    
    For example (simplified):
    - "king" might be [0.8, 0.6, 0.2, ...]
    - "queen" might be [0.7, 0.5, 0.3, ...]
    - "car" might be [0.1, 0.2, 0.9, ...]
    
    Notice how king and queen have similar numbers? That's the magic! ✨
    """
    
    def __init__(self):
        """Initialize our similarity calculator"""
        print("✅ Semantic Similarity Calculator ready!")
    
    def cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """
        Calculate cosine similarity between two vectors.
        
        🎯 What is cosine similarity?
        It measures the angle between two vectors. The smaller the angle,
        the more similar they are!
        
        The formula:
        similarity = (A · B) / (||A|| × ||B||)
        
        Where:
        - A · B is the dot product (multiply corresponding numbers and sum)
        - ||A|| is the magnitude (length) of vector A
        - ||B|| is the magnitude (length) of vector B
        
        Returns:
            float: A score between -1 and 1
                   1.0  = Identical (same direction)
                   0.5+ = Very similar
                   0.0  = Completely different (perpendicular)
                  -1.0  = Opposite
        
        Example:
            >>> sim = SemanticSimilarity()
            >>> vec_cat = [0.8, 0.6]
            >>> vec_dog = [0.7, 0.5]
            >>> sim.cosine_similarity(vec_cat, vec_dog)
            0.996  # Very similar! Both are pets
        """
        # Ensure vectors are the same length
        if len(vec1) != len(vec2):
            raise ValueError(f"Vectors must be same length! Got {len(vec1)} and {len(vec2)}")
        
        # Step 1: Calculate dot product (A · B)
        # Multiply corresponding elements and sum them up
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        
        # Step 2: Calculate magnitude of vec1 (||A||)
        # Square each element, sum them, then take square root
        magnitude1 = math.sqrt(sum(a * a for a in vec1))
        
        # Step 3: Calculate magnitude of vec2 (||B||)
        magnitude2 = math.sqrt(sum(b * b for b in vec2))
        
        # Handle edge case: zero vectors
        if magnitude1 == 0 or magnitude2 == 0:
            return 0.0
        
        # Step 4: Calculate cosine similarity
        similarity = dot_product / (magnitude1 * magnitude2)
        
        return similarity
    
    def interpret_similarity(self, score: float) -> str:
        """
        Convert similarity score to human-readable interpretation.
        
        This helps you understand what the numbers mean!
        """
        if score >= 0.9:
            return "Nearly identical! 🎯"
        elif score >= 0.7:
            return "Very similar 👍"
        elif score >= 0.5:
            return "Somewhat similar 🤔"
        elif score >= 0.3:
            return "A bit related 🤷"
        else:
            return "Quite different 🔀"
    
    def compare_multiple(self, base_vec: List[float], compare_vecs: dict) -> dict:
        """
        Compare one vector against multiple others.
        
        Useful for finding the most similar item!
        
        Args:
            base_vec: The reference vector
            compare_vecs: Dict of {name: vector} to compare against
        
        Returns:
            Dict of {name: similarity_score} sorted by similarity
        """
        results = {}
        
        for name, vec in compare_vecs.items():
            similarity = self.cosine_similarity(base_vec, vec)
            results[name] = similarity
        
        # Sort by similarity (highest first)
        sorted_results = dict(sorted(results.items(), key=lambda x: x[1], reverse=True))
        
        return sorted_results


# ============================================================================
# DEMO: Let's explore similarity with real examples! 🚀
# ============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("SEMANTIC SIMILARITY DEMO - Understanding Meaning Through Math!")
    print("=" * 70 + "\n")
    
    sim = SemanticSimilarity()
    
    # In real AI, these would be generated by language models
    # We're using simple 2D vectors for easy visualization
    print("📊 Our simplified vector space (2D for easy understanding):\n")
    
    # Royalty cluster
    vec_king = [0.8, 0.6]
    vec_queen = [0.7, 0.5]
    
    # Vehicle cluster
    vec_car = [0.2, 0.9]
    vec_truck = [0.3, 0.8]
    
    # Animal cluster  
    vec_dog = [0.6, 0.3]
    vec_cat = [0.5, 0.2]
    
    print("Royalty:")
    print(f"  👑 King:  {vec_king}")
    print(f"  👑 Queen: {vec_queen}\n")
    
    print("Vehicles:")
    print(f"  🚗 Car:   {vec_car}")
    print(f"  🚚 Truck: {vec_truck}\n")
    
    print("Animals:")
    print(f"  🐕 Dog:   {vec_dog}")
    print(f"  🐈 Cat:   {vec_cat}\n")
    
    print("-" * 70)
    print("🧪 EXPERIMENT 1: Comparing within categories")
    print("-" * 70 + "\n")
    
    # Similar concepts should have high similarity
    similarity_king_queen = sim.cosine_similarity(vec_king, vec_queen)
    similarity_car_truck = sim.cosine_similarity(vec_car, vec_truck)
    similarity_dog_cat = sim.cosine_similarity(vec_dog, vec_cat)
    
    print(f"King 👑 <-> Queen 👑:  {similarity_king_queen:.3f} - {sim.interpret_similarity(similarity_king_queen)}")
    print(f"Car 🚗 <-> Truck 🚚:   {similarity_car_truck:.3f} - {sim.interpret_similarity(similarity_car_truck)}")
    print(f"Dog 🐕 <-> Cat 🐈:     {similarity_dog_cat:.3f} - {sim.interpret_similarity(similarity_dog_cat)}")
    
    print("\n" + "-" * 70)
    print("🧪 EXPERIMENT 2: Comparing across categories")
    print("-" * 70 + "\n")
    
    # Different concepts should have lower similarity
    similarity_king_car = sim.cosine_similarity(vec_king, vec_car)
    similarity_queen_dog = sim.cosine_similarity(vec_queen, vec_dog)
    similarity_car_cat = sim.cosine_similarity(vec_car, vec_cat)
    
    print(f"King 👑 <-> Car 🚗:    {similarity_king_car:.3f} - {sim.interpret_similarity(similarity_king_car)}")
    print(f"Queen 👑 <-> Dog 🐕:   {similarity_queen_dog:.3f} - {sim.interpret_similarity(similarity_queen_dog)}")
    print(f"Car 🚗 <-> Cat 🐈:     {similarity_car_cat:.3f} - {sim.interpret_similarity(similarity_car_cat)}")
    
    print("\n" + "-" * 70)
    print("🧪 EXPERIMENT 3: Finding most similar words")
    print("-" * 70 + "\n")
    
    # What's most similar to "king"?
    comparisons = {
        "queen": vec_queen,
        "car": vec_car,
        "dog": vec_dog,
        "truck": vec_truck
    }
    
    results = sim.compare_multiple(vec_king, comparisons)
    
    print("What words are most similar to 'King'? 👑\n")
    for word, score in results.items():
        print(f"  {word:10} → {score:.3f} - {sim.interpret_similarity(score)}")
    
    print("\n" + "=" * 70)
    print("💡 KEY TAKEAWAY:")
    print("=" * 70)
    print("""
This is how AI understands meaning! Real language models use vectors 
with 768 or even 1536 dimensions (not just 2), but the concept is the same:
- Similar meanings = Similar vectors = High cosine similarity
- Different meanings = Different vectors = Low cosine similarity

This is the foundation of:
✓ Search engines finding relevant documents
✓ Recommendation systems suggesting similar items  
✓ Chatbots understanding your questions
✓ Translation systems matching meanings across languages
    """)
    
    print("\n" + "=" * 70)
    print("✨ Excellent work! You've mastered semantic similarity!")
    print("Next up: FAQ Finder - Putting it all together!")
    print("=" * 70 + "\n")