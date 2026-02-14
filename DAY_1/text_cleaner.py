"""
Day 1 - Exercise 1: Text Cleaning Utility
==========================================

Welcome to your first step in building an AI agent! 🎉

Think about it: When you read messy text with WEIRD capitalization, 
lots of    spaces, and special characters!!!, your brain automatically 
cleans it up to understand the meaning. We're teaching Python to do the same!

Why is this important for AI?
- AI models work better with clean, consistent text
- Removes noise that can confuse the model
- Makes text easier to compare and analyze

Learning Goals:
✓ String manipulation in Python
✓ Regular expressions (regex) basics
✓ Object-oriented programming (classes)
"""

import re
import string

class TextCleaner:
    """
    A friendly text cleaning assistant!
    
    This class helps us clean messy text, just like a helpful editor
    would clean up a rough draft before publishing.
    """
    
    def __init__(self):
        """
        Initialize our cleaner with Python's built-in punctuation list.
        
        Fun fact: string.punctuation contains: !"#$%&()*+,-./:;<=>?@[\]^_`{|}~
        """
        self.punctuation = string.punctuation
        print("✅ TextCleaner ready! Let's clean some text!")
    
    def clean_text(self, text):
        """
        Clean text: lowercase, strip whitespace, remove special characters
        
        Think of this as the "basic cleanup" - like tidying your room:
        1. Make everything lowercase (consistency is key!)
        2. Remove extra spaces (no clutter)
        3. Keep only letters, numbers, and spaces
        
        Args:
            text (str): The messy text you want to clean
            
        Returns:
            str: Sparkling clean text! ✨
        
        Example:
            >>> cleaner = TextCleaner()
            >>> cleaner.clean_text("  Hello, World!!!  ")
            'hello world'
        """
        # Step 1: Lowercase everything for consistency
        text = text.lower()
        
        # Step 2: Remove leading/trailing whitespace
        text = text.strip()
        
        # Step 3: Keep only alphanumeric characters and spaces
        # [^a-z0-9\s] means "anything that's NOT a letter, number, or space"
        text = re.sub(r'[^a-z0-9\s]', '', text)
        
        # Step 4: Replace multiple spaces with a single space
        # \s+ means "one or more whitespace characters"
        text = re.sub(r'\s+', ' ', text)
        
        return text
    
    def tokenize(self, text):
        """
        Split text into individual words (tokens).
        
        Tokenization is a fancy word for "breaking text into pieces."
        It's like separating a sentence into individual words so we can
        analyze them one by one.
        
        Args:
            text (str): Text to split into words
            
        Returns:
            list: A list of clean words
        
        Example:
            >>> cleaner.tokenize("Hello, wonderful world!")
            ['hello', 'wonderful', 'world']
        """
        cleaned = self.clean_text(text)
        tokens = cleaned.split()
        return tokens
    
    def get_word_count(self, text):
        """
        Count how many words are in the text.
        
        This is super useful for understanding document length!
        """
        tokens = self.tokenize(text)
        return len(tokens)


# ============================================================================
# DEMO: Let's see our TextCleaner in action! 🚀
# ============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("TEXT CLEANING DEMO - Let's clean some messy text!")
    print("=" * 70 + "\n")
    
    # Create our cleaner
    cleaner = TextCleaner()
    
    # Test with some real-world messy examples
    test_cases = [
        "  Hello, World!!!  ",
        "Email: support@gdg.dev",
        "Price: $99.99 (AMAZING Deal!!!)",
        "Python     is     AWESOME!!!",
        "Check out: https://gdg.community.dev 🚀"
    ]
    
    print("Let's clean some messy text:\n")
    
    for i, messy_text in enumerate(test_cases, 1):
        clean_text = cleaner.clean_text(messy_text)
        word_count = cleaner.get_word_count(messy_text)
        
        print(f"Example {i}:")
        print(f"  Original: '{messy_text}'")
        print(f"  Cleaned:  '{clean_text}'")
        print(f"  Words:    {word_count}")
        print()

