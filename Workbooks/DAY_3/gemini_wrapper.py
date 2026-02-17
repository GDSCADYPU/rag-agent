"""
Gemini API Wrapper for RAG Systems - WORKBOOK 🎓
=================================================
Fill in every blank marked  ___  to complete the code.
Run demo() at the bottom to test your work!

HOW TO USE THIS WORKBOOK
-------------------------
  • Find every  ___  blank and replace it with the correct value
  • Read the  # ✏️ FILL IN  comment above each blank — it tells you what to write
  • Read the  # 💡 HINT  comment for extra guidance
  • Run the file after each section to catch mistakes early!
===================================================

A clean, production-ready wrapper for Google's Gemini AI API.
Designed specifically for building Retrieval-Augmented Generation (RAG) systems.

Author: GDG Workshop Team
License: MIT
"""

import google.genai as genai
from typing import Optional, List, Dict
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class GeminiWrapper:
    """
    Simple, robust wrapper for Google's Gemini AI API.

    Features:
    - Easy initialization with API key management
    - Configurable temperature and model selection
    - System persona support for consistent responses
    - Conversation history tracking
    - Built-in error handling
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: str = "gemini-2.5-flash",
        temperature: float = 0.7,
        verbose: bool = True
    ):
        """
        Initialize the Gemini wrapper.

        Args:
            api_key: Gemini API key. If None, reads from GEMINI_API_KEY env variable
            model_name: Gemini model to use (default: 'gemini-2.5-flash')
            temperature: Response randomness (0.0=deterministic, 1.0=creative)
            verbose: Whether to print initialization messages

        Raises:
            ValueError: If no API key is provided or found in environment
        """
        # Get API key
        # ✏️ FILL IN: Replace ___ so it uses api_key if given, otherwise reads GEMINI_API_KEY from environment
        # 💡 HINT: use "or"  →  self.api_key = api_key or os.getenv('GEMINI_API_KEY')
        self.api_key = ___
        # ✏️ FILL IN: Replace ___ with the condition that checks the key is missing
        # 💡 HINT: not self.api_key  → True when api_key is None or empty
        if ___:
            raise ValueError(
                "No Gemini API key provided.\n"
                "Either pass api_key parameter or set GEMINI_API_KEY environment variable.\n"
                "Get your key at: https://makersuite.google.com/app/apikey"
            )

        # Configure Gemini client (google.genai)
        # google.genai uses a Client; there is no global configure()
        # ✏️ FILL IN: Replace ___ with genai.Client(api_key=self.api_key)
        # 💡 HINT: This creates a connection to the Gemini service
        self.client = ___

        # Store configuration
        self.model_name = model_name
        self.temperature = temperature
        self.verbose = verbose

        # Initialize tracking
        self.history = []
        self.persona = None

        if self.verbose:
            print(f"✅ Gemini initialized: {model_name} (temp={temperature})")

    def set_persona(self, persona_description: str) -> None:
        """
        Set the AI's system persona/role.

        This defines how the AI should behave and respond. Useful for:
        - Setting response style (formal, casual, technical)
        - Defining expertise area
        - Enforcing response guidelines (e.g., "always cite sources")

        Args:
            persona_description: Description of the AI's role and behavior
        """
        # ✏️ FILL IN: Replace ___ to save the persona description
        # 💡 HINT: self.persona = persona_description
        self.persona = ___
        # ✏️ FILL IN: Replace ___ with the condition to check if we should print
        # 💡 HINT: self.verbose
        if ___:
            preview = persona_description[:80] + "..." if len(persona_description) > 80 else persona_description
            print(f"✅ Persona set: {preview}")

    def generate(
        self,
        prompt: str,
        temperature: Optional[float] = None,
        max_tokens: int = 2048
    ) -> str:
        """
        Generate a response from Gemini.

        Args:
            prompt: The input prompt/question
            temperature: Override default temperature for this request
            max_tokens: Maximum response length (default: 2048)

        Returns:
            Generated text response

        Raises:
            Exception: If API call fails (returns error message as string)
        """
        # Build full prompt with persona if set
        # ✏️ FILL IN: Replace ___ with the condition — do we have a persona?
        # 💡 HINT: full_prompt = f"SYSTEM: {self.persona}\n\nUSER: {prompt}" if self.persona else prompt
        full_prompt = f"SYSTEM: {self.persona}\n\nUSER: {prompt}" if ___ else prompt
        # ✏️ FILL IN: Replace ___ — use the passed-in temperature, or fall back to self.temperature
        # 💡 HINT: temperature if temperature is not None else self.temperature
        temp = ___

        try:
            # google.genai generation call (use generate_content)
            resp = self.client.models.generate_content(
                model=self.model_name,
                contents=full_prompt,
                config={
                    "temperature": temp,
                    "max_output_tokens": max_tokens,
                    "top_p": 0.95,
                    "top_k": 40,
                },
            )

            # Extract text
            text = ""
            if hasattr(resp, "text") and isinstance(resp.text, str):
                text = resp.text
            elif hasattr(resp, "candidates") and resp.candidates:
                for cand in resp.candidates:
                    if getattr(cand, "content", None):
                        parts = getattr(cand.content, "parts", [])
                        for p in parts:
                            if getattr(p, "text", None):
                                text += p.text
                text = text.strip()

            # Track in history
            # ✏️ FILL IN: Fill in the 4 values inside this dictionary
            # 💡 HINT: keys are 'prompt', 'response', 'temperature', 'model'
            #          values are: prompt, text, temp, self.model_name
            self.history.append({
                'prompt': ___,
                'response': ___,
                'temperature': ___,
                'model': ___
            })

            return text or ""
        except Exception as e:
            error_msg = f"Error calling Gemini API: {str(e)}"
            if self.verbose:
                print(f"❌ {error_msg}")
            return error_msg

    def chat(self, message: str) -> str:
        """
        Simple chat using a running transcript.
        """
        # Maintain a naive chat transcript for context
        if not hasattr(self, '_chat_transcript'):
            self._chat_transcript = []

        # ✏️ FILL IN: Replace ___ with the message the user sent
        # 💡 HINT: self._chat_transcript.append({"role": "user", "text": message})
        self._chat_transcript.append({"role": "user", "text": ___})
        # Build a chat-style prompt
        convo = []
        if self.persona:
            convo.append(f"SYSTEM: {self.persona}")
        for turn in self._chat_transcript[-10:]:  # last 10 turns
            prefix = "USER" if turn["role"] == "user" else "ASSISTANT"
            convo.append(f"{prefix}: {turn['text']}")
        convo.append("ASSISTANT:")
        prompt = "\n\n".join(convo)

        # ✏️ FILL IN: Replace ___ with the prompt variable built above
        # 💡 HINT: self.generate(prompt)
        reply = self.generate(___)
        # ✏️ FILL IN: Replace ___ with the reply we just got
        # 💡 HINT: self._chat_transcript.append({"role": "assistant", "text": reply})
        self._chat_transcript.append({"role": "assistant", "text": ___})
        return reply

    def clear_history(self) -> None:
        """
        Clear conversation history and reset chat session.

        Useful when starting a new conversation topic.
        """
        # ✏️ FILL IN: Replace ___ with an empty list to clear history
        # 💡 HINT: self.history = []
        self.history = ___
        if hasattr(self, '_chat_transcript'):
            self._chat_transcript = []
        if self.verbose:
            print("✅ History cleared")

    def get_history(self) -> List[Dict]:
        """
        Get the conversation history.

        Returns:
            List of dictionaries containing prompt, response, temperature, and model
        """
        # ✏️ FILL IN: Replace ___ with the history list
        # 💡 HINT: return self.history
        return ___

    def get_stats(self) -> Dict:
        """
        Get wrapper statistics.

        Returns:
            Dictionary with model info and usage stats
        """
        # ✏️ FILL IN: Fill in all 4 values in this dictionary
        # 💡 HINT: model=self.model_name, temperature=self.temperature,
        #          total_interactions=len(self.history), has_persona=self.persona is not None
        return {
            'model': ___,
            'temperature': ___,
            'total_interactions': ___,
            'has_persona': ___
        }


# ============================================================================
# Demo & Testing
# ============================================================================

def demo():
    """Run a simple demo of the Gemini wrapper."""
    print("\n" + "="*70)
    print("GEMINI WRAPPER DEMO")
    print("="*70 + "\n")

    try:
        # Initialize
        llm = GeminiWrapper(temperature=0.7)

        # Basic generation
        print("1. Basic Generation")
        print("-" * 70)
        response = llm.generate("What is Python in one sentence?")
        print(f"Q: What is Python in one sentence?")
        print(f"A: {response}\n")

        # With persona
        print("2. With Persona")
        print("-" * 70)
        llm.set_persona(
            "You are a helpful teacher who explains concepts using simple analogies."
        )
        response = llm.generate("What is machine learning?")
        print(f"Q: What is machine learning?")
        print(f"A: {response}\n")

        # Chat mode
        print("3. Chat Mode (Multi-turn)")
        print("-" * 70)
        print("User: My favorite color is blue")
        r1 = llm.chat("My favorite color is blue")
        print(f"AI: {r1}\n")

        print("User: What's my favorite color?")
        r2 = llm.chat("What's my favorite color?")
        print(f"AI: {r2}\n")

    except ValueError as e:
        print(f"\n❌ Error: {e}\n")
        print("Setup Instructions:")
        print("1. Get API key: https://makersuite.google.com/app/apikey")
        print("2. Create .env file with: GEMINI_API_KEY=your_key_here")
        print("3. Run again\n")


if __name__ == "__main__":
    demo()