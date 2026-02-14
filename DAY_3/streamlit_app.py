"""
Day 3 - Exercise 3: Streamlit Web Interface
===========================================

🌟 The Grand Finale - Share Your Creation with the World! 🌟

What is Streamlit?
Streamlit is a Python library that turns your code into beautiful,
interactive web apps in minutes - no HTML, CSS, or JavaScript needed!

Think of it as: "Python scripts → Beautiful web apps"

Why Streamlit for AI apps?
✅ Dead simple to use
✅ Perfect for data science and ML demos
✅ Real-time interactivity
✅ Free hosting available
✅ Looks professional out of the box

What you'll learn:
✓ Building web UIs with Streamlit
✓ Chat interfaces
✓ File uploads
✓ Session state management
✓ Deploying AI applications

Real-world use:
This is how many AI startups build their MVPs!
"""

import streamlit as st
import sys
import os

# Add project root to sys.path so package imports like DAY_2.knowledge_base work
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Try standard package imports first; fallback to direct module import if needed
try:
    from DAY_3.rag_agent import RAGAgent
    from DAY_2.knowledge_base import KnowledgeBase
except ModuleNotFoundError:
    # Fallback for environments running the file directly where packages aren't resolved
    sys.path.insert(0, os.path.join(project_root, 'DAY_2'))
    sys.path.insert(0, os.path.join(project_root, 'DAY_3'))
    from rag_agent import RAGAgent
    from knowledge_base import KnowledgeBase


def init_session_state():
    """
    Initialize Streamlit session state.
    
    Session state = variables that persist across page reloads.
    Think of it as the app's memory!
    
    We track:
    - RAG agent instance
    - Chat message history
    - Knowledge base instance
    """
    if 'agent' not in st.session_state:
        st.session_state.agent = None
    
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    
    if 'kb' not in st.session_state:
        st.session_state.kb = None


def main():
    """
    Main application function.
    
    This is where we build our beautiful web interface!
    """
    
    # Configure the page
    st.set_page_config(
        page_title="GDG Knowledge Agent",
        page_icon="🤖",
        layout="wide",  # Use full screen width
        initial_sidebar_state="expanded"
    )
    
    # Initialize session state
    init_session_state()
    
    # =================================================================
    # HEADER
    # =================================================================
    
    st.title("🤖 GDG Knowledge Agent")
    st.markdown("*Powered by Retrieval-Augmented Generation (RAG) with Gemini AI*")
    st.markdown("---")
    
    # =================================================================
    # SIDEBAR - Configuration & Setup
    # =================================================================
    
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # API Key input
        api_key = st.text_input(
            "Gemini API Key",
            type="password",
            help="Get your free key from https://makersuite.google.com/app/apikey",
            placeholder="Enter your API key here..."
        )
        
        # Temperature slider
        temperature = st.slider(
            "Response Temperature",
            min_value=0.0,
            max_value=1.0,
            value=0.3,
            step=0.1,
            help="Lower = more factual, Higher = more creative. For RAG, keep it low!"
        )
        
        st.markdown("**💡 Tip:** For factual Q&A, use temperature ≤ 0.3")
        
        # Initialize button
        if st.button("🚀 Initialize Agent", type="primary", use_container_width=True):
            if not api_key:
                st.error("⚠️ Please provide your Gemini API key!")
            else:
                with st.spinner("Initializing RAG Agent... This may take a moment..."):
                    try:
                        # Create knowledge base
                        st.session_state.kb = KnowledgeBase("gdg_streamlit_agent")
                        
                        # Add sample data
                        sample_data = """
                        GDG (Google Developer Groups) events are completely free for all students.
                        Registration is done through gdg.community.dev website.
                        
                        Workshop Schedule:
                        - Day 1: Python Basics and NLP (9 AM - 5 PM)
                        - Day 2: Vector Databases (9 AM - 5 PM)
                        - Day 3: RAG Systems with Gemini (9 AM - 5 PM)
                        
                        What to bring:
                        - Laptop with Python 3.8+
                        - Charger
                        - Enthusiasm to learn!
                        
                        Lunch is provided at 12:30 PM each day.
                        Coffee and snacks available throughout.
                        WiFi and power outlets at all seats.
                        
                        Certificates provided upon completion.
                        """
                        
                        st.session_state.kb.add_document(
                            sample_data,
                            metadata={'source': 'GDG Workshop Guide', 'type': 'official'}
                        )
                        
                        # Initialize RAG agent
                        st.session_state.agent = RAGAgent(
                            gemini_api_key=api_key,
                            knowledge_base=st.session_state.kb,
                            temperature=temperature
                        )
                        
                        st.success("✅ Agent initialized successfully!")
                        st.balloons()  # Celebration! 🎉
                        
                    except Exception as e:
                        st.error(f"❌ Error: {str(e)}")
        
        st.markdown("---")
        
        # Document upload section
        st.header("📄 Upload Documents")
        
        uploaded_files = st.file_uploader(
            "Upload text files to expand knowledge base",
            accept_multiple_files=True,
            type=['txt', 'md'],
            help="Upload .txt or .md files containing information you want the agent to learn"
        )
        
        if uploaded_files and st.button("Process Documents", use_container_width=True):
            if st.session_state.kb is None:
                st.error("⚠️ Please initialize the agent first!")
            else:
                with st.spinner(f"Processing {len(uploaded_files)} files..."):
                    try:
                        for file in uploaded_files:
                            # Read file content
                            text = file.read().decode('utf-8')
                            
                            # Add to knowledge base
                            st.session_state.kb.add_document(
                                text,
                                metadata={'source': file.name, 'type': 'user-uploaded'}
                            )
                        
                        st.success(f"✅ Processed {len(uploaded_files)} documents successfully!")
                    
                    except Exception as e:
                        st.error(f"❌ Error processing files: {str(e)}")
        
        st.markdown("---")
        
        # Statistics section
        if st.session_state.kb:
            st.header("📊 Knowledge Base Stats")
            stats = st.session_state.kb.get_stats()
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Total Chunks", stats['total_chunks'])
            
            with col2:
                st.metric("Embedding Dim", stats['embedding_dimension'])
            
            st.caption(f"Model: {stats['embedding_model']}")
        
        st.markdown("---")
        
        # Help section
        with st.expander("ℹ️ How to Use"):
            st.markdown("""
            **Getting Started:**
            1. Enter your Gemini API key
            2. Click "Initialize Agent"
            3. Start asking questions!
            
            **Tips:**
            - Be specific in your questions
            - Upload relevant documents for better answers
            - Check sources to verify information
            
            **Example Questions:**
            - How do I register?
            - What's the schedule?
            - Is there a fee?
            """)
    
    # =================================================================
    # MAIN AREA - Chat Interface
    # =================================================================
    
    if st.session_state.agent is None:
        # Show welcome screen before initialization
        st.info("👈 Please configure and initialize the agent in the sidebar to begin")

        st.markdown("## Ask Questions about GDG here once the agent is ready!")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("""
            ### 🔍 Retrieval
            Search your knowledge base for relevant information
            """)

        with col2:
            st.markdown("""
            ### 🔗 Augmented
            Combine retrieved info with AI
            """)

        with col3:
            st.markdown("""
            ### 💬 Generation
            Create accurate, sourced answers
            """)
        
        st.markdown("---")
        
        st.markdown("""
        ### ✨ Benefits of RAG
        
        - **✅ Up-to-date:** No need to retrain models
        - **✅ Accurate:** Cites real sources
        - **✅ Transparent:** Shows where info comes from
        - **✅ Cost-effective:** Works with any LLM
        
        ### 🚀 How It Works
        
        1. **Upload** your documents
        2. **Ask** questions naturally
        3. **Receive** accurate answers with sources
        4. **Verify** information from citations
        """)
        
    else:
        # Chat interface
        st.header("💬 Ask Me Anything!")
        
        # Display chat history
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                
                # Show sources for assistant messages
                if message["role"] == "assistant" and "sources" in message:
                    if message["sources"]:
                        with st.expander(f"📚 View {len(message['sources'])} Sources"):
                            for i, source in enumerate(message['sources'], 1):
                                similarity = source.get('similarity', 0) * 100
                                
                                st.markdown(f"**Source {i}:** {source['metadata'].get('source', 'Unknown')}")
                                st.caption(f"Relevance: {similarity:.1f}%")
                                st.text(source['text'][:200] + "...")
                                st.markdown("---")
        
        # Chat input
        if prompt := st.chat_input("Ask about GDG events, workshops, or anything in the knowledge base..."):
            # Add user message to chat
            st.session_state.messages.append({
                "role": "user",
                "content": prompt
            })
            
            # Display user message
            with st.chat_message("user"):
                st.markdown(prompt)
            
            # Generate and display assistant response
            with st.chat_message("assistant"):
                with st.spinner("🤔 Thinking..."):
                    # Get RAG response
                    result = st.session_state.agent.answer(prompt, verbose=False)
                    
                    # Display answer
                    st.markdown(result['answer'])
                    
                    # Display sources
                    if result['sources']:
                        with st.expander(f"📚 View {len(result['sources'])} Sources"):
                            for i, source in enumerate(result['sources'], 1):
                                similarity = source.get('similarity', 0) * 100
                                
                                st.markdown(f"**Source {i}:** {source['metadata'].get('source', 'Unknown')}")
                                st.caption(f"Relevance: {similarity:.1f}%")
                                st.text(source['text'][:200] + "...")
                                st.markdown("---")
                    else:
                        st.caption("ℹ️ No sources found in knowledge base")
            
            # Add assistant message to chat history
            st.session_state.messages.append({
                "role": "assistant",
                "content": result['answer'],
                "sources": result['sources']
            })
        
        # Clear chat button
        if st.session_state.messages:
            if st.button("🗑️ Clear Chat History"):
                st.session_state.messages = []
                st.rerun()


# =================================================================
# RUN THE APP
# =================================================================

if __name__ == "__main__":
    main()