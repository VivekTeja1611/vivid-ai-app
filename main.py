import os
import time
import hashlib
from pathlib import Path
from typing import List, Tuple, Optional
from dotenv import load_dotenv
import streamlit as st
from streamlit_chat import message
from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document
from together_chat import TogetherAIChat
# Load environment variables
load_dotenv()
# Configuration
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
VECTOR_PATH = "vector_store"
DOCS_PATH = "Vivid"
CHUNK_SIZE = 1200
CHUNK_OVERLAP = 150
MAX_CONTEXT_LENGTH = 8000
RETRIEVAL_K = 6
# Initialize components
@st.cache_resource
def get_embedding_model():
    """Cache the embedding model to avoid reloading."""
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
@st.cache_resource
def get_chat_model():
    """Cache the chat model."""
    return TogetherAIChat()
@st.cache_data
def get_directory_hash(directory: str) -> str:
    """Generate hash of directory contents to detect changes."""
    hash_md5 = hashlib.md5()
    try:
        for root, dirs, files in os.walk(directory):
            for file in sorted(files):
                if file.endswith('.py'):
                    filepath = os.path.join(root, file)
                    try:
                        with open(filepath, 'rb') as f:
                            hash_md5.update(f.read())
                    except (IOError, OSError):
                        continue
    except (IOError, OSError):
        return ""
    return hash_md5.hexdigest()
@st.cache_resource
def load_or_create_vector_db():
    """Load existing vector DB or create new one with change detection."""
    embedding = get_embedding_model()
    
    # Check if vector store exists and is up to date
    current_hash = get_directory_hash(DOCS_PATH)
    hash_file = Path(VECTOR_PATH) / "content_hash.txt"
    
    if os.path.exists(VECTOR_PATH) and hash_file.exists():
        try:
            with open(hash_file, 'r') as f:
                stored_hash = f.read().strip()
            
            if stored_hash == current_hash:
                st.success("📚 Loading existing vector database...")
                return FAISS.load_local(VECTOR_PATH, embedding, allow_dangerous_deserialization=True)
        except:
            pass
    
    # Create new vector store
    st.info("🔄 Building vector database from codebase...")
    
    if not os.path.exists(DOCS_PATH):
        st.error(f"❌ Directory '{DOCS_PATH}' not found!")
        return None
    
    try:
        loader = DirectoryLoader(
            DOCS_PATH, 
            glob="**/*.py",
            show_progress=True,
            use_multithreading=True
        )
        docs = loader.load()
        
        if not docs:
            st.warning(f"⚠️ No Python files found in '{DOCS_PATH}'")
            return None
        
        # Enhanced text splitter
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            separators=["\n\nclass ", "\n\ndef ", "\n\n", "\n", " ", ""],
            keep_separator=True
        )
        
        # Add metadata to chunks
        enhanced_docs = []
        for doc in docs:
            chunks = splitter.split_documents([doc])
            for chunk in chunks:
                chunk.metadata.update({
                    'file_name': Path(doc.metadata['source']).name,
                    'file_path': doc.metadata['source'],
                    'chunk_size': len(chunk.page_content)
                })
                enhanced_docs.append(chunk)
        
        # Create vector store
        progress_bar = st.progress(0)
        st.write(f"Processing {len(enhanced_docs)} code chunks...")
        
        db = FAISS.from_documents(enhanced_docs, embedding)
        
        # Save vector store and hash
        os.makedirs(VECTOR_PATH, exist_ok=True)
        db.save_local(VECTOR_PATH)
        
        with open(hash_file, 'w') as f:
            f.write(current_hash)
        
        progress_bar.progress(1.0)
        st.success(f"✅ Vector database created with {len(enhanced_docs)} chunks!")
        
        return db
        
    except Exception as e:
        st.error(f"❌ Error creating vector database: {str(e)}")
        return None
def format_context(docs: List[Document], max_length: int = MAX_CONTEXT_LENGTH) -> str:
    """Format retrieved documents with better structure and length control."""
    context_parts = []
    current_length = 0
    
    for i, doc in enumerate(docs):
        file_info = f"📁 {doc.metadata.get('file_name', 'Unknown file')}"
        content = f"{file_info}\n```python\n{doc.page_content}\n```\n"
        
        if current_length + len(content) > max_length:
            break
            
        context_parts.append(content)
        current_length += len(content)
    
    return "\n---\n".join(context_parts)
def get_response_with_context(user_input: str, chat_history: List[Tuple[str, str]]) -> Tuple[str, List[str]]:
    """Generate response with retrieved context and source files."""
    db = st.session_state.get('vector_db')
    if not db:
        return "❌ Vector database not available. Please check your setup.", []
    
    try:
        # Retrieve relevant documents
        retriever = db.as_retriever(
            search_type="mmr",  # Maximal Marginal Relevance for diversity
            search_kwargs={
                "k": RETRIEVAL_K,
                "fetch_k": RETRIEVAL_K * 2,
                "lambda_mult": 0.7
            }
        )
        
        docs = retriever.invoke(user_input)
        
        if not docs:
            return "🤔 I couldn't find relevant code for your question. Try rephrasing or asking about specific files/functions.", []
        
        context = format_context(docs)
        source_files = list(set([doc.metadata.get('file_name', 'Unknown') for doc in docs]))
        
        # Build conversation context
        conversation_context = ""
        if chat_history:
            recent_history = chat_history[-3:]  # Last 3 exchanges
            for user_q, assistant_a in recent_history:
                conversation_context += f"Previous Q: {user_q}\nPrevious A: {assistant_a[:200]}...\n\n"
        
        # Enhanced system prompt
        system_prompt = """You are an expert code analyst and software engineer. Your role is to:
1. Analyze the provided code context carefully
2. Give clear, accurate explanations with specific examples
3. Reference exact file names and line numbers when possible
4. Provide actionable insights and suggestions
5. Use proper formatting with code blocks for better readability
6. If the context doesn't fully answer the question, clearly state what's missing
Format your response professionally with:
- Clear headings and structure
- Code examples in proper markdown
- Bullet points for lists
- **Bold** for important concepts"""
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"""
Context from codebase:
{context}
{f"Recent conversation context: {conversation_context}" if conversation_context else ""}
Current question: {user_input}
Please provide a comprehensive answer based on the code context above."""}
        ]
        
        chat_model = get_chat_model()
        response = chat_model.chat(messages)
        
        return response, source_files
        
    except Exception as e:
        return f"❌ Error generating response: {str(e)}", []
# Streamlit Configuration
st.set_page_config(
    page_title="CodeGPT Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)
# Custom CSS for modern UI
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 1rem 0;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    
    .metric-container {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #667eea;
    }
    
    .source-files {
        background: #e3f2fd;
        padding: 0.5rem;
        border-radius: 5px;
        border-left: 3px solid #2196f3;
        margin: 0.5rem 0;
    }
    
    .chat-container {
        max-height: 600px;
        overflow-y: auto;
        padding: 1rem;
        border: 1px solid #e0e0e0;
        border-radius: 10px;
        background: #fafafa;
    }
    
    .stAlert > div {
        padding: 0.5rem 1rem;
    }
    
    .stButton > button {
        width: 100%;
        border-radius: 20px;
        border: none;
        background: linear-gradient(45deg, #667eea, #764ba2);
        color: white;
        font-weight: bold;
    }
    
    .stTextInput > div > div > input {
        border-radius: 20px;
    }
</style>
""", unsafe_allow_html=True)
# Initialize session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "vector_db" not in st.session_state:
    st.session_state.vector_db = None
# Header
st.markdown("""
<div class="main-header">
    <h1>🤖 CodeGPT Assistant</h1>
    <p>Intelligent Python Codebase Explorer powered by RAG & AI</p>
</div>
""", unsafe_allow_html=True)
# Sidebar
with st.sidebar:
    st.markdown("### 📊 System Status")
    
    # Initialize vector DB
    if st.session_state.vector_db is None:
        with st.spinner("Loading vector database..."):
            st.session_state.vector_db = load_or_create_vector_db()
    
    if st.session_state.vector_db:
        st.success("✅ Vector DB Ready")
        
        # Stats
        try:
            total_vectors = st.session_state.vector_db.index.ntotal
            st.metric("📚 Code Chunks", total_vectors)
        except:
            st.metric("📚 Code Chunks", "Unknown")
    else:
        st.error("❌ Vector DB Failed")
    
    st.markdown("---")
    
    # Controls
    st.markdown("### ⚙️ Controls")
    
    if st.button("🔄 Refresh Database"):
        st.cache_resource.clear()
        st.cache_data.clear()
        st.session_state.vector_db = None
        st.rerun()
    
    if st.button("🗑️ Clear Chat History"):
        st.session_state.chat_history = []
        st.rerun()
    
    st.markdown("---")
    
    # Tips
    with st.expander("💡 Usage Tips"):
        st.markdown("""
        **Good Questions:**
        - "Explain the authentication flow"
        - "How is data validation handled?"
        - "Show me the API endpoints"
        - "What design patterns are used?"
        
        **Features:**
        - 🔍 Semantic code search
        - 📁 Multi-file context
        - 🧠 Conversation memory
        - ⚡ Smart caching
        """)
# Main content area
col1, col2 = st.columns([2, 1])
with col1:
    st.markdown("### 💬 Chat Interface")
    
    # Chat history display
    if st.session_state.chat_history:
        chat_container = st.container()
        with chat_container:
            for i, (user_msg, bot_msg) in enumerate(st.session_state.chat_history):
                message(user_msg, is_user=True, key=f"user_{i}")
                message(bot_msg, is_user=False, key=f"bot_{i}")
    else:
        st.info("👋 Ask me anything about your Python codebase!")
    
    # Input
    user_input = st.chat_input("Type your question here...", key="main_input")
with col2:
    st.markdown("### 📈 Session Info")
    
    # Metrics
    st.metric("💬 Messages", len(st.session_state.chat_history))
    
    if st.session_state.chat_history:
        st.markdown("### 📁 Recent Sources")
        # Show source files from last query (you'd need to track this)
        st.info("Source files will appear here after queries")
# Process user input
if user_input and st.session_state.vector_db:
    with st.spinner("🤔 Analyzing codebase..."):
        start_time = time.time()
        response, source_files = get_response_with_context(user_input, st.session_state.chat_history)
        response_time = time.time() - start_time
    
    # Add to chat history
    st.session_state.chat_history.append((user_input, response))
    
    # Show source files
    if source_files:
        with col2:
            st.markdown("### 📁 Sources Used")
            for file in source_files[:5]:  # Show top 5
                st.markdown(f"📄 `{file}`")
            
            st.caption(f"⏱️ Response time: {response_time:.2f}s")
    
    st.rerun()
elif user_input and not st.session_state.vector_db:
    st.error("❌ Please wait for the vector database to load before asking questions.")
# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 1rem;'>
    <small>🚀 Enhanced CodeGPT Assistant | Built with Streamlit & LangChain</small>
</div>
""", unsafe_allow_html=True)
