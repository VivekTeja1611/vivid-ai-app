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
CHUNK_SIZE = 800  # Reduced for better granularity
CHUNK_OVERLAP = 100  # Reduced overlap
MAX_CONTEXT_LENGTH = 12000  # Increased context length
RETRIEVAL_K = 8  # Increased retrieval count

# File types to include in vector store
SUPPORTED_EXTENSIONS = {
    # Code files
    '.py': 'python',
    '.js': 'javascript', 
    '.jsx': 'javascript',
    '.ts': 'typescript',
    '.tsx': 'typescript',
    '.java': 'java',
    '.cpp': 'cpp',
    '.c': 'c',
    '.h': 'c',
    '.cs': 'csharp',
    '.php': 'php',
    '.rb': 'ruby',
    '.go': 'go',
    '.rs': 'rust',
    '.swift': 'swift',
    '.kt': 'kotlin',
    '.scala': 'scala',
    '.r': 'r',
    '.sql': 'sql',
    '.sh': 'bash',
    '.bat': 'batch',
    '.ps1': 'powershell',
    
    # Documentation files
    '.md': 'markdown',
    '.txt': 'text',
    '.rst': 'restructuredtext',
    '.tex': 'latex',
    '.org': 'org',
    
    # Configuration files
    '.json': 'json',
    '.yaml': 'yaml',
    '.yml': 'yaml',
    '.toml': 'toml',
    '.ini': 'ini',
    '.cfg': 'ini',
    '.conf': 'ini',
    '.xml': 'xml',
    '.html': 'html',
    '.htm': 'html',
    '.css': 'css',
    '.scss': 'scss',
    '.sass': 'sass',
    '.less': 'less',
    
    # Data files
    '.csv': 'csv',
    '.tsv': 'csv',
    
    # Docker and deployment
    'dockerfile': 'dockerfile',
    '.dockerfile': 'dockerfile',
    'docker-compose.yml': 'yaml',
    'docker-compose.yaml': 'yaml',
    
    # Other important files
    'makefile': 'makefile',
    'cmakelist.txt': 'cmake',
    'requirements.txt': 'text',
    'pipfile': 'toml',
    'poetry.lock': 'toml',
    'package.json': 'json',
    'yarn.lock': 'text',
    'gemfile': 'ruby',
    'cargo.toml': 'toml',
    '.gitignore': 'text',
    '.env': 'text',
    'license': 'text',
    'readme': 'text',
    'changelog': 'text',
    'contributing': 'text',
    'authors': 'text',
    'contributors': 'text'
}

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

def is_supported_file(filename: str) -> bool:
    """Check if file is supported based on extension or special names."""
    filename_lower = filename.lower()
    
    # Check special filenames (README, LICENSE, etc.)
    special_names = ['readme', 'license', 'changelog', 'contributing', 'authors', 
                    'contributors', 'makefile', 'dockerfile', 'pipfile', 'gemfile']
    
    for special in special_names:
        if filename_lower.startswith(special):
            return True
    
    # Check extensions
    for ext in SUPPORTED_EXTENSIONS:
        if filename_lower.endswith(ext):
            return True
    
    # Check files without extensions that might be important
    if '.' not in filename and filename_lower in ['dockerfile', 'makefile', 'license', 'readme']:
        return True
    
    return False

def get_file_language(filename: str) -> str:
    """Get the language/type of a file for better processing."""
    filename_lower = filename.lower()
    
    # Check special filenames first
    special_mappings = {
        'readme': 'markdown',
        'license': 'text', 
        'changelog': 'markdown',
        'contributing': 'markdown',
        'authors': 'text',
        'contributors': 'text',
        'makefile': 'makefile',
        'dockerfile': 'dockerfile',
        'pipfile': 'toml',
        'gemfile': 'ruby'
    }
    
    for special, lang in special_mappings.items():
        if filename_lower.startswith(special):
            return lang
    
    # Check extensions
    for ext, lang in SUPPORTED_EXTENSIONS.items():
        if filename_lower.endswith(ext):
            return lang
    
    return 'text'

@st.cache_data
def get_directory_hash(directory: str) -> str:
    """Generate hash of directory contents to detect changes."""
    hash_md5 = hashlib.md5()
    try:
        for root, dirs, files in os.walk(directory):
            # Skip hidden directories and common ignore patterns
            dirs[:] = [d for d in dirs if not d.startswith('.') and d not in ['__pycache__', 'node_modules', '.git']]
            
            for file in sorted(files):
                if is_supported_file(file):
                    filepath = os.path.join(root, file)
                    try:
                        with open(filepath, 'rb') as f:
                            hash_md5.update(f.read())
                    except (IOError, OSError, UnicodeDecodeError):
                        continue
    except (IOError, OSError):
        return ""
    return hash_md5.hexdigest()

def load_documents_from_directory(directory: str) -> List[Document]:
    """Load all supported documents from directory with enhanced metadata."""
    documents = []
    
    if not os.path.exists(directory):
        st.error(f"❌ Directory '{directory}' not found!")
        return documents
    
    file_count_by_type = {}
    
    for root, dirs, files in os.walk(directory):
        # Skip hidden directories and common ignore patterns
        dirs[:] = [d for d in dirs if not d.startswith('.') and d not in ['__pycache__', 'node_modules', '.git', 'venv', 'env']]
        
        for file in files:
            if not is_supported_file(file):
                continue
                
            filepath = os.path.join(root, file)
            relative_path = os.path.relpath(filepath, directory)
            file_lang = get_file_language(file)
            
            # Count files by type for stats
            file_count_by_type[file_lang] = file_count_by_type.get(file_lang, 0) + 1
            
            try:
                # Try to read file with different encodings
                content = None
                encodings = ['utf-8', 'utf-8-sig', 'latin-1', 'cp1252']
                
                for encoding in encodings:
                    try:
                        with open(filepath, 'r', encoding=encoding) as f:
                            content = f.read()
                        break
                    except UnicodeDecodeError:
                        continue
                
                if content is None:
                    st.warning(f"⚠️ Could not read file: {relative_path}")
                    continue
                
                # Skip empty files
                if not content.strip():
                    continue
                
                # Create document with enhanced metadata
                doc = Document(
                    page_content=content,
                    metadata={
                        'source': filepath,
                        'relative_path': relative_path,
                        'file_name': file,
                        'file_type': file_lang,
                        'file_size': len(content),
                        'directory': os.path.dirname(relative_path) or 'root'
                    }
                )
                documents.append(doc)
                
            except Exception as e:
                st.warning(f"⚠️ Error reading {relative_path}: {str(e)}")
                continue
    
    # Display file type statistics
    if file_count_by_type:
        st.info(f"📁 Found files: {dict(sorted(file_count_by_type.items()))}")
    
    return documents

def get_text_splitter_for_type(file_type: str) -> RecursiveCharacterTextSplitter:
    """Get appropriate text splitter based on file type with better preservation."""
    
    # Define separators for different file types - more conservative splitting
    separators_map = {
        'python': ["\n\nclass ", "\n\ndef ", "\n\nasync def ", "\n\nif __name__", "\n\n# ", "\n\n", "\n", " "],
        'javascript': ["\n\nfunction ", "\n\nconst ", "\n\nlet ", "\n\nvar ", "\n\nclass ", "\n\n// ", "\n\n", "\n", " "],
        'typescript': ["\n\nfunction ", "\n\nconst ", "\n\nlet ", "\n\nvar ", "\n\nclass ", "\n\ninterface ", "\n\ntype ", "\n\n// ", "\n\n", "\n", " "],
        'java': ["\n\npublic class ", "\n\nclass ", "\n\npublic ", "\n\nprivate ", "\n\nprotected ", "\n\n// ", "\n\n", "\n", " "],
        'markdown': ["\n\n## ", "\n\n### ", "\n\n#### ", "\n\n- ", "\n\n* ", "\n\n", "\n", " "],
        'text': ["\n\n", "\n", ". ", " "],
        'json': ["},\n", "}\n", "\n", " "],
        'yaml': ["\n\n", "\n- ", "\n", " "],
        'html': ["\n\n<div", "\n\n<section", "\n\n<article", "\n\n", "\n", " "],
        'css': ["\n\n.", "\n\n#", "\n\n@", "\n\n", "\n", " "]
    }
    
    separators = separators_map.get(file_type, ["\n\n", "\n", ". ", " "])
    
    return RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=separators,
        keep_separator=True,
        length_function=len
    )

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
        # Load all supported documents
        docs = load_documents_from_directory(DOCS_PATH)
        
        if not docs:
            st.warning(f"⚠️ No supported files found in '{DOCS_PATH}'")
            return None
        
        st.success(f"📚 Loaded {len(docs)} files from codebase")
        
        # Process documents by file type for better chunking
        enhanced_docs = []
        file_type_counts = {}
        
        for doc in docs:
            file_type = doc.metadata.get('file_type', 'text')
            file_type_counts[file_type] = file_type_counts.get(file_type, 0) + 1
            
            # Get appropriate splitter for this file type
            splitter = get_text_splitter_for_type(file_type)
            chunks = splitter.split_documents([doc])
            
            # Enhance chunk metadata
            for i, chunk in enumerate(chunks):
                chunk.metadata.update({
                    'chunk_index': i,
                    'total_chunks': len(chunks),
                    'chunk_size': len(chunk.page_content),
                    'file_type': file_type
                })
                enhanced_docs.append(chunk)
        
        # Display processing statistics
        st.info(f"📊 Processing {len(enhanced_docs)} chunks from {len(docs)} files")
        
        # Show file type breakdown
        type_breakdown = ", ".join([f"{k}: {v}" for k, v in sorted(file_type_counts.items())])
        st.caption(f"File types: {type_breakdown}")
        
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
    """Format retrieved documents with better structure and complete content preservation."""
    context_parts = []
    current_length = 0
    
    # Sort documents by relevance score if available, then by file type priority
    file_priority = {'markdown': 0, 'text': 1, 'python': 2, 'javascript': 3, 'json': 4, 'yaml': 5}
    
    sorted_docs = sorted(docs, key=lambda x: (
        file_priority.get(x.metadata.get('file_type', 'unknown'), 10),
        -len(x.page_content)  # Prefer longer, more complete chunks
    ))
    
    for i, doc in enumerate(sorted_docs):
        file_name = doc.metadata.get('file_name', 'Unknown file')
        file_type = doc.metadata.get('file_type', 'text')
        relative_path = doc.metadata.get('relative_path', file_name)
        chunk_index = doc.metadata.get('chunk_index', 0)
        total_chunks = doc.metadata.get('total_chunks', 1)
        
        # Create comprehensive file header
        chunk_info = f" (chunk {chunk_index + 1}/{total_chunks})" if total_chunks > 1 else ""
        file_header = f"=== FILE: {relative_path} ({file_type}){chunk_info} ==="
        
        # Format content with minimal processing to preserve original structure
        content_block = f"{file_header}\n{doc.page_content}\n{'=' * len(file_header)}\n"
        
        if current_length + len(content_block) > max_length:
            # If we can't fit the whole chunk, try to fit a partial one
            remaining_space = max_length - current_length - len(file_header) - 100
            if remaining_space > 200:  # Only include if we have reasonable space
                partial_content = doc.page_content[:remaining_space] + "\n... [truncated]"
                content_block = f"{file_header}\n{partial_content}\n{'=' * len(file_header)}\n"
                context_parts.append(content_block)
            break
            
        context_parts.append(content_block)
        current_length += len(content_block)
    
    return "\n".join(context_parts)

def get_response_with_context(user_input: str, chat_history: List[Tuple[str, str]]) -> Tuple[str, List[str]]:
    """Generate response with retrieved context and source files."""
    db = st.session_state.get('vector_db')
    if not db:
        return "❌ Vector database not available. Please check your setup.", []
    
    try:
        # Enhanced retrieval with better parameters
        retriever = db.as_retriever(
            search_type="mmr",  # Maximal Marginal Relevance for diversity
            search_kwargs={
                "k": RETRIEVAL_K,
                "fetch_k": RETRIEVAL_K * 3,  # Fetch more candidates
                "lambda_mult": 0.5  # More diversity
            }
        )
        
        # Also try similarity search as backup
        docs_mmr = retriever.invoke(user_input)
        docs_similarity = db.similarity_search(user_input, k=RETRIEVAL_K//2)
        
        # Combine and deduplicate
        all_docs = docs_mmr + docs_similarity
        seen_content = set()
        unique_docs = []
        
        for doc in all_docs:
            content_hash = hashlib.md5(doc.page_content.encode()).hexdigest()
            if content_hash not in seen_content:
                seen_content.add(content_hash)
                unique_docs.append(doc)
        
        docs = unique_docs[:RETRIEVAL_K]
        
        if not docs:
            return "🤔 I couldn't find relevant code for your question. Try rephrasing or asking about specific files/functions.", []
        
        context = format_context(docs)
        source_files = []
        
        # Enhanced source file information
        for doc in docs:
            file_info = {
                'name': doc.metadata.get('file_name', 'Unknown'),
                'path': doc.metadata.get('relative_path', 'Unknown'),
                'type': doc.metadata.get('file_type', 'unknown')
            }
            if file_info not in source_files:
                source_files.append(file_info)
        
        # Build conversation context (reduced to avoid overwhelming)
        conversation_context = ""
        if chat_history:
            recent_history = chat_history[-2:]  # Last 2 exchanges only
            for user_q, assistant_a in recent_history:
                conversation_context += f"Previous Q: {user_q}\nPrevious A: {assistant_a[:150]}...\n\n"
        
        # Enhanced system prompt with strict instructions
        system_prompt = """You are a precise code analyst. Your primary job is to provide accurate information based EXACTLY on the provided code context.

CRITICAL INSTRUCTIONS:
1. Base your answers ONLY on the exact content provided in the context
2. Quote directly from the code/documentation when possible
3. If the context doesn't contain the specific information requested, clearly state this
4. Do NOT make assumptions or add information not present in the context
5. When discussing README or documentation, reference the EXACT points mentioned
6. Use the exact terminology, function names, and structure from the provided code
7. If multiple files are relevant, clearly distinguish information from each file

Response format:
- Start with the most relevant information
- Use direct quotes when appropriate: "According to the README: '[exact quote]'"
- Reference specific files: "In file `filename.py`..."
- Be precise about what the code actually does vs what you think it might do"""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"""
CODEBASE CONTEXT:
{context}

{f"RECENT CONVERSATION:\n{conversation_context}" if conversation_context else ""}

USER QUESTION: {user_input}

Provide a precise answer based strictly on the provided context. If the context doesn't fully address the question, explicitly state what information is missing."""}
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
        - "What exactly does the README say about installation?"
        - "Show me the exact authentication flow from the code"
        - "List the exact API endpoints defined"
        - "What are the specific features mentioned in documentation?"
        
        **Features:**
        - 🔍 Semantic code search
        - 📁 Multi-file context
        - 🧠 Conversation memory
        - ⚡ Smart caching
        - 🎯 Exact content matching
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
        st.info("👋 Ask me anything about your Python codebase! I'll provide exact information from your files.")
    
    # Input
    user_input = st.chat_input("Type your question here...", key="main_input")

with col2:
    st.markdown("### 📈 Session Info")
    
    # Metrics
    st.metric("💬 Messages", len(st.session_state.chat_history))
    
    if st.session_state.chat_history:
        st.markdown("### 📁 Recent Sources")
        # Show source files from last query
        if hasattr(st.session_state, 'last_sources'):
            for source in st.session_state.last_sources[:3]:
                st.markdown(f"📄 `{source['path']}`")

# Process user input
if user_input and st.session_state.vector_db:
    with st.spinner("🤔 Analyzing codebase..."):
        start_time = time.time()
        response, source_files = get_response_with_context(user_input, st.session_state.chat_history)
        response_time = time.time() - start_time
    
    # Add to chat history
    st.session_state.chat_history.append((user_input, response))
    st.session_state.last_sources = source_files
    
    # Show source files
    if source_files:
        with col2:
            st.markdown("### 📁 Sources Used")
            for file in source_files[:5]:  # Show top 5
                st.markdown(f"📄 `{file['path']}` ({file['type']})")
            
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
