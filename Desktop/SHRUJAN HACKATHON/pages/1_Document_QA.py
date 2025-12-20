import streamlit as st
import time
from utils.rag_qa import process_documents, get_answer

# Initialize theme in session state
if 'dark_mode' not in st.session_state:
    st.session_state.dark_mode = False

# ================================
# SESSION STATE INITIALIZATION
# ================================
if 'uploaded_files' not in st.session_state:
    st.session_state.uploaded_files = []
if 'documents_processed' not in st.session_state:
    st.session_state.documents_processed = False
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'current_answer' not in st.session_state:
    st.session_state.current_answer = None
if 'current_sources' not in st.session_state:
    st.session_state.current_sources = []
if 'current_retrieved_chunks' not in st.session_state:
    st.session_state.current_retrieved_chunks = []
if 'show_debug_view' not in st.session_state:
    st.session_state.show_debug_view = False
if 'processing_error' not in st.session_state:
    st.session_state.processing_error = None



# ================================
# MAIN PAGE CONTROLS
# ================================

# Page Header
st.markdown("""
    <div class="section-header">
        <h1 style="margin: 0; font-size: 2.5em; display: flex; align-items: center;">
            <span style="margin-right: 15px;">🤖</span>
            Document Q&A
        </h1>
        <p style="margin: 10px 0 0 0; opacity: 0.9; font-size: 1.1em;">Ask questions about your documents with AI-powered answers</p>
    </div>
""", unsafe_allow_html=True)

# Theme Toggle at the top
col1, col2, col3 = st.columns([1, 1, 1])
with col2:
    theme_icon = "🌙" if not st.session_state.dark_mode else "☀️"
    if st.button(f"{theme_icon} Toggle Theme", use_container_width=True, help="Switch between light and dark mode"):
        st.session_state.dark_mode = not st.session_state.dark_mode
        st.rerun()

st.markdown("---")

# ================================
# DOCUMENT MANAGEMENT SECTION
# ================================
st.markdown("""
    <div class="content-card">
        <h2 style="margin: 0 0 20px 0; font-size: 1.8em; display: flex; align-items: center;">
            <span style="margin-right: 12px;">📤</span>
            Document Management
        </h2>
    </div>
""", unsafe_allow_html=True)

# File uploader
uploaded_files = st.file_uploader(
    "Upload documents (PDF, DOCX, TXT)",
    type=["pdf", "docx", "txt"],
    accept_multiple_files=True,
    key="file_uploader",
    help="Select multiple documents to analyze"
)

# Update session state with uploaded files
if uploaded_files:
    st.session_state.uploaded_files = uploaded_files

# Display uploaded files and status
col1, col2 = st.columns([2, 1])

with col1:
    if st.session_state.uploaded_files:
        st.markdown("**📋 Uploaded Files:**")
        file_cols = st.columns(min(len(st.session_state.uploaded_files), 3))
        for i, file in enumerate(st.session_state.uploaded_files):
            col_idx = i % 3
            with file_cols[col_idx]:
                file_icon = "📄" if file.name.endswith('.pdf') else "📝" if file.name.endswith('.docx') else "📃"
                st.markdown(f"{file_icon} **{file.name}**")

with col2:
    # Status indicator
    if st.session_state.uploaded_files:
        status_color = "#28a745" if st.session_state.documents_processed else "#ffc107"
        status_text = "Ready to Process" if not st.session_state.documents_processed else "Documents Processed"
        status_icon = "⏳" if not st.session_state.documents_processed else "✅"
        st.markdown(f"### Status\n{status_icon} {status_text}")
    else:
        st.markdown("### Status\n⚠️ No documents uploaded")

# Process documents button
if st.button("🚀 Process Documents", use_container_width=True, type="primary"):
    with st.spinner("🔄 Processing documents..."):
        progress_bar = st.progress(0)
        for i in range(100):
            time.sleep(0.02)
            progress_bar.progress(i + 1)

        try:
            process_documents(st.session_state.uploaded_files)
            st.session_state.documents_processed = True
            st.success("✅ Documents processed successfully!")
            st.balloons()
        except Exception as e:
            st.error(f"❌ Error processing documents: {str(e)}")
        finally:
            progress_bar.empty()

st.markdown("---")

# ================================
# CONTROLS SECTION
# ================================
st.markdown("""
    <div class="content-card">
        <h2 style="margin: 0 0 20px 0; font-size: 1.8em; display: flex; align-items: center;">
            <span style="margin-right: 12px;">⚙️</span>
            Controls & Settings
        </h2>
    </div>
""", unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("### 📄 Document Status")
    if st.session_state.documents_processed:
        st.success("✅ Documents Processed")
        st.info(f"📊 {len(st.session_state.uploaded_files)} file(s) loaded")
    else:
        st.warning("⚠️ No documents processed yet")

    if st.session_state.processing_error:
        st.error(f"❌ Processing Error: {st.session_state.processing_error}")

with col2:
    st.markdown("### 💬 Chat Controls")
    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.chat_history = []
        st.session_state.current_answer = None
        st.session_state.current_sources = []
        st.session_state.current_retrieved_chunks = []
        st.success("Chat history cleared!")

    st.markdown("### 🔧 Debug Options")
    debug_toggle = st.checkbox("Show Debug View", value=st.session_state.show_debug_view, key="debug_toggle")
    if debug_toggle != st.session_state.show_debug_view:
        st.session_state.show_debug_view = debug_toggle

with col3:
    st.markdown("### 📈 Quick Stats")
    if st.session_state.chat_history:
        total_questions = len([msg for msg in st.session_state.chat_history if msg['role'] == 'user'])
        st.metric("Questions Asked", total_questions)
    else:
        st.metric("Questions Asked", 0)

st.markdown("---")

# Modern CSS with Enhanced Dark Mode Support
def get_css(dark_mode=False):
    if dark_mode:
        return """
        <style>
        :root {
            --bg-primary: #0f0f0f;
            --bg-secondary: #1a1a1a;
            --bg-tertiary: #2a2a2a;
            --bg-card: #1f1f1f;
            --text-primary: #ffffff;
            --text-secondary: #b0b0b0;
            --text-muted: #808080;
            --accent-primary: #4f46e5;
            --accent-secondary: #7c3aed;
            --accent-success: #059669;
            --accent-warning: #d97706;
            --accent-error: #dc2626;
            --border-color: #404040;
            --border-light: #555555;
            --card-shadow: 0 8px 32px rgba(0,0,0,0.4);
            --hover-shadow: 0 12px 40px rgba(0,0,0,0.5);
        }

        .main {
            background: linear-gradient(135deg, #0f0f0f 0%, #1a1a1a 100%);
            color: var(--text-primary);
        }

        .stButton>button {
            border-radius: 12px;
            font-weight: 600;
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
            border: none;
            background: linear-gradient(135deg, var(--accent-primary) 0%, var(--accent-secondary) 100%);
            color: white;
            box-shadow: 0 4px 15px rgba(79, 70, 229, 0.4);
        }

        .stButton>button:hover {
            transform: translateY(-2px);
            box-shadow: 0 8px 25px rgba(79, 70, 229, 0.5);
        }

        .stTextInput>div>div>input {
            border-radius: 12px;
            border: 2px solid var(--border-color);
            padding: 16px;
            font-size: 16px;
            background-color: var(--bg-secondary);
            color: var(--text-primary);
            transition: all 0.3s ease;
        }

        .stTextInput>div>div>input:focus {
            border-color: var(--accent-primary);
            box-shadow: 0 0 0 3px rgba(79, 70, 229, 0.3);
            background-color: var(--bg-tertiary);
        }

        .section-header {
            background: linear-gradient(135deg, #1e293b 0%, #334155 100%);
            color: white;
            padding: 25px;
            border-radius: 15px;
            margin-bottom: 25px;
            box-shadow: var(--card-shadow);
        }

        .content-card {
            background-color: var(--bg-card);
            padding: 30px;
            border-radius: 12px;
            margin-bottom: 25px;
            box-shadow: 0 4px 20px rgba(0,0,0,0.3);
            border: 1px solid var(--border-color);
        }

        .chat-message {
            padding: 20px;
            border-radius: 12px;
            margin-bottom: 15px;
            border: 1px solid var(--border-color);
        }

        .user-message {
            background: linear-gradient(135deg, var(--accent-primary) 0%, var(--accent-secondary) 100%);
            color: white;
            margin-left: 20%;
        }

        .assistant-message {
            background-color: var(--bg-card);
            color: var(--text-primary);
            margin-right: 20%;
        }

        .source-card {
            background-color: var(--bg-secondary);
            padding: 15px;
            border-radius: 8px;
            margin-bottom: 10px;
            border-left: 4px solid var(--accent-primary);
        }
        </style>
        """
    else:
        return """
        <style>
        :root {
            --bg-primary: #ffffff;
            --bg-secondary: #f8fafc;
            --bg-tertiary: #f1f5f9;
            --bg-card: #ffffff;
            --text-primary: #1e293b;
            --text-secondary: #64748b;
            --text-muted: #94a3b8;
            --accent-primary: #4f46e5;
            --accent-secondary: #7c3aed;
            --accent-success: #059669;
            --accent-warning: #d97706;
            --accent-error: #dc2626;
            --border-color: #e2e8f0;
            --border-light: #cbd5e1;
            --card-shadow: 0 8px 32px rgba(0,0,0,0.08);
            --hover-shadow: 0 12px 40px rgba(0,0,0,0.12);
        }

        .main {
            background: linear-gradient(135deg, #f8fafc 0%, #ffffff 100%);
            color: var(--text-primary);
        }

        .stButton>button {
            border-radius: 12px;
            font-weight: 600;
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
            border: none;
            background: linear-gradient(135deg, var(--accent-primary) 0%, var(--accent-secondary) 100%);
            color: white;
            box-shadow: 0 4px 15px rgba(79, 70, 229, 0.3);
        }

        .stButton>button:hover {
            transform: translateY(-2px);
            box-shadow: 0 8px 25px rgba(79, 70, 229, 0.4);
        }

        .stTextInput>div>div>input {
            border-radius: 12px;
            border: 2px solid var(--border-color);
            padding: 16px;
            font-size: 16px;
            background-color: var(--bg-primary);
            color: var(--text-primary);
            transition: all 0.3s ease;
        }

        .stTextInput>div>div>input:focus {
            border-color: var(--accent-primary);
            box-shadow: 0 0 0 3px rgba(79, 70, 229, 0.2);
            background-color: var(--bg-primary);
        }

        .section-header {
            background: linear-gradient(135deg, #1e293b 0%, #334155 100%);
            color: white;
            padding: 25px;
            border-radius: 15px;
            margin-bottom: 25px;
            box-shadow: var(--card-shadow);
        }

        .content-card {
            background-color: var(--bg-card);
            padding: 30px;
            border-radius: 12px;
            margin-bottom: 25px;
            box-shadow: 0 4px 20px rgba(0,0,0,0.08);
            border: 1px solid var(--border-color);
        }

        .chat-message {
            padding: 20px;
            border-radius: 12px;
            margin-bottom: 15px;
            border: 1px solid var(--border-color);
        }

        .user-message {
            background: linear-gradient(135deg, var(--accent-primary) 0%, var(--accent-secondary) 100%);
            color: white;
            margin-left: 20%;
        }

        .assistant-message {
            background-color: var(--bg-card);
            color: var(--text-primary);
            margin-right: 20%;
        }

        .source-card {
            background-color: var(--bg-secondary);
            padding: 15px;
            border-radius: 8px;
            margin-bottom: 10px;
            border-left: 4px solid var(--accent-primary);
        }
        </style>
        """

