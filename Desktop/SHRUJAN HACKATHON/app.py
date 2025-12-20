import streamlit as st
import time

# ================================
# PAGE SETUP
# ================================
st.set_page_config(
    page_title="RAG Document QA System",
    page_icon="🏠",
    layout="wide"
)

# Initialize theme in session state
if 'dark_mode' not in st.session_state:
    st.session_state.dark_mode = False

# ================================
# MODERN CSS WITH ENHANCED DARK MODE SUPPORT
# ================================
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
        }

        body {
            background-color: var(--bg-primary);
            color: var(--text-primary);
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        }

        .hero-section {
            background: linear-gradient(135deg, var(--accent-primary) 0%, var(--accent-secondary) 100%);
            color: white;
            padding: 40px;
            border-radius: 15px;
            margin-bottom: 30px;
            box-shadow: 0 8px 32px rgba(0,0,0,0.3);
            text-align: center;
        }

        .feature-card {
            background-color: var(--bg-card);
            padding: 25px;
            border-radius: 12px;
            margin-bottom: 20px;
            box-shadow: 0 4px 20px rgba(0,0,0,0.2);
            border: 1px solid var(--border-color);
            transition: transform 0.3s ease, box-shadow 0.3s ease;
        }

        .feature-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 8px 30px rgba(0,0,0,0.3);
        }

        .nav-button {
            background: linear-gradient(135deg, var(--accent-primary) 0%, var(--accent-secondary) 100%);
            color: white;
            padding: 15px 30px;
            border: none;
            border-radius: 10px;
            font-size: 16px;
            font-weight: 600;
            cursor: pointer;
            text-decoration: none;
            display: inline-block;
            margin: 10px;
            transition: transform 0.3s ease, box-shadow 0.3s ease;
            box-shadow: 0 4px 15px rgba(0,0,0,0.2);
        }

        .nav-button:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(0,0,0,0.3);
        }

        .stats-card {
            background: linear-gradient(135deg, var(--accent-primary) 0%, var(--accent-secondary) 100%);
            color: white;
            padding: 20px;
            border-radius: 10px;
            text-align: center;
            margin: 10px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.2);
        }

        .animate-fade-in {
            animation: fadeIn 0.5s ease-in;
        }

        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(20px); }
            to { opacity: 1; transform: translateY(0); }
        }
        </style>
        """
    else:
        return """
        <style>
        :root {
            --bg-primary: #ffffff;
            --bg-secondary: #f8f9fa;
            --bg-tertiary: #e9ecef;
            --bg-card: #ffffff;
            --text-primary: #212529;
            --text-secondary: #6c757d;
            --text-muted: #adb5bd;
            --accent-primary: #4f46e5;
            --accent-secondary: #7c3aed;
            --accent-success: #059669;
            --accent-warning: #d97706;
            --accent-error: #dc2626;
            --border-color: #dee2e6;
        }

        body {
            background-color: var(--bg-primary);
            color: var(--text-primary);
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        }

        .hero-section {
            background: linear-gradient(135deg, var(--accent-primary) 0%, var(--accent-secondary) 100%);
            color: white;
            padding: 40px;
            border-radius: 15px;
            margin-bottom: 30px;
            box-shadow: 0 8px 32px rgba(0,0,0,0.1);
            text-align: center;
        }

        .feature-card {
            background-color: var(--bg-card);
            padding: 25px;
            border-radius: 12px;
            margin-bottom: 20px;
            box-shadow: 0 4px 20px rgba(0,0,0,0.08);
            border: 1px solid var(--border-color);
            transition: transform 0.3s ease, box-shadow 0.3s ease;
        }

        .feature-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 8px 30px rgba(0,0,0,0.15);
        }

        .nav-button {
            background: linear-gradient(135deg, var(--accent-primary) 0%, var(--accent-secondary) 100%);
            color: white;
            padding: 15px 30px;
            border: none;
            border-radius: 10px;
            font-size: 16px;
            font-weight: 600;
            cursor: pointer;
            text-decoration: none;
            display: inline-block;
            margin: 10px;
            transition: transform 0.3s ease, box-shadow 0.3s ease;
            box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        }

        .nav-button:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(0,0,0,0.15);
        }

        .stats-card {
            background: linear-gradient(135deg, var(--accent-primary) 0%, var(--accent-secondary) 100%);
            color: white;
            padding: 20px;
            border-radius: 10px;
            text-align: center;
            margin: 10px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        }

        .animate-fade-in {
            animation: fadeIn 0.5s ease-in;
        }

        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(20px); }
            to { opacity: 1; transform: translateY(0); }
        }
        </style>
        """

# Apply CSS
st.markdown(get_css(st.session_state.dark_mode), unsafe_allow_html=True)

# ================================
# SIDEBAR
# ================================
with st.sidebar:
    st.markdown("### 🎛️ Control Panel")

    # Theme toggle
    theme_icon = "🌙" if not st.session_state.dark_mode else "☀️"
    if st.button(f"{theme_icon} Toggle Theme", use_container_width=True):
        st.session_state.dark_mode = not st.session_state.dark_mode
        st.rerun()

    st.markdown("---")

    # Navigation
    st.markdown("### 🧭 Navigation")
    if st.button("🏠 Home", use_container_width=True):
        st.switch_page("app.py")
    if st.button("🤖 Document Q&A", use_container_width=True):
        st.switch_page("pages/1_Document_QA.py")
    if st.button("📊 Document Summary", use_container_width=True):
        st.switch_page("pages/2_Document_Summary.py")
    if st.button("📈 Analytics Dashboard", use_container_width=True):
        st.switch_page("pages/3_Analytics_Dashboard.py")

    st.markdown("---")

    # About section
    st.markdown("### ℹ️ About")
    st.markdown("""
    **RAG Document QA System**

    - 🤖 AI-powered Q&A from documents
    - 📄 Support for PDF, DOCX, TXT files
    - 🌙 Dark/Light theme support
    - 📊 Analytics and insights
    - 🔍 Source citations
    """)

# ================================
# MAIN CONTENT
# ================================
st.markdown("""
<div class="hero-section animate-fade-in">
    <h1 style="margin: 0; font-size: 3em; font-weight: 700;">🤖 RAG Document QA System</h1>
    <p style="margin: 10px 0 0 0; font-size: 1.2em; opacity: 0.9;">
        Intelligent Document Analysis & Question Answering
    </p>
</div>
""", unsafe_allow_html=True)

# Quick Stats
col1, col2, col3 = st.columns(3)
with col1:
    st.markdown("""
    <div class="stats-card animate-fade-in">
        <h3 style="margin: 0; font-size: 2em;">📄</h3>
        <p style="margin: 5px 0 0 0; font-size: 0.9em;">Documents Processed</p>
        <p style="margin: 0; font-size: 1.5em; font-weight: bold;">0</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="stats-card animate-fade-in">
        <h3 style="margin: 0; font-size: 2em;">❓</h3>
        <p style="margin: 5px 0 0 0; font-size: 0.9em;">Questions Answered</p>
        <p style="margin: 0; font-size: 1.5em; font-weight: bold;">0</p>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="stats-card animate-fade-in">
        <h3 style="margin: 0; font-size: 2em;">📊</h3>
        <p style="margin: 5px 0 0 0; font-size: 0.9em;">Summaries Generated</p>
        <p style="margin: 0; font-size: 1.5em; font-weight: bold;">0</p>
    </div>
    """, unsafe_allow_html=True)

# Feature Overview
st.markdown("## 🚀 Features Overview")

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    <div class="feature-card animate-fade-in">
        <h3 style="margin: 0 0 15px 0; color: var(--accent-primary);">🤖 Document Q&A</h3>
        <p style="margin: 0; color: var(--text-secondary);">
            Upload documents and ask intelligent questions. Get AI-powered answers with source citations and retrieval transparency.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="feature-card animate-fade-in">
        <h3 style="margin: 0 0 15px 0; color: var(--accent-primary);">📊 Document Summary</h3>
        <p style="margin: 0; color: var(--text-secondary);">
            Generate comprehensive summaries of your documents with customizable options and export capabilities.
        </p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="feature-card animate-fade-in">
        <h3 style="margin: 0 0 15px 0; color: var(--accent-primary);">📈 Analytics Dashboard</h3>
        <p style="margin: 0; color: var(--text-secondary);">
            View detailed analytics about your usage, question patterns, and document insights with interactive charts.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="feature-card animate-fade-in">
        <h3 style="margin: 0 0 15px 0; color: var(--accent-primary);">🌙 Theme Support</h3>
        <p style="margin: 0; color: var(--text-secondary);">
            Switch between dark and light themes for comfortable viewing in any environment.
        </p>
    </div>
    """, unsafe_allow_html=True)

# Quick Navigation
st.markdown("## 🧭 Quick Navigation")

col1, col2, col3, col4 = st.columns(4)

with col1:
    if st.button("🏠 Home", use_container_width=True, type="primary"):
        st.switch_page("app.py")

with col2:
    if st.button("🤖 Document Q&A", use_container_width=True, type="secondary"):
        st.switch_page("pages/1_Document_QA.py")

with col3:
    if st.button("📊 Document Summary", use_container_width=True, type="secondary"):
        st.switch_page("pages/2_Document_Summary.py")

with col4:
    if st.button("📈 Analytics", use_container_width=True, type="secondary"):
        st.switch_page("pages/3_Analytics_Dashboard.py")

# Getting Started Guide
st.markdown("## 📚 Getting Started")

st.markdown("""
<div class="feature-card animate-fade-in">
    <h3 style="margin: 0 0 15px 0; color: var(--accent-primary);">Welcome to RAG Document QA System!</h3>
    <ol style="margin: 0; color: var(--text-secondary);">
        <li><strong>Navigate:</strong> Use the sidebar or quick navigation buttons above to access different features</li>
        <li><strong>Upload Documents:</strong> Start with the Document Q&A page to upload PDF, DOCX, or TXT files</li>
        <li><strong>Ask Questions:</strong> Once documents are processed, ask questions in natural language</li>
        <li><strong>Explore Features:</strong> Try document summarization and view analytics for insights</li>
        <li><strong>Customize Theme:</strong> Use the theme toggle in the sidebar for your preferred viewing mode</li>
    </ol>
</div>
""", unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; padding: 20px; color: var(--text-secondary);">
    <p style="margin: 0; font-size: 0.9em;">
        🤖 <strong>RAG Document QA System</strong> | Built with Streamlit & OpenAI GPT
    </p>
    <p style="margin: 5px 0 0 0; font-size: 0.8em; opacity: 0.7;">
        Intelligent Document Analysis • AI-Powered Q&A • Advanced Analytics
    </p>
</div>
""", unsafe_allow_html=True)
