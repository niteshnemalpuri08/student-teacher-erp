import streamlit as st
import time
from utils.rag_qa import process_documents, summarize_documents

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
if 'document_summary' not in st.session_state:
    st.session_state.document_summary = None

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

        .summary-stats {
            background: linear-gradient(135deg, var(--accent-primary) 0%, var(--accent-secondary) 100%);
            color: white;
            padding: 20px;
            border-radius: 12px;
            margin-bottom: 20px;
            text-align: center;
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

        .section-header {
            background: linear-gradient(135deg, #3b82f6 0%, #1d4ed8 100%);
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
            box-shadow: 0 4px 20px rgba(0,0,0,0.06);
            border: 1px solid var(--border-color);
        }

        .summary-stats {
            background: linear-gradient(135deg, var(--accent-primary) 0%, var(--accent-secondary) 100%);
            color: white;
            padding: 20px;
            border-radius: 12px;
            margin-bottom: 20px;
            text-align: center;
        }
        </style>
        """

# Apply CSS based on theme
st.markdown(get_css(st.session_state.dark_mode), unsafe_allow_html=True)

# ================================
# SIDEBAR: THEME TOGGLE
# ================================
with st.sidebar:
    st.markdown("### 🎨 Theme")
    theme_icon = "🌙" if not st.session_state.dark_mode else "☀️"
    if st.button(theme_icon, help="Toggle dark/light mode", key="theme_toggle"):
        st.session_state.dark_mode = not st.session_state.dark_mode
        st.rerun()

# ================================
# MAIN CONTENT
# ================================

# Page Header
st.markdown("""
    <div class="section-header">
        <h1 style="margin: 0; font-size: 2.5em; display: flex; align-items: center;">
            <span style="margin-right: 15px;">📊</span>
            Document Summary
        </h1>
        <p style="margin: 10px 0 0 0; opacity: 0.9; font-size: 1.1em;">AI-generated overview of your documents</p>
    </div>
""", unsafe_allow_html=True)

# Quick Stats Dashboard
if st.session_state.documents_processed:
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("📄 Documents", len(st.session_state.uploaded_files))
    with col2:
        st.metric("📊 Summaries Generated", 1 if st.session_state.document_summary else 0)
    with col3:
        st.metric("⚡ Processing Status", "Complete" if st.session_state.documents_processed else "Pending")

st.markdown("---")

# ================================
# SIDEBAR: DOCUMENT MANAGEMENT
# ================================
with st.sidebar:
    st.markdown("### 📤 Document Management")

    # File uploader
    uploaded_files = st.file_uploader(
        "Upload documents (PDF, DOCX, TXT)",
        type=["pdf", "docx", "txt"],
        accept_multiple_files=True,
        key="file_uploader",
        help="Select multiple documents to summarize"
    )

    # Update session state with uploaded files
    if uploaded_files:
        st.session_state.uploaded_files = uploaded_files

    # Display uploaded files
    if st.session_state.uploaded_files:
        st.markdown("**📋 Uploaded Files:**")
        for file in st.session_state.uploaded_files:
            file_icon = "📄" if file.name.endswith('.pdf') else "📝" if file.name.endswith('.docx') else "📃"
            st.markdown(f"{file_icon} {file.name}")

    # Status indicator
    if st.session_state.uploaded_files:
        status_color = "#28a745" if st.session_state.documents_processed else "#ffc107"
        status_text = "Ready to Process" if not st.session_state.documents_processed else "Documents Processed"
        status_icon = "⏳" if not st.session_state.documents_processed else "✅"
        st.markdown(f"**Status:** {status_icon} {status_text}")

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

# ================================
# SUMMARY GENERATION SECTION
# ================================
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("""
        <div class="section-header">
            <h2 style="margin: 0; font-size: 1.8em; display: flex; align-items: center;">
                <span style="margin-right: 10px;">🤖</span>
                Generate Summary
            </h2>
            <p style="margin: 8px 0 0 0; opacity: 0.9; font-size: 1em;">Create an AI-powered summary of your documents</p>
        </div>
    """, unsafe_allow_html=True)

    # Summary generation options
    summary_type = st.selectbox(
        "Summary Type",
        ["Comprehensive Overview", "Key Points", "Executive Summary", "Technical Summary"],
        help="Choose the type of summary you want to generate"
    )

    summary_length = st.selectbox(
        "Summary Length",
        ["Short (100-200 words)", "Medium (300-500 words)", "Long (600-800 words)"],
        index=1,
        help="Select the desired length of the summary"
    )

    # Generate summary button
    generate_disabled = not st.session_state.documents_processed

    if generate_disabled:
        st.warning("⚠️ Please upload and process your documents first using the sidebar.")

    if st.button("🚀 Generate Summary", disabled=generate_disabled, use_container_width=True, type="primary"):
        with st.spinner("🤖 Analyzing documents and generating summary..."):
            progress_bar = st.progress(0)
            for i in range(100):
                time.sleep(0.01)
                progress_bar.progress(i + 1)

            try:
                summary = summarize_documents()
                st.session_state.document_summary = summary
                st.success("✅ Document summary generated successfully!")
                st.balloons()

            except Exception as e:
                st.error(f"❌ Error generating summary: {str(e)}")
            finally:
                progress_bar.empty()

with col2:
    st.markdown("<br>", unsafe_allow_html=True)

    # Summary statistics
    if st.session_state.document_summary:
        word_count = len(st.session_state.document_summary.split())
        char_count = len(st.session_state.document_summary)

        st.markdown(f"""
            <div class="summary-stats">
                <h3 style="margin: 0 0 10px 0; font-size: 1.2em;">📈 Summary Stats</h3>
                <div style="display: flex; justify-content: space-around;">
                    <div>
                        <div style="font-size: 1.5em; font-weight: bold;">{word_count}</div>
                        <div style="font-size: 0.9em; opacity: 0.9;">Words</div>
                    </div>
                    <div>
                        <div style="font-size: 1.5em; font-weight: bold;">{char_count}</div>
                        <div style="font-size: 0.9em; opacity: 0.9;">Characters</div>
                    </div>
                </div>
            </div>
        """, unsafe_allow_html=True)

    # Export options
    if st.session_state.document_summary:
        st.markdown("### 💾 Export Options")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("📄 Download as TXT", use_container_width=True):
                st.download_button(
                    label="Download Summary",
                    data=st.session_state.document_summary,
                    file_name="document_summary.txt",
                    mime="text/plain"
                )
        with col2:
            if st.button("📋 Copy to Clipboard", use_container_width=True):
                st.code(st.session_state.document_summary, language=None)

# ================================
# SUMMARY DISPLAY
# ================================
if st.session_state.document_summary:
    st.markdown("""
        <div class="section-header">
            <h2 style="margin: 0; font-size: 1.8em; display: flex; align-items: center;">
                <span style="margin-right: 12px;">📋</span>
                Generated Summary
            </h2>
            <p style="margin: 8px 0 0 0; opacity: 0.9; font-size: 1em;">AI-generated document overview</p>
        </div>
    """, unsafe_allow_html=True)

    # Summary content in a professional card
    st.markdown(f"""
        <div class="content-card">
            <div style="font-size: 18px; line-height: 1.7; color: var(--text-primary);">
                {st.session_state.document_summary}
            </div>
        </div>
    """, unsafe_allow_html=True)

    # Summary analysis
    st.markdown("### 📊 Summary Analysis")

    col1, col2, col3 = st.columns(3)

    with col1:
        # Word frequency analysis (simple)
        words = st.session_state.document_summary.lower().split()
        word_freq = {}
        for word in words:
            if len(word) > 3:  # Only count meaningful words
                word_freq[word] = word_freq.get(word, 0) + 1

        top_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:5]

        st.markdown("**🔤 Top Keywords:**")
        for word, count in top_words:
            st.markdown(f"• {word} ({count})")

    with col2:
        # Readability metrics
        sentences = st.session_state.document_summary.split('.')
        avg_sentence_length = sum(len(s.split()) for s in sentences) / len(sentences) if sentences else 0

        st.markdown("**📏 Readability:**")
        st.markdown(f"• Sentences: {len(sentences)}")
        st.markdown(".1f")
        st.markdown(f"• Words: {len(words)}")

    with col3:
        # Content type detection
        content_indicators = {
            "Technical": ["algorithm", "system", "process", "method", "analysis"],
            "Business": ["strategy", "market", "revenue", "growth", "plan"],
            "Academic": ["research", "study", "findings", "conclusion", "methodology"]
        }

        detected_types = []
        summary_lower = st.session_state.document_summary.lower()

        for content_type, keywords in content_indicators.items():
            if any(keyword in summary_lower for keyword in keywords):
                detected_types.append(content_type)

        st.markdown("**🎯 Content Type:**")
        if detected_types:
            for content_type in detected_types:
                st.markdown(f"• {content_type}")
        else:
            st.markdown("• General Content")

# ================================
# SUMMARY HISTORY
# ================================
if st.session_state.document_summary:
    st.markdown("---")
    st.markdown("""
        <div class="section-header">
            <h2 style="margin: 0; font-size: 1.8em; display: flex; align-items: center;">
                <span style="margin-right: 12px;">📚</span>
                Summary History
            </h2>
            <p style="margin: 8px 0 0 0; opacity: 0.9; font-size: 1em;">Previously generated summaries</p>
        </div>
    """, unsafe_allow_html=True)

    # For now, just show the current summary
    # In a full implementation, this would store multiple summaries
    with st.expander("📄 Latest Summary", expanded=True):
        st.markdown(f"""
            <div style="background-color: var(--bg-secondary); padding: 20px; border-radius: 10px;">
                <div style="font-size: 16px; line-height: 1.6; color: var(--text-primary);">
                    {st.session_state.document_summary}
                </div>
                <div style="margin-top: 15px; padding-top: 10px; border-top: 1px solid var(--border-color);">
                    <small style="color: var(--text-secondary);">
                        Generated on {time.strftime('%Y-%m-%d %H:%M:%S')} |
                        Type: {summary_type} |
                        Length: {summary_length}
                    </small>
                </div>
            </div>
        """, unsafe_allow_html=True)

    # Clear summary button
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        if st.button("🗑️ Clear Summary", use_container_width=True, type="secondary"):
            st.session_state.document_summary = None
            st.success("🧹 Summary cleared successfully!")

# ================================
# FOOTER
# ================================
st.markdown("---")
st.markdown("""
    <div style="text-align: center; padding: 20px; color: var(--text-secondary);">
        <p style="margin: 0; font-size: 0.9em;">
            📊 <strong>Document Summary</strong> | Powered by AI & Document Analysis
        </p>
        <p style="margin: 5px 0 0 0; font-size: 0.8em; opacity: 0.7;">
            Upload • Process • Summarize | Intelligent Document Analysis
        </p>
    </div>
""", unsafe_allow_html=True)
