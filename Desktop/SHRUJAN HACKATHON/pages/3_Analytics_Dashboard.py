import streamlit as st
import time
from utils.rag_qa import process_documents, get_answer
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# ================================
# PAGE SETUP
# ================================
st.set_page_config(
    page_title="Analytics Dashboard - RAG System",
    page_icon="📈",
    layout="wide"
)

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
if 'analytics_data' not in st.session_state:
    st.session_state.analytics_data = None

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

        .metric-card {
            background-color: var(--bg-card);
            padding: 20px;
            border-radius: 12px;
            margin-bottom: 15px;
            box-shadow: 0 4px 20px rgba(0,0,0,0.3);
            border: 1px solid var(--border-color);
            text-align: center;
        }

        .chart-container {
            background-color: var(--bg-card);
            padding: 25px;
            border-radius: 12px;
            margin-bottom: 25px;
            box-shadow: 0 4px 20px rgba(0,0,0,0.3);
            border: 1px solid var(--border-color);
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

        .metric-card {
            background-color: var(--bg-card);
            padding: 20px;
            border-radius: 12px;
            margin-bottom: 15px;
            box-shadow: 0 4px 20px rgba(0,0,0,0.06);
            border: 1px solid var(--border-color);
            text-align: center;
        }

        .chart-container {
            background-color: var(--bg-card);
            padding: 25px;
            border-radius: 12px;
            margin-bottom: 25px;
            box-shadow: 0 4px 20px rgba(0,0,0,0.06);
            border: 1px solid var(--border-color);
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
# ANALYTICS FUNCTIONS
# ================================
def generate_analytics_data():
    """Generate analytics data from chat history and documents"""
    if not st.session_state.chat_history:
        return None

    analytics = {
        'total_questions': len(st.session_state.chat_history),
        'total_answers': len([h for h in st.session_state.chat_history if h['answer']]),
        'avg_answer_length': 0,
        'question_types': {},
        'response_times': [],
        'document_usage': {},
        'question_length_distribution': [],
        'answer_quality_scores': []
    }

    # Calculate metrics
    total_answer_length = 0
    for entry in st.session_state.chat_history:
        if entry['answer']:
            total_answer_length += len(entry['answer'].split())
            analytics['question_length_distribution'].append(len(entry['question'].split()))

            # Categorize question types
            question_lower = entry['question'].lower()
            if any(word in question_lower for word in ['what', 'how', 'why', 'when', 'where', 'who']):
                q_type = 'WH-Questions'
            elif any(word in question_lower for word in ['explain', 'describe', 'define']):
                q_type = 'Explanatory'
            elif any(word in question_lower for word in ['compare', 'contrast', 'difference']):
                q_type = 'Comparative'
            else:
                q_type = 'General'

            analytics['question_types'][q_type] = analytics['question_types'].get(q_type, 0) + 1

            # Track document usage
            if entry['sources']:
                for source in entry['sources']:
                    doc_name = source['document']
                    analytics['document_usage'][doc_name] = analytics['document_usage'].get(doc_name, 0) + 1

    if analytics['total_answers'] > 0:
        analytics['avg_answer_length'] = total_answer_length / analytics['total_answers']

    return analytics

# ================================
# MAIN CONTENT
# ================================

# Page Header
st.markdown("""
    <div class="section-header">
        <h1 style="margin: 0; font-size: 2.5em; display: flex; align-items: center;">
            <span style="margin-right: 15px;">📈</span>
            Analytics Dashboard
        </h1>
        <p style="margin: 10px 0 0 0; opacity: 0.9; font-size: 1.1em;">Insights and analytics for your document interactions</p>
    </div>
""", unsafe_allow_html=True)

# Check if we have data to analyze
if not st.session_state.chat_history:
    st.warning("⚠️ No interaction data available. Please use the Document Q&A page first to generate analytics.")

    st.markdown("""
        <div class="chart-container">
            <h3 style="text-align: center; color: var(--text-secondary);">Getting Started</h3>
            <p style="text-align: center; color: var(--text-muted);">
                Visit the <strong>Document Q&A</strong> page to ask questions and generate analytics data.
            </p>
        </div>
    """, unsafe_allow_html=True)
else:
    # Generate analytics data
    if st.session_state.analytics_data is None:
        with st.spinner("🔄 Analyzing your interaction data..."):
            st.session_state.analytics_data = generate_analytics_data()

    analytics = st.session_state.analytics_data

    # ================================
    # KEY METRICS
    # ================================
    st.markdown("### 📊 Key Metrics")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(f"""
            <div class="metric-card">
                <h3 style="margin: 0 0 10px 0; color: var(--accent-primary); font-size: 2em;">{analytics['total_questions']}</h3>
                <p style="margin: 0; color: var(--text-secondary); font-size: 0.9em;">Total Questions</p>
            </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
            <div class="metric-card">
                <h3 style="margin: 0 0 10px 0; color: var(--accent-success); font-size: 2em;">{analytics['total_answers']}</h3>
                <p style="margin: 0; color: var(--text-secondary); font-size: 0.9em;">Answers Generated</p>
            </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
            <div class="metric-card">
                <h3 style="margin: 0 0 10px 0; color: var(--accent-warning); font-size: 2em;">{analytics['avg_answer_length']:.0f}</h3>
                <p style="margin: 0; color: var(--text-secondary); font-size: 0.9em;">Avg Answer Length</p>
            </div>
        """, unsafe_allow_html=True)

    with col4:
        success_rate = (analytics['total_answers'] / analytics['total_questions'] * 100) if analytics['total_questions'] > 0 else 0
        st.markdown(f"""
            <div class="metric-card">
                <h3 style="margin: 0 0 10px 0; color: var(--accent-secondary); font-size: 2em;">{success_rate:.1f}%</h3>
                <p style="margin: 0; color: var(--text-secondary); font-size: 0.9em;">Success Rate</p>
            </div>
        """, unsafe_allow_html=True)

    # ================================
    # CHARTS SECTION
    # ================================
    st.markdown("### 📈 Analytics Charts")

    col1, col2 = st.columns(2)

    with col1:
        # Question Types Distribution
        if analytics['question_types']:
            st.markdown('<div class="chart-container">', unsafe_allow_html=True)
            st.markdown("#### ❓ Question Types")

            question_types_df = pd.DataFrame(
                list(analytics['question_types'].items()),
                columns=['Question Type', 'Count']
            )

            fig = px.pie(
                question_types_df,
                values='Count',
                names='Question Type',
                title="Question Type Distribution",
                color_discrete_sequence=px.colors.qualitative.Set3
            )
            fig.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font_color='var(--text-primary)' if not st.session_state.dark_mode else 'white'
            )
            st.plotly_chart(fig, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        # Document Usage
        if analytics['document_usage']:
            st.markdown('<div class="chart-container">', unsafe_allow_html=True)
            st.markdown("#### 📄 Document Usage")

            doc_usage_df = pd.DataFrame(
                list(analytics['document_usage'].items()),
                columns=['Document', 'References']
            )

            fig = px.bar(
                doc_usage_df,
                x='Document',
                y='References',
                title="Document Reference Frequency",
                color='References',
                color_continuous_scale='Blues'
            )
            fig.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font_color='var(--text-primary)' if not st.session_state.dark_mode else 'white'
            )
            st.plotly_chart(fig, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

    # Question Length Distribution
    if analytics['question_length_distribution']:
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        st.markdown("#### 📏 Question Length Distribution")

        fig = px.histogram(
            analytics['question_length_distribution'],
            nbins=10,
            title="Distribution of Question Lengths",
            labels={'value': 'Words in Question', 'count': 'Frequency'},
            color_discrete_sequence=['var(--accent-primary)']
        )
        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font_color='var(--text-primary)' if not st.session_state.dark_mode else 'white'
        )
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # ================================
    # DETAILED ANALYTICS
    # ================================
    st.markdown("### 📋 Detailed Analytics")

    tab1, tab2, tab3 = st.tabs(["📊 Summary", "❓ Questions", "📄 Documents"])

    with tab1:
        st.markdown("#### 📊 Session Summary")

        summary_data = {
            "Metric": ["Total Sessions", "Questions Asked", "Answers Generated", "Success Rate", "Avg Answer Length"],
            "Value": [
                len(st.session_state.chat_history),
                analytics['total_questions'],
                analytics['total_answers'],
                ".1f",
                ".0f"
            ]
        }

        summary_df = pd.DataFrame(summary_data)
        st.dataframe(summary_df, use_container_width=True)

    with tab2:
        st.markdown("#### ❓ Question Analysis")

        if st.session_state.chat_history:
            # Create a table of questions and their types
            question_data = []
            for i, entry in enumerate(st.session_state.chat_history):
                question_lower = entry['question'].lower()
                if any(word in question_lower for word in ['what', 'how', 'why', 'when', 'where', 'who']):
                    q_type = 'WH-Questions'
                elif any(word in question_lower for word in ['explain', 'describe', 'define']):
                    q_type = 'Explanatory'
                elif any(word in question_lower for word in ['compare', 'contrast', 'difference']):
                    q_type = 'Comparative'
                else:
                    q_type = 'General'

                question_data.append({
                    "Session": i + 1,
                    "Question": entry['question'][:50] + "..." if len(entry['question']) > 50 else entry['question'],
                    "Type": q_type,
                    "Answered": "Yes" if entry['answer'] else "No",
                    "Sources": len(entry['sources']) if entry['sources'] else 0
                })

            questions_df = pd.DataFrame(question_data)
            st.dataframe(questions_df, use_container_width=True)

    with tab3:
        st.markdown("#### 📄 Document Analysis")

        if analytics['document_usage']:
            doc_analysis_data = []
            for doc, refs in analytics['document_usage'].items():
                doc_analysis_data.append({
                    "Document": doc,
                    "Total References": refs,
                    "Usage Percentage": ".1f"
                })

            doc_df = pd.DataFrame(doc_analysis_data)
            st.dataframe(doc_df, use_container_width=True)

            # Document usage chart
            fig = px.bar(
                doc_df,
                x='Document',
                y='Total References',
                title="Document Usage Overview",
                color='Usage Percentage',
                color_continuous_scale='Greens'
            )
            fig.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font_color='var(--text-primary)' if not st.session_state.dark_mode else 'white'
            )
            st.plotly_chart(fig, use_container_width=True)

    # ================================
    # EXPORT ANALYTICS
    # ================================
    st.markdown("### 💾 Export Analytics")

    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("📊 Export Summary CSV", use_container_width=True):
            summary_csv = summary_df.to_csv(index=False)
            st.download_button(
                label="Download Summary",
                data=summary_csv,
                file_name="analytics_summary.csv",
                mime="text/csv"
            )

    with col2:
        if st.button("📋 Export Questions CSV", use_container_width=True):
            questions_csv = questions_df.to_csv(index=False)
            st.download_button(
                label="Download Questions",
                data=questions_csv,
                file_name="questions_analysis.csv",
                mime="text/csv"
            )

    with col3:
        if st.button("🗑️ Clear Analytics", use_container_width=True, type="secondary"):
            st.session_state.analytics_data = None
            st.success("🧹 Analytics data cleared successfully!")
            st.rerun()

# ================================
# FOOTER
# ================================
st.markdown("---")
st.markdown("""
    <div style="text-align: center; padding: 20px; color: var(--text-secondary);">
        <p style="margin: 0; font-size: 0.9em;">
            📈 <strong>Analytics Dashboard</strong> | Data-Driven Document Insights
        </p>
        <p style="margin: 5px 0 0 0; font-size: 0.8em; opacity: 0.7;">
            Analyze • Visualize • Optimize | Intelligent Document Analytics
        </p>
    </div>
""", unsafe_allow_html=True)
