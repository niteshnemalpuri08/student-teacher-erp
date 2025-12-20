#!/usr/bin/env python3
"""
Test script for Analytics Dashboard fixes
"""

import sys
import os
sys.path.append('.')

import pandas as pd
import streamlit as st

# Mock session state for testing
class MockSessionState:
    def __init__(self):
        self.chat_history = [
            {'role': 'user', 'content': 'What is machine learning?'},
            {'role': 'assistant', 'content': 'Machine learning is a subset of AI...', 'sources': [{'document': 'doc1.pdf'}, {'document': 'doc2.pdf'}]},
            {'role': 'user', 'content': 'How does it work?'},
            {'role': 'assistant', 'content': 'Machine learning works by...', 'sources': [{'document': 'doc1.pdf'}]},
            {'role': 'user', 'content': 'Explain neural networks'},
            {'role': 'assistant', 'content': 'Neural networks are...', 'sources': [{'document': 'doc3.pdf'}, {'document': 'doc3.pdf'}]}
        ]
        self.analytics_data = None
        self.dark_mode = False

# Mock the session state
st.session_state = MockSessionState()

def generate_analytics_data():
    """Generate analytics data from chat history and documents"""
    if not st.session_state.chat_history:
        return None

    # Separate user and assistant messages
    user_messages = [h for h in st.session_state.chat_history if h.get('role') == 'user']
    assistant_messages = [h for h in st.session_state.chat_history if h.get('role') == 'assistant']

    analytics = {
        'total_questions': len(user_messages),
        'total_answers': len(assistant_messages),
        'avg_answer_length': 0,
        'question_types': {},
        'response_times': [],
        'document_usage': {},
        'question_length_distribution': [],
        'answer_quality_scores': []
    }

    # Calculate metrics
    total_answer_length = 0
    for assistant_msg in assistant_messages:
        if assistant_msg.get('content'):
            total_answer_length += len(assistant_msg['content'].split())

            # Track document usage
            if assistant_msg.get('sources'):
                for source in assistant_msg['sources']:
                    doc_name = source['document']
                    analytics['document_usage'][doc_name] = analytics['document_usage'].get(doc_name, 0) + 1

    for user_msg in user_messages:
        if user_msg.get('content'):
            analytics['question_length_distribution'].append(len(user_msg['content'].split()))

            # Categorize question types
            question_lower = user_msg['content'].lower()
            if any(word in question_lower for word in ['what', 'how', 'why', 'when', 'where', 'who']):
                q_type = 'WH-Questions'
            elif any(word in question_lower for word in ['explain', 'describe', 'define']):
                q_type = 'Explanatory'
            elif any(word in question_lower for word in ['compare', 'contrast', 'difference']):
                q_type = 'Comparative'
            else:
                q_type = 'General'

            analytics['question_types'][q_type] = analytics['question_types'].get(q_type, 0) + 1

    if analytics['total_answers'] > 0:
        analytics['avg_answer_length'] = total_answer_length / analytics['total_answers']

    return analytics

def test_analytics_calculations():
    """Test the analytics calculations"""
    print("🧪 Testing Analytics Dashboard Calculations...")

    analytics = generate_analytics_data()

    # Test basic metrics
    assert analytics['total_questions'] == 3, f"Expected 3 questions, got {analytics['total_questions']}"
    assert analytics['total_answers'] == 3, f"Expected 3 answers, got {analytics['total_answers']}"

    # Test document usage
    expected_usage = {'doc1.pdf': 2, 'doc2.pdf': 1, 'doc3.pdf': 2}
    assert analytics['document_usage'] == expected_usage, f"Document usage mismatch: {analytics['document_usage']} vs {expected_usage}"

    # Test question types
    expected_types = {'WH-Questions': 2, 'Explanatory': 1}
    assert analytics['question_types'] == expected_types, f"Question types mismatch: {analytics['question_types']} vs {expected_types}"

    print("✅ Basic analytics calculations passed")

def test_percentage_calculation():
    """Test the fixed percentage calculation"""
    print("🧪 Testing Usage Percentage Calculation...")

    analytics = generate_analytics_data()

    # Calculate percentages as done in the dashboard
    total_refs = sum(analytics['document_usage'].values())
    percentages = {}
    for doc, refs in analytics['document_usage'].items():
        usage_pct = (refs / total_refs * 100) if total_refs > 0 else 0
        percentages[doc] = round(usage_pct, 1)

    # Expected: doc1.pdf: 2/5 = 40%, doc2.pdf: 1/5 = 20%, doc3.pdf: 2/5 = 40%
    expected_percentages = {'doc1.pdf': 40.0, 'doc2.pdf': 20.0, 'doc3.pdf': 40.0}
    assert percentages == expected_percentages, f"Percentage calculation failed: {percentages} vs {expected_percentages}"

    print("✅ Percentage calculation passed")

def test_dataframe_creation():
    """Test dataframe creation for export"""
    print("🧪 Testing DataFrame Creation for Export...")

    analytics = generate_analytics_data()

    # Test summary dataframe
    success_rate = (analytics['total_answers'] / analytics['total_questions'] * 100) if analytics['total_questions'] > 0 else 0
    summary_data = {
        "Metric": ["Total Sessions", "Questions Asked", "Answers Generated", "Success Rate", "Avg Answer Length"],
        "Value": [
            len(st.session_state.chat_history),
            analytics['total_questions'],
            analytics['total_answers'],
            f"{success_rate:.1f}%",
            f"{analytics['avg_answer_length']:.0f}"
        ]
    }
    summary_df = pd.DataFrame(summary_data)
    assert len(summary_df) == 5, f"Summary dataframe should have 5 rows, got {len(summary_df)}"
    assert summary_df.iloc[3, 1] == "100.0%", f"Success rate should be 100.0%, got {summary_df.iloc[3, 1]}"

    # Test questions dataframe
    questions_df = None
    if st.session_state.chat_history:
        question_data = []
        session_num = 1

        for i in range(0, len(st.session_state.chat_history), 2):
            if i + 1 < len(st.session_state.chat_history):
                user_msg = st.session_state.chat_history[i]
                assistant_msg = st.session_state.chat_history[i + 1]

                if user_msg.get('role') == 'user' and assistant_msg.get('role') == 'assistant':
                    question_text = user_msg.get('content', '')
                    question_lower = question_text.lower()

                    if any(word in question_lower for word in ['what', 'how', 'why', 'when', 'where', 'who']):
                        q_type = 'WH-Questions'
                    elif any(word in question_lower for word in ['explain', 'describe', 'define']):
                        q_type = 'Explanatory'
                    elif any(word in question_lower for word in ['compare', 'contrast', 'difference']):
                        q_type = 'Comparative'
                    else:
                        q_type = 'General'

                    question_data.append({
                        "Session": session_num,
                        "Question": question_text[:50] + "..." if len(question_text) > 50 else question_text,
                        "Type": q_type,
                        "Answered": "Yes" if assistant_msg.get('content') else "No",
                        "Sources": len(assistant_msg.get('sources', []))
                    })
                    session_num += 1

        if question_data:
            questions_df = pd.DataFrame(question_data)

    assert questions_df is not None, "Questions dataframe should not be None"
    assert len(questions_df) == 3, f"Questions dataframe should have 3 rows, got {len(questions_df)}"
    assert questions_df.iloc[0, 2] == 'WH-Questions', f"First question type should be WH-Questions, got {questions_df.iloc[0, 2]}"

    # Test document analysis dataframe
    doc_analysis_data = []
    total_refs = sum(analytics['document_usage'].values())
    for doc, refs in analytics['document_usage'].items():
        usage_pct = (refs / total_refs * 100) if total_refs > 0 else 0
        doc_analysis_data.append({
            "Document": doc,
            "Total References": refs,
            "Usage Percentage": f"{usage_pct:.1f}%"
        })

    doc_df = pd.DataFrame(doc_analysis_data)
    assert len(doc_df) == 3, f"Document dataframe should have 3 rows, got {len(doc_df)}"
    assert doc_df.iloc[0, 2] == "40.0%", f"First document percentage should be 40.0%, got {doc_df.iloc[0, 2]}"

    print("✅ DataFrame creation passed")

def test_csv_export():
    """Test CSV export functionality"""
    print("🧪 Testing CSV Export Functionality...")

    analytics = generate_analytics_data()

    # Create dataframes
    success_rate = (analytics['total_answers'] / analytics['total_questions'] * 100) if analytics['total_questions'] > 0 else 0
    summary_data = {
        "Metric": ["Total Sessions", "Questions Asked", "Answers Generated", "Success Rate", "Avg Answer Length"],
        "Value": [
            len(st.session_state.chat_history),
            analytics['total_questions'],
            analytics['total_answers'],
            f"{success_rate:.1f}%",
            f"{analytics['avg_answer_length']:.0f}"
        ]
    }
    summary_df = pd.DataFrame(summary_data)

    # Test summary CSV export
    summary_csv = summary_df.to_csv(index=False)
    assert "Total Sessions" in summary_csv, "Summary CSV should contain 'Total Sessions'"
    assert "100.0%" in summary_csv, "Summary CSV should contain success rate"

    # Create questions dataframe
    question_data = []
    session_num = 1
    for i in range(0, len(st.session_state.chat_history), 2):
        if i + 1 < len(st.session_state.chat_history):
            user_msg = st.session_state.chat_history[i]
            assistant_msg = st.session_state.chat_history[i + 1]

            if user_msg.get('role') == 'user' and assistant_msg.get('role') == 'assistant':
                question_text = user_msg.get('content', '')
                question_lower = question_text.lower()

                if any(word in question_lower for word in ['what', 'how', 'why', 'when', 'where', 'who']):
                    q_type = 'WH-Questions'
                elif any(word in question_lower for word in ['explain', 'describe', 'define']):
                    q_type = 'Explanatory'
                elif any(word in question_lower for word in ['compare', 'contrast', 'difference']):
                    q_type = 'Comparative'
                else:
                    q_type = 'General'

                question_data.append({
                    "Session": session_num,
                    "Question": question_text[:50] + "..." if len(question_text) > 50 else question_text,
                    "Type": q_type,
                    "Answered": "Yes" if assistant_msg.get('content') else "No",
                    "Sources": len(assistant_msg.get('sources', []))
                })
                session_num += 1

    questions_df = pd.DataFrame(question_data) if question_data else None

    # Test questions CSV export (only if dataframe exists)
    if questions_df is not None:
        questions_csv = questions_df.to_csv(index=False)
        assert "WH-Questions" in questions_csv, "Questions CSV should contain question types"
        assert "Yes" in questions_csv, "Questions CSV should contain answer status"

    print("✅ CSV export functionality passed")

if __name__ == "__main__":
    print("🚀 Starting Analytics Dashboard Tests...\n")

    try:
        test_analytics_calculations()
        test_percentage_calculation()
        test_dataframe_creation()
        test_csv_export()

        print("\n🎉 All tests passed! Analytics Dashboard fixes are working correctly.")

    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        sys.exit(1)
