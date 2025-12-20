#!/usr/bin/env python3
"""
Test script for the summarize_documents function
"""

import sys
import os

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from utils.rag_qa import summarize_documents, load_chunks

    print("Testing summarize_documents function...")

    # Check if chunks exist
    chunks = load_chunks()
    if chunks is None:
        print("❌ No chunks loaded. Please process documents first.")
        sys.exit(1)

    print(f"✅ Loaded {len(chunks)} chunks")

    # Test summarize_documents
    try:
        summary = summarize_documents()
        print("✅ Summary generated successfully!")
        print("\n" + "="*50)
        print("SUMMARY:")
        print("="*50)
        print(summary)
        print("="*50)

        # Basic validation
        if not summary or summary.strip() == "":
            print("❌ Summary is empty!")
        elif "Error" in summary:
            print("❌ Summary contains error message!")
        else:
            print("✅ Summary appears valid")

    except Exception as e:
        print(f"❌ Error generating summary: {str(e)}")
        import traceback
        traceback.print_exc()

except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)
