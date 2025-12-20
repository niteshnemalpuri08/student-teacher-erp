import os
import io
import pickle
import numpy as np
import faiss
import logging
import streamlit as st
from pypdf import PdfReader
from docx import Document
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set your OpenAI API key here - REPLACE WITH YOUR ACTUAL KEY
os.environ["OPENAI_API_KEY"] = "your_openai_api_key_here"

# Demo mode: allow running without real API key
DEMO_MODE = os.environ.get("OPENAI_API_KEY") == "your_openai_api_key_here"

if DEMO_MODE:
    print("⚠️ Running in demo mode without OpenAI API key. Answers will be simulated.")
    openai = None  # Will use simulated responses
else:
    import openai
    print("✅ OpenAI API key detected. Full functionality available.")

@st.cache_resource
def load_sentence_transformer():
    """Load and cache the sentence transformer model"""
    return SentenceTransformer('all-mpnet-base-v2')

@st.cache_data
def process_documents(files):
    """
    Process uploaded files: extract text, chunk it, create embeddings, and store in FAISS vector database.
    Input: list of Streamlit UploadedFile objects
    Output: Saves FAISS index and chunks to disk
    """
    documents = []

    # Extract text from each file with error handling
    for file in files:
        try:
            if file.size > 10 * 1024 * 1024:  # 10MB limit
                raise ValueError(f"File {file.name} is too large (>10MB)")

            if file.name.endswith('.pdf'):
                pdf = PdfReader(io.BytesIO(file.read()))
                for page_num, page in enumerate(pdf.pages):
                    text = page.extract_text()
                    if text.strip():  # Only add non-empty pages
                        documents.append({
                            'text': text,
                            'metadata': {'source': file.name, 'page': page_num + 1}
                        })
            elif file.name.endswith('.docx'):
                doc = Document(io.BytesIO(file.read()))
                text = '\n'.join([para.text for para in doc.paragraphs])
                documents.append({
                    'text': text,
                    'metadata': {'source': file.name, 'page': 'N/A'}
                })
            elif file.name.endswith('.txt'):
                text = file.read().decode('utf-8', errors='ignore')
                documents.append({
                    'text': text,
                    'metadata': {'source': file.name, 'page': 'N/A'}
                })
        except Exception as e:
            raise ValueError(f"Error processing {file.name}: {str(e)}")

    if not documents:
        raise ValueError("No valid text extracted from uploaded files")

    # Optimized chunking for better performance and accuracy
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,  # Reduced from 1200 for better semantic granularity
        chunk_overlap=200,  # Adjusted from 250 for optimal context continuity
        separators=["\n\n", "\n", ". ", " ", ""],  # Prioritize paragraph breaks
        length_function=len,
        is_separator_regex=False
    )
    chunks = []
    for doc in documents:
        splits = text_splitter.split_text(doc['text'])
        for split in splits:
            if split.strip():  # Skip empty chunks
                chunks.append({
                    'page_content': split,
                    'metadata': doc['metadata']
                })

    if not chunks:
        raise ValueError("No text chunks created from documents")

    # Create embeddings with progress indicators
    model = load_sentence_transformer()

    # Add progress bar for embedding generation
    progress_bar = st.progress(0, text="Generating embeddings...")
    total_chunks = len(chunks)
    embeddings = []

    for i, chunk in enumerate(chunks):
        embedding = model.encode(chunk['page_content'])
        embeddings.append(embedding)
        progress_bar.progress((i + 1) / total_chunks, text=f"Generating embeddings... ({i + 1}/{total_chunks})")

    progress_bar.empty()  # Remove progress bar when done

    # Create FAISS index with cosine similarity
    dimension = len(embeddings[0])
    index = faiss.IndexFlatIP(dimension)  # Inner Product for cosine similarity
    embeddings_array = np.array(embeddings)
    faiss.normalize_L2(embeddings_array)  # Normalize for cosine similarity
    index.add(embeddings_array)

    # Save index and chunks
    faiss.write_index(index, "faiss_index.idx")
    with open("chunks.pkl", "wb") as f:
        pickle.dump(chunks, f)

def preprocess_question(question):
    """Clean and optimize question for retrieval with enhanced query expansion"""
    # Remove extra whitespace
    question = " ".join(question.split())

    # Enhanced query expansion for better retrieval
    question_lower = question.lower()

    # Add context hints for common question types
    if question_lower.startswith(("what is", "what are", "define")):
        question += " Provide details and explanation from the documents."
    elif question_lower.startswith(("how does", "how do", "explain")):
        question += " Provide step-by-step details from the documents."
    elif question_lower.startswith(("why", "when", "where")):
        question += " Provide context and reasoning from the documents."
    elif question_lower.startswith(("who", "which")):
        question += " Provide specific information from the documents."

    # Add synonyms and related terms for common concepts
    expansions = {
        "machine learning": ["ML", "artificial intelligence", "AI"],
        "data science": ["analytics", "data analysis"],
        "python": ["programming", "coding"],
        "algorithm": ["method", "technique", "approach"],
        "performance": ["efficiency", "speed", "accuracy"],
        "model": ["system", "framework", "architecture"]
    }

    # Expand question with synonyms if found
    expanded_terms = []
    for term, synonyms in expansions.items():
        if term in question_lower:
            expanded_terms.extend(synonyms[:2])  # Limit to 2 synonyms per term

    if expanded_terms:
        question += f" Related terms: {', '.join(expanded_terms)}."

    return question

@st.cache_resource
def load_faiss_index():
    """Load and cache the FAISS index"""
    if os.path.exists("faiss_index.idx"):
        return faiss.read_index("faiss_index.idx")
    return None

@st.cache_data
def load_chunks():
    """Load and cache the document chunks"""
    if os.path.exists("chunks.pkl"):
        with open("chunks.pkl", "rb") as f:
            return pickle.load(f)
    return None

def retrieve_chunks(question, k=5, fetch_k=15, lambda_mult=0.7):
    """
    Retrieve top-k relevant chunks for a given question using MMR for diversity.
    MMR balances relevance and diversity to avoid redundant chunks.

    Parameters:
    - question (str): The query question
    - k (int): Final number of chunks to return (default: 5)
    - fetch_k (int): Number of initial candidates to fetch (default: 20)
    - lambda_mult (float): MMR balance parameter (0.5 = equal relevance/diversity)

    Returns:
    - list of chunk dictionaries with metadata and similarity scores
    """
    index = load_faiss_index()
    chunks = load_chunks()

    if index is None or chunks is None:
        raise ValueError("Documents not processed yet. Please upload and process documents first.")

    try:
        # Preprocess question
        question = preprocess_question(question)

        # Use cached model
        model = load_sentence_transformer()
        query_embedding = model.encode(question)
        query_embedding = np.array([query_embedding])
        faiss.normalize_L2(query_embedding)  # Normalize for cosine similarity

        # Get initial candidates (more than k for MMR selection)
        candidates_k = min(fetch_k, len(chunks))
        distances, indices = index.search(query_embedding, candidates_k)

        # Create candidates with embeddings and similarity scores
        candidates = []
        candidate_embeddings = []

        for dist, idx in zip(distances[0], indices[0]):
            if idx < len(chunks):
                chunk_copy = chunks[idx].copy()
                chunk_copy['similarity'] = float(dist)  # Cosine similarity to query

                # Get chunk embedding from FAISS index (reconstruct from search)
                chunk_embedding = index.reconstruct(idx)
                chunk_copy['embedding'] = chunk_embedding

                candidates.append(chunk_copy)
                candidate_embeddings.append(chunk_embedding)

        if not candidates:
            return []

        # Apply MMR (Maximal Marginal Relevance) for diversity
        selected = []
        remaining = candidates.copy()

        # First selection: highest relevance to query
        if remaining:
            best_first = max(remaining, key=lambda x: x['similarity'])
            selected.append(best_first)
            remaining.remove(best_first)

        # Subsequent selections using MMR
        while len(selected) < k and remaining:
            best_score = -float('inf')
            best_chunk = None

            for chunk in remaining:
                # Relevance score (similarity to query)
                rel_score = chunk['similarity']

                # Diversity penalty (max similarity to already selected chunks)
                max_sim_to_selected = 0.0
                if selected:
                    chunk_emb = chunk['embedding']
                    similarities_to_selected = [
                        np.dot(chunk_emb, sel['embedding']) for sel in selected
                    ]
                    max_sim_to_selected = max(similarities_to_selected)

                # MMR score: balance relevance vs diversity
                mmr_score = lambda_mult * rel_score - (1 - lambda_mult) * max_sim_to_selected

                if mmr_score > best_score:
                    best_score = mmr_score
                    best_chunk = chunk

            if best_chunk:
                selected.append(best_chunk)
                remaining.remove(best_chunk)

        # Remove embedding from final results (not needed for output)
        for chunk in selected:
            if 'embedding' in chunk:
                del chunk['embedding']

        return selected

    except Exception as e:
        logger.error(f"Error in MMR retrieval: {str(e)}")
        # Fallback to simple similarity search if MMR fails
        try:
            distances, indices = index.search(query_embedding, k)
            fallback_chunks = []
            for dist, idx in zip(distances[0], indices[0]):
                if idx < len(chunks):
                    chunk_copy = chunks[idx].copy()
                    chunk_copy['similarity'] = float(dist)
                    fallback_chunks.append(chunk_copy)
            return fallback_chunks
        except Exception as fallback_error:
            raise ValueError(f"Error retrieving chunks: {str(e)} (fallback also failed: {str(fallback_error)})")

def check_context_sufficiency(retrieved_chunks, similarity_threshold=0.3):
    """
    Check if retrieved chunks are sufficiently relevant to answer the question.
    Input: retrieved_chunks (list), similarity_threshold (float)
    Output: bool - True if context is sufficient, False otherwise
    """
    if not retrieved_chunks:
        return False

    # Check if any chunk has sufficient similarity score
    max_similarity = max(chunk.get('similarity', 0) for chunk in retrieved_chunks)
    return max_similarity >= similarity_threshold

def validate_answer_against_context(answer, retrieved_chunks):
    """
    Validate that the answer is grounded in the retrieved context.
    Input: answer (str), retrieved_chunks (list)
    Output: bool - True if answer is valid, False if hallucinated
    """
    if not answer or not retrieved_chunks:
        return False

    # Check for hallucination indicators
    hallucination_indicators = [
        "i think", "i believe", "probably", "likely", "generally",
        "typically", "usually", "often", "sometimes", "maybe",
        "perhaps", "could be", "might be", "as far as i know"
    ]

    answer_lower = answer.lower()
    if any(indicator in answer_lower for indicator in hallucination_indicators):
        return False

    # Check if answer contains text from context (basic validation)
    context_text = " ".join(chunk['page_content'] for chunk in retrieved_chunks).lower()
    answer_words = set(answer_lower.split())

    # At least 20% of answer words should appear in context
    context_words = set(context_text.split())
    overlap_ratio = len(answer_words & context_words) / len(answer_words) if answer_words else 0

    return overlap_ratio >= 0.2

def generate_answer(question, retrieved_chunks):
    """
    Generate an answer with STRICT ZERO-HALLUCINATION MODE.
    Input: question (str), retrieved_chunks (list)
    Output: answer (str)
    """
    # 2️⃣ CONTEXT SUFFICIENCY CHECK
    if not check_context_sufficiency(retrieved_chunks, similarity_threshold=0.5):
        return "Answer not found in the provided documents."

    # Sort chunks by relevance score
    retrieved_chunks.sort(key=lambda x: x.get('similarity', 0), reverse=True)

    # Create context with source attribution
    context_parts = []
    for i, chunk in enumerate(retrieved_chunks):
        context_parts.append(f"[Chunk {i+1} - {chunk['metadata']['source']} p.{chunk['metadata']['page']}]: {chunk['page_content']}")
    context = "\n\n".join(context_parts)

    # Demo mode: provide simulated response with source attribution
    if DEMO_MODE or openai is None:
        # Generate simulated answer with inline citations
        chunk = retrieved_chunks[0]
        doc_name = chunk['metadata']['source']
        page = chunk['metadata']['page']
        snippet = chunk['page_content'][:150] + "..." if len(chunk['page_content']) > 150 else chunk['page_content']

        if "what" in question.lower():
            answer = f"Based on the documents, \"{snippet}\" [{doc_name}, Page {page}]"
        elif "how" in question.lower():
            answer = f"The documents explain: \"{snippet}\" [{doc_name}, Page {page}]"
        else:
            answer = f"According to the source material: \"{snippet}\" [{doc_name}, Page {page}]"

        # 3️⃣ LLM OUTPUT VALIDATION (for demo mode)
        if validate_answer_against_context(answer, retrieved_chunks):
            return answer
        else:
            return "Answer not found in the provided documents."

    # 1️⃣ STRICT RAG PROMPT - ZERO HALLUCINATION VERSION
    prompt = "You are a STRICT document analysis system. You MUST answer using ONLY the provided context.\n\n" + \
"ABSOLUTE RULES (VIOLATION = REFUSAL):\n" + \
"1. NEVER use external knowledge or general knowledge\n" + \
"2. NEVER guess, assume, or speculate\n" + \
"3. NEVER provide information not explicitly in the context\n" + \
"4. If question cannot be answered from context → respond ONLY with: \"Answer not found in the provided documents.\"\n" + \
"5. Do NOT explain why you cannot answer - just refuse\n\n" + \
f"Context chunks:\n{context}\n\n" + \
f"Question: {question}\n\n" + \
"Answer using ONLY the context above. If insufficient context, respond with exactly: \"Answer not found in the provided documents.\""

    try:
        # Use GPT-4 for better reasoning if available, fallback to GPT-3.5-turbo
        model = "gpt-4" if not DEMO_MODE else "gpt-3.5-turbo"

        response = openai.ChatCompletion.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a zero-hallucination document analysis system. Refuse to answer if information is not in context."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.0,  # Zero temperature for consistency
            max_tokens=600,
            presence_penalty=0.0  # No creativity
        )
        answer = response.choices[0].message.content.strip()

        # 3️⃣ LLM OUTPUT VALIDATION
        if not validate_answer_against_context(answer, retrieved_chunks):
            return "Answer not found in the provided documents."

        # Check for refusal message in response
        if "Answer not found in the provided documents" in answer:
            return "Answer not found in the provided documents."

        # Post-processing: ensure citations
        if retrieved_chunks and not any(chunk['metadata']['source'] in answer for chunk in retrieved_chunks):
            answer += f"\n\nSource: {retrieved_chunks[0]['metadata']['source']}"

        return answer if answer else "Answer not found in the provided documents."
    except Exception as e:
        logger.error(f"Error generating answer: {str(e)}")
        return "Answer not found in the provided documents."

def extract_sources(retrieved_chunks, max_sources=5):
    """
    Extract and deduplicate sources from retrieved chunks.
    Input: retrieved_chunks (list), max_sources (int)
    Output: list of unique sources with document, page, and relevant snippet
    """
    # Sort chunks by relevance score (highest first)
    sorted_chunks = sorted(retrieved_chunks, key=lambda x: x.get('similarity', 0), reverse=True)

    # Deduplicate by document and page
    seen_sources = set()
    sources = []

    for chunk in sorted_chunks:
        source_key = (chunk['metadata']['source'], chunk['metadata']['page'])

        # Skip if we've already included this document/page
        if source_key in seen_sources:
            continue

        # Extract relevant snippet (first 300-400 chars, or full chunk if shorter)
        snippet_length = min(350, len(chunk['page_content']))
        snippet = chunk['page_content'][:snippet_length]
        if len(chunk['page_content']) > snippet_length:
            snippet += "..."

        sources.append({
            "document": chunk['metadata']['source'],
            "page": chunk['metadata']['page'],
            "snippet": snippet
        })

        seen_sources.add(source_key)

        # Limit to max_sources
        if len(sources) >= max_sources:
            break

    return sources

def get_answer(question):
    """
    Main function to get answer, sources, and retrieved chunks for a question.
    Input: question (str)
    Output: dict with answer, sources, and retrieved_chunks
    """
    chunks = retrieve_chunks(question)
    answer = generate_answer(question, chunks)

    # Extract deduplicated sources
    sources = extract_sources(chunks, max_sources=5)

    return {
        "answer": answer,
        "sources": sources,
        "retrieved_chunks": chunks
    }

def log_retrieval(question, retrieved_chunks, answer):
    """Log retrieval and generation details for debugging"""
    logger.info(f"Question: {question}")
    logger.info(f"Retrieved {len(retrieved_chunks)} chunks:")
    for i, chunk in enumerate(retrieved_chunks):
        logger.info(f"  Chunk {i+1}: {chunk['metadata']} - Similarity: {chunk.get('similarity', 'N/A')}")
    logger.info(f"Generated Answer: {answer}")

def evaluate_retrieval(question, retrieved_chunks, ground_truth_chunks):
    """Evaluate retrieval accuracy"""
    retrieved_texts = {chunk['page_content'] for chunk in retrieved_chunks}
    ground_truth_texts = {chunk['page_content'] for chunk in ground_truth_chunks}

    precision = len(retrieved_texts & ground_truth_texts) / len(retrieved_texts) if retrieved_texts else 0
    recall = len(retrieved_texts & ground_truth_texts) / len(ground_truth_texts) if ground_truth_texts else 0

    return {"precision": precision, "recall": recall, "f1": 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0}

def select_representative_chunks(chunks, chunks_per_doc=3):
    """
    Select representative chunks from all documents for summarization.
    Input: chunks (list), chunks_per_doc (int)
    Output: list of selected chunks
    """
    # Group chunks by document
    docs_chunks = {}
    for chunk in chunks:
        doc_name = chunk['metadata']['source']
        if doc_name not in docs_chunks:
            docs_chunks[doc_name] = []
        docs_chunks[doc_name].append(chunk)

    selected_chunks = []

    # For each document, select representative chunks
    for doc_name, doc_chunks in docs_chunks.items():
        # Sort by content length (prefer substantial chunks) and position
        sorted_chunks = sorted(doc_chunks, key=lambda x: len(x['page_content']), reverse=True)

        # Take top chunks_per_doc from each document
        selected_chunks.extend(sorted_chunks[:chunks_per_doc])

    return selected_chunks

@st.cache_data
def summarize_documents():
    """
    Generate a concise, structured summary of all uploaded documents.
    Uses representative chunks from all documents to create a summary.
    Output: summary string
    """
    chunks = load_chunks()

    if chunks is None:
        raise ValueError("No documents processed yet. Please upload and process documents first.")

    try:
        if not chunks:
            raise ValueError("No document chunks available for summarization.")

        # Select representative chunks
        representative_chunks = select_representative_chunks(chunks, chunks_per_doc=3)

        # Create context from selected chunks
        context_parts = []
        for i, chunk in enumerate(representative_chunks):
            context_parts.append(f"[Document: {chunk['metadata']['source']}, Page {chunk['metadata']['page']}]: {chunk['page_content']}")
        context = "\n\n".join(context_parts)

        # Strict summarization prompt
        prompt = "You are a STRICT document summarization system. You MUST summarize using ONLY the provided document content.\n\n" + \
"ABSOLUTE RULES:\n" + \
"1. NEVER use external knowledge or general knowledge\n" + \
"2. NEVER add information not explicitly in the documents\n" + \
"3. NEVER speculate or assume\n" + \
"4. Create a concise summary of 5-7 bullet points covering key topics, sections, or objectives\n" + \
"5. If insufficient content → respond with: \"Unable to generate summary from available documents.\"\n\n" + \
f"Document Content:\n{context}\n\n" + \
"Summary (5-7 bullet points):"

        # Demo mode: provide simulated summary
        if DEMO_MODE or openai is None:
            # Extract document names and create detailed summary from actual content
            doc_names = list(set(chunk['metadata']['source'] for chunk in representative_chunks))

            # Analyze content to create more detailed summary
            all_text = " ".join([chunk['page_content'] for chunk in representative_chunks])
            word_count = len(all_text.split())

            # Extract key phrases and topics (simple keyword extraction)
            common_words = {}
            stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'must', 'can', 'shall'}

            for word in all_text.lower().split():
                word = word.strip('.,!?;:()[]{}"\'')
                if len(word) > 3 and word not in stop_words:
                    common_words[word] = common_words.get(word, 0) + 1

            # Get top keywords
            top_keywords = sorted(common_words.items(), key=lambda x: x[1], reverse=True)[:8]
            keywords_list = [word for word, count in top_keywords if count > 1]

            # Create detailed summary points
            summary_points = [
                f"• **Documents Analyzed**: {', '.join(doc_names)} ({len(representative_chunks)} sections processed)",
                f"• **Content Overview**: {word_count} words total across all documents",
                f"• **Key Topics Identified**: {', '.join(keywords_list[:5]) if keywords_list else 'Various topics covered'}",
                "• **Document Structure**: Content organized into representative sections for efficient analysis",
                "• **Analysis Ready**: System prepared for detailed question answering and information retrieval"
            ]

            # Add content preview if available
            if representative_chunks:
                first_chunk = representative_chunks[0]
                preview = first_chunk['page_content'][:200] + "..." if len(first_chunk['page_content']) > 200 else first_chunk['page_content']
                summary_points.insert(2, f"• **Content Preview**: {preview}")

            return "\n".join(summary_points)

        # Use LLM for summarization
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a zero-hallucination document summarization system. Only summarize what is explicitly in the provided content."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,  # Low temperature for consistency
            max_tokens=500
        )

        summary = response.choices[0].message.content.strip()

        # Validate summary doesn't hallucinate
        if not summary or "Unable to generate summary" in summary:
            return "Unable to generate summary from available documents."

        # Basic validation - check if summary mentions document content
        context_lower = context.lower()
        summary_lower = summary.lower()

        # Ensure at least some overlap with original content
        context_words = set(context_lower.split())
        summary_words = set(summary_lower.split())
        overlap_ratio = len(context_words & summary_words) / len(summary_words) if summary_words else 0

        if overlap_ratio < 0.1:
            return "Unable to generate summary from available documents."

        return summary

    except Exception as e:
        logger.error(f"Error generating document summary: {str(e)}")
        return "Error generating document summary. Please try again."
