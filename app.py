import streamlit as st
import os
import asyncio
from utils import save_uploaded_files, load_pdfs
from qa_engine import build_qa_chain
from cache_manager import SemanticCache
from evaluator import evaluate_rag 

# Page Config
st.set_page_config(page_title="Groq-Powered Doc Q&A", layout="centered")
st.title("⚡ Ultra-Fast Doc Q&A (Multi-PDF + Eval)")

st.info("""
### ℹ️ System Logic & Performance Note
* **Routing Logic:** To optimize for cost and speed, documents with **>20 pages** are processed using PyPDFLoader (local), while smaller, complex files use LlamaParse (AI-powered).
* **Evaluation Metrics:** You may occasionally see NaN for Faithfulness or Relevancy. This is a known behavior of the RAGAS framework when the LLM 'Judge' encounteres formatting issues or empty contexts, actively working on a more robust local evaluation model.
""", icon="💡")

# Session State
if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None
if "retriever_func" not in st.session_state:
    st.session_state.retriever_func = None
if "processing" not in st.session_state:
    st.session_state.processing = False
if "cache" not in st.session_state:
    st.session_state.cache = SemanticCache(threshold=0.2)
if "uploader_key" not in st.session_state:
    st.session_state.uploader_key = 0
if "last_retrieved_docs" not in st.session_state:
    st.session_state.last_retrieved_docs = []

# Sidebar Logic 
def clear_everything():
    st.session_state.qa_chain = None
    st.session_state.retriever_func = None
    st.session_state.cache = SemanticCache()
    st.session_state.processing = False
    st.session_state["input_box"] = ""
    st.session_state.last_retrieved_docs = []
    st.session_state.uploader_key += 1 



async def process_documents(file_paths):
    loop = asyncio.get_event_loop()
    docs = await loop.run_in_executor(None, load_pdfs, file_paths)
    chain, retriever_fn = await loop.run_in_executor(None, build_qa_chain, docs)
    return chain, retriever_fn

# Main Interface 
uploaded_files = st.file_uploader(
    "Upload PDF file(s)", 
    type=["pdf"], 
    accept_multiple_files=True,
    key=f"uploader_{st.session_state.uploader_key}",
    max_upload_size=10
)
status_container = st.empty()

if uploaded_files:
    if st.session_state.qa_chain is None and not st.session_state.processing:
        st.session_state.processing = True
        file_paths = save_uploaded_files(uploaded_files)
        status_container.info(f"🚀 Processing {len(uploaded_files)} files... (Async)")
        
        try:
            chain, retriever_fn = asyncio.run(process_documents(file_paths))
            st.session_state.qa_chain = chain
            st.session_state.retriever_func = retriever_fn
            status_container.success("✅ Knowledge Base Ready!")
        except Exception as e:
            status_container.error(f"Error: {e}")
        
        st.session_state.processing = False

user_query = st.text_input("Ask a question:", key="input_box")
button = st.button("Enter")

if user_query and st.session_state.qa_chain:
    st.markdown("### ▌ Answer:")
    with st.chat_message("assistant"):
        # 1. CHECK CACHE
        cached_data = st.session_state.cache.get_cached_response(user_query)
        
        if cached_data:
            st.success("⚡ Served from Semantic Cache")
            full_response = cached_data["response"]
            # Restore context from cache!
            st.session_state.last_retrieved_docs = cached_data.get("context", [])
            st.markdown(full_response)
        else:
            # 2. IF MISS, RETRIEVE NEW
            try:
                # Capture docs
                relevant_docs = st.session_state.retriever_func(user_query)
                st.session_state.last_retrieved_docs = relevant_docs 
            except Exception as e:
                st.error(f"Retrieval Error: {e}")
                relevant_docs = []

            # Generate
            full_response = ""
            response_placeholder = st.empty()
            try:
                for chunk in st.session_state.qa_chain.stream(user_query):
                    full_response += chunk
                    response_placeholder.markdown(full_response + "▌")
                response_placeholder.markdown(full_response)
                
                # SAVE TO CACHE 
                st.session_state.cache.add_to_cache(
                    user_query, 
                    full_response, 
                    st.session_state.last_retrieved_docs
                )
            except Exception as e:
                st.error(f"Generation Error: {e}")

        # EVALUATION SECTION
        st.divider()
        if st.button("📊 Evaluate Answer (RAGAS)"):
            with st.spinner("⚖️  Judge Llama-3 is evaluating..."):
                try:
                    # Use the context we just loaded (either from Fresh Retrieval OR Cache)
                    raw_context = st.session_state.last_retrieved_docs
                    
                    # Normalization: Handle if it's Document objects or just strings from cache
                    context_texts = []
                    for item in raw_context:
                        if isinstance(item, str):
                            context_texts.append(item)
                        elif hasattr(item, 'page_content'):
                            context_texts.append(item.page_content)

                    # Debug Info
                    if not context_texts:
                        st.warning("⚠️ Context is empty. Scores will be 0.")
                    else:
                        st.info(f"📝 Evaluating against {len(context_texts)} context chunks.")

                    scores = evaluate_rag(
                        question=user_query,
                        answer=full_response,
                        contexts=context_texts
                    )
                    
                    c1, c2 = st.columns(2)
                    
                    def safe_score(val):
                        try:
                            if isinstance(val, list): val = val[0]
                            return float(val)
                        except:
                            return 0.0

                    f_score = safe_score(scores.get('faithfulness', 0))
                    r_score = safe_score(scores.get('answer_relevancy', 0))

                    c1.metric("Faithfulness", f"{f_score:.2f}")
                    c2.metric("Relevancy", f"{r_score:.2f}")
                    
                except Exception as e:
                    st.error(f"Evaluation Failed: {e}")

# SIDEBAR 
with st.sidebar:
    st.header("🔧 Settings")
    st.button("Clear Documents & Cache", on_click=clear_everything)
    st.markdown("---")
    st.markdown("**Status:**")
    if st.session_state.qa_chain:
        st.success("System Ready ✅")
    else:
        st.warning("No Documents ❌")
