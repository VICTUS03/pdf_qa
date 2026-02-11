# for qdrant vector database
import os
import json
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from langchain_community.retrievers import BM25Retriever
from langchain_community.document_compressors.flashrank_rerank import FlashrankRerank
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
import streamlit as st

from dotenv import load_dotenv
load_dotenv()

def get_key(key_name):
    try:
        if key_name in st.secrets:
            return st.secrets[key_name]
        
        return os.environ.get(key_name)
        
    except Exception:
        return None


def build_qa_chain(docs):
    # SETUP 
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=300)
    chunks = splitter.split_documents(docs)

    embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    # Qdrant
    qdrant_url = get_key("QDRANT_URL")
    qdrant_key = get_key("QDRANT_API_KEY")

    if qdrant_url and qdrant_key:
        print("☁️ Connecting to Qdrant Cloud...")
        vectorstore = QdrantVectorStore.from_documents(
            documents=chunks,
            embedding=embedding_model,
            url=qdrant_url,
            api_key=qdrant_key,
            collection_name="demo_rag_collection",
            force_recreate=True
        )
    else:
        print("💾 Using Local Disk Qdrant (./qdrant_db)...")
        vectorstore = QdrantVectorStore.from_documents(
            documents=chunks,
            embedding=embedding_model,
            path="./qdrant_db",
            collection_name="local_collection",
            force_recreate=True
        )
    
    qdrant_retriever = vectorstore.as_retriever(search_kwargs={"k": 20}) 
    bm25_retriever = BM25Retriever.from_documents(chunks)
    bm25_retriever.k = 20 
    reranker = FlashrankRerank(model="ms-marco-MiniLM-L-12-v2")

    llm = ChatGroq(
        api_key=get_key("GROQ_API_KEY"),
        model="openai/gpt-oss-120b",
        temperature=0
    )

    # QUERY 
    query_gen_prompt = ChatPromptTemplate.from_template("""
    You are an AI search optimizer. Your goal is to break down a complex user query into 3 distinct search variations to capture different technical aspects of the documentation.
    Rules:
    - Variation 1: Focus on core technical specifications/algorithms.
    - Variation 2: Focus on hardware and system requirements.
    - Variation 3: A broader semantic rephrasing of the entire intent.
    - Output ONLY a JSON list of strings.
    Original Query: {question}
    """)

    query_chain = query_gen_prompt | llm | StrOutputParser()

    # RETRIEVAL CORE (Returns List[Doc]) 
    def get_relevant_docs(question):
        if isinstance(question, dict):
            original_query = question.get("question", "")
        else:
            original_query = question
        
        print(f"🔍 Analyzing Query: '{original_query}'")

        # A. Generate Sub-Queries
        try:
            response = query_chain.invoke({"question": original_query})
            
            if hasattr(response, 'content'):
                sub_queries_json = response.content
            elif isinstance(response, dict):
                sub_queries_json = response.get("text") or response.get("content") or str(response)
            else:
                sub_queries_json = str(response)
            
            clean_json = sub_queries_json.replace("```json", "").replace("```", "").strip()
            sub_queries = json.loads(clean_json)
        except Exception as e:
            # print(f"⚠️ Sub-query generation failed: {e}")
            sub_queries = [original_query]

        # B. Collect Docs
        all_docs = []
        for q in sub_queries:
            if isinstance(q, str) and q.strip():
                docs_semantic = qdrant_retriever.invoke(q)
                docs_keyword = bm25_retriever.invoke(q)
                all_docs.extend(docs_semantic + docs_keyword)

        # C. Deduplicate
        unique_docs_map = {doc.page_content: doc for doc in all_docs}
        unique_docs = list(unique_docs_map.values())
        
        print(f"📥 Retrieved {len(unique_docs)} raw documents.")

        if not unique_docs:
            return []
            
        # D. Rerank (With Safety Valve)
        try:
            reranked_docs = reranker.compress_documents(unique_docs, original_query)
            
            # [CRITICAL FIX] If Reranker filters everything out, use the raw docs!
            if not reranked_docs:
                print("⚠️ Reranker dropped all docs. Using raw fallback.")
                return unique_docs[:10]
                
            print(f"✅ Reranker kept {len(reranked_docs)} most relevant docs.")
            return reranked_docs[:10]
            
        except Exception as e:
            print(f"⚠️ Rerank failed: {e}. Returning raw docs.")
            return unique_docs[:10]

    # FORMATTING
    def format_docs_for_chain(question):
        docs = get_relevant_docs(question)
        context_str = "\n\n".join(doc.page_content for doc in docs)
        return context_str if context_str else "No relevant information found in documents."
    

    # CHAIN 
    final_prompt = ChatPromptTemplate.from_template("""
    You are an expert analyst. Provide a detailed answer based STRICTLY on the context.
    Directives:
    1. **Focus:** Read the context thoroughly, do not skip any context. Answer the specific question asked, look carefully for the answer to the specific question, do not miss any information about the question in the context. Do not drift into unrelated topics. (Improves Relevancy)
    2. **Depth:** Explain the 'how' and 'why' using details strictly from the context only, do not use your thinking or outside knowledge to answer the queations, answer should be from the context even if it is too small or to big use the same context and nothing extra. (Improves Size)
    3. **Evidence:** Cite numbers, metrics, or quotes from the context to support your answer and mention the exact relevant page number for the context where the answer to the question is metioned(if present). (Improves Faithfulness)
    4. **Safety:** If the context is empty or irrelevant, say "Data not available", and do not say " No additional quantitative data or detailed descriptions are provided in the context."
    Context:
    {context}
    
    Question: {question}
    Answer:
    """)

    chain = (
        {"context": RunnableLambda(format_docs_for_chain), "question": RunnablePassthrough()}
        | final_prompt
        | llm
        | StrOutputParser()
    )

    return chain, get_relevant_docs


