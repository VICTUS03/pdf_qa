import os
from ragas import evaluate
from ragas.metrics import AnswerRelevancy, Faithfulness
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from datasets import Dataset
import pandas as pd 
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


# Judge 
judge_llm = ChatGroq(
    api_key= get_key("GROQ_API_KEY"),
    model="openai/gpt-oss-safeguard-20b",
    temperature=0 
    )

embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Configuration 
faithfulness_metric = Faithfulness(llm=judge_llm)
answer_relevancy_metric = AnswerRelevancy(llm=judge_llm, embeddings=embedding_model)
answer_relevancy_metric.n = 1

def evaluate_rag(question, answer, contexts):
    
    data = {
        "question": [question],
        "answer": [answer],
        "contexts": [contexts],
    }
    dataset = Dataset.from_dict(data)

    print("⚖️  Judge Llama-3 is evaluating...")

    try:
        results = evaluate(
            dataset=dataset,
            metrics=[faithfulness_metric, answer_relevancy_metric],
            llm=judge_llm,
            embeddings=embedding_model,
            raise_exceptions=False
        )
        

        df = results.to_pandas()
        
        output_dict = df.iloc[0].to_dict()
        
        print(f"✅ Evaluation Success: {output_dict}")
        return output_dict

    except Exception as e:
        print(f"❌ RAGAS Eval Failed: {e}")
        return {"faithfulness": 0.0, "answer_relevancy": 0.0}
