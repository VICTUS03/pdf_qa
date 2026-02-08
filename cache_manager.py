import time
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

class SemanticCache:
    def __init__(self, threshold=0.35):
        self.threshold = threshold
        self.embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        self.vectorstore = None 

    def get_cached_response(self, question):
        if self.vectorstore is None:
            return None
        
        try:
            results_with_score = self.vectorstore.similarity_search_with_score(question, k=1)
            if not results_with_score:
                return None
                
            stored_doc, score = results_with_score[0]
            
            if score < self.threshold:
                print(f"✅ Cache Hit! (Score: {score:.4f})")
                return {
                    "response": stored_doc.metadata["response"],
                    "context": stored_doc.metadata.get("context", []) # Retrieve Context too!
                }
            
            print(f"❌ Cache Miss (Score: {score:.4f})")
            return None
        except Exception as e:
            print(f"Cache Error: {e}")
            return None

    def add_to_cache(self, question, response, context_docs):
        # We store the Context in metadata
        doc = Document(
            page_content=question,
            metadata={
                "response": response,
                "context": [d.page_content for d in context_docs] # Store text only
            }
        )
        
        if self.vectorstore is None:
            self.vectorstore = FAISS.from_documents([doc], self.embedding_model)
        else:
            self.vectorstore.add_documents([doc])
        
        print(f"💾 Saved to Cache: '{question}'")