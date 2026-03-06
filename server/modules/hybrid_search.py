from rank_bm25 import BM250kapi
from langchain_core.documents import Document

class HybridRetriever:
    def __init__(self,documents):
        self.documents = documents
        corpus = [doc.page_content.split() for doc in documents]
        self.bm25 = BM250kapi(corpus)

    def keyword_search(self,query,k = 3):
        tokenized_query = query.split()
        scores = self.bm25.get_scores(tokenized_query)

        ranked_docs = sorted(
            zip(self.documents,scores),
            key = lambda x: x[1],
            reverse = True
        )

        return [doc for doc, _ in ranked_docs[:k]]