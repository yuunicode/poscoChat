# infra/retriever.py 내부
from abc import ABC, abstractmethod
from typing import List, Dict 
from config.embedding_config import EmbeddingType
from infra.embedding import BaseEmbedding
from infra.vectorstore import Qdrant
from infra.vectorizer import BaseSparseVectorizer
from sentence_transformers import CrossEncoder
from qdrant_client.http.models import PointStruct, SparseVector, ScoredPoint
import logging

class SearchModule(ABC):
    def __init__(self, 
                 vectorstore: Qdrant):
        self.vectorstore = vectorstore

    @abstractmethod
    def search(self, query: str) -> List[ScoredPoint]: pass

    def apply_cross_encoder_rerank(self,
                                   model_name: str,
                                   query: str,
                                   results: List[PointStruct]):
        
        cross_encoder = CrossEncoder(model_name)
        pairs = [(query, r.payload['content']) if r.payload and r.payload.get('content') else (query,"") for r in results]  # (query, doc) pair 생성
        scores = cross_encoder.predict(pairs)  # Cross-Encoder로 점수 예측
        sorted_results = [x for _, x in sorted(zip(scores, results), key = lambda y: y[0], reverse = True)]  # 점수로 정렬
        return sorted_results
    
class SparseSearch(SearchModule):
    def __init__(self, sparse_vectorizer: BaseSparseVectorizer, collection_name: str, vectorstore: Qdrant):
        super().__init__(vectorstore)
        self.sparse_vectorizer = sparse_vectorizer
        self.collection_name = collection_name
        
    def search(self, query: str) -> List[ScoredPoint]:
        sparse_vectors = self.sparse_vectorizer.transform([query])
        # Convert the dict to a SparseVector if necessary
        sparse_vector = SparseVector(**sparse_vectors[0]) if isinstance(sparse_vectors[0], dict) else sparse_vectors[0]

        result = self.vectorstore.client.query_points(
            collection_name = self.collection_name,
            query = sparse_vector,
            using = EmbeddingType.SPARSE.value,
            limit = 20
        ).points

        return result

class DenseSearch(SearchModule):
    def __init__(self, embedding: BaseEmbedding, collection_name: str, vectorstore: Qdrant):
        super().__init__(vectorstore)
        self.embedding = embedding
        self.collection_name = collection_name
        
    def search(self, query: str) -> List:
        embedding_vector = self.embedding.encode([query])[0]
        result = self.vectorstore.client.query_points(
            collection_name = self.collection_name,
            query = embedding_vector,
            using = self.embedding.embedding_type.value,
            limit = 20
        ).points

        return result
