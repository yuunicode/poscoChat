# infra/retriever.py 내부
from abc import ABC, abstractmethod
from typing import List, Dict 
from config.embedding_config import EmbeddingType
from infra.embedding import BaseEmbedding
from infra.vectorstore import Qdrant
from infra.vectorizer import BaseSparseVectorizer
from sentence_transformers import CrossEncoder
from qdrant_client.http.models import PointStruct, SparseVector, ScoredPoint, Fusion, FusionQuery, Prefetch
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
        
    def search(self, query: str) -> List[ScoredPoint]:
        embedding_vector = list(self.embedding.encode([query]))[0]
        # List[float] 타입 커버
        if hasattr(embedding_vector, "tolist"):
            embedding_vector = embedding_vector.tolist()
        if not isinstance(embedding_vector, list):
            embedding_vector = list(embedding_vector)

        result = self.vectorstore.client.query_points(
            collection_name = self.collection_name,
            query = embedding_vector,
            using = self.embedding.embedding_type.value,
            limit = 20
        ).points

        return result

class HybridSearch(SearchModule):
    def __init__(self, sparse_vectorizer: BaseSparseVectorizer, embedding: BaseEmbedding, collection_name: str, vectorstore: Qdrant):
        super().__init__(vectorstore)
        self.embedding = embedding
        self.sparse_vectorizer = sparse_vectorizer
        self.collection_name = collection_name
        
    def search(self, query: str) -> List:
        # 희소벡터
        sparse_vectors = self.sparse_vectorizer.transform([query])
        sparse_vector = SparseVector(**sparse_vectors[0]) if isinstance(sparse_vectors[0], dict) else sparse_vectors[0]

        # 밀집벡터
        embedding_vector = list(self.embedding.encode([query]))[0]
        # List[float] 타입 커버
        if hasattr(embedding_vector, "tolist"):
            embedding_vector = embedding_vector.tolist()
        if not isinstance(embedding_vector, list):
            embedding_vector = list(embedding_vector)

        result = self.vectorstore.client.query_points(
            collection_name = self.collection_name,
            prefetch = [
                Prefetch(
                    query = sparse_vector,
                    using = EmbeddingType.SPARSE.value,
                    limit = 20,
                ),
                Prefetch(
                    query = embedding_vector,
                    using = self.embedding.embedding_type.value,
                    limit = 20

                ),
            ],
            query = FusionQuery(fusion = Fusion.RRF)
        ).points
        
        return result

class MultiStageSearch(SearchModule):
    def __init__(self, dense_embedding: BaseEmbedding, colbert_embedding: BaseEmbedding, collection_name: str, vectorstore: Qdrant):
        super().__init__(vectorstore)
        self.dense_embedding = dense_embedding
        self.colbert_embedding = colbert_embedding
        self.collection_name = collection_name
        
    def search(self, query: str) -> List:

        # 밀집벡터
        dense_embedding_vector = list(self.dense_embedding.encode([query]))[0]
        # List[float] 타입 커버
        if hasattr(dense_embedding_vector, "tolist"):
            dense_embedding_vector = dense_embedding_vector.tolist()
        if not isinstance(dense_embedding_vector, list):
            dense_embedding_vector = list(dense_embedding_vector)

        # 코버트 벡터
        colbert_embedding_vector = (list(self.colbert_embedding.encode([query]))[0]).tolist()
        if hasattr(colbert_embedding_vector, "tolist"):
            colbert_embedding_vector = colbert_embedding_vector.tolist()
        if not isinstance(colbert_embedding_vector, list):
            colbert_embedding_vector = list(colbert_embedding_vector)
                
        result = self.vectorstore.client.query_points(
            collection_name = self.collection_name,
            prefetch =
                Prefetch(
                    query = dense_embedding_vector,
                    using = self.dense_embedding.embedding_type.value,
                    limit = 100,
                ),
            query = colbert_embedding_vector,
            using = self.colbert_embedding.embedding_type.value,
            limit = 20
        ).points
        
        return result

