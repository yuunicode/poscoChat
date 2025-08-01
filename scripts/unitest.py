import pytest
from unittest.mock import MagicMock
from infra.vectorstore import Qdrant
from infra.vectorizer import BaseSparseVectorizer
from infra.embedding import BaseEmbedding
from config.embedding_config import EmbeddingType
from sentence_transformers import CrossEncoder
from qdrant_client.http.models import PointStruct, SparseVector, ScoredPoint
from infra.retriever import SparseSearch, DenseSearch


# Qdrant의 mock 객체 생성
@pytest.fixture
def mock_qdrant():
    # Qdrant의 mock 객체를 생성
    mock_qdrant_instance = MagicMock(spec=Qdrant)
    
    # Qdrant의 client 속성을 mock
    mock_qdrant_instance.client = MagicMock()
    
    # mock for client.query_points method
    # 여기서 query_points가 반환하는 값을 mock 설정
    mock_qdrant_instance.client.query_points.return_value = MagicMock(
        points=[ScoredPoint(id="1", version = 1, score=0.9, vector=[], payload={})]
    )
    
    # mock for client.upsert method
    mock_qdrant_instance.client.upsert.return_value = None
    
    # mock for client.update_vectors method
    mock_qdrant_instance.client.update_vectors.return_value = None
    
    return mock_qdrant_instance

@pytest.fixture
def mock_embedding():
    # BaseEmbedding의 mock 객체 생성
    mock_embedding = MagicMock(spec=BaseEmbedding)
    mock_embedding.encode.return_value = [[0.1, 0.2, 0.3]]  # 예시 벡터 반환
    mock_embedding.embedding_type = EmbeddingType.DENSE_LARGE
    return mock_embedding

@pytest.fixture
def mock_sparse_vectorizer():
    # BaseSparseVectorizer의 mock 객체 생성
    mock_sparse_vectorizer = MagicMock(spec=BaseSparseVectorizer)
    mock_sparse_vectorizer.transform.return_value = [{"indices": [1, 2], "values": [0.5, 0.6]}]
    return mock_sparse_vectorizer

def test_sparse_search(mock_qdrant, mock_sparse_vectorizer):
    # SparseSearch 인스턴스 생성
    sparse_search = SparseSearch(sparse_vectorizer=mock_sparse_vectorizer, collection_name="test_collection", vectorstore=mock_qdrant)

    # search 메서드 실행
    results = sparse_search.search("test query")

    # 쿼리 결과 검증
    assert len(results) == 1  # 결과는 하나의 ScoredPoint 객체여야 함
    assert isinstance(results[0], ScoredPoint)  # 결과는 ScoredPoint 객체여야 함
    assert results[0].id == "1"  # ID 검증
    assert results[0].score == 0.9  # 점수 검증

    # `query_points`가 올바르게 호출되었는지 검증
    mock_qdrant.client.query_points.assert_called_once_with(
        collection_name="test_collection",
        query=SparseVector(indices=[1, 2], values=[0.5, 0.6]),
        using="sparse",
        limit=20
    )

def test_dense_search(mock_qdrant, mock_embedding):
    # DenseSearch 인스턴스 생성
    dense_search = DenseSearch(embedding=mock_embedding, collection_name="test_collection", vectorstore=mock_qdrant)

    # search 메서드 실행
    results = dense_search.search("test query")

    # 쿼리 결과 검증
    assert len(results) == 1  # 결과는 하나의 ScoredPoint 객체여야 함
    assert isinstance(results[0], ScoredPoint)  # 결과는 ScoredPoint 객체여야 함
    assert results[0].id == "1"  # ID 검증
    assert results[0].score == 0.9  # 점수 검증

    # `query_points`가 올바르게 호출되었는지 검증
    mock_qdrant.client.query_points.assert_called_once_with(
        collection_name="test_collection",
        query=[0.1, 0.2, 0.3],
        using="dense_large",  # 예상되는 임베딩 타입
        limit=20
    )
