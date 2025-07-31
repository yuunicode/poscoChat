import pytest
from unittest.mock import MagicMock
from infra.vectorstore import Qdrant
from infra.vectorizer import BaseSparseVectorizer
from infra.embedding import BaseEmbedding
from infra.retriever import SparseSearch, DenseSearch
from sentence_transformers import CrossEncoder
from qdrant_client.http.models import PointStruct, SparseVector

# 테스트용 클래스 정의
class TestRagPipeline:
    @pytest.fixture
    def mock_qdrant(self):
        # Qdrant의 mock 객체를 생성
        mock_qdrant_instance = MagicMock(spec=Qdrant)
        mock_qdrant_instance.client.query_points.return_value = MagicMock(points=[])
        return mock_qdrant_instance
    
    @pytest.fixture
    def mock_embedding(self):
        # BaseEmbedding의 mock 객체 생성
        mock_embedding = MagicMock(spec=BaseEmbedding)
        mock_embedding.encode.return_value = [[0.1, 0.2, 0.3]]  # 예시 벡터 반환
        return mock_embedding
    
    @pytest.fixture
    def mock_sparse_vectorizer(self):
        # BaseSparseVectorizer의 mock 객체 생성
        mock_sparse_vectorizer = MagicMock(spec=BaseSparseVectorizer)
        mock_sparse_vectorizer.transform.return_value = [{"indices": [1, 2], "values": [0.5, 0.6]}]
        return mock_sparse_vectorizer
    
    def test_sparse_search(self, mock_qdrant, mock_sparse_vectorizer):
        # SparseSearch 인스턴스 생성
        sparse_search = SparseSearch(sparse_vectorizer=mock_sparse_vectorizer, collection_name="test_collection", vectorstore=mock_qdrant)

        # search 메서드 실행
        results = sparse_search.search("test query")

        # 쿼리 결과 검증
        assert results == []  # 결과가 mock으로 설정되어 빈 리스트가 반환됨을 예상

        # `query_points`가 올바르게 호출되었는지 검증
        mock_qdrant.client.query_points.assert_called_once_with(
            collection_name="test_collection",
            query=SparseVector(indices=[1, 2], values=[0.5, 0.6]),
            using="sparse",
            limit=20
        )

    def test_dense_search(self, mock_qdrant, mock_embedding):
        # DenseSearch 인스턴스 생성
        dense_search = DenseSearch(embedding=mock_embedding, collection_name="test_collection", vectorstore=mock_qdrant)

        # search 메서드 실행
        results = dense_search.search("test query")

        # 쿼리 결과 검증
        assert results == []  # 결과가 mock으로 설정되어 빈 리스트가 반환됨을 예상

        # `query_points`가 올바르게 호출되었는지 검증
        mock_qdrant.client.query_points.assert_called_once_with(
            collection_name="test_collection",
            query=[0.1, 0.2, 0.3],
            using="dense_large",  # 예상되는 임베딩 타입
            limit=20
        )
