import pytest
from unittest.mock import MagicMock
from infra.retriever import SparseSearch, Qdrant
from infra.vectorizer import BaseSparseVectorizer
from config.embedding_config import EmbeddingType
from sentence_transformers import CrossEncoder
from typing import List, Dict

# Mock Vectorizer class
class MockSparseVectorizer(BaseSparseVectorizer):
    def __init__(self):
        self.vocab = {"term1": 1, "term2": 2, "term3": 3}
    
    def transform(self, query: List[str]) -> List[Dict[str, any]]:
        # Return mock sparse vector with indices and values
        return [{"indices": [1, 2], "values": [0.5, 0.3]}]

# Mock Qdrant client
class MockQdrant(Qdrant):
    def __init__(self, qdrant_url: str, api_key: str, collection_name: str):
        self.collection_name = collection_name
        self.client = MagicMock()
    
    def query_points(self, collection_name: str, query: List[float], using: str, limit: int):
        # Simulate a response from Qdrant
        return MagicMock(points=[{"payload": {"content": "Mock document 1"}}])

# Test for SparseSearch
def test_sparse_search():
    # Mock Qdrant client and SparseVectorizer
    mock_qdrant = MockQdrant(qdrant_url="mock_url", api_key="mock_api_key", collection_name="test_collection")
    mock_vectorizer = MockSparseVectorizer()

    # Initialize SparseSearch with mocked Qdrant and Vectorizer
    sparse_search = SparseSearch(sparse_vectorizer=mock_vectorizer, collection_name="test_collection", vectorstore=mock_qdrant)
    
    # Run search
    query = "term1 term2"
    results = sparse_search.search(query)

    # Assertions
    assert len(results) == 1  # One result should be returned
    assert results[0]['payload']['content'] == "Mock document 1"  # Check the content of the returned result

    # Check if Qdrant's query_points was called with the correct parameters
    mock_qdrant.client.query_points.assert_called_with(
        collection_name="test_collection", 
        query=[0.5, 0.3],  # The transformed sparse vector 
        using=EmbeddingType.SPARSE.value, 
        limit=20
    )

# Test for Cross-Encoder reranking
def test_apply_cross_encoder_rerank():
    # Mock results for reranking
    results = [
        {"payload": {"content": "Mock document 1"}},
        {"payload": {"content": "Mock document 2"}}
    ]
    query = "term1 term2"

    # Initialize Cross-Encoder mock
    cross_encoder = MagicMock()
    cross_encoder.predict.return_value = [0.9, 0.5]  # Mock prediction scores

    # Apply Cross-Encoder reranking
    reranked_results = SparseSearch.apply_cross_encoder_rerank(
        None,  # `self` is not needed for static method
        cross_encoder,
        query,
        results
    )

    # Check if results are sorted based on scores
    assert reranked_results[0]['payload']['content'] == "Mock document 1"  # Higher score
    assert reranked_results[1]['payload']['content'] == "Mock document 2"  # Lower score

# Run tests
pytest.main()