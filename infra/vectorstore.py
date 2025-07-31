# infra/vectorstore.py
from typing import Dict, Optional, List
from config.embedding_config import EmbeddingType, EmbeddingModelConfig
import logging, time
from qdrant_client import QdrantClient
from qdrant_client.http.models import (
    Distance,
    HnswConfigDiff,
    MultiVectorComparator,
    MultiVectorConfig,
    PointStruct,
    PointVectors,
    SparseIndexParams,
    SparseVectorParams,
    VectorParams,
)

class Qdrant():
   
    def __init__(self, qdrant_url: str, qdrant_api_key: Optional[str] = None):
        self.client = QdrantClient(
                url     = qdrant_url,
                api_key = qdrant_api_key,
        )
        
    def create_collection(self,
                      collection_name: str,
                      model_config: EmbeddingModelConfig,
                      hnsw_config: Optional[HnswConfigDiff] = None):
        try:
            dense_vector_params = {}
            sparse_vector_params = {}

            if model_config.enabled.get(EmbeddingType.DENSE_SMALL):
                dense_vector_params[EmbeddingType.DENSE_SMALL.value] = VectorParams(
                    size = model_config.dimensions[EmbeddingType.DENSE_SMALL],
                    distance = Distance[model_config.dense_distance.upper()]
                )

            if model_config.enabled.get(EmbeddingType.DENSE_LARGE):
                dense_vector_params[EmbeddingType.DENSE_LARGE.value] = VectorParams(
                    size = model_config.dimensions[EmbeddingType.DENSE_LARGE],
                    distance = Distance[model_config.dense_distance.upper()]
                )

            if model_config.enabled.get(EmbeddingType.COLBERT):
                dense_vector_params[EmbeddingType.COLBERT.value] = VectorParams(
                    size = model_config.dimensions[EmbeddingType.COLBERT],
                    distance = Distance.COSINE,
                    multivector_config = MultiVectorConfig(comparator = MultiVectorComparator.MAX_SIM),
                    hnsw_config = HnswConfigDiff(m = 0)  # HNSW 비활성화
                )

            if model_config.enabled.get(EmbeddingType.SPARSE):
                sparse_vector_params[EmbeddingType.SPARSE.value] = SparseVectorParams(
                    index = SparseIndexParams(on_disk = False)
                )

            self.client.create_collection(
                collection_name = collection_name,
                vectors_config = dense_vector_params or None,
                sparse_vectors_config = sparse_vector_params or None,
                hnsw_config = hnsw_config or None
            )

            logging.info(f"컬렉션 생성 완료: {collection_name}")
    
        except Exception as e:
            logging.error(f"컬렉션 생성 실패: {e}")
            raise

    def delete_collection(self, collection_name: str) -> None:
        try:
            # 현재 존재하는 컬렉션 목록 확인
            collections = [c.name for c in self.client.get_collections().collections]
            if collection_name not in collections:
                logging.info(f"'{collection_name}' 컬렉션이 존재하지 않습니다. 삭제하지 않습니다.")
                return

            # 컬렉션 삭제 요청
            self.client.delete_collection(collection_name = collection_name)
            
            # 삭제 완료 확인 (최대 5초 대기)
            max_attempts = 5
            for _ in range(max_attempts):
                collections = [c.name for c in self.client.get_collections().collections]
                if collection_name not in collections:
                    logging.info(f"'{collection_name}' 컬렉션이 성공적으로 삭제됐습니다.")
                    return
                time.sleep(0.1)
            
            logging.warning(f"'{collection_name}' 삭제 {max_attempts * 0.1} 초 대기")
        
        except Exception as e:
            logging.error(f"컬렉션을 삭제하지 못했습니다. '{collection_name}': {e}")
            raise

    def upload_vectors(self, collection_name: str, points: List[PointStruct]) -> None:
        self.client.upsert(collection_name = collection_name, points = points, wait = True)
    # -------------------------------------------------------------
    #           멀티벡터를 지원하기 때문에 Colbart 벡터 업로드
    # -------------------------------------------------------------
    def update_colbert_vectors(self, collection_name: str, vectors: List[PointVectors]) -> None:
        self.client.update_vectors(collection_name = collection_name, points = vectors, wait = True)