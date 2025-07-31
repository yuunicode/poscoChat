# config/embedding_config.py
from config.embedding_config import EmbeddingType
from typing import Dict, Optional, Literal
from dataclasses import dataclass
from enum import Enum

class EmbeddingType(str, Enum): 
    DENSE_SMALL = "dense_small"
    DENSE_LARGE = "dense_large"
    SPARSE = "sparse"
    COLBERT = "colbert"

# 유저가 어떤 벡터를 쓸지, 차원과 distance 설정을 담습니다.
@dataclass
class EmbeddingModelConfig:
    enabled: Dict[EmbeddingType, bool]       # 어떤 임베딩 타입을 활성화할지
    dimensions: Dict[EmbeddingType, int]     # 각 타입의 차원 수
    dense_distance: str                      # Cosine, Dot, Euclid 등 (Qdrant Distance)

# 각 임베딩 타입에 대해 어떤 라이브러리/모델을 쓸지 결정합니다.
@dataclass
class EmbeddingRoutingConfig:
    provider: Literal["huggingface", "fastembed", "vectorizer"]
    model_name: Optional[str] = None   # dense만 사용
    vectorizer_type: Optional[str] = None # "bm25" or "tfidf"
    language: Optional[str] = "en"     # "en" or "ko" (토크나이저 결정)

# 전처리, 검색 양쪽 모두에서 재사용할 config입니다.
# 현재 build_index.py에서 쓰고있습니다.
EMBEDDING_TYPE_CONFIGS: Dict[EmbeddingType, EmbeddingRoutingConfig] = {
    EmbeddingType.SPARSE: EmbeddingRoutingConfig(
        provider = "vectorizer",
        vectorizer_type = "bm25",
        language = "en"
    ),
    EmbeddingType.DENSE_LARGE: EmbeddingRoutingConfig(
        provider = "fastembed",
        model_name = "BAAI/bge-small-en-v1.5"
    ),
    EmbeddingType.COLBERT: EmbeddingRoutingConfig(
        provider = "fastembed",
        model_name = "colbert-ir/colbertv2.0"
    ),    
}

