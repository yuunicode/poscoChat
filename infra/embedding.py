# infra/embedding.py 내부
from abc import ABC, abstractmethod
from config.embedding_config import EmbeddingType
from typing import List
import numpy as np


class BaseEmbedding(ABC):
    def __init__(self, model_name: str, embedding_type: EmbeddingType):
        self.model_name = model_name
        self.embedding_type = embedding_type

    @abstractmethod
    def encode(self, texts: List[str]) -> List[np.ndarray]: pass

    @abstractmethod
    def get_dimension(self) -> int: pass

class HuggingFaceEmbedding(BaseEmbedding):
    def __init__(self, model_name: str, embedding_type: EmbeddingType):
        super().__init__(model_name, embedding_type)
        from transformers import AutoModelForMaskedLM, AutoTokenizer
        self.model = AutoModelForMaskedLM.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model.eval()

    def encode(self, texts: List[str]) -> List[np.ndarray]:
        import torch
        inputs = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            outputs = self.model(**inputs)

            if self.embedding_type in (EmbeddingType.DENSE_SMALL, EmbeddingType.DENSE_LARGE):
                # [CLS] 토큰 임베딩 사용 (BERT 계열 기준)
                embeddings = outputs.last_hidden_state[:, 0, :]  # shape: (batch_size, hidden_dim)
                return [vec.cpu().numpy() for vec in embeddings]

            elif self.embedding_type == EmbeddingType.COLBERT:
                # Token-level embeddings 사용
                token_embeddings = outputs.last_hidden_state  # shape: (batch_size, seq_len, hidden_dim)
                return [sentence.cpu().numpy() for sentence in token_embeddings]  # List[np.ndarray] per sentence

            else:
                raise ValueError(f"Unsupported embedding type for HuggingFace: {self.embedding_type}")

    def get_dimension(self) -> int:
        return self.model.config.hidden_size
    
class FastEmbedEmbedding(BaseEmbedding):
    def __init__(self, model_name: str, embedding_type: EmbeddingType, use_cuda: bool = False):
        super().__init__(model_name, embedding_type)
        from fastembed import TextEmbedding, LateInteractionTextEmbedding
        self.embedding_type = embedding_type
        self.model_name = model_name
        provider = ["CUDAExecutionProvider"] if use_cuda else None

        if embedding_type in (EmbeddingType.DENSE_SMALL, EmbeddingType.DENSE_LARGE):
            if provider:
                self.model = TextEmbedding(model_name, providers = provider)
            else:
                self.model = TextEmbedding(model_name)

        elif embedding_type == EmbeddingType.COLBERT:
            if provider:
                self.model = LateInteractionTextEmbedding(model_name, providers = provider)
            else:
                self.model = LateInteractionTextEmbedding(model_name)
        
        else:
            raise ValueError(f"Unsupported embedding type: {embedding_type}")

        self._embedding_type = embedding_type
        self._dimension = self.get_dimension()

    def encode(self, texts: List[str]) -> List[np.ndarray]:
        return list(self.model.embed(texts))

    def get_dimension(self) -> int:
        return self.model.embedding_size