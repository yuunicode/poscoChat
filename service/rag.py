# service/rag_service.py
from typing import List, Tuple
import time
from infra.retriever import SearchModule, SparseSearch, DenseSearch, HybridSearch, MultiStageSearch
from infra.generator import BaseLLMGenerator, OllamaGenerator, OpenRouterGenerator
from infra.vectorstore import Qdrant
from infra.vectorizer import BaseSparseVectorizer, BM25Vectorizer, TfidfSparseVectorizer
from infra.embedding import FastEmbedEmbedding
from config.embedding_config import EMBEDDING_TYPE_CONFIGS, EmbeddingType
from config.settings import get_settings

class RagService:
    def __init__(self, search_module: SearchModule, generator: BaseLLMGenerator):
        self.search_module = search_module
        self.generator = generator

    def retrieve_and_generate(
        self, 
        query: str, 
        prompt_type: str
    ) -> Tuple[List, List, float, float]:
        """
        query: 사용자의 질문
        prompt_type: configuration / workflow / definition 등
        """
        start_time = time.time()
        results = self.search_module.search(query)
        print(results)
        search_time = time.time() - start_time
        prompt_template = self.generator.select_prompt(prompt_type)

        start_time = time.time()
        answers = self.generator.generate_answers(results, query, prompt_template)
        generation_time = time.time() - start_time

        return results, answers, search_time, generation_time


if __name__ == "__main__":
    settings = get_settings()
    collection_name = settings.COLLECTION_NAME
    qdrant = Qdrant(settings.QDRANT_URL)

    postfix = "kr"
    # 벡터라이저와 임베딩 초기화
    bm25_vectorizer = BM25Vectorizer(settings.PROCESSED_DATA_PATH / "pickles" / f"{settings.FILENAME}_bm25_{postfix}.pkl", language="kr")
    bm25_vectorizer.load()
    embedding_dense = FastEmbedEmbedding(
        model_name      = EMBEDDING_TYPE_CONFIGS[EmbeddingType.DENSE_LARGE].model_name or "default_model_name",
        embedding_type  = EmbeddingType.DENSE_LARGE
    )
    embedding_colbert = FastEmbedEmbedding(
        model_name      = EMBEDDING_TYPE_CONFIGS[EmbeddingType.COLBERT].model_name or "default_model_name",
        embedding_type  = EmbeddingType.COLBERT
    )


    # 하이브리드 검색과 Ollama 생성기 구성
    search_module = MultiStageSearch(embedding_dense, embedding_colbert, collection_name, qdrant)
    generator = OllamaGenerator("qwen3:4b")

    # 서비스 실행
    rag = RagService(search_module, generator)
    query = "가중치가 신경망을 생성할 때 어떤 값을 가져?"
    prompt_type = "workflow"

    results, answers, search_time, generation_time = rag.retrieve_and_generate(query, prompt_type)

    print(f"검색 시간: {search_time:.2f}s")
    print(f"생성 시간: {generation_time:.2f}s")
    print(" 답변:\n", answers[0]["answer"])
