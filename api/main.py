# api/main.py
import logging
import sys, os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

from fastapi import FastAPI, Request, Form, HTTPException
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from service.rag import RagService
from config import settings
from infra.vectorstore import Qdrant
from infra.retriever import SparseSearch, DenseSearch, HybridSearch, MultiStageSearch
from infra.generator import OllamaGenerator, OpenRouterGenerator
from infra.embedding import FastEmbedEmbedding
from infra.vectorizer import BM25Vectorizer
from config.embedding_config import EMBEDDING_TYPE_CONFIGS, EmbeddingType
from datetime import datetime
import json
from markdown import markdown

# FastAPI 애플리케이션 초기화
app = FastAPI()

# Jinja2 템플릿 엔진 설정
templates = Jinja2Templates(directory="api/templates")
templates.env.filters['markdown'] = lambda text: markdown(text)

# settings 초기화
settings = settings.get_settings()
collection_name = settings.COLLECTION_NAME
qdrant = Qdrant(settings.QDRANT_URL)

# 벡터라이저와 임베딩 초기화
postfix = "kr"
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

# Retriever 초기화
sparse_search = SparseSearch(bm25_vectorizer, collection_name, qdrant)
dense_search = DenseSearch(embedding_dense, collection_name, qdrant)
hybrid_search = HybridSearch(bm25_vectorizer, embedding_dense, collection_name, qdrant)
multistage_search = MultiStageSearch(embedding_dense, embedding_colbert, collection_name, qdrant)

# generator = OllamaGenerator("qwen3:4b")
generator = OpenRouterGenerator("qwen/qwen3-8b:free")

# 전역 리스트 - 로그 저장을 위해
answers = []

def append_loggings(query, answer, results, search_type, prompt_type, use_cross_encoder, search_time, generation_time, collection_name):
    
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    # ScoredPoint 객체 각각을 직접 dict로 변환
    results_dict = [{
        "score": getattr(r, "score", None),
        "payload": getattr(r, "payload", None),
    } for r in results]
    
    # answer는 list[dict] 형태이므로 첫 번째 항목에서 answer 값을 바로 저장
    answer_text = answer[0].get("answer", "")  # 첫 번째 항목에서 "answer" 추출, 없으면 빈 문자열

    #search_time, generation_time 조정
    search_time_formatted = f"{search_time:.2f}초"
    generation_time_formatted = f"{generation_time:.2f}초"

    answers.append({
        "query": query,
        "answer": answer_text,
        "results": results_dict,
        "timestamp": current_time,  # 현재 시간
        "search_time": search_time_formatted,  # 검색 시간
        "generation_time": generation_time_formatted,  # 생성 시간
        "collection_name": collection_name,  # 컬렉션 이름
        "search_method": search_type,  # 검색 방법
        "prompt_type": prompt_type,  # 프롬프트 종류
        "use_cross_encoder": use_cross_encoder  # Cross encoder 사용 여부
    })

    
def get_all_answers():
    return answers

def clear_answers():
    answers.clear()

@app.get("/", response_class = HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", 
                                        {
                                            "request": request, 
                                            "answers": None, 
                                            "query": "", 
                                            "search_type": "COLBERT_RERANK", 
                                            "prompt_type": "default",
                                            "use_cross_encoder": False # 이래야 체크박스가 풀려있음
                                        }
                                      )

@app.post("/", response_class = HTMLResponse)
async def search(
    request: Request, 
    query: str = Form(...),
    search_type: str = Form("COLBERT_RERANK"),
    prompt_type: str = Form("default"),
    use_cross_encoder: bool = Form(None) #값이 on일 때만 true로 변환해서 서비스에 넘김
):
    
    # 체크박스 처리: 체크시 "on", 해제시 None
    use_cross_encoder_bool = (use_cross_encoder == "on") # 템플릿 변수로 얘는 bool로 넘겨야 jinja2 문법 동작
    
    # search_type에 따라 적절한 RagService 초기화
    if search_type == "SPARSE":
        rag_service = RagService(sparse_search, generator)
    elif search_type == "DENSE_LARGE":
        rag_service = RagService(dense_search, generator)
    elif search_type == "HYBRID":
        rag_service = RagService(hybrid_search, generator)
    elif search_type == "COLBERT_RERANK":
        rag_service = RagService(multistage_search, generator)
    else:
        raise HTTPException(status_code=400, detail="Invalid search type")

    results, answer, search_time, generation_time = rag_service.retrieve_and_generate(query, use_cross_encoder_bool, prompt_type)

    # 서버 램에 저장되는 로그파일 
    append_loggings(query, answer, results, search_type, prompt_type, use_cross_encoder_bool, search_time, generation_time, collection_name)
    answers_json = json.dumps(get_all_answers(), ensure_ascii = False)
    single_answer = [{"query": query, "answer": answer[0].get("answer", "")}]
    
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request, 
            "answers": answer, 
            "answers_json": answers_json,
            "single_answer": single_answer,
            "query": query,
            "search_type": search_type,
            "prompt_type": prompt_type,
            "use_cross_encoder": use_cross_encoder_bool,
        }
    )