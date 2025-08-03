# scripts/build_index.py
from pathlib import Path
from config.embedding_config import EMBEDDING_TYPE_CONFIGS, EmbeddingModelConfig, EmbeddingType
from config.settings import get_settings
from infra.vectorstore import Qdrant
from infra.embedding import HuggingFaceEmbedding, FastEmbedEmbedding
from infra.vectorizer import BM25Vectorizer, TfidfSparseVectorizer
from langchain_core.documents import Document
from qdrant_client.models import HnswConfigDiff, PointStruct, SparseVector, PointVectors
from typing import List, Dict, Any, Optional
import json, uuid

def main():
    settings = get_settings()
    jsonl_path = settings.PROCESSED_DATA_FILE_PATH
    pkl_path = settings.PROCESSED_DATA_PATH / "pickles"
    pkl_path.mkdir(parents=True, exist_ok=True)

    # --------------------------
    # STEP 0: 사용자 설정 및 준비
    # --------------------------

    # collection_name = settings.COLLECTION_NAME
    # TODO: 아래는 임시방편으로, 실제로는 settings에서 가져와야 함
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    collection_name = f"{settings.COLLECTION_NAME}"

    enabled_types = [
        EmbeddingType.SPARSE,
        EmbeddingType.DENSE_LARGE,
        EmbeddingType.COLBERT,
    ]

    indexing_type = "hnsw" # or "flat"
    distance_type = "Cosine"  # or "Dot", "Euclidean"
    USE_CUDA = False # CUDA 사용 여부

    batch_size          = 2
    update_colbert      = True
    colbert_batch_size  = 16

    # --------------------------
    # STEP 1: 각 임베딩 타입별 벡터 준비
    # --------------------------
    embedding_instances = {}
    dimensions = {}
    sparse_vectors_by_type = {}

    for etype in enabled_types:
        routing = EMBEDDING_TYPE_CONFIGS[etype]

        if routing.provider == "vectorizer":
            # 피클 경로 결정
            if routing.vectorizer_type == "bm25":
                language = routing.language if routing.language is not None else "unknown"
                postfix = language
                pickle_name = f"{settings.FILENAME}_bm25_{postfix}.pkl"
                vectorizer = BM25Vectorizer(pkl_path = pkl_path / pickle_name,
                                            language = language)
                
            elif routing.vectorizer_type == "tfidf":
                language = routing.language if routing.language is not None else "unknown"
                postfix = language
                pickle_name = f"{settings.FILENAME}_tfidf_{postfix}.pkl"
                vectorizer = TfidfSparseVectorizer(pkl_path = pkl_path / pickle_name,
                                                   language = language)
            else:
                raise ValueError(f"Unknown vectorizer type: {routing.vectorizer_type}")

            # 피클 로드 실패 시 학습
            if not vectorizer.load():
                with open(jsonl_path, "r", encoding="utf-8") as f:
                    texts = [json.loads(line)["context"] for line in f if "context" in json.loads(line)]
                vectorizer.fit(texts)

            # 벡터 생성
            with open(jsonl_path, "r", encoding="utf-8") as f:
                texts = [json.loads(line)["context"] for line in f if "context" in json.loads(line)]
                
            sparse_vectors = vectorizer.transform(texts)
            sparse_vectors_by_type[etype] = sparse_vectors
            dimensions[etype] = len(sparse_vectors[0]["values"]) if sparse_vectors else 0

        else:
            # Dense embedding 로딩
            if routing.provider == "huggingface":
                model_name = routing.model_name if routing.model_name is not None else "unknown"
                model = HuggingFaceEmbedding(model_name, etype)
            elif routing.provider == "fastembed":
                model_name = routing.model_name if routing.model_name is not None else "unknown"
                model = FastEmbedEmbedding(model_name, etype, use_cuda = USE_CUDA)
            else:
                raise ValueError(f"Unknown provider: {routing.provider}")
            
            embedding_instances[etype] = model
            dimensions[etype] = model.get_dimension()

    # --------------------------
    # STEP 2: EmbeddingModelConfig 구성
    # --------------------------
    model_config = EmbeddingModelConfig(
        enabled         = {etype: True for etype in enabled_types},
        dimensions      = dimensions,
        dense_distance  = distance_type # Dot, Euclideanm Cosine
    )

    # --------------------------
    # STEP 3: HNSW 설정 (필요 시)
    # --------------------------
    # HNSW 설정 객체 생성 (HNSW 인덱스 사용 시)
    if indexing_type == "hnsw":
        hnsw_config_obj = HnswConfigDiff( 
                m                         = 32 # 한 노드 당 최대 연결 수 (그래프의 밀도), 커질수록 정확도가 올라가지만 경로가 많아지고 구조가 복잡해질 수 있음. 정확도가 중요하면 32~64
                , ef_construct            = 200 # 각 벡터에 대해 탐색할 후보 노드 수 (exploration factor for index building) 크게 설정할수록 더 정확한 그래프 생성, but 구축시간 느려짐. 대규모나 고정밀은 200~500까지 ㄱㄴ 
                , full_scan_threshold     = 10000 # 데이터 개수가 이 값보다 적으면 FLAT 스캔 수행. 작은 데이터라도 HNSW 쓰고싶으면 0으로 설정
                , payload_m               = 32 // 2 
        )
    else:
        hnsw_config_obj = None # Flat Index 자동적용

    # --------------------------
    # STEP 4: Qdrant 컬렉션 생성
    # --------------------------
    qdrant = Qdrant(
        qdrant_url     = settings.QDRANT_URL,
        qdrant_api_key = None  # 필요시 적용
    )

    qdrant.create_collection(
        collection_name = collection_name,
        model_config    = model_config,
        hnsw_config     = hnsw_config_obj
    )

    # --------------------------
    # STEP 5: 컬렉션에 벡터 업로드
    # --------------------------
    document_list = load_documents_from_jsonl(jsonl_path)

    sparse_vectors_data = sparse_vectors_by_type.get(EmbeddingType.SPARSE, [])
    
    if EmbeddingType.COLBERT not in enabled_types:
        update_colbert = False
    
    upload_all_vectors_to_qdrant(
        qdrant              = qdrant,
        collection_name     = collection_name,
        document_list       = document_list,
        embedding_models    = embedding_instances,
        sparse_vectors_data = sparse_vectors_data,
        batch_size          = batch_size,
        update_colbert      = update_colbert,
        colbert_batch_size  = colbert_batch_size
    )

# ----------------------------------------------------------------------------------

def load_documents_from_jsonl(jsonl_path: Path) -> List[Document]:
    documents = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            if "context" in data:
                documents.append(
                    Document(page_content=data["context"], metadata=data.get("metadata", {}))
                )
    return documents

def upload_all_vectors_to_qdrant(
    qdrant: Qdrant,
    collection_name: str,
    document_list: List[Document],
    embedding_models: Dict[EmbeddingType, Any],
    sparse_vectors_data: List[Dict[str, Any]],
    batch_size: int,
    update_colbert: bool,
    colbert_batch_size: Optional[int] 
):
    if update_colbert and colbert_batch_size is None:
        raise ValueError("colbert_batch_size must be set when updating ColBERT vectors.")
    
    uploaded_doc_info = []
    total_uploaded_points = 0

    for i in range(0, len(document_list), batch_size):
        batch_docs = document_list[i : i + batch_size]
        contents = [doc.page_content for doc in batch_docs]
        metas = [doc.metadata for doc in batch_docs]

        # 임베딩 벡터 계산
        dense_vecs_by_type = {}
        for etype, model in embedding_models.items():
            if etype in [EmbeddingType.DENSE_SMALL, EmbeddingType.DENSE_LARGE]:
                dense_vecs_by_type[etype] = list(model.encode(contents))  # List[List[float]]

        # 희소벡터 계산
        current_sparse_batch_data = sparse_vectors_data[i : i + len(batch_docs)]
                
        # 포인트 구성
        points: List[PointStruct] = []
        for j in range(len(batch_docs)):
            point_id = str(uuid.uuid4())

            # Sparse
            current_sparse_data = current_sparse_batch_data[j]
            sparse_vecs = None
            if current_sparse_data and current_sparse_data["indices"] and current_sparse_data["values"]:
                sparse_vecs = SparseVector(
                    indices = current_sparse_data["indices"],
                    values  = current_sparse_data["values"]
                )
            
            all_vectors = {}
            # Dense
            for etype in [EmbeddingType.DENSE_SMALL, EmbeddingType.DENSE_LARGE]:
                if etype in dense_vecs_by_type:
                    vec = dense_vecs_by_type[etype][j]
                    all_vectors[etype.value] = vec.tolist() if hasattr(vec, "tolist") else list(vec)
            # Sparse
            if sparse_vecs is not None:
                all_vectors[EmbeddingType.SPARSE.value] = sparse_vecs

            points.append(PointStruct(
                id      = point_id,
                vector  = all_vectors,
                payload = {
                    "content": contents[j],
                    "metadata": metas[j]
                }
            ))
            uploaded_doc_info.append((point_id, contents[j]))

        # 업로드
        if points:
            qdrant.upload_vectors(collection_name, points)
            total_uploaded_points += len(points)

    print(f"Dense/Sparse 벡터 업로드 완료: {total_uploaded_points} points")

    # ColBERT 업데이트
    if update_colbert and uploaded_doc_info:
        if colbert_batch_size is not None:
            total_updated = 0
            for i in range(0, len(uploaded_doc_info), colbert_batch_size):
                batch_infos = uploaded_doc_info[i : i + colbert_batch_size]
                point_ids = [pid for pid, _ in batch_infos]
                texts = [txt for _, txt in batch_infos]

                colbert_embs = list(embedding_models[EmbeddingType.COLBERT].encode(texts))

                colbert_points = [
                    PointVectors(
                        id      = pid,
                        vector  = {EmbeddingType.COLBERT.value: colbert_embs[j]}
                    )
                    for j, pid in enumerate(point_ids)
                ]

                if colbert_points:
                    qdrant.update_colbert_vectors(collection_name, colbert_points)
                    total_updated += len(colbert_points)

            print(f"ColBERT 벡터 업데이트 완료: {total_updated} points")

if __name__ == "__main__":
    main()