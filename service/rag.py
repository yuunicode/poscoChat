from infra.vectorstore import Qdrant

class RagPipeline:
    def __init__(self, searchModule: ):

    def run(self, query: str, search_type: str, use_cross_encoder: bool):
            
        # --------------------------
        # STEP 1: 사용자 쿼리 입력
        # --------------------------

        # --------------------------
        # STEP 2: 쿼리에 따른 서치 기법
        # --------------------------

        # --------------------------
        # STEP 3: 해당 서치 기법에 맞는 쿼리 임베딩 적용
        # --------------------------

        # --------------------------
        # STEP 4: 크로스인코더 사용 유무
        # --------------------------

if __name__ == "__main__":
    from infra.vectorstore import Vectorstore
    vector_store = Vectorstore()
    rag_service = RagService(vector_store)
    
    # Example usage
    query = "What is the capital of France?"
    search_type = "sparse"
    use_cross_encoder = False
    rag_service.run(query, search_type, use_cross_encoder)