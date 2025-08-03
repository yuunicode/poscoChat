## 1. PYTHONPATH 환경변수 설정
경로: .vscode/settings.json
{
    "python-envs.pythonProjects": [],
    "terminal.integrated.env.linux": {
        "PYTHONPATH": "/home/jykoo/vscodeproject/poscoChat" # 자기 프로젝트 경로 환경변수에 추가
    },
  "python.analysis.extraPaths": ["."]
}

## 2. Qdrant 연결
1. docker pull qdrant/qdrant
2. docker run -d   --name <원하는 큐드런트 이름>   -p 6333:6333   -v $(pwd)/data/index:/qdrant/storage   qdrant/qdrant

## 3. 사용자가 볼 것
- embedding_config에서 임베딩 모델 설정
- build_index의 main() 함수 안 인자 조절
- config/settings 에 filename 설정 (.docx, pptx 중 현재 .docx만 지원)

## 작업트리
📂 chatbot_project/
 ├── api/
 ├──── main.py                # FastAPI 앱 실행 진입점 (백엔드 서버)
 ├──── routes.py              # 라우팅(페이지 URL, API URL 연결)
 ├──── service/
 │        └── rag.py    # RAG 로직 (검색 + 생성)
 ├──── templates/
 │        └── index.html        # 프론트엔드 HTML (UI 뼈대)
 ├──── static/
 │        ├── style.css         # CSS (UI 스타일)
 │        └── script.js         # JS (이벤트, API 호출)
 ├── config/
 ├──── main.py                # FastAPI 앱 실행 진입점 (백엔드 서버)
 ├──── routes.py              # 라우팅(페이지 URL, API URL 연결)