# config/settings.py
from pydantic_settings import BaseSettings, SettingsConfigDict
from pathlib import Path
from typing import List

class Settings(BaseSettings):
    """
        설정값의 유효성 검사
        api/routes.py는 사용자 요청의 유효성을 검사한다는 차이가 있다.
    """
    model_config = SettingsConfigDict(extra="ignore")
    
    # ----------------------------------------- 
    #               사용자 설정 파트   
    # -----------------------------------------
    COLLECTION_NAME: str = "test_ragservice"
    # 단일 파일 호환을 위해 유지
    FILENAME: str = "dev_doc.docx"
    # 여러 파일 입력 지원
    FILENAMES: List[str] = ["dev_doc.docx", "table_test.docx", "course_material.pptx"]
    # ----------------------------------------- 
    #               경로   
    # -----------------------------------------    
    RAW_DATA_PATH: Path = Path("data/raw")
    PROCESSED_DATA_PATH: Path = Path("data/processed")

    QDRANT_URL: str = "http://localhost:6333"
    INDEX_DIR: Path = Path("data/index")
    
# settings 객체를 한 번만 초기화
_settings_instance = None

def get_settings() -> Settings:
    global _settings_instance
    if _settings_instance is None:
        _settings_instance = Settings()  # 객체가 없으면 한 번만 생성
    return _settings_instance