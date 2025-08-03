# infra/vectorizer.py
from abc import ABC, abstractmethod
import json
import logging
from typing import List, Dict, Optional, Any
from pathlib import Path
from langchain_core.documents import Document
from sklearn.feature_extraction.text import TfidfVectorizer
from rank_bm25 import BM25Okapi
import numpy as np
import pickle
import nltk
from urllib.error import URLError
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# # --- 필요한 NLTK 데이터 다운로드 (최초 1회 실행) ---
# # 이 코드는 스크립트 실행 환경에서 한 번만 실행되도록 관리하는 것이 좋습니다.
# # 예를 들어, 애플리케이션 시작 시점에 호출하거나, Dockerfile에 포함할 수 있습니다.
# import nltk

# # punkt tokenizer
# try:
#     nltk.data.find('tokenizers/punkt')
# except LookupError:
#     nltk.download('punkt')

# # stopwords
# try:
#     nltk.data.find('corpora/stopwords')
# except LookupError:
#     nltk.download('stopwords')

# # wordnet
# try:
#     nltk.data.find('corpora/wordnet')
# except LookupError:
#     nltk.download('wordnet')

# # omw-1.4 (WordNetLemmatizer에 필요)
# try:
#     nltk.data.find('corpora/omw-1.4')
# except LookupError:
#     nltk.download('omw-1.4')

class BaseSparseVectorizer(ABC):
    """ 모든 sparse 벡터라이저의 기본 인터페이스 """

    def __init__(self, pkl_path: Path, language: str):
        self.pkl_path = pkl_path
        self.language = language

    @abstractmethod
    def fit(self, texts: List[str]):
        """주어진 텍스트 컬렉션에 벡터라이저를 학습시킵니다."""
        pass

    @abstractmethod
    def transform(self, texts: List[str]) -> List[Dict[str, Any]]:
        """
        주어진 텍스트 리스트를 Qdrant SparseVectors 형식에 맞는
        { "indices": [...], "values": [...] } 리스트로 변환합니다.
        """
        pass

    @abstractmethod
    def save(self):
        """학습된 벡터라이저를 저장합니다."""
        pass

    @abstractmethod
    def load(self) -> bool:
        """저장된 벡터라이저를 로드합니다."""
        pass


_lemmatizer = None
_stop_words = None

def get_lemmatizer():
    """WordNetLemmatizer 인스턴스를 한 번만 생성하여 반환합니다."""
    global _lemmatizer
    if _lemmatizer is None:
        _lemmatizer = WordNetLemmatizer()
    return _lemmatizer

def get_stop_words():
    """영어 불용어 집합을 한 번만 생성하여 반환합니다."""
    global _stop_words
    if _stop_words is None:
        _stop_words = set(stopwords.words('english'))
    return _stop_words

def nltk_en_tokenizer(text: str) -> List[str]:
    """
    BM25Vectorizer에 적합하도록 영어 텍스트를 전처리하여 토큰 리스트를 반환합니다.
    - 소문자 변환
    - 단어 토큰화
    - 알파벳이 아닌 문자 및 불용어 제거
    - 표제어 추출
    """
    lemmatizer = get_lemmatizer()
    stop_words = get_stop_words()

    # 1. 소문자 변환 및 토큰화
    tokens = word_tokenize(text.lower())

    processed_tokens = []
    for word in tokens:
        # 2. 알파벳만 남기고 불용어 제거
        if word.isalpha() and word not in stop_words:
            # 3. 표제어 추출
            processed_tokens.append(lemmatizer.lemmatize(word))
            
    return processed_tokens


class BM25Vectorizer(BaseSparseVectorizer):
    def __init__(self, pkl_path: Path, language: str = "en", k1: float = 1.5, b: float = 0.75):
        super().__init__(pkl_path, language)
        self.k1 = k1
        self.b = b
        self.bm25_model = None
        self.vocab = {} # 용어-인덱스 매핑 (BM25 계산에 필요)
        self.tokenizer = self._get_tokenizer_for_language(language)
        self.doc_lengths = []
        self.avg_doc_len = 0.0
        

    def fit(self, texts: List[str]):
        logging.info("[BM25Vectorizer] BM25 모델 학습을 시작합니다.")
        tokenized_corpus = [self.tokenizer(text) for text in texts]
        self.bm25_model = BM25Okapi(tokenized_corpus, k1=self.k1, b=self.b)
        
        self.vocab = self._build_vocab(tokenized_corpus)
        self.doc_freqs = self._calculate_doc_frequencies(tokenized_corpus)
        self.idf_scores = self._calculate_idf_scores(len(texts))

        self.doc_lengths = [len(doc) for doc in tokenized_corpus]
        self.avg_doc_len = sum(self.doc_lengths) / len(self.doc_lengths) if self.doc_lengths else 0.0

        logging.info("[BM25Vectorizer] BM25 모델 학습 완료.")
        self.save()

    def _get_tokenizer_for_language(self, lang: str):
        if lang == "en":
            return nltk_en_tokenizer
        elif lang == "kr":
            return nltk_en_tokenizer # TODO: 한국어 토크나이저 추가
        else:
            raise ValueError(f"[BM25Vectorizer] Unsupported language: {lang}")
        
    def _build_vocab(self, tokenized_corpus: List[List[str]]) -> Dict[str, int]:
        vocab = {}
        for doc_tokens in tokenized_corpus:
            for token in doc_tokens:
                if token not in vocab:
                    vocab[token] = len(vocab)
        return vocab

    def _calculate_doc_frequencies(self, tokenized_corpus: List[List[str]]) -> Dict[str, int]:
        doc_freqs = {term: 0 for term in self.vocab}
        for doc_tokens in tokenized_corpus:
            for term in set(doc_tokens):
                doc_freqs[term] += 1
        return doc_freqs

    def _calculate_idf_scores(self, num_documents: int) -> Dict[str, float]:
        idf_scores = {}
        for term, df in self.doc_freqs.items():
            idf_scores[term] = np.log((num_documents - df + 0.5) / (df + 0.5) + 1)
        return idf_scores

    def transform(self, texts: List[str]) -> List[Dict[str, Any]]:
        sparse_vectors = []
        for i, text in enumerate(texts):
            tokenized_doc = self.tokenizer(text)
            
            term_freq = {}
            for term in tokenized_doc:
                term_freq[term] = term_freq.get(term, 0) + 1    
            indices = []
            values = []
            
            current_doc_len = len(tokenized_doc)
           
            for term, tf in term_freq.items():
        
                if term in self.vocab:
                    term_id = self.vocab[term]
                    idf = self.idf_scores.get(term, 0.0)
                    
                    numerator = tf * (self.k1 + 1)
                    denominator = tf + self.k1 * (1 - self.b + self.b * (current_doc_len / self.avg_doc_len))
                    
                    bm25_term_score = idf * (numerator / denominator)

                    if bm25_term_score > 0:
                        indices.append(term_id)
                        values.append(float(bm25_term_score))

            sparse_vectors.append({
                "indices": indices,
                "values": values
            })

        return sparse_vectors

    def save(self):
        save_data = {
            "vocab": self.vocab,
            "doc_freqs": self.doc_freqs,
            "idf_scores": self.idf_scores,
            "k1": self.k1,
            "b": self.b,
            "doc_lengths": self.doc_lengths,
            "avg_doc_len": self.avg_doc_len
        }
        try:
            with open(self.pkl_path, "wb") as f:
                pickle.dump(save_data, f)
            logging.info(f"[BM25Vectorizer] BM25 관련 데이터 저장 완료: {self.pkl_path}")
        except Exception as e:
            logging.error(f"[BM25Vectorizer] BM25 데이터 저장 중 오류 발생: {e}")

    def load(self):
        try:
            with open(self.pkl_path, "rb") as f:
                loaded_data = pickle.load(f)
                self.vocab = loaded_data.get("vocab", {})
                self.doc_freqs = loaded_data.get("doc_freqs", {})
                self.idf_scores = loaded_data.get("idf_scores", {})
                self.k1 = loaded_data.get("k1", 1.5)
                self.b = loaded_data.get("b", 0.75)
                self.doc_lengths = loaded_data.get("doc_lengths", [])
                self.avg_doc_len = loaded_data.get("avg_doc_len", 0.0)
                logging.info(f"[BM25Vectorizer] BM25 관련 데이터 로드 완료: {self.pkl_path}")
            return True
        except (FileNotFoundError, EOFError, pickle.UnpicklingError) as e:
            logging.warning(f"[BM25Vectorizer] BM25 데이터 로드 실패 (파일 없거나 손상됨): {e}")
            return False
        except Exception as e:
            logging.error(f"[BM25Vectorizer] BM25 데이터 로드 중 예상치 못한 오류 발생: {e}")
            return False

# --- TfidfVectorizer 구현 시작 ---
class TfidfSparseVectorizer(BaseSparseVectorizer):
    def __init__(self, pkl_path: Path, language: str = "en", max_features: Optional[int] = None):
        super().__init__(pkl_path, language)
        self.tfidf_vectorizer = None
        self.max_features = max_features

    def fit(self, texts: List[str]):
        logging.info("[TfidfSparseVectorizer] TF-IDF 모델 학습을 시작합니다.")
        # TfidfVectorizer는 자체적으로 토큰화 및 IDF 계산을 처리합니다.
        self.tfidf_vectorizer = TfidfVectorizer(max_features=self.max_features)
        self.tfidf_vectorizer.fit(texts)
        
        # Qdrant sparse vector 생성을 위해 어휘-인덱스 매핑 저장
        self.vocab = self.tfidf_vectorizer.vocabulary_ 
        logging.info("[TfidfSparseVectorizer] TF-IDF 모델 학습 완료.")
        self.save()

    def transform(self, texts: List[str]) -> List[Dict[str, Any]]:
        if self.tfidf_vectorizer is None:
            logging.error("[TfidfSparseVectorizer] TF-IDF 벡터라이저가 학습되지 않았습니다.")
            return []

        # TfidfVectorizer를 사용하여 sparse 행렬 얻기
        tfidf_matrix = self.tfidf_vectorizer.transform(texts)
        
        sparse_vectors_qdrant_format = []
        # scipy.sparse 행렬을 Qdrant SparseVectors 형식으로 변환
        for i in range(tfidf_matrix.shape[0]):
            row = tfidf_matrix.getrow(i) # 각 문서에 대한 sparse row
            indices = row.indices.tolist()
            values = row.data.tolist()
            
            sparse_vectors_qdrant_format.append({
                "indices": indices,
                "values": values
            })
        return sparse_vectors_qdrant_format

    def save(self):
        try:
            with open(self.pkl_path, "wb") as f:
                pickle.dump(self.tfidf_vectorizer, f) # sklearn 벡터라이저 객체 전체 저장
            logging.info(f"[TfidfSparseVectorizer] TF-IDF 벡터라이저 저장 완료: {self.pkl_path}")
        except Exception as e:
            logging.error(f"[TfidfSparseVectorizer] TF-IDF 벡터라이저 저장 중 오류 발생: {e}")

    def load(self):
        try:
            with open(self.pkl_path, "rb") as f:
                self.tfidf_vectorizer = pickle.load(f)
            self.vocab = self.tfidf_vectorizer.vocabulary_ # 로드 후 vocab 업데이트
            logging.info(f"[TfidfSparseVectorizer] TF-IDF 벡터라이저 로드 완료: {self.pkl_path}")
            return True
        except (FileNotFoundError, EOFError, pickle.UnpicklingError) as e:
            logging.warning(f"[TfidfSparseVectorizer] TF-IDF 데이터 로드 실패 (파일 없거나 손상됨): {e}")
            return False
        except Exception as e:
            logging.error(f"[TfidfSparseVectorizer] TF-IDF 데이터 로드 중 예상치 못한 오류 발생: {e}")
            return False
           