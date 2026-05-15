"""
텍스트 임베딩 모듈
──────────────────
sentence-transformers를 사용해 텍스트를 벡터로 변환합니다.
입력 텍스트 품질이 임베딩 품질을 결정합니다. (원칙 14번)
"""

from pathlib import Path
import re

import numpy as np
from sentence_transformers import SentenceTransformer


# 무료로 사용할 수 있는 다국어 지원 모델
# 한국어 텍스트도 어느 정도 지원합니다.
DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


class TextEmbedder:
    """
    텍스트를 임베딩 벡터로 변환합니다.

    사용 예시:
        embedder = TextEmbedder()
        vectors = embedder.encode(["수면 부족은 불안을 유발합니다.", "운동은 스트레스를 줄입니다."])
    """

    def __init__(self, model_name: str = DEFAULT_MODEL):
        print(f"[임베딩] 모델 로드 중: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.model_name = model_name
        print(f"[임베딩] 모델 로드 완료 (차원: {self.model.get_sentence_embedding_dimension()})")

    def encode(self, texts: list[str], batch_size: int = 32, show_progress: bool = False) -> np.ndarray:
        """
        텍스트 리스트를 임베딩 배열로 변환합니다.

        Args:
            texts: 임베딩할 텍스트 리스트
            batch_size: 한 번에 처리할 텍스트 수
            show_progress: 진행 바 표시 여부

        Returns:
            shape: (len(texts), embedding_dim)
        """
        if not texts:
            raise ValueError("빈 텍스트 리스트는 임베딩할 수 없습니다.")

        # 입력 품질 확인: 빈 텍스트 필터링
        cleaned = [self.clean_text(t) for t in texts]
        empty_count = sum(1 for t in cleaned if not t.strip())
        if empty_count > 0:
            print(f"  ⚠️  빈 텍스트 {empty_count}개 발견. 건너뜁니다.")
            cleaned = [t if t.strip() else "unknown" for t in cleaned]

        vectors = self.model.encode(
            cleaned,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            normalize_embeddings=True,  # 코사인 유사도 계산에 유리
        )
        return vectors

    @staticmethod
    def clean_text(text: str) -> str:
        """
        임베딩 전 텍스트 정제.
        HTML 태그, 특수문자, 과도한 공백 제거.
        (원칙: 가공되지 않은 HTML은 임베딩에 적합하지 않음)
        """
        if not isinstance(text, str):
            return ""
        # HTML 태그 제거
        text = re.sub(r"<[^>]+>", " ", text)
        # URL 제거
        text = re.sub(r"http\S+|www\S+", " ", text)
        # 연속 공백 제거
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def get_embedding_dim(self) -> int:
        return self.model.get_sentence_embedding_dimension()
