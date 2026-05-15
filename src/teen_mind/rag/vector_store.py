"""
벡터 스토어 모듈 (FAISS)
─────────────────────────
지식 베이스 텍스트를 청크로 나누어 FAISS 인덱스로 저장하고
유사도 검색을 수행합니다.

핵심 원칙: 청크 크기가 RAG 품질을 결정합니다. (원칙 10번)
"""

from pathlib import Path
import json
import pickle

import faiss
import numpy as np

from teen_mind.rag.embeddings import TextEmbedder


class VectorStore:
    """
    FAISS 기반 벡터 스토어

    사용 예시:
        store = VectorStore(embedder)
        store.add_documents(texts, metadatas)
        results = store.search("수면 부족이 미치는 영향", k=3)
        store.save("data/knowledge_base/faiss_index")
    """

    def __init__(self, embedder: TextEmbedder, chunk_size: int = 512, chunk_overlap: int = 50):
        """
        Args:
            embedder: TextEmbedder 인스턴스
            chunk_size: 청크 당 최대 글자 수 (토큰 단위가 아닌 글자 단위)
                        실험 결과: 256 / 512 / 1024 중 512가 최적
            chunk_overlap: 청크 간 겹치는 글자 수 (문맥 연속성 유지)
        """
        self.embedder = embedder
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        self.index: faiss.IndexFlatIP | None = None  # Inner Product (코사인 유사도)
        self.documents: list[str] = []   # 원본 청크 텍스트
        self.metadatas: list[dict] = []  # 소스 파일명, 제목 등 메타데이터
        self.dim: int = embedder.get_embedding_dim()

    # ─────────────────────────────────────────
    # 청크 분할
    # ─────────────────────────────────────────
    def _split_into_chunks(self, text: str, source: str = "") -> list[tuple[str, dict]]:
        """
        텍스트를 chunk_size 단위로 분할합니다.
        chunk_overlap으로 청크 사이에 겹치는 부분을 만들어 문맥이 끊기지 않게 합니다.

        Returns:
            [(청크 텍스트, 메타데이터), ...]
        """
        chunks = []
        start = 0
        chunk_idx = 0

        while start < len(text):
            end = start + self.chunk_size
            chunk_text = text[start:end].strip()
            if chunk_text:
                chunks.append((
                    chunk_text,
                    {
                        "source": source,
                        "chunk_index": chunk_idx,
                        "start_char": start,
                        "chunk_size": self.chunk_size,
                    }
                ))
                chunk_idx += 1
            # 겹침 적용: end - overlap 위치부터 다음 청크 시작
            start = end - self.chunk_overlap

        return chunks

    # ─────────────────────────────────────────
    # 문서 추가
    # ─────────────────────────────────────────
    def add_documents(self, texts: list[str], sources: list[str] = None) -> None:
        """
        여러 텍스트 문서를 청크로 나누어 인덱스에 추가합니다.

        Args:
            texts: 텍스트 리스트 (파일 내용 전체)
            sources: 각 텍스트의 출처 이름 (파일명, 문서 제목 등)
        """
        sources = sources or [f"document_{i}" for i in range(len(texts))]
        all_chunks = []
        all_metas = []

        print(f"\n[벡터스토어] 문서 청킹 시작 (chunk_size={self.chunk_size}, overlap={self.chunk_overlap})")
        for text, source in zip(texts, sources):
            chunk_pairs = self._split_into_chunks(text, source)
            for chunk_text, meta in chunk_pairs:
                all_chunks.append(chunk_text)
                all_metas.append(meta)

        print(f"  총 청크 수: {len(all_chunks)}")

        # 임베딩 생성
        print("  임베딩 생성 중...")
        vectors = self.embedder.encode(all_chunks, show_progress=True)

        # FAISS 인덱스 초기화 및 추가
        if self.index is None:
            self.index = faiss.IndexFlatIP(self.dim)

        self.index.add(vectors.astype(np.float32))
        self.documents.extend(all_chunks)
        self.metadatas.extend(all_metas)
        print(f"[벡터스토어] 인덱스 구축 완료. 총 {len(self.documents)}개 청크 저장")

    # ─────────────────────────────────────────
    # 유사도 검색
    # ─────────────────────────────────────────
    def search(self, query: str, k: int = 3) -> list[dict]:
        """
        쿼리와 가장 유사한 상위 k개의 청크를 반환합니다.

        Returns:
            [{"text": ..., "score": ..., "metadata": {...}}, ...]
        """
        if self.index is None or self.index.ntotal == 0:
            raise RuntimeError("인덱스가 비어 있습니다. add_documents()를 먼저 실행하세요.")

        # 쿼리 임베딩
        query_vec = self.embedder.encode([query]).astype(np.float32)

        # 검색 (k가 인덱스 크기를 초과하지 않도록)
        actual_k = min(k, self.index.ntotal)
        scores, indices = self.index.search(query_vec, actual_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx >= 0:
                results.append({
                    "text": self.documents[idx],
                    "score": float(score),
                    "metadata": self.metadatas[idx],
                })
        return results

    # ─────────────────────────────────────────
    # 저장 / 불러오기
    # ─────────────────────────────────────────
    def save(self, directory: str | Path) -> None:
        """인덱스와 문서 목록을 저장합니다."""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)

        faiss.write_index(self.index, str(directory / "index.faiss"))
        with open(directory / "documents.pkl", "wb") as f:
            pickle.dump({"documents": self.documents, "metadatas": self.metadatas}, f)
        with open(directory / "config.json", "w", encoding="utf-8") as f:
            json.dump({"chunk_size": self.chunk_size, "chunk_overlap": self.chunk_overlap, "dim": self.dim}, f)

        print(f"[저장] 벡터스토어 → {directory}")

    @classmethod
    def load(cls, directory: str | Path, embedder: TextEmbedder) -> "VectorStore":
        """저장된 벡터스토어를 불러옵니다."""
        directory = Path(directory)

        with open(directory / "config.json", encoding="utf-8") as f:
            config = json.load(f)

        obj = cls(embedder, chunk_size=config["chunk_size"], chunk_overlap=config.get("chunk_overlap", 50))
        obj.index = faiss.read_index(str(directory / "index.faiss"))

        with open(directory / "documents.pkl", "rb") as f:
            data = pickle.load(f)
        obj.documents = data["documents"]
        obj.metadatas = data["metadatas"]

        print(f"[로드] 벡터스토어 ← {directory} ({len(obj.documents)}개 청크)")
        return obj
