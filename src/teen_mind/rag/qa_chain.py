"""
RAG Q&A 체인 모듈
──────────────────
사용자 질문 → 관련 문서 검색 → LLM으로 답변 생성

OpenAI API 키가 없으면 자동으로 로컬 템플릿 기반 답변으로 폴백합니다.
"""

import os
from pathlib import Path

from teen_mind.rag.vector_store import VectorStore


SYSTEM_PROMPT = """당신은 청소년 정신건강 분야의 친절한 AI 상담 도우미입니다.
제공된 참고 자료를 바탕으로 질문에 답변하세요.
중요: 이 서비스는 의료 진단이 아닙니다. 심각한 증상이 있다면 전문가 상담을 권장하세요.
답변은 한국어로, 이해하기 쉽고 따뜻한 어조로 작성하세요."""


class QAChain:
    """
    RAG 기반 Q&A 체인

    1. VectorStore에서 질문과 유사한 문서를 검색
    2. 검색된 문서를 컨텍스트로 LLM에 전달
    3. LLM이 컨텍스트를 참고해 답변 생성

    OpenAI API 키가 있으면 GPT-3.5, 없으면 템플릿 기반 폴백 답변.
    """

    def __init__(self, vector_store: VectorStore, top_k: int = 3, use_openai: bool = None):
        """
        Args:
            vector_store: 구축된 VectorStore 인스턴스
            top_k: 검색할 관련 문서 수
            use_openai: True=OpenAI API 사용, None=자동 감지
        """
        self.vector_store = vector_store
        self.top_k = top_k

        # OpenAI API 키 자동 감지
        api_key = os.getenv("OPENAI_API_KEY", "")
        if use_openai is None:
            self.use_openai = bool(api_key)
        else:
            self.use_openai = use_openai

        self.openai_client = None
        if self.use_openai:
            try:
                from openai import OpenAI
                self.openai_client = OpenAI(api_key=api_key)
                print("[RAG] OpenAI API 사용 모드")
            except Exception as e:
                print(f"[RAG] OpenAI 초기화 실패: {e}. 폴백 모드로 전환합니다.")
                self.use_openai = False
        else:
            print("[RAG] 템플릿 기반 폴백 모드 (OpenAI API 키 없음)")

    def ask(self, question: str) -> dict:
        """
        질문에 대한 답변을 반환합니다.

        Returns:
            {
                "answer": 답변 텍스트,
                "sources": [{"text": ..., "source": ..., "score": ...}, ...],
                "used_openai": True/False
            }
        """
        # 관련 문서 검색
        search_results = self.vector_store.search(question, k=self.top_k)

        if not search_results:
            return {
                "answer": "관련 정보를 찾지 못했습니다. 질문을 다르게 표현해 보세요.",
                "sources": [],
                "used_openai": False,
            }

        context = self._build_context(search_results)

        if self.use_openai and self.openai_client:
            answer = self._generate_openai_answer(question, context)
            used_openai = True
        else:
            answer = self._generate_fallback_answer(question, context, search_results)
            used_openai = False

        sources = [
            {
                "text": r["text"][:200] + "..." if len(r["text"]) > 200 else r["text"],
                "source": r["metadata"].get("source", "알 수 없음"),
                "score": round(r["score"], 4),
            }
            for r in search_results
        ]

        return {"answer": answer, "sources": sources, "used_openai": used_openai}

    def _build_context(self, search_results: list[dict]) -> str:
        """검색 결과를 하나의 컨텍스트 문자열로 합칩니다."""
        context_parts = []
        for i, result in enumerate(search_results, 1):
            source = result["metadata"].get("source", "참고자료")
            context_parts.append(f"[참고자료 {i} - {source}]\n{result['text']}")
        return "\n\n".join(context_parts)

    def _generate_openai_answer(self, question: str, context: str) -> str:
        """OpenAI API로 답변을 생성합니다."""
        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": f"참고 자료:\n{context}\n\n질문: {question}",
                    },
                ],
                temperature=0.7,
                max_tokens=600,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"[RAG] OpenAI API 오류: {e}")
            return self._generate_fallback_answer(question, context, [])

    def _generate_fallback_answer(
        self, question: str, context: str, search_results: list[dict]
    ) -> str:
        """
        OpenAI API 없이 검색된 문서를 그대로 요약해서 답변합니다.
        """
        if not search_results:
            return "관련 정보를 찾을 수 없습니다."

        top_text = search_results[0]["text"]
        source = search_results[0]["metadata"].get("source", "참고자료")

        answer = (
            f"'{question}'에 관한 정보를 찾았습니다.\n\n"
            f"📌 {source}에서 발췌:\n"
            f"{top_text}\n\n"
            f"⚠️ 이 답변은 데이터베이스 검색 기반입니다. "
            f"더 정확한 상담은 전문가를 찾아주세요."
        )
        return answer

    def get_context_preview(self, question: str) -> str:
        """디버깅용: 검색된 컨텍스트만 반환합니다."""
        results = self.vector_store.search(question, k=self.top_k)
        return self._build_context(results)
