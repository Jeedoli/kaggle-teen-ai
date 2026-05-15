"""
Tab 3: AI 상담 챗봇 페이지 (RAG 기반)
──────────────────────────────────────
사용자 질문 → FAISS 벡터 검색 → 답변 생성 + 출처 표시
"""

from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).parent.parent.parent
KB_DIR = PROJECT_ROOT / "data" / "knowledge_base"
FAISS_DIR = KB_DIR / "faiss_index"

DISCLAIMER = (
    "⚠️ **면책 고지**: 이 챗봇은 교육 목적의 AI입니다. "
    "의료 진단 또는 전문 심리 상담을 대체하지 않습니다. "
    "심각한 증상이 지속된다면 반드시 전문가와 상담하세요."
)

EXAMPLE_QUESTIONS = [
    "소셜미디어를 많이 쓰면 우울증이 생기나요?",
    "청소년에게 권장되는 수면 시간은 얼마인가요?",
    "운동이 불안감을 줄이는 데 도움이 되나요?",
    "스트레스를 줄이는 방법이 있나요?",
    "이 데이터셋에서 가장 중요한 위험 요인은 무엇인가요?",
]


@st.cache_resource
def load_rag_chain():
    """RAG 체인을 초기화합니다. (앱 시작 시 1회만 실행)"""
    try:
        from teen_mind.rag.embeddings import TextEmbedder
        from teen_mind.rag.vector_store import VectorStore
        from teen_mind.rag.qa_chain import QAChain

        embedder = TextEmbedder()

        # 저장된 인덱스가 있으면 로드, 없으면 새로 빌드
        if FAISS_DIR.exists() and (FAISS_DIR / "index.faiss").exists():
            vector_store = VectorStore.load(FAISS_DIR, embedder)
        else:
            vector_store = _build_vector_store(embedder)

        qa_chain = QAChain(vector_store, top_k=3)
        return qa_chain, None

    except Exception as e:
        return None, str(e)


def _build_vector_store(embedder):
    """지식 베이스 텍스트 파일을 읽어 FAISS 인덱스를 구축합니다."""
    from teen_mind.rag.vector_store import VectorStore

    txt_files = sorted(KB_DIR.glob("*.txt"))
    if not txt_files:
        raise FileNotFoundError(f"지식 베이스 텍스트 파일이 없습니다. ({KB_DIR})")

    texts, sources = [], []
    for path in txt_files:
        with open(path, encoding="utf-8") as f:
            texts.append(f.read())
        sources.append(path.stem)

    vector_store = VectorStore(embedder, chunk_size=512, chunk_overlap=50)
    vector_store.add_documents(texts, sources)
    vector_store.save(FAISS_DIR)
    return vector_store


def render():
    st.header("💬 AI 상담 챗봇")
    st.caption("청소년 정신건강에 대해 궁금한 것을 물어보세요.")
    st.info(DISCLAIMER)

    qa_chain, error = load_rag_chain()

    if error:
        st.error(f"RAG 초기화 실패: {error}")
        st.markdown("sentence-transformers와 faiss-cpu가 설치되어 있는지 확인하세요.")
        return

    # ── 채팅 기록 초기화
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # ── 예시 질문 버튼
    st.subheader("💡 예시 질문")
    cols = st.columns(len(EXAMPLE_QUESTIONS))
    for i, (col, q) in enumerate(zip(cols, EXAMPLE_QUESTIONS)):
        if col.button(f"Q{i+1}", help=q, use_container_width=True):
            _process_question(qa_chain, q)

    st.divider()

    # ── 채팅 기록 표시
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant" and "sources" in msg:
                _show_sources(msg["sources"])

    # ── 입력창
    user_input = st.chat_input("질문을 입력하세요...")
    if user_input:
        _process_question(qa_chain, user_input)

    # ── 초기화 버튼
    if st.session_state.chat_history:
        if st.button("🗑️ 대화 초기화", use_container_width=False):
            st.session_state.chat_history = []
            st.rerun()


def _process_question(qa_chain, question: str):
    """질문을 처리하고 채팅 기록에 추가합니다."""
    # 사용자 메시지 추가
    st.session_state.chat_history.append({"role": "user", "content": question})

    with st.spinner("답변 생성 중..."):
        result = qa_chain.ask(question)

    # 어시스턴트 메시지 추가
    badge = "🤖 GPT-3.5" if result.get("used_openai") else "📚 지식베이스"
    answer = f"{result['answer']}\n\n---\n*{badge} 기반 답변*"
    st.session_state.chat_history.append({
        "role": "assistant",
        "content": answer,
        "sources": result.get("sources", []),
    })
    st.rerun()


def _show_sources(sources: list[dict]):
    """참고 출처를 접기 형태로 표시합니다."""
    if not sources:
        return
    with st.expander(f"📚 참고 출처 ({len(sources)}개)"):
        for i, src in enumerate(sources, 1):
            st.markdown(f"**[{i}] {src['source']}** (유사도: {src['score']:.3f})")
            st.caption(src["text"])
            st.divider()
