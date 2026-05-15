"""
Teen Mind AI — 메인 앱
─────────────────────
탭 1: 🧠 정신건강 위험도 진단
탭 2: 📊 데이터 인사이트
탭 3: 💬 AI 상담 챗봇

실행: streamlit run app/streamlit_app.py
"""

import streamlit as st

st.set_page_config(
    page_title="Teen Mind AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS 스타일
st.markdown("""
<style>
    .main-title { font-size: 2.5rem; font-weight: 700; color: #2c3e50; }
    .subtitle { font-size: 1.1rem; color: #7f8c8d; margin-bottom: 2rem; }
    .risk-low { background-color: #d5f5e3; padding: 10px 20px; border-radius: 8px; color: #1e8449; font-weight: bold; }
    .risk-medium { background-color: #fef9e7; padding: 10px 20px; border-radius: 8px; color: #b7950b; font-weight: bold; }
    .risk-high { background-color: #fadbd8; padding: 10px 20px; border-radius: 8px; color: #c0392b; font-weight: bold; }
    .disclaimer { font-size: 0.8rem; color: #95a5a6; border-left: 3px solid #bdc3c7; padding-left: 10px; margin-top: 20px; }
</style>
""", unsafe_allow_html=True)

# ── 헤더
st.markdown('<p class="main-title">🧠 Teen Mind AI</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="subtitle">소셜미디어 사용 패턴과 생활 습관으로 청소년 정신건강 위험도를 분석합니다.</p>',
    unsafe_allow_html=True,
)

# ── 탭 구성
tab1, tab2, tab3 = st.tabs(["🔍 위험도 진단", "📊 데이터 인사이트", "💬 AI 상담 챗봇"])

with tab1:
    from app.pages import diagnosis_page
    diagnosis_page.render()

with tab2:
    from app.pages import analysis_page
    analysis_page.render()

with tab3:
    from app.pages import chatbot_page
    chatbot_page.render()

# ── 공통 푸터
st.divider()
st.markdown(
    '<p class="disclaimer">⚠️ 이 서비스는 교육 및 연구 목적으로 제작되었습니다. '
    '의료 진단이나 전문 심리상담을 대체하지 않습니다. '
    '심각한 증상이 있다면 반드시 전문가와 상담하세요.</p>',
    unsafe_allow_html=True,
)
