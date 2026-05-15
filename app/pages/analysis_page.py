"""
Tab 2: 데이터 인사이트 대시보드
─────────────────────────────
특성 중요도, 상관관계, 분포 시각화
"""

from pathlib import Path

import pandas as pd
import streamlit as st

from teen_mind.utils.visualization import (
    plot_class_distribution,
    plot_correlation_heatmap,
    plot_feature_importance,
    plot_model_comparison,
    plot_social_media_vs_mental_health,
)

PROJECT_ROOT = Path(__file__).parent.parent.parent
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
MODEL_DIR = PROJECT_ROOT / "models" / "saved"


@st.cache_data
def load_data():
    csv_files = list(RAW_DATA_DIR.glob("*.csv"))
    if not csv_files:
        return None
    return pd.read_csv(csv_files[0])


@st.cache_data
def load_results():
    results_path = MODEL_DIR / "model_comparison.csv"
    if results_path.exists():
        return pd.read_csv(results_path)
    return None


@st.cache_data
def load_importance():
    importance_path = MODEL_DIR / "feature_importance.csv"
    if importance_path.exists():
        return pd.read_csv(importance_path)
    return None


def render():
    st.header("📊 데이터 인사이트 대시보드")
    st.caption("데이터셋 분포와 ML 모델 성능을 탐색합니다.")

    df = load_data()
    results_df = load_results()
    importance_df = load_importance()

    # ── 데이터셋 없을 때 안내
    if df is None:
        st.warning("⚠️ 데이터가 없습니다. `data/raw/` 폴더에 CSV 파일을 넣어주세요.")
        st.markdown("""
        **데이터 다운로드 방법:**
        ```bash
        kaggle datasets download -d algozee/teenager-menthal-healy
        unzip teenager-menthal-healy.zip -d data/raw/
        ```
        """)
        return

    # ── 기본 통계
    st.subheader("📈 데이터셋 개요")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("총 샘플 수", f"{len(df):,}개")
    col2.metric("피처 수", f"{df.shape[1] - 1}개")
    target_col = "mental_health_risk"
    if target_col in df.columns:
        col3.metric("클래스 수", "3개 (Low/Medium/High)")
        col4.metric("결측치", f"{df.isnull().sum().sum()}개")

    # ── 탭으로 구분
    viz_tab1, viz_tab2, viz_tab3, viz_tab4 = st.tabs(
        ["📊 분포", "🔗 상관관계", "🤖 모델 성능", "🔍 특성 중요도"]
    )

    with viz_tab1:
        col_a, col_b = st.columns(2)
        with col_a:
            if target_col in df.columns:
                fig = plot_class_distribution(df[target_col])
                st.plotly_chart(fig, use_container_width=True)
        with col_b:
            if "social_media_hours" in df.columns and target_col in df.columns:
                fig2 = plot_social_media_vs_mental_health(df)
                st.plotly_chart(fig2, use_container_width=True)

    with viz_tab2:
        fig = plot_correlation_heatmap(df)
        st.pyplot(fig)

    with viz_tab3:
        if results_df is not None:
            fig = plot_model_comparison(results_df)
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(results_df, use_container_width=True)
        else:
            st.info("모델 비교 결과가 없습니다. `03_ml_modeling.ipynb`를 실행하세요.")

    with viz_tab4:
        if importance_df is not None:
            top_n = st.slider("Top N 피처", 5, min(20, len(importance_df)), 10)
            fig = plot_feature_importance(importance_df, top_n=top_n)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("특성 중요도 데이터가 없습니다. `03_ml_modeling.ipynb`를 실행하세요.")

    # ── 원본 데이터 미리보기
    with st.expander("🗄️ 원본 데이터 미리보기"):
        st.dataframe(df.head(20), use_container_width=True)
        st.markdown(f"**데이터 통계 요약**")
        st.dataframe(df.describe(), use_container_width=True)
