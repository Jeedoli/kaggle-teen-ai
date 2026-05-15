"""
Tab 1: 정신건강 위험도 진단 페이지
─────────────────────────────────
생활 패턴 입력 → ML 모델 예측 → 위험도 결과 + 개선 제안
"""

from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from teen_mind.utils.visualization import plot_radar_chart, plot_risk_gauge


# 저장된 모델 경로
PROJECT_ROOT = Path(__file__).parent.parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "saved"
PREPROCESSOR_PATH = MODEL_PATH / "preprocessor.pkl"
ML_MODEL_PATH = MODEL_PATH / "best_ml_model.pkl"

RISK_LABELS = ["Low", "Medium", "High"]
RISK_KO = {"Low": "낮음 🟢", "Medium": "보통 🟡", "High": "높음 🔴"}

IMPROVEMENT_TIPS = {
    "Low": [
        "✅ 현재 건강한 생활 습관을 잘 유지하고 있어요!",
        "💪 지금처럼 규칙적인 수면과 운동을 계속 유지하세요.",
        "📱 소셜미디어 사용 시간을 지금 수준으로 유지하면 좋아요.",
    ],
    "Medium": [
        "⚠️ 몇 가지 생활 습관을 조금씩 개선하면 더 건강해질 수 있어요.",
        "😴 수면 시간을 하루 8시간 이상으로 늘려보세요.",
        "📵 저녁 10시 이후에는 소셜미디어 앱 알림을 꺼두세요.",
        "🚶 매일 30분 이상 가볍게 걷는 습관을 만들어보세요.",
    ],
    "High": [
        "🚨 정신건강에 주의가 필요한 상태입니다. 아래 조언을 참고하세요.",
        "📵 소셜미디어 사용 시간을 하루 1~2시간으로 줄여보세요.",
        "🛌 취침 1시간 전 스크린 사용을 완전히 중단하세요.",
        "🏃 주 3회 이상 규칙적인 운동이 스트레스와 불안을 크게 줄여줍니다.",
        "👨‍👩‍👧 가족이나 친한 친구와 오프라인 대화를 늘려보세요.",
        "🩺 증상이 2주 이상 지속된다면 전문 상담사나 의사를 찾아가세요.",
    ],
}


@st.cache_resource
def load_models():
    """저장된 전처리기와 모델을 로드합니다. (앱 시작 시 1회만 실행)"""
    try:
        from teen_mind.data.preprocessor import MentalHealthPreprocessor
        from teen_mind.models.ml_classifier import MLClassifier

        preprocessor = MentalHealthPreprocessor.load(PREPROCESSOR_PATH)
        ml_model = MLClassifier.load_model(ML_MODEL_PATH)
        return preprocessor, ml_model, None
    except FileNotFoundError as e:
        return None, None, str(e)


def render():
    st.header("🔍 정신건강 위험도 진단")
    st.caption("생활 패턴을 입력하면 AI가 위험도를 분석해드립니다.")

    preprocessor, ml_model, error = load_models()

    if error:
        st.warning(
            "⚠️ 아직 모델이 학습되지 않았습니다. "
            "`notebooks/03_ml_modeling.ipynb`를 먼저 실행해 모델을 저장하세요.",
            icon="🔧",
        )
        st.caption(f"(오류 세부: {error})")
        _show_demo_mode()
        return

    _show_input_form(preprocessor, ml_model)


def _show_input_form(preprocessor, ml_model):
    """입력 폼과 예측 결과를 표시합니다."""
    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        st.subheader("📝 생활 패턴 입력")

        age = st.slider("나이", min_value=13, max_value=19, value=16)
        gender = st.selectbox("성별", ["Male", "Female", "Other"])
        social_media_hours = st.slider("하루 소셜미디어 사용 시간 (시간)", 0.0, 12.0, 3.0, 0.5)
        sleep_hours = st.slider("하루 평균 수면 시간 (시간)", 3.0, 12.0, 7.0, 0.5)
        physical_activity = st.slider("주간 운동 시간 (시간)", 0.0, 21.0, 3.0, 0.5)

        st.divider()
        st.subheader("💭 심리 상태 (1=매우 낮음, 10=매우 높음)")

        depression_score = st.slider("우울감", 1, 10, 4)
        anxiety_score = st.slider("불안감", 1, 10, 4)
        stress_level = st.slider("스트레스 수준", 1, 10, 5)
        self_esteem = st.slider("자존감", 1, 10, 6)
        family_support = st.slider("가족 지지도", 1, 10, 6)

        st.divider()
        academic = st.selectbox("학업 성적", ["Poor", "Average", "Good", "Excellent"])
        online_support = st.radio("온라인 정신건강 자원 접근 가능 여부", [0, 1], format_func=lambda x: "예" if x else "아니오")

        predict_btn = st.button("🔍 위험도 분석하기", use_container_width=True, type="primary")

    with col2:
        if predict_btn:
            _run_prediction(
                preprocessor, ml_model,
                {
                    "age": age, "gender": gender,
                    "social_media_hours": social_media_hours,
                    "sleep_hours": sleep_hours,
                    "physical_activity_hours": physical_activity,
                    "depression_score": depression_score,
                    "anxiety_score": anxiety_score,
                    "stress_level": stress_level,
                    "self_esteem_score": self_esteem,
                    "family_support_score": family_support,
                    "academic_performance": academic,
                    "online_support_access": online_support,
                }
            )
        else:
            st.info("⬅️ 왼쪽에서 생활 패턴을 입력하고 분석 버튼을 눌러보세요.")


def _run_prediction(preprocessor, ml_model, input_data: dict):
    """입력 데이터로 예측을 실행하고 결과를 표시합니다."""
    input_df = pd.DataFrame([input_data])
    X = preprocessor.transform(input_df)
    proba = ml_model.predict_proba(X)[0]
    pred_idx = np.argmax(proba)
    pred_label = RISK_LABELS[pred_idx]
    confidence = proba[pred_idx]

    st.subheader("📊 분석 결과")

    # 게이지 차트
    gauge_fig = plot_risk_gauge(confidence, pred_label)
    st.plotly_chart(gauge_fig, use_container_width=True)

    # 위험도 확률 바
    st.markdown("**클래스별 확률**")
    for i, label in enumerate(RISK_LABELS):
        st.progress(float(proba[i]), text=f"{label}: {proba[i]*100:.1f}%")

    # 레이더 차트 (나 vs 데이터셋 평균)
    user_radar = {
        "소셜미디어": min(input_data["social_media_hours"] / 12 * 10, 10),
        "수면": input_data["sleep_hours"] / 10 * 10,
        "운동": min(input_data["physical_activity_hours"] / 21 * 10, 10),
        "가족지지": input_data["family_support_score"],
        "자존감": input_data["self_esteem_score"],
    }
    avg_radar = {"소셜미디어": 3.0, "수면": 7.0, "운동": 4.0, "가족지지": 6.5, "자존감": 6.0}
    radar_fig = plot_radar_chart(user_radar, avg_radar)
    st.plotly_chart(radar_fig, use_container_width=True)

    # 개선 제안
    st.subheader("💡 개선 제안")
    for tip in IMPROVEMENT_TIPS[pred_label]:
        st.markdown(f"- {tip}")


def _show_demo_mode():
    """모델 미학습 시 데모 UI만 표시"""
    st.subheader("🎮 데모 미리보기")
    st.markdown("모델 학습 완료 후 이런 화면이 나타납니다.")
    sample_fig = plot_risk_gauge(0.65, "Medium")
    st.plotly_chart(sample_fig, use_container_width=True)
