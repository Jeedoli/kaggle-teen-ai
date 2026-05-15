"""
시각화 유틸리티 모듈
──────────────────────
EDA, 모델 성능, 특성 중요도 등 프로젝트 전반에 사용하는 시각화 함수 모음
"""

from typing import Optional
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix


# 한글 폰트 설정 (macOS)
matplotlib.rcParams["axes.unicode_minus"] = False
try:
    matplotlib.rcParams["font.family"] = "AppleGothic"
except Exception:
    pass

RISK_COLORS = {0: "#2ecc71", 1: "#e74c3c"}  # 0=정상, 1=우울증위험
RISK_LABELS = {0: "정상", 1: "우울증 위험"}
PALETTE = ["#3498db", "#e74c3c", "#2ecc71", "#9b59b6", "#f39c12"]


# ─────────────────────────────────────────
# EDA 시각화
# ─────────────────────────────────────────

def plot_class_distribution(y: pd.Series | np.ndarray, class_names: list[str] = None) -> go.Figure:
    """타겟 클래스 분포 파이 차트"""
    if isinstance(y, np.ndarray):
        counts = pd.Series(y).value_counts().sort_index()
    else:
        counts = y.value_counts().sort_index()

    labels = class_names or ["정상", "우울증 위험"]
    fig = px.pie(
        values=counts.values,
        names=labels[:len(counts)],
        title="우울증 위험 여부 분포",
        color_discrete_sequence=list(RISK_COLORS.values()),
        hole=0.4,
    )
    fig.update_traces(textposition="inside", textinfo="percent+label")
    return fig


def plot_correlation_heatmap(df: pd.DataFrame, figsize: tuple = (12, 10)) -> plt.Figure:
    """수치형 피처 간 상관관계 히트맵"""
    numeric_df = df.select_dtypes(include=[np.number])
    corr = numeric_df.corr()

    fig, ax = plt.subplots(figsize=figsize)
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(
        corr, mask=mask, annot=True, fmt=".2f",
        cmap="RdBu_r", center=0, square=True,
        linewidths=0.5, ax=ax,
        vmin=-1, vmax=1,
    )
    ax.set_title("피처 간 상관관계 히트맵", fontsize=15, pad=15)
    plt.tight_layout()
    return fig


def plot_feature_distributions(df: pd.DataFrame, columns: list[str], figsize: tuple = (16, 12)) -> plt.Figure:
    """여러 피처의 분포를 한 번에 시각화"""
    n_cols = 3
    n_rows = (len(columns) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten()

    for i, col in enumerate(columns):
        if col in df.columns:
            axes[i].hist(df[col].dropna(), bins=30, color=PALETTE[i % len(PALETTE)], edgecolor="white", alpha=0.8)
            axes[i].set_title(col)
            axes[i].set_xlabel("값")
            axes[i].set_ylabel("빈도")

    # 빈 서브플롯 숨기기
    for i in range(len(columns), len(axes)):
        axes[i].set_visible(False)

    fig.suptitle("피처 분포", fontsize=16, y=1.02)
    plt.tight_layout()
    return fig


def plot_social_media_vs_mental_health(df: pd.DataFrame) -> go.Figure:
    """소셔미디어 사용시간 vs 불안 수준 산점도"""
    df_plot = df.copy()
    df_plot["우울증여부"] = df_plot["depression_label"].map(RISK_LABELS)
    fig = px.scatter(
        df_plot,
        x="daily_social_media_hours",
        y="anxiety_level",
        color="우울증여부",
        color_discrete_map={"정상": RISK_COLORS[0], "우울증 위험": RISK_COLORS[1]},
        size="stress_level",
        hover_data=["age", "sleep_hours", "addiction_level"],
        title="소셜미디어 사용시간 vs 불안 수준",
        labels={
            "daily_social_media_hours": "일일 소셜미디어 사용 시간 (시간)",
            "anxiety_level": "불안 수준",
            "우울증여부": "우울증 여부",
        },
        template="plotly_white",
    )
    return fig


# ─────────────────────────────────────────
# 모델 성능 시각화
# ─────────────────────────────────────────

def plot_model_comparison(results_df: pd.DataFrame) -> go.Figure:
    """ML 모델 성능 비교 바 차트"""
    metrics = ["Accuracy", "F1 (binary)", "AUC-ROC"]
    fig = go.Figure()

    for metric in metrics:
        if metric in results_df.columns:
            fig.add_trace(go.Bar(
                name=metric,
                x=results_df["Model"],
                y=results_df[metric],
                text=results_df[metric].round(4),
                textposition="outside",
            ))

    fig.update_layout(
        title="ML 모델 성능 비교",
        xaxis_title="모델",
        yaxis_title="점수",
        barmode="group",
        yaxis=dict(range=[0, 1.1]),
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    return fig


def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, class_names: list[str] = None) -> plt.Figure:
    """혼동 행렬 히트맵"""
    cm = confusion_matrix(y_true, y_pred)
    class_names = class_names or [str(i) for i in range(cm.shape[0])]

    fig, ax = plt.subplots(figsize=(8, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title("혼동 행렬 (Confusion Matrix)", fontsize=14)
    plt.tight_layout()
    return fig


def plot_feature_importance(importance_df: pd.DataFrame, top_n: int = 10) -> go.Figure:
    """특성 중요도 수평 바 차트"""
    top_df = importance_df.head(top_n).sort_values("Importance")

    fig = px.bar(
        top_df,
        x="Importance",
        y="Feature",
        orientation="h",
        title=f"특성 중요도 Top {top_n}",
        labels={"Importance": "중요도", "Feature": "피처"},
        color="Importance",
        color_continuous_scale="Viridis",
        template="plotly_white",
    )
    fig.update_layout(showlegend=False, yaxis=dict(categoryorder="total ascending"))
    return fig


def plot_training_history(history: dict) -> go.Figure:
    """DL 모델 학습 곡선 (Train/Val Loss)"""
    fig = go.Figure()
    epochs = list(range(1, len(history.get("train_loss", [])) + 1))

    if "train_loss" in history:
        fig.add_trace(go.Scatter(x=epochs, y=history["train_loss"], mode="lines", name="Train Loss", line=dict(color="#3498db")))
    if "val_loss" in history:
        fig.add_trace(go.Scatter(x=epochs, y=history["val_loss"], mode="lines", name="Val Loss", line=dict(color="#e74c3c")))

    fig.update_layout(
        title="학습 곡선 (Learning Curve)",
        xaxis_title="Epoch",
        yaxis_title="Loss",
        template="plotly_white",
        legend=dict(orientation="h"),
    )
    return fig


# ─────────────────────────────────────────
# 서비스 시각화 (Streamlit 앱용)
# ─────────────────────────────────────────

def plot_radar_chart(user_values: dict, avg_values: dict) -> go.Figure:
    """
    나 vs 평균 레이더 차트
    사용자 입력값과 데이터셋 평균을 비교합니다.
    """
    categories = list(user_values.keys())
    user_vals = list(user_values.values())
    avg_vals = [avg_values.get(k, 0) for k in categories]

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(r=user_vals, theta=categories, fill="toself", name="나의 습관", line_color="#e74c3c"))
    fig.add_trace(go.Scatterpolar(r=avg_vals, theta=categories, fill="toself", name="전체 평균", line_color="#3498db", opacity=0.5))

    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 10])),
        showlegend=True,
        title="나의 생활 습관 vs 전체 평균",
        template="plotly_white",
    )
    return fig


def plot_risk_gauge(probability: float, risk_label: str) -> go.Figure:
    """우울증 위험 게이지 차트 (0~1 확률값을 시각적으로 표시)"""
    color = RISK_COLORS.get(1 if risk_label == "우울증 위험" else 0, "#95a5a6")
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=round(probability * 100, 1),
        domain={"x": [0, 1], "y": [0, 1]},
        title={"text": f"우울증 위험도: {risk_label}", "font": {"size": 20}},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1},
            "bar": {"color": color},
            "steps": [
                {"range": [0, 50], "color": "#d5f5e3"},
                {"range": [50, 100], "color": "#fadbd8"},
            ],
            "threshold": {
                "line": {"color": "black", "width": 4},
                "thickness": 0.75,
                "value": probability * 100,
            },
        },
        number={"suffix": "%"},
    ))
    fig.update_layout(height=300, template="plotly_white")
    return fig
