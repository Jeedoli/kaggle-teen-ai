"""
머신러닝 분류 모델 모듈
────────────────────────
Logistic Regression(베이스라인) → Random Forest → XGBoost 순서로
모델을 학습하고 성능을 비교합니다.
"""

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score
from xgboost import XGBClassifier


class MLClassifier:
    """
    여러 ML 모델을 한 번에 학습하고 성능을 비교하는 클래스

    사용 예시:
        clf = MLClassifier()
        clf.fit(X_train, y_train)
        report = clf.evaluate(X_test, y_test)
        clf.save_best("models/saved/best_ml_model.pkl")
    """

    def __init__(self, random_state: int = 42, n_jobs: int = -1):
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.models: dict[str, Any] = {}
        self.results: dict[str, dict] = {}
        self.best_model_name: str | None = None
        self._build_models()

    def _build_models(self) -> None:
        """모델 딕셔너리 초기화. 여기에 모델을 추가하거나 교체할 수 있습니다."""
        self.models = {
            "Logistic Regression (Baseline)": LogisticRegression(
                max_iter=1000,
                random_state=self.random_state,
                n_jobs=self.n_jobs,
            ),
            "Random Forest": RandomForestClassifier(
                n_estimators=200,
                max_depth=None,
                min_samples_split=5,
                random_state=self.random_state,
                n_jobs=self.n_jobs,
            ),
            "XGBoost": XGBClassifier(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                eval_metric="mlogloss",
                random_state=self.random_state,
                n_jobs=self.n_jobs,
                verbosity=0,
            ),
        }

    # ─────────────────────────────────────────
    # 학습
    # ─────────────────────────────────────────
    def fit(self, X_train: np.ndarray, y_train: np.ndarray, cv: int = 5) -> None:
        """
        모든 모델을 학습하고 교차검증 점수를 계산합니다.

        Args:
            X_train: 학습 피처 배열
            y_train: 학습 타겟 배열
            cv: 교차검증 폴드 수 (기본 5)
        """
        print(f"\n{'='*60}")
        print("ML 모델 학습 시작")
        print(f"{'='*60}")
        print(f"학습 데이터: {X_train.shape[0]}행 × {X_train.shape[1]}열")
        print(f"교차검증: {cv}-Fold Stratified\n")

        skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=self.random_state)

        for name, model in self.models.items():
            print(f"▶ {name} 학습 중...")
            model.fit(X_train, y_train)

            # 교차검증
            cv_scores = cross_val_score(
                model, X_train, y_train, cv=skf, scoring="f1_weighted", n_jobs=self.n_jobs
            )
            self.results[name] = {
                "model": model,
                "cv_f1_mean": cv_scores.mean(),
                "cv_f1_std": cv_scores.std(),
            }
            print(f"  CV F1 (weighted): {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

        # 가장 높은 CV F1 점수 모델을 베스트로 선택
        self.best_model_name = max(
            self.results, key=lambda k: self.results[k]["cv_f1_mean"]
        )
        print(f"\n🏆 최고 성능 모델: {self.best_model_name}")

    # ─────────────────────────────────────────
    # 평가
    # ─────────────────────────────────────────
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> pd.DataFrame:
        """
        테스트 데이터로 모든 모델 성능을 평가하고 비교 테이블을 반환합니다.
        """
        rows = []
        print(f"\n{'='*60}")
        print("테스트 데이터 평가 결과")
        print(f"{'='*60}")

        for name, info in self.results.items():
            model = info["model"]
            y_pred = model.predict(X_test)

            acc = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average="weighted")

            # AUC (다중 클래스 지원)
            try:
                y_prob = model.predict_proba(X_test)
                auc = roc_auc_score(y_test, y_prob, multi_class="ovr", average="weighted")
            except Exception:
                auc = float("nan")

            rows.append({
                "Model": name,
                "Accuracy": round(acc, 4),
                "F1 (weighted)": round(f1, 4),
                "AUC (OvR)": round(auc, 4),
                "CV F1 Mean": round(info["cv_f1_mean"], 4),
                "CV F1 Std": round(info["cv_f1_std"], 4),
            })

            marker = " ← BEST" if name == self.best_model_name else ""
            print(f"\n{name}{marker}")
            print(f"  Accuracy: {acc:.4f} | F1: {f1:.4f} | AUC: {auc:.4f}")

        results_df = pd.DataFrame(rows).sort_values("F1 (weighted)", ascending=False)
        return results_df

    def get_best_model(self) -> Any:
        """가장 성능이 좋은 모델 객체를 반환합니다."""
        if self.best_model_name is None:
            raise RuntimeError("먼저 fit()을 실행하세요.")
        return self.results[self.best_model_name]["model"]

    def get_feature_importance(self, feature_names: list[str]) -> pd.DataFrame:
        """
        Random Forest 또는 XGBoost의 특성 중요도를 반환합니다.
        어떤 생활 습관이 정신건강에 가장 큰 영향을 주는지 파악할 수 있습니다.
        """
        best = self.get_best_model()
        if not hasattr(best, "feature_importances_"):
            # Logistic Regression일 경우 계수 절댓값 사용
            importances = np.abs(best.coef_[0]) if hasattr(best, "coef_") else None
            if importances is None:
                return pd.DataFrame()
        else:
            importances = best.feature_importances_

        df = pd.DataFrame({
            "Feature": feature_names,
            "Importance": importances,
        }).sort_values("Importance", ascending=False)
        return df

    def predict(self, X: np.ndarray, model_name: str = None) -> np.ndarray:
        """예측 수행. model_name 미지정 시 최고 성능 모델 사용."""
        name = model_name or self.best_model_name
        return self.results[name]["model"].predict(X)

    def predict_proba(self, X: np.ndarray, model_name: str = None) -> np.ndarray:
        """확률 예측. model_name 미지정 시 최고 성능 모델 사용."""
        name = model_name or self.best_model_name
        return self.results[name]["model"].predict_proba(X)

    # ─────────────────────────────────────────
    # 저장 / 불러오기
    # ─────────────────────────────────────────
    def save_best(self, path: str | Path) -> None:
        """최고 성능 모델을 파일로 저장합니다."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.get_best_model(), path)
        print(f"[저장] {self.best_model_name} → {path}")

    def save_all(self, directory: str | Path) -> None:
        """모든 모델을 저장합니다."""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        for name, info in self.results.items():
            safe_name = name.replace(" ", "_").replace("(", "").replace(")", "").lower()
            filepath = directory / f"{safe_name}.pkl"
            joblib.dump(info["model"], filepath)
            print(f"[저장] {name} → {filepath}")

    @staticmethod
    def load_model(path: str | Path) -> Any:
        """저장된 모델 파일을 불러옵니다."""
        model = joblib.load(path)
        print(f"[로드] 모델 ← {path}")
        return model
