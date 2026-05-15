"""
데이터 전처리 모듈
──────────────────
결측치 처리, 스케일링, 인코딩, Train/Test 분할을 담당합니다.
파이프라인은 교체 가능하도록 각 단계를 분리해서 구성했습니다.
"""

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
import joblib

from teen_mind.data.loader import (
    CATEGORICAL_COLUMNS,
    NUMERIC_COLUMNS,
    PROCESSED_DATA_DIR,
    TARGET_COLUMN,
)


class MentalHealthPreprocessor:
    """
    청소년 정신건강 데이터 전처리 파이프라인

    사용 예시:
        prep = MentalHealthPreprocessor()
        X_train, X_test, y_train, y_test = prep.fit_transform(df)
        prep.save("models/saved/preprocessor.pkl")
    """

    def __init__(self, scaler_type: str = "standard", test_size: float = 0.2, random_state: int = 42):
        """
        Args:
            scaler_type: "standard" (StandardScaler) 또는 "minmax" (MinMaxScaler)
            test_size: 테스트 데이터 비율 (기본 20%)
            random_state: 재현성을 위한 랜덤 시드
        """
        self.scaler_type = scaler_type
        self.test_size = test_size
        self.random_state = random_state

        # 각 단계별 변환기 (교체 가능하도록 분리)
        self.scaler = StandardScaler() if scaler_type == "standard" else MinMaxScaler()
        self.label_encoders: dict[str, LabelEncoder] = {}
        self.is_fitted = False
        self.class_weights: np.ndarray | None = None  # 클래스 불균형 가중치

    # ─────────────────────────────────────────
    # Step 1: 결측치 처리
    # ─────────────────────────────────────────
    def _handle_missing(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        수치형: 중앙값으로 대체
        범주형: 최빈값으로 대체
        """
        df = df.copy()
        for col in NUMERIC_COLUMNS:
            if col in df.columns and df[col].isnull().any():
                median_val = df[col].median()
                df[col] = df[col].fillna(median_val)
                print(f"  결측치 처리: {col} → 중앙값({median_val:.2f})으로 대체")

        for col in CATEGORICAL_COLUMNS:
            if col in df.columns and df[col].isnull().any():
                mode_val = df[col].mode()[0]
                df[col] = df[col].fillna(mode_val)
                print(f"  결측치 처리: {col} → 최빈값({mode_val})으로 대체")

        return df

    # ─────────────────────────────────────────
    # Step 2: 이상치 처리 (IQR 방식)
    # ─────────────────────────────────────────
    def _handle_outliers(self, df: pd.DataFrame, columns: list[str] = None) -> pd.DataFrame:
        """
        IQR(사분위범위) 방식으로 이상치를 클리핑합니다.
        이상치를 제거하는 대신 경계값으로 대체합니다 (데이터 손실 방지).
        """
        df = df.copy()
        target_cols = columns or NUMERIC_COLUMNS

        for col in target_cols:
            if col not in df.columns:
                continue
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR
            outlier_count = ((df[col] < lower) | (df[col] > upper)).sum()
            if outlier_count > 0:
                df[col] = df[col].clip(lower=lower, upper=upper)
                print(f"  이상치 처리: {col} → {outlier_count}개 클리핑 [{lower:.2f}, {upper:.2f}]")

        return df

    # ─────────────────────────────────────────
    # Step 3: 범주형 인코딩
    # ─────────────────────────────────────────
    def _encode_categoricals(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """
        범주형 컬럼을 레이블 인코딩합니다.
        fit=True: 학습 데이터 기준으로 인코더를 학습
        fit=False: 이미 학습된 인코더를 적용 (테스트 데이터에 사용)
        """
        df = df.copy()
        for col in CATEGORICAL_COLUMNS:
            if col not in df.columns:
                continue
            if fit:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
                self.label_encoders[col] = le
            else:
                le = self.label_encoders[col]
                # 학습 시 보지 못한 범주는 최빈값으로 대체
                known = set(le.classes_)
                df[col] = df[col].astype(str).apply(
                    lambda x: x if x in known else le.classes_[0]
                )
                df[col] = le.transform(df[col])
        return df

    # ─────────────────────────────────────────
    # Step 4: 수치형 스케일링
    # ─────────────────────────────────────────
    def _scale_numerics(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """
        수치형 컬럼을 스케일링합니다.
        fit=True: 학습 데이터에서 평균/표준편차를 계산
        fit=False: 이미 계산된 값으로 변환만 수행
        """
        df = df.copy()
        cols_to_scale = [c for c in NUMERIC_COLUMNS if c in df.columns]
        if fit:
            df[cols_to_scale] = self.scaler.fit_transform(df[cols_to_scale])
        else:
            df[cols_to_scale] = self.scaler.transform(df[cols_to_scale])
        return df

    # ─────────────────────────────────────────
    # 메인 메서드
    # ─────────────────────────────────────────
    def fit_transform(
        self, df: pd.DataFrame
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        전체 전처리 파이프라인을 실행하고 Train/Test 분할 결과를 반환합니다.

        Returns:
            (X_train, X_test, y_train, y_test)
        """
        print("\n[전처리 시작]")

        # 타겟 컬럼 (이미 0/1 정수 — 별도 인코딩 불필요)
        y = df[TARGET_COLUMN].astype(int).values
        X = df.drop(columns=[TARGET_COLUMN])

        # 파이프라인 순서대로 실행
        X = self._handle_missing(X)
        X = self._handle_outliers(X)
        X = self._encode_categoricals(X, fit=True)
        X = self._scale_numerics(X, fit=True)

        # Train/Test 분할 (stratified: 클래스 비율 유지)
        X_train, X_test, y_train, y_test = train_test_split(
            X.values, y,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=y,
        )

        self.feature_names = X.columns.tolist()
        self.is_fitted = True

        # 클래스 불균형 가중치 계산 (1169:31 극심한 불균형 대응)
        from sklearn.utils.class_weight import compute_class_weight
        classes = np.unique(y_train)
        weights = compute_class_weight("balanced", classes=classes, y=y_train)
        self.class_weights = weights

        neg, pos = np.bincount(y_train)
        print(f"\n[전처리 완료]")
        print(f"  학습 데이터: {X_train.shape[0]}행 × {X_train.shape[1]}열")
        print(f"  테스트 데이터: {X_test.shape[0]}행 × {X_test.shape[1]}열")
        print(f"  클래스 분포 (학습): 정상={neg}, 우울증위험={pos}")
        print(f"  클래스 가중치: {dict(zip(classes.tolist(), weights.round(2).tolist()))}")

        return X_train, X_test, y_train, y_test

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """
        이미 학습된 파이프라인으로 새로운 데이터를 변환합니다.
        (배포 시 사용자 입력값 변환에 사용)
        """
        if not self.is_fitted:
            raise RuntimeError("먼저 fit_transform()을 실행해야 합니다.")
        X = df.copy()
        X = self._handle_missing(X)
        X = self._handle_outliers(X)
        X = self._encode_categoricals(X, fit=False)
        X = self._scale_numerics(X, fit=False)
        return X.values

    def save(self, path: str | Path) -> None:
        """전처리 파이프라인(스케일러, 인코더 등)을 파일로 저장합니다."""
        joblib.dump(self, path)
        print(f"[저장] 전처리 파이프라인 → {path}")

    @classmethod
    def load(cls, path: str | Path) -> "MentalHealthPreprocessor":
        """저장된 파이프라인을 불러옵니다."""
        obj = joblib.load(path)
        print(f"[로드] 전처리 파이프라인 ← {path}")
        return obj

    def save_processed_data(
        self,
        X_train: np.ndarray,
        X_test: np.ndarray,
        y_train: np.ndarray,
        y_test: np.ndarray,
    ) -> None:
        """전처리된 데이터를 CSV로 저장합니다. (원시 데이터와 별도 보관)"""
        PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

        pd.DataFrame(X_train, columns=self.feature_names).assign(
            target=y_train
        ).to_csv(PROCESSED_DATA_DIR / "train.csv", index=False)

        pd.DataFrame(X_test, columns=self.feature_names).assign(
            target=y_test
        ).to_csv(PROCESSED_DATA_DIR / "test.csv", index=False)

        print(f"[저장] 전처리 데이터 → {PROCESSED_DATA_DIR}")
