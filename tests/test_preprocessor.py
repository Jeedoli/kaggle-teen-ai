"""
전처리기 단위 테스트
──────────────────
pytest로 MentalHealthPreprocessor의 핵심 기능을 검증합니다.

실행: pytest tests/ -v
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from teen_mind.data.preprocessor import MentalHealthPreprocessor
from teen_mind.data.loader import COLUMN_NAMES, TARGET_COLUMN, NUMERIC_COLUMNS, CATEGORICAL_COLUMNS


def _make_sample_df(n: int = 100) -> pd.DataFrame:
    """테스트용 더미 데이터프레임 생성 (실제 데이터셋 구조 반영)."""
    rng = np.random.default_rng(42)
    # 극심한 불균형 (97%:3%) 반영: n=100이면 0이 97개, 1이 3개
    n_pos = max(int(n * 0.03), 1)
    n_neg = n - n_pos
    labels = np.array([0] * n_neg + [1] * n_pos)
    rng.shuffle(labels)
    data = {
        "age": rng.integers(13, 20, n),
        "gender": rng.choice(["male", "female"], n),
        "daily_social_media_hours": rng.uniform(0, 12, n),
        "platform_usage": rng.choice(["Instagram", "TikTok", "Both"], n),
        "sleep_hours": rng.uniform(4, 11, n),
        "screen_time_before_sleep": rng.uniform(0, 5, n),
        "academic_performance": rng.uniform(1, 10, n),
        "physical_activity": rng.uniform(0, 15, n),
        "social_interaction_level": rng.choice(["low", "medium", "high"], n),
        "stress_level": rng.integers(1, 11, n),
        "anxiety_level": rng.integers(1, 11, n),
        "addiction_level": rng.integers(1, 11, n),
        "depression_label": labels,
    }
    return pd.DataFrame(data)


# ─────────────────────────────────────────
# fit_transform 테스트
# ─────────────────────────────────────────

class TestFitTransform:
    def setup_method(self):
        self.df = _make_sample_df(100)
        self.preprocessor = MentalHealthPreprocessor(test_size=0.2, random_state=42)

    def test_output_shapes(self):
        """fit_transform 출력 형태 검증."""
        X_train, X_test, y_train, y_test = self.preprocessor.fit_transform(self.df)
        assert X_train.shape[0] == len(y_train), "X_train과 y_train 행 수가 다릅니다"
        assert X_test.shape[0] == len(y_test), "X_test와 y_test 행 수가 다릅니다"
        assert X_train.shape[1] == X_test.shape[1], "train/test 피처 수가 다릅니다"

    def test_train_test_split_ratio(self):
        """train/test 분할 비율 검증 (80:20)."""
        X_train, X_test, y_train, y_test = self.preprocessor.fit_transform(self.df)
        total = len(y_train) + len(y_test)
        test_ratio = len(y_test) / total
        assert 0.15 <= test_ratio <= 0.25, f"테스트 비율 {test_ratio:.2f}이 0.2에서 벗어납니다"

    def test_no_missing_values_after_transform(self):
        """전처리 후 결측치가 없어야 합니다."""
        X_train, X_test, y_train, y_test = self.preprocessor.fit_transform(self.df)
        assert not np.isnan(X_train).any(), "X_train에 결측치가 있습니다"
        assert not np.isnan(X_test).any(), "X_test에 결측치가 있습니다"

    def test_target_encoding(self):
        """타겟 변수가 0 또는 1 정수여야 합니다 (이진 분류)."""
        _, _, y_train, y_test = self.preprocessor.fit_transform(self.df)
        all_labels = np.concatenate([y_train, y_test])
        assert set(np.unique(all_labels)).issubset({0, 1}), \
            f"예상치 못한 타겟 값: {np.unique(all_labels)}"

    def test_scaling_applied(self):
        """스케일링 후 값이 표준 범위 내에 있어야 합니다."""
        X_train, _, _, _ = self.preprocessor.fit_transform(self.df)
        # 표준화 후 대부분 값이 [-10, 10] 범위 내에 있어야 함
        assert X_train.max() < 20, "스케일링이 적용되지 않았을 수 있습니다 (최대값이 너무 큼)"


# ─────────────────────────────────────────
# transform (배포용 단일 샘플) 테스트
# ─────────────────────────────────────────

class TestTransform:
    def setup_method(self):
        self.df = _make_sample_df(100)
        self.preprocessor = MentalHealthPreprocessor(test_size=0.2, random_state=42)
        self.preprocessor.fit_transform(self.df)  # fit 먼저 수행

    def test_single_sample_transform(self):
        """단일 샘플 변환이 정상 작동해야 합니다."""
        sample = pd.DataFrame([{
            "age": 16, "gender": "male",
            "daily_social_media_hours": 4.0, "platform_usage": "Instagram",
            "sleep_hours": 7.0, "screen_time_before_sleep": 1.0,
            "academic_performance": 6.0, "physical_activity": 3.0,
            "social_interaction_level": "medium",
            "stress_level": 5, "anxiety_level": 5, "addiction_level": 4,
        }])
        result = self.preprocessor.transform(sample)
        assert result.shape[0] == 1, "출력 행 수가 1이어야 합니다"
        assert not np.isnan(result).any(), "변환 결과에 결측치가 있습니다"

    def test_unknown_category_handled(self):
        """알 수 없는 범주형 값도 오류 없이 처리되어야 합니다."""
        sample = pd.DataFrame([{
            "age": 15, "gender": "unknown_gender",  # 학습 시 본 적 없는 값
            "daily_social_media_hours": 3.0, "platform_usage": "YouTube",
            "sleep_hours": 8.0, "screen_time_before_sleep": 0.5,
            "academic_performance": 7.0, "physical_activity": 2.0,
            "social_interaction_level": "very_high",
            "stress_level": 4, "anxiety_level": 3, "addiction_level": 2,
        }])
        try:
            result = self.preprocessor.transform(sample)
            assert result is not None
        except Exception as e:
            pytest.fail(f"알 수 없는 범주 처리 중 예외 발생: {e}")


# ─────────────────────────────────────────
# 저장 / 불러오기 테스트
# ─────────────────────────────────────────

class TestSaveLoad:
    def test_save_and_load(self, tmp_path):
        """저장 후 불러온 전처리기가 동일하게 작동해야 합니다."""
        df = _make_sample_df(80)
        preprocessor = MentalHealthPreprocessor(test_size=0.2, random_state=42)
        X_train, X_test, y_train, y_test = preprocessor.fit_transform(df)

        save_path = tmp_path / "preprocessor.pkl"
        preprocessor.save(save_path)

        loaded = MentalHealthPreprocessor.load(save_path)

        # 동일한 변환 결과를 내는지 확인
        sample = df.drop(columns=[TARGET_COLUMN]).iloc[:5]
        result_original = preprocessor.transform(sample)
        result_loaded = loaded.transform(sample)

        np.testing.assert_array_almost_equal(
            result_original, result_loaded,
            err_msg="저장/로드 후 변환 결과가 다릅니다"
        )
