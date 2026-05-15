"""
데이터 로더 모듈
────────────────
원시 CSV 파일을 읽어서 DataFrame으로 반환합니다.
원칙: 원시 데이터는 절대 수정하지 않고 그대로 보존합니다.
"""

from pathlib import Path
import pandas as pd


# 프로젝트 루트 기준으로 경로를 잡습니다.
PROJECT_ROOT = Path(__file__).resolve().parents[4]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
KNOWLEDGE_BASE_DIR = PROJECT_ROOT / "data" / "knowledge_base"

# 데이터셋 컬럼 정의 (13개)
COLUMN_NAMES = [
    "age",
    "gender",
    "social_media_hours",
    "sleep_hours",
    "physical_activity_days",
    "stress_level",
    "anxiety_score",
    "depression_score",
    "screen_time_hours",
    "number_of_platforms",
    "online_interaction_frequency",
    "academic_performance",
    "mental_health_risk",
]

# 타겟 컬럼
TARGET_COLUMN = "mental_health_risk"

# 수치형 컬럼 (스케일링 대상)
NUMERIC_COLUMNS = [
    "age",
    "social_media_hours",
    "sleep_hours",
    "physical_activity_days",
    "stress_level",
    "anxiety_score",
    "depression_score",
    "screen_time_hours",
    "number_of_platforms",
    "online_interaction_frequency",
]

# 범주형 컬럼 (인코딩 대상)
CATEGORICAL_COLUMNS = ["gender", "academic_performance"]


def load_raw_data(filename: str = "Teen_Mental_Health_Dataset.csv") -> pd.DataFrame:
    """
    원시 CSV 파일을 읽어 DataFrame으로 반환합니다.

    Args:
        filename: data/raw/ 폴더 내의 파일명

    Returns:
        원시 DataFrame (전처리 없음)

    Raises:
        FileNotFoundError: 파일이 없을 때
    """
    filepath = RAW_DATA_DIR / filename
    if not filepath.exists():
        raise FileNotFoundError(
            f"데이터 파일을 찾을 수 없습니다: {filepath}\n"
            "Kaggle에서 다운로드 후 data/raw/ 폴더에 넣어주세요.\n"
            "URL: https://www.kaggle.com/datasets/algozee/teenager-menthal-healy"
        )
    df = pd.read_csv(filepath)
    print(f"[로드 완료] {filename} → {df.shape[0]}행 × {df.shape[1]}열")
    return df


def load_processed_data(filename: str = "processed_data.csv") -> pd.DataFrame:
    """
    전처리 완료된 CSV 파일을 읽어 반환합니다.
    """
    filepath = PROCESSED_DATA_DIR / filename
    if not filepath.exists():
        raise FileNotFoundError(
            f"전처리된 데이터가 없습니다: {filepath}\n"
            "notebooks/02_preprocessing.ipynb 를 먼저 실행해주세요."
        )
    return pd.read_csv(filepath)


def validate_raw_data(df: pd.DataFrame) -> dict:
    """
    학습 전 데이터 기본 검증 (원칙: 학습 후가 아닌 학습 전에 검증)

    반환값:
        {
            "is_valid": True/False,
            "row_count": 행 수,
            "missing_ratio": 결측값 비율,
            "issues": [발견된 문제 목록]
        }
    """
    issues = []
    report = {
        "row_count": len(df),
        "column_count": len(df.columns),
        "missing_ratio": df.isnull().mean().to_dict(),
        "dtype_summary": df.dtypes.astype(str).to_dict(),
        "issues": issues,
    }

    # 최소 행 수 확인
    if len(df) < 100:
        issues.append(f"데이터 행이 너무 적습니다: {len(df)}행")

    # 타겟 컬럼 존재 여부
    if TARGET_COLUMN not in df.columns:
        issues.append(f"타겟 컬럼 '{TARGET_COLUMN}' 이 없습니다.")

    # 전체 결측 비율
    total_missing_ratio = df.isnull().mean().mean()
    if total_missing_ratio > 0.3:
        issues.append(f"결측값 비율이 너무 높습니다: {total_missing_ratio:.1%}")

    # 클래스 불균형 확인
    if TARGET_COLUMN in df.columns:
        class_counts = df[TARGET_COLUMN].value_counts(normalize=True)
        min_ratio = class_counts.min()
        if min_ratio < 0.1:
            issues.append(f"클래스 불균형이 심합니다. 최소 클래스 비율: {min_ratio:.1%}")
        report["class_distribution"] = class_counts.to_dict()

    report["is_valid"] = len(issues) == 0

    # 결과 출력
    print(f"\n{'='*50}")
    print("데이터 검증 결과")
    print(f"{'='*50}")
    print(f"행 수: {report['row_count']}")
    print(f"열 수: {report['column_count']}")
    print(f"전체 결측 비율: {total_missing_ratio:.2%}")
    if "class_distribution" in report:
        print(f"클래스 분포: {report['class_distribution']}")
    if issues:
        print("\n⚠️  발견된 문제:")
        for issue in issues:
            print(f"  - {issue}")
    else:
        print("\n✅ 데이터 검증 통과")
    print(f"{'='*50}\n")

    return report
