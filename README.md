# 🧠 Teen Mind AI

> **청소년 정신건강 위험도 예측 서비스**  
> 소셜미디어 사용 패턴과 생활 습관 데이터로 정신건강 위험도를 분석하고, RAG 기반 AI 챗봇이 관련 정보를 제공합니다.

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.35+-red.svg)](https://streamlit.io)
[![License](https://img.shields.io/badge/License-Apache_2.0-green.svg)](LICENSE)

---

## 📌 프로젝트 소개

안녕하세요! 이 프로젝트는 취업용 ML/DL 포트폴리오로 만들었습니다.

요즘 청소년들이 소셜미디어를 얼마나 많이 쓰는지, 수면은 얼마나 자는지... 이런 생활 패턴이 정신건강에 어떤 영향을 미치는지 데이터로 분석해봤어요. 딱딱한 논문처럼 결과만 보여주는 게 아니라, 실제로 "내가 입력하면 결과가 나오는" 서비스를 만들고 싶었습니다.

**왜 이 주제를 선택했나요?**  
AICE Associate 자격을 취득하면서 배운 ML/DL 기술을 사회적으로 의미 있는 주제에 적용해보고 싶었습니다. 청소년 정신건강은 전 세계적으로 중요한 이슈이고, 데이터 기반 인사이트가 실제 도움이 될 수 있다고 생각했어요.

---

## ✨ 주요 기능

| 기능 | 설명 |
|------|------|
| 🔍 **위험도 진단** | 나이, 수면, 운동, 소셜미디어 시간 등을 입력하면 Low/Medium/High 위험도 예측 |
| 📊 **데이터 인사이트** | 원본 데이터셋 분포, 모델 성능 비교, 특성 중요도 시각화 |
| 💬 **AI 상담 챗봇** | FAISS 벡터 검색 + LLM으로 청소년 정신건강 관련 Q&A |

---

## 🏗️ 기술 스택

### ML/DL
- **Scikit-learn**: Logistic Regression, Random Forest (기준 모델)
- **XGBoost**: 고성능 그래디언트 부스팅
- **PyTorch**: Tabular Neural Network (FC + BatchNorm + ReLU + Dropout)

### RAG (검색 증강 생성)
- **sentence-transformers**: 텍스트 임베딩 (`all-MiniLM-L6-v2`)
- **FAISS**: 코사인 유사도 기반 벡터 검색
- **OpenAI API**: 답변 생성 (없으면 자동으로 폴백 모드)

### 서비스
- **Streamlit**: 웹 앱 (3탭 구성)
- **Plotly**: 인터랙티브 시각화

---

## 🚀 빠른 시작

### 1. 저장소 클론

```bash
git clone https://github.com/YOUR_USERNAME/teen-mind-ai.git
cd teen-mind-ai
```

### 2. 의존성 설치 (Poetry)

```bash
# Poetry가 없다면 먼저 설치
curl -sSL https://install.python-poetry.org | python3 -

# 패키지 설치
poetry install
poetry shell
```

### 3. 데이터 다운로드

```bash
# Kaggle CLI 필요 (pip install kaggle)
kaggle datasets download -d algozee/teenager-menthal-healy
unzip teenager-menthal-healy.zip -d data/raw/
```

### 4. 환경변수 설정 (선택사항)

```bash
cp .env.example .env
# .env 파일에 OPENAI_API_KEY를 추가하면 GPT-3.5 기반 답변 활성화
# 없어도 템플릿 기반 폴백 답변으로 작동합니다
```

### 5. 모델 학습

```bash
# Jupyter Notebook으로 순서대로 실행
jupyter notebook notebooks/
# 01_eda.ipynb → 02_preprocessing.ipynb → 03_ml_modeling.ipynb → 04_dl_modeling.ipynb → 05_rag_pipeline.ipynb
```

### 6. 앱 실행

```bash
streamlit run app/streamlit_app.py
```

---

## 📁 프로젝트 구조

```
teen-mind-ai/
├── app/
│   ├── streamlit_app.py          # 메인 앱 진입점
│   └── pages/
│       ├── diagnosis_page.py     # Tab 1: 위험도 진단
│       ├── analysis_page.py      # Tab 2: 데이터 인사이트
│       └── chatbot_page.py       # Tab 3: AI 챗봇
├── data/
│   ├── raw/                      # 원본 CSV (gitignore)
│   ├── processed/                # 전처리된 데이터 (gitignore)
│   └── knowledge_base/           # RAG 지식 베이스 텍스트 파일
├── models/saved/                 # 학습된 모델 파일 (gitignore)
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_ml_modeling.ipynb
│   ├── 04_dl_modeling.ipynb
│   └── 05_rag_pipeline.ipynb
├── src/teen_mind/
│   ├── data/
│   │   ├── loader.py             # 데이터 로드 및 검증
│   │   └── preprocessor.py      # 전처리 파이프라인
│   ├── models/
│   │   ├── ml_classifier.py      # LR + RF + XGBoost
│   │   └── dl_classifier.py      # PyTorch Neural Network
│   ├── rag/
│   │   ├── embeddings.py         # sentence-transformers 임베딩
│   │   ├── vector_store.py       # FAISS 벡터 스토어
│   │   └── qa_chain.py           # Q&A 체인 (OpenAI + 폴백)
│   └── utils/
│       └── visualization.py      # Plotly/seaborn 시각화
├── tests/
│   └── test_preprocessor.py
├── docs/
│   └── GUIDEBOOK.md              # 개발 가이드 (gitignore)
├── pyproject.toml
└── README.md
```

---

## 📊 데이터셋

- **출처**: [Kaggle - Social Media Impact on Teen Mental Health](https://www.kaggle.com/datasets/algozee/teenager-menthal-healy)
- **라이선스**: Apache 2.0
- **크기**: 약 700+ 샘플, 13개 컬럼
- **타겟**: `mental_health_risk` (Low / Medium / High)

주요 피처:

| 피처 | 설명 |
|------|------|
| `social_media_hours` | 하루 소셜미디어 사용 시간 |
| `sleep_hours` | 하루 평균 수면 시간 |
| `physical_activity_hours` | 주간 운동 시간 |
| `depression_score` | 우울감 자가 평가 (1-10) |
| `anxiety_score` | 불안감 자가 평가 (1-10) |
| `family_support_score` | 가족 지지도 (1-10) |

---

## 🤖 모델 성능 (참고)

학습 후 `notebooks/03_ml_modeling.ipynb`에서 상세 결과를 확인할 수 있습니다.

| 모델 | Accuracy | F1 (weighted) |
|------|----------|---------------|
| Logistic Regression | - | - |
| Random Forest | - | - |
| XGBoost | - | - |
| Neural Network | - | - |

*직접 학습 후 결과를 채워보세요!*

---

## 🧪 테스트

```bash
pytest tests/ -v
```

---

## ⚠️ 면책 고지

이 서비스는 교육 및 연구 목적으로 제작된 프로토타입입니다. **의료 진단 또는 전문 심리 상담을 대체하지 않습니다.** 심각한 정신건강 문제가 있다면 반드시 전문가와 상담하세요.

---

## 📝 License

Apache 2.0 — 데이터셋 라이선스를 따릅니다.

---

*Made with ❤️ as a portfolio project*
