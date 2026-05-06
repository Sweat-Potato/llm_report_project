# LLM Report Project

국내 증권사 리서치 리포트를 수집·분석하고, RAG 기반으로 질의응답 및 리포트를 생성하는 시스템입니다.

---

## 주요 기능

- 네이버 증권 업종분석 리포트 수집 (PDF)
- PDF 파싱 → 텍스트 정제 → 청킹 → 임베딩 → 벡터 DB 저장
- 질의 의도 분류 및 증권사 필터링 기반 스마트 검색
- 앙상블 / 균형 리트리버 + 리랭커 조합 검색
- 질문 유형별 특화 리포트 생성 (팩트 조회, 컨센서스, 타임라인 등 7종)
- Streamlit 웹 UI 제공

---

## 디렉토리 구조

```
llm_report_project/
├── app.py                      # Streamlit 웹 UI
├── pyproject.toml              # 의존성 (uv)
├── .env                        # API 키 설정
│
├── pipeline/
│   ├── ingest.py               # 전처리 파이프라인 (Load → Clean → Chunk → Embed → Store)
│   ├── main.py                 # 검색 및 질의응답 실행
│   └── inspect_chunks.py       # 청킹 결과 확인
│
├── src/
│   ├── collector/              # 데이터 수집
│   ├── processing/             # 텍스트 정제 및 청킹 전략
│   │   └── chunking/           # recursive / semantic / hybrid / sentence
│   ├── embedding/              # 임베딩 (OpenAI)
│   ├── vectorstore/            # 벡터 DB (ChromaDB)
│   ├── retriever/              # 라우터, 앙상블/균형 리트리버
│   ├── reranker/               # CrossEncoder / Cohere 리랭커
│   └── reportcreator/          # 리포트 생성 체인
│
├── data/                       # 수집 PDF, 청킹 결과, 벡터 DB 등
├── eval/                       # 평가 파이프라인
├── test/                       # 유닛 테스트
└── notebook/                   # 실험용 Jupyter 노트북
```

---

## 시작하기

### 1. 환경 설정

```bash
cp .env.example .env
# .env에 API 키 입력
```

`.env` 필수 키:

| 키 | 설명 |
|----|------|
| `OPENAI_API_KEY` | 임베딩 및 LLM 사용 |
| `LANGSMITH_API_KEY` | LangSmith 트레이싱 (선택) |

### 2. 의존성 설치

```bash
uv sync
```

### 3. 벡터 DB 구축 (최초 1회)

`data/reports/reports_naver_industry/` 에 PDF 리포트를 넣은 후:

```bash
uv run python pipeline/ingest.py
```

### 4. 실행

**웹 UI:**
```bash
uv run streamlit run app.py
```

**CLI 검색:**
```bash
uv run python pipeline/main.py
uv run python pipeline/main.py --ask "하나증권과 키움증권의 의견 차이"
```

---

## 전략 구성

전략별로 모듈을 교체할 수 있습니다. 벡터 DB 경로는 선택된 전략 조합으로 자동 생성됩니다.

| 단계 | 선택지 |
|------|--------|
| 청킹 | `recursive` / `semantic` / `hybrid` / `sentence` |
| 임베딩 | `openai` (text-embedding-3-small) |
| 벡터 DB | `chroma` |
| 리랭커 | `crossencoder` (무료) / `cohere` (유료) |
| 리트리버 | `ensemble` (점수 기반) / `balanced` (증권사 균형) |

---

## 리포트 생성 질문 유형

| 유형 | 설명 |
|------|------|
| `fact_lookup` | 수치·근거 조회 |
| `coverage_summary` | 커버리지 현황 요약 |
| `timeline` | 시간순 의견 변화 |
| `broker_comparison` | 증권사별 의견 비교 |
| `risk` | 리스크 요인 정리 |
| `consensus` | 공통 주제·컨센서스 |
| `other` | 복합 질문 → 전체 리포트 생성 |

---

## 평가

```bash
uv run python eval/03_eval_rag.py     # RAG 검색 평가
uv run python eval/04_eval_report.py  # 리포트 품질 평가
```
