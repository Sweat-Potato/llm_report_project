"""
src/reportcreator/freeform_chain.py
자유형 질문 라우터 + 응답 체인

입력 질문을 분류하여:
  - Q&A 유형  → freeform 답변 (증권사 비교, 시점 분석, 리스크 정리, 컨센서스 요약 등)
  - 리포트 유형 → report_chain.generate_report() 위임

사용 예시:
    from src.reportcreator.freeform_chain import handle

    result = handle(retriever, "하나증권과 키움증권의 3월 의견 차이를 설명해줘")
    result = handle(retriever, "AI 반도체 업황 전망 리포트 작성해줘")  # → report_chain
"""
from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path

from langchain.schema import Document
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate

from src.retriever.ensemble_retriever import retrieve
from src.reranker.reranker import rerank


# ── LLM ──────────────────────────────────────────────────────────────────────

def _llm_fast()   -> ChatOpenAI: return ChatOpenAI(model="gpt-4o-mini", temperature=0)
def _llm_strong() -> ChatOpenAI: return ChatOpenAI(model="gpt-4o",      temperature=0.2)


# ── Step 1: 라우터 (Q&A vs 리포트 생성) ──────────────────────────────────────

_ROUTER_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """당신은 금융 리포트 시스템의 라우터입니다.
사용자 입력이 아래 두 가지 중 어디에 해당하는지 판별하세요.

mode 값:
  "qa"     - 특정 질문에 대한 분석/비교/요약 답변이 필요한 경우
  "report" - 특정 주제에 대해 처음부터 종합 리포트를 작성해 달라는 경우

판별 기준:
  qa     → 증권사 비교, 특정 시점 의견, 리스크 나열, 컨센서스 요약,
            애널리스트 입장 정리, "설명해줘 / 비교해줘 / 정리해줘 / 알려줘"
  report → "리포트 써줘 / 분석해줘 / 종합해줘 / 전망 작성해줘",
            특정 섹터·종목에 대한 포괄적 분석 요청

JSON으로만 답변하세요:
{{"mode": "qa" or "report", "topic": "리포트 생성 시 주제 (qa면 null)"}}"""),

    ("human",  "하나증권과 키움증권의 3월 의견 차이를 설명해줘"),
    ("ai",     '{{"mode": "qa", "topic": null}}'),

    ("human",  "이번 달 반도체 섹터 투자의견 변화 알려줘"),
    ("ai",     '{{"mode": "qa", "topic": null}}'),

    ("human",  "조선업에서 언급된 리스크 요인 정리해줘"),
    ("ai",     '{{"mode": "qa", "topic": null}}'),

    ("human",  "AI 인프라에 대해 증권사들이 공통으로 강조하는 게 뭐야"),
    ("ai",     '{{"mode": "qa", "topic": null}}'),

    ("human",  "각 증권사 목표주가 상향 근거가 뭐야"),
    ("ai",     '{{"mode": "qa", "topic": null}}'),

    ("human",  "AI 반도체 업황 전망 리포트 작성해줘"),
    ("ai",     '{{"mode": "report", "topic": "AI 반도체 업황 전망"}}'),

    ("human",  "조선업 수주 동향 종합 분석해줘"),
    ("ai",     '{{"mode": "report", "topic": "조선업 수주 동향"}}'),

    ("human",  "2차전지 시장 투자 리포트 써줘"),
    ("ai",     '{{"mode": "report", "topic": "2차전지 시장"}}'),

    ("human",  "방산 섹터 전망 정리해줘"),
    ("ai",     '{{"mode": "report", "topic": "방산 섹터 전망"}}'),

    ("human",  "{question}"),
])


def _route(question: str) -> dict:
    chain = _ROUTER_PROMPT | _llm_fast()
    raw   = chain.invoke({"question": question}).content.strip()
    raw   = re.sub(r'^```json\s*', '', raw)
    raw   = re.sub(r'\s*```$',     '', raw)
    try:
        return json.loads(raw)
    except Exception:
        return {"mode": "qa", "topic": None}


# ── Step 2: 의도 분석 — 5개 유형 few-shot ────────────────────────────────────

_INTENT_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """당신은 증권사 리서치 리포트 검색 전략가입니다.
사용자 질문을 분석하여 JSON으로 응답하세요.

{{
  "question_type": "broker_comparison | timeline | risk | consensus | valuation | other",
  "target_brokers": ["언급된 증권사 목록, 없으면 빈 배열"],
  "target_sector":  "언급된 섹터/종목, 없으면 null",
  "target_period":  "언급된 기간 (예: 2026-03), 없으면 null",
  "search_queries": ["쿼리1", "쿼리2", "쿼리3"],
  "structure_hint": "답변 구성 방향 한 문장"
}}

JSON만 반환 (설명 없이)."""),

    # broker_comparison
    ("human", "하나증권과 키움증권의 3월 반도체 의견 차이를 설명해줘"),
    ("ai", json.dumps({
        "question_type":  "broker_comparison",
        "target_brokers": ["하나증권", "키움증권"],
        "target_sector":  "반도체",
        "target_period":  "2026-03",
        "search_queries": [
            "하나증권 반도체 투자의견 3월",
            "키움증권 반도체 투자의견 3월",
            "반도체 섹터 목표주가 컨센서스 2026",
        ],
        "structure_hint": "두 증권사의 투자의견·목표주가·논거를 나란히 비교하고 핵심 이견 포인트를 강조",
    }, ensure_ascii=False)),

    # timeline
    ("human", "이번 달 반도체 섹터 투자의견 변화 알려줘"),
    ("ai", json.dumps({
        "question_type":  "timeline",
        "target_brokers": [],
        "target_sector":  "반도체",
        "target_period":  "2026-04",
        "search_queries": [
            "반도체 투자의견 변화 2026년 4월",
            "반도체 목표주가 상향 하향 최근",
            "반도체 업황 전망 최신",
        ],
        "structure_hint": "기간 내 의견 변화 흐름을 시간 순으로 정리하고 변화 원인을 분석",
    }, ensure_ascii=False)),

    # risk
    ("human", "조선업에서 언급된 리스크 요인 정리해줘"),
    ("ai", json.dumps({
        "question_type":  "risk",
        "target_brokers": [],
        "target_sector":  "조선",
        "target_period":  None,
        "search_queries": [
            "조선업 리스크 요인",
            "조선 수주 불확실성 원가 상승",
            "조선 섹터 하방 리스크 시나리오",
        ],
        "structure_hint": "리스크를 단기/구조적으로 분류하고 각 리스크의 발생 조건과 영향도를 서술",
    }, ensure_ascii=False)),

    # consensus
    ("human", "AI 인프라에 대해 증권사들이 공통으로 강조하는 게 뭐야"),
    ("ai", json.dumps({
        "question_type":  "consensus",
        "target_brokers": [],
        "target_sector":  "AI 인프라",
        "target_period":  None,
        "search_queries": [
            "AI 인프라 투자 증설 데이터센터",
            "AI 인프라 섹터 투자의견 컨센서스",
            "AI 반도체 전력 네트워크 성장 전망",
        ],
        "structure_hint": "여러 증권사가 공통으로 강조하는 핵심 논거를 항목별로 정리하고 수치로 뒷받침",
    }, ensure_ascii=False)),

    # valuation
    ("human", "각 증권사 목표주가 상향 근거가 뭐야"),
    ("ai", json.dumps({
        "question_type":  "valuation",
        "target_brokers": [],
        "target_sector":  None,
        "target_period":  None,
        "search_queries": [
            "목표주가 상향 근거 밸류에이션",
            "PER PBR 상향 조정 이유",
            "실적 추정치 상향 목표주가",
        ],
        "structure_hint": "증권사별 목표주가와 상향 근거(밸류에이션 방법론, 실적 추정치 변화)를 비교 정리",
    }, ensure_ascii=False)),

    ("human", "{question}"),
])


def _analyze_intent(question: str) -> dict:
    chain = _INTENT_PROMPT | _llm_fast()
    raw   = chain.invoke({"question": question}).content.strip()
    raw   = re.sub(r'^```json\s*', '', raw)
    raw   = re.sub(r'\s*```$',     '', raw)
    try:
        return json.loads(raw)
    except Exception:
        return {
            "question_type":  "other",
            "target_brokers": [],
            "target_sector":  None,
            "target_period":  None,
            "search_queries": [question],
            "structure_hint": "질문에 직접 답변",
        }


# ── Step 3: 다중 쿼리 검색 + 중복 제거 + Rerank ───────────────────────────────

def _collect_chunks(
    retriever,
    queries:        list[str],
    target_brokers: list[str],
    k_per_query:    int = 15,
    top_n:          int = 12,
) -> list[Document]:
    all_candidates: list[Document] = []
    seen: set[tuple] = set()

    for q in queries:
        for doc in retrieve(retriever, q, k=k_per_query):
            key = (
                doc.metadata.get("filename",    ""),
                doc.metadata.get("chunk_index", ""),
            )
            if key not in seen:
                seen.add(key)
                all_candidates.append(doc)

    if target_brokers:
        normalized = [b.replace("증권", "").replace(" ", "") for b in target_brokers]
        pinned = [
            d for d in all_candidates
            if any(nb in (d.metadata.get("source_firm", "") or "").replace(" ", "")
                   for nb in normalized)
        ]
        rest = [d for d in all_candidates if d not in pinned]
        all_candidates = pinned + rest

    return rerank(queries[0], all_candidates, top_n=top_n)


# ── Step 4: 증권사별 컨텍스트 구성 ───────────────────────────────────────────

def _build_context(docs: list[Document], target_brokers: list[str]) -> str:
    broker_chunks: dict[str, list[str]] = {}
    broker_dates:  dict[str, set[str]]  = {}

    for doc in docs:
        firm = doc.metadata.get("source_firm", "알 수 없음")
        date = doc.metadata.get("report_date", "")
        broker_chunks.setdefault(firm, []).append(doc.page_content)
        broker_dates.setdefault(firm, set())
        if date:
            broker_dates[firm].add(date)

    if target_brokers:
        normalized = [b.replace("증권", "").replace(" ", "") for b in target_brokers]
        order = sorted(
            broker_chunks.keys(),
            key=lambda f: 0 if any(nb in f.replace(" ", "") for nb in normalized) else 1,
        )
    else:
        order = list(broker_chunks.keys())

    parts = []
    for firm in order:
        date_str = ", ".join(sorted(broker_dates[firm])) or "날짜 미상"
        content  = "\n\n---\n\n".join(broker_chunks[firm])[:4000]
        parts.append(f"### [{firm}] (리포트 날짜: {date_str})\n{content}")

    return "\n\n".join(parts)


# ── Step 5: 답변 생성 — 5개 유형 few-shot ────────────────────────────────────

_ANSWER_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """당신은 15년 경력의 금융 리서치 전문 애널리스트입니다.
제공된 증권사 리포트 발췌문을 바탕으로 질문에 답변하세요.

**절대 원칙**
- 발췌문에 있는 구체적 수치(목표주가, 성장률, EPS 등)를 반드시 인용할 것
- 일반론·상식·교과서 내용 금지. 이 발췌문에만 있는 내용에 집중
- 발췌문에 없는 정보는 "리포트에서 확인되지 않음"으로 명시
- 참고한 증권사명과 날짜를 본문에 반드시 표기
- 마크다운으로 작성, 질문에 맞는 구조로 자유롭게 설계"""),

    # ── few-shot 1: broker_comparison ────────────────────────────────────
    ("human", """\
[구조 가이드] 두 증권사의 투자의견과 핵심 논거를 나란히 비교
[질문] 하나증권과 키움증권의 3월 반도체 의견 차이를 설명해줘
[발췌]
### [하나증권] (리포트 날짜: 2026-03-10)
HBM3e 공급 부족이 2026년 하반기까지 지속. 삼성전자 목표주가 105,000원 유지, 투자의견 매수. AI 서버 수요 폭발로 D램 업황 상저하고 전망.
### [키움증권] (리포트 날짜: 2026-03-15)
중국 레거시 DRAM 공급 과잉이 단기 가격 압박 요인. 삼성전자 목표주가 88,000원으로 하향. 상반기 실적 부진 후 하반기 반등 가능성 유지. 투자의견 중립."""),
    ("ai", """\
## 하나증권 vs 키움증권 — 2026년 3월 반도체 의견 비교

### 투자의견 및 목표주가

| 구분 | 하나증권 (2026-03-10) | 키움증권 (2026-03-15) |
|------|----------------------|----------------------|
| 투자의견 | **매수** | **중립** |
| 삼성전자 목표주가 | 105,000원 | 88,000원 (하향) |
| 핵심 전망 | 상저하고 | 단기 부진 후 하반기 반등 |

### 핵심 논거 비교

**하나증권 — 강세론**
- HBM3e 공급 부족이 2026년 하반기까지 지속될 것으로 판단
- AI 서버 수요 폭발이 D램 업황 회복의 핵심 촉매

**키움증권 — 중립론**
- 중국 레거시 DRAM 공급 과잉이 단기 가격 압박 요인
- 상반기 실적 부진 가능성을 주요 리스크로 강조

### 핵심 이견 포인트

두 증권사 모두 하반기 반등 가능성은 동의하나 **단기 리스크 평가**에서 갈린다.
하나증권은 HBM 수요 중심으로 낙관하는 반면, 키움증권은 레거시 DRAM 가격 하락을 더 무겁게 본다.
목표주가 차이(105,000원 vs 88,000원)가 이 시각 차이를 직접 반영한다."""),

    # ── few-shot 2: timeline ──────────────────────────────────────────────
    ("human", """\
[구조 가이드] 기간 내 의견 변화 흐름을 시간 순으로 정리하고 변화 원인을 분석
[질문] 이번 달 반도체 섹터 투자의견 변화 알려줘
[발췌]
### [미래에셋증권] (리포트 날짜: 2026-04-03)
삼성전자 투자의견 중립 → 매수로 상향. 목표주가 82,000원 → 95,000원. HBM4 양산 일정 확인이 상향 트리거.
### [KB증권] (리포트 날짜: 2026-04-18)
SK하이닉스 목표주가 210,000원 → 230,000원으로 상향. 1Q26 영업이익 7.4조로 컨센서스(6.1조) 대폭 상회 반영."""),
    ("ai", """\
## 2026년 4월 반도체 섹터 투자의견 변화

### 변화 타임라인

| 날짜 | 증권사 | 종목 | 변화 | 트리거 |
|------|--------|------|------|--------|
| 2026-04-03 | 미래에셋증권 | 삼성전자 | 중립 → **매수**, 목표주가 82,000 → **95,000원** | HBM4 양산 일정 확인 |
| 2026-04-18 | KB증권 | SK하이닉스 | 목표주가 210,000 → **230,000원** | 1Q26 어닝 서프라이즈 (+21% vs 컨센서스) |

### 상향 공통 배경

- 미래에셋증권: HBM4 양산 가시화 → 삼성전자 HBM 경쟁력 재평가
- KB증권: 1Q26 영업이익 7.4조 (예상 6.1조 대비 +21%) → 연간 추정치 상향

### 시사점

이달 변화는 일제히 상향 방향으로, 반도체 업황 회복을 시장이 공식 반영하기 시작한 시점으로 볼 수 있다.
삼성전자 목표주가(95,000원)가 SK하이닉스 대비 보수적인 수준을 유지하는 것은 HBM 점유율 격차 우려가 아직 완전히 해소되지 않았음을 시사한다."""),

    # ── few-shot 3: risk ──────────────────────────────────────────────────
    ("human", """\
[구조 가이드] 리스크를 단기/구조적으로 분류하고 발생 조건과 영향도를 서술
[질문] 조선업에서 언급된 리스크 요인 정리해줘
[발췌]
### [삼성증권] (리포트 날짜: 2026-02-20)
후판 가격 상승이 2026년 영업이익률을 1.5~2.0%p 압박할 전망. LNG선 발주 지연 우려.
### [대신증권] (리포트 날짜: 2026-02-28)
중국 조선소 수주 경쟁력 강화로 국내 수주 점유율 하락 가능성. 원/달러 환율 하락 시 수익성 악화."""),
    ("ai", """\
## 조선업 리스크 요인 정리

### 단기 리스크 (3~6개월)

**1. 원자재 비용 상승 (삼성증권, 2026-02-20)**
- 후판 가격 상승이 영업이익률을 **1.5~2.0%p** 직접 압박
- 고정 계약 비중이 높아 비용 전가 어려움

**2. LNG선 발주 지연 (삼성증권, 2026-02-20)**
- 글로벌 LNG 프로젝트 FID 지연으로 신규 수주 모멘텀 둔화 가능

### 구조적 리스크 (1년 이상)

**3. 중국 조선소 경쟁 심화 (대신증권, 2026-02-28)**
- 중국의 가격 경쟁력 강화 → 국내 수주 점유율 하락 우려
- 벌크선·컨테이너선 부문 마진 경쟁 불가피

**4. 환율 리스크 (대신증권, 2026-02-28)**
- 수주는 달러 기준, 비용은 원화 기준 → 원/달러 하락 시 수익성 직접 악화
- 환율 1,300원 하회 시 수익성 재평가 필요

### 종합 평가
단기는 후판 원가와 LNG 발주 지연이 실적 변수이며,
중장기로는 중국과의 수주 경쟁이 구조적 수익성에 더 큰 영향을 줄 것으로 판단된다."""),

    # ── few-shot 4: consensus ─────────────────────────────────────────────
    ("human", """\
[구조 가이드] 여러 증권사가 공통으로 강조하는 핵심 논거를 항목별로 정리하고 수치로 뒷받침
[질문] AI 인프라에 대해 증권사들이 공통으로 강조하는 게 뭐야
[발췌]
### [신한투자증권] (리포트 날짜: 2026-03-05)
빅테크 AI 설비투자(Capex) 합산 2026년 $3,200억 전망. 전력·냉각 인프라 부족이 데이터센터 확장 병목.
### [NH투자증권] (리포트 날짜: 2026-03-12)
글로벌 AI 서버 출하량 2026년 YoY +68% 전망. 전력 소비 급증으로 전력 인프라 수혜 부각.
### [한국투자증권] (리포트 날짜: 2026-03-20)
빅테크 Capex 가이던스 상향이 수주 증가로 연결. 전력망·냉각·네트워크 장비 동반 수혜."""),
    ("ai", """\
## AI 인프라 — 증권사 공통 강조 논거

### 공통 포인트

**① 빅테크 Capex 급증 (3/3 증권사 공통)**
- 신한투자증권: 2026년 빅테크 AI 설비투자 합산 **$3,200억** 전망
- NH투자증권: AI 서버 출하량 **YoY +68%** 예상
- 한국투자증권: Capex 가이던스 상향 → AI 인프라 전반 수주 증가로 연결

**② 전력·냉각 인프라 병목 → 관련 섹터 수혜 (3/3 증권사 공통)**
- 신한투자증권: 전력·냉각 부족이 데이터센터 확장의 실질 병목
- NH투자증권: 전력 소비 급증 → 전력 인프라 직접 수혜
- 한국투자증권: 전력망·냉각·네트워크 장비 동반 수혜 강조

### 컨센서스 요약

| 논거 | 언급 수 | 핵심 수치 |
|------|---------|-----------|
| 빅테크 Capex 급증 | 3/3 | $3,200억 (2026년) |
| AI 서버 출하 고성장 | 2/3 | YoY +68% |
| 전력·냉각 병목·수혜 | 3/3 | — |

### 컨센서스 이후 남은 불확실성
Capex 규모와 전력 수혜에는 컨센서스가 강하게 형성됐으나,
전력망 vs 냉각 장비 vs 네트워크 중 **어느 서브섹터의 수혜가 더 클지**는 증권사별 시각이 갈린다."""),

    # ── few-shot 5: valuation ─────────────────────────────────────────────
    ("human", """\
[구조 가이드] 증권사별 목표주가와 상향 근거(밸류에이션 방법론, 실적 추정치 변화)를 비교 정리
[질문] 각 증권사 목표주가 상향 근거가 뭐야
[발췌]
### [대신증권] (리포트 날짜: 2026-03-08)
SK하이닉스 목표주가 200,000원 → 220,000원. 2026년 EPS 추정치 18,500원으로 상향. 12개월 Fwd PER 11.9배 적용.
### [메리츠증권] (리포트 날짜: 2026-03-14)
SK하이닉스 목표주가 195,000원 → 225,000원. HBM 매출 비중 2026년 45% 반영, EV/EBITDA 6.5배로 산출."""),
    ("ai", """\
## 증권사별 목표주가 상향 근거 비교

### 목표주가 변화

| 증권사 | 기존 | 변경 | 상향폭 | 방법론 |
|--------|------|------|--------|--------|
| 대신증권 (2026-03-08) | 200,000원 | **220,000원** | +10% | PER |
| 메리츠증권 (2026-03-14) | 195,000원 | **225,000원** | +15.4% | EV/EBITDA |

### 상향 근거 비교

**대신증권 — EPS 기반 PER**
- 2026년 EPS 추정치 **18,500원**으로 상향
- 12개월 Fwd PER **11.9배** 적용 → 목표주가 220,000원
- 실적 추정치 개선이 직접 드라이버

**메리츠증권 — HBM 믹스 반영 EV/EBITDA**
- HBM 매출 비중 2026년 **45%** 반영 (프리미엄 믹스 개선)
- EV/EBITDA **6.5배** 적용으로 225,000원 산출
- 단순 이익 개선이 아닌 제품 믹스 프리미엄을 별도 부여

### 방법론 차이의 함의

메리츠의 상향폭(+15.4%)이 더 큰 이유는 HBM 비중 확대를 별도 프리미엄으로 계산했기 때문이다.
두 방법 모두 2026년 실적 개선을 가정하므로 **HBM 출하 지연 시 공통 하향 압력**이 발생할 수 있다."""),

    # ── 실제 질문 ─────────────────────────────────────────────────────────
    ("human", """\
[구조 가이드] {structure_hint}
[질문] {question}
[발췌]
{context}"""),
])


def _generate_answer(question: str, context: str, structure_hint: str) -> str:
    chain = _ANSWER_PROMPT | _llm_strong()
    return chain.invoke({
        "question":       question,
        "context":        context,
        "structure_hint": structure_hint,
    }).content


# ── 공통 저장 ─────────────────────────────────────────────────────────────────

def _save(question: str, content: str, sources: list[str], output_dir: str, prefix: str) -> None:
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    safe   = re.sub(r'[\\/:*?"<>|]', "_", question[:40])
    ts     = datetime.now().strftime("%Y%m%d_%H%M")
    path   = Path(output_dir) / f"{prefix}_{safe}_{ts}.md"
    header = f"# Q: {question}\n\n> 참고 증권사: {', '.join(sources)}\n\n---\n\n"
    path.write_text(header + content, encoding="utf-8")
    print(f"  저장 완료: {path}")


# ── 메인 진입점 ───────────────────────────────────────────────────────────────

def handle(
    retriever,
    question:    str,
    k_per_query: int  = 15,
    top_n:       int  = 12,
    output_dir:  str  = "./data/reports_output",
    save:        bool = True,
) -> dict:
    """
    질문 유형 자동 판별 후 적합한 체인으로 라우팅

    Q&A  → freeform 답변 (증권사 비교, 시점 분석, 리스크, 컨센서스, 밸류에이션)
    리포트 → report_chain.generate_report() 위임

    Returns:
        {
            "mode":          "qa" | "report",
            "question":      원본 질문,
            "question_type": 세부 유형 (qa일 때),
            "answer":        생성된 답변 (마크다운),
            "sources":       참고 증권사 목록,
            "chunk_count":   사용한 청크 수,
        }
    """
    print("\n" + "=" * 60)
    print(f"입력: {question}")
    print("=" * 60)

    # 라우팅
    print("\n[Router] 질문 유형 판별 중...")
    route = _route(question)
    mode  = route.get("mode", "qa")
    print(f"  -> mode: {mode}")

    # 리포트 생성 위임
    if mode == "report":
        topic = route.get("topic") or question
        print(f"  -> report_chain 위임: '{topic}'")
        from src.reportcreator.report_chain import generate_report
        answer = generate_report(retriever, topic, output_dir=output_dir)
        return {
            "mode":          "report",
            "question":      question,
            "question_type": "report",
            "answer":        answer,
            "sources":       [],
            "chunk_count":   0,
        }

    # Q&A 처리
    print("\n[Step 1] 의도 분석 중...")
    intent = _analyze_intent(question)
    print(f"  -> 유형: {intent['question_type']}")
    print(f"  -> 대상 증권사: {intent['target_brokers'] or '전체'}")
    print(f"  -> 검색 쿼리: {intent['search_queries']}")

    print("\n[Step 2] 청크 수집 중...")
    docs = _collect_chunks(
        retriever,
        queries        = intent["search_queries"],
        target_brokers = intent["target_brokers"],
        k_per_query    = k_per_query,
        top_n          = top_n,
    )
    print(f"  -> {len(docs)}개 청크 확보")

    if not docs:
        return {
            "mode":          "qa",
            "question":      question,
            "question_type": intent["question_type"],
            "answer":        "관련 리포트를 찾을 수 없습니다.",
            "sources":       [],
            "chunk_count":   0,
        }

    sources = sorted({d.metadata.get("source_firm", "알 수 없음") for d in docs})
    print(f"  -> 참고 증권사: {sources}")

    print("\n[Step 3] 컨텍스트 구성 중...")
    context = _build_context(docs, intent["target_brokers"])

    print("\n[Step 4] 답변 생성 중... (gpt-4o)")
    answer = _generate_answer(
        question       = question,
        context        = context,
        structure_hint = intent.get("structure_hint", "질문에 맞게 자유롭게 구성"),
    )
    print("  -> 완료")

    if save:
        _save(question, answer, sources, output_dir, prefix="freeform")

    print("=" * 60)

    return {
        "mode":          "qa",
        "question":      question,
        "question_type": intent["question_type"],
        "answer":        answer,
        "sources":       sources,
        "chunk_count":   len(docs),
    }
