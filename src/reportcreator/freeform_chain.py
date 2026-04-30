"""
src/reportcreator/freeform_chain.py
자유형 질문 → 리포트 생성 체인

report_chain.py 의 고정 7섹션 구조와 달리,
질문 유형에 맞게 구조를 설계하되 아래 3개 섹션은 모든 유형에서 반드시 포함:
  1. 시장 분석 (현황·배경)
  2. 증권사별 관점 차이
  3. 주요 논거 포인트

지원 질문 유형:
  - broker_comparison : "하나증권과 키움증권의 3월 의견 차이"
  - timeline          : "이번 달 반도체 섹터 투자의견 변화"
  - valuation         : "각 증권사 목표주가 상향 근거"
  - risk              : "조선업에서 언급된 리스크 요인 정리"
  - consensus         : "AI 인프라에 대해 증권사들이 공통으로 강조하는 것"

사용 예시:
    from src.reportcreator.freeform_chain import answer_question

    result = answer_question(retriever, "하나증권과 키움증권의 3월 의견 차이를 설명해줘")
    print(result["answer"])
"""
from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path

from langchain.schema import Document
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate

from src.retriever.retriever_01_ensemble import retrieve
from src.reranker.reranker_01_crossencoder import rerank


# ── LLM ──────────────────────────────────────────────────────────────────────

def _llm_fast()   -> ChatOpenAI: return ChatOpenAI(model="gpt-4o-mini", temperature=0)
def _llm_strong() -> ChatOpenAI: return ChatOpenAI(model="gpt-4o",      temperature=0.2)


# ── Step 1: 질문 유형 분류 + 검색 쿼리 생성 (few-shot) ────────────────────────

_INTENT_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """당신은 증권사 리서치 리포트 검색 전략가입니다.
사용자 질문을 분석하여 JSON으로 응답하세요.

{{
  "question_type": "broker_comparison | timeline | valuation | risk | consensus | other",
  "target_brokers": ["언급된 증권사 목록, 없으면 빈 배열"],
  "target_sector":  "언급된 섹터/종목, 없으면 null",
  "target_period":  "언급된 기간 (예: 2026-03), 없으면 null",
  "search_queries": ["쿼리1", "쿼리2", "쿼리3"],
  "structure_hint": "질문 특성에 맞는 답변 구성 방향 한 문장"
}}

JSON만 반환 (설명 없이)."""),

    ("human", "하나증권과 키움증권의 3월 반도체 의견 차이를 설명해줘"),
    ("ai", json.dumps({
        "question_type":  "broker_comparison",
        "target_brokers": ["하나증권", "키움증권"],
        "target_sector":  "반도체",
        "target_period":  "2026-03",
        "search_queries": [
            "하나증권 반도체 투자의견 3월",
            "키움증권 반도체 투자의견 3월",
            "반도체 섹터 목표주가 업황 전망",
        ],
        "structure_hint": "두 증권사의 투자의견·목표주가·핵심 논거를 대조하고, 이견의 근본 원인까지 분석",
    }, ensure_ascii=False)),

    ("human", "이번 달 반도체 섹터 투자의견 변화 알려줘"),
    ("ai", json.dumps({
        "question_type":  "timeline",
        "target_brokers": [],
        "target_sector":  "반도체",
        "target_period":  "2026-04",
        "search_queries": [
            "반도체 투자의견 변화 2026년 4월",
            "반도체 목표주가 상향 하향 최근",
            "반도체 업황 회복 전망",
        ],
        "structure_hint": "기간 내 의견 변화 흐름을 시간 순으로 정리하고, 변화를 유발한 시장 배경과 논거를 분석",
    }, ensure_ascii=False)),

    ("human", "각 증권사 목표주가 상향 근거가 뭐야"),
    ("ai", json.dumps({
        "question_type":  "valuation",
        "target_brokers": [],
        "target_sector":  None,
        "target_period":  None,
        "search_queries": [
            "목표주가 상향 밸류에이션 근거",
            "PER PBR EV/EBITDA 상향 조정",
            "실적 추정치 상향 목표주가 변화",
        ],
        "structure_hint": "증권사별 밸류에이션 방법론과 실적 추정치 변화를 비교하고, 방법론 차이가 목표주가 격차에 미치는 함의 분석",
    }, ensure_ascii=False)),

    ("human", "조선업에서 언급된 리스크 요인 정리해줘"),
    ("ai", json.dumps({
        "question_type":  "risk",
        "target_brokers": [],
        "target_sector":  "조선",
        "target_period":  None,
        "search_queries": [
            "조선업 리스크 요인 불확실성",
            "조선 원가 상승 수주 지연 리스크",
            "조선 섹터 하방 시나리오",
        ],
        "structure_hint": "리스크를 단기·구조적으로 분류하고, 각 리스크의 발생 조건·영향도·증권사별 온도 차를 서술",
    }, ensure_ascii=False)),

    ("human", "AI 인프라에 대해 증권사들이 공통으로 강조하는 게 뭐야"),
    ("ai", json.dumps({
        "question_type":  "consensus",
        "target_brokers": [],
        "target_sector":  "AI 인프라",
        "target_period":  None,
        "search_queries": [
            "AI 인프라 데이터센터 투자 전망",
            "AI 인프라 증권사 컨센서스 수혜",
            "AI 전력 냉각 네트워크 장비 성장",
        ],
        "structure_hint": "공통 논거를 항목별로 수치와 함께 정리하고, 컨센서스가 형성된 배경과 아직 이견이 있는 영역을 구분",
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


# ── Step 2: 다중 쿼리 검색 + 중복 제거 + Rerank ───────────────────────────────

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


# ── Step 3: 증권사별 컨텍스트 구성 ───────────────────────────────────────────

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


# ── Step 4: 답변 생성 — 5개 유형 few-shot ────────────────────────────────────
# 모든 예시에 공통 3개 섹션 포함:
#   ## 시장 분석
#   ## 증권사별 관점 차이
#   ## 주요 논거 포인트

_ANSWER_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """당신은 15년 경력의 금융 리서치 전문 애널리스트입니다.
제공된 증권사 리포트 발췌문을 바탕으로 질문에 답변하는 리포트를 작성하세요.

**모든 답변에 반드시 포함할 3개 섹션**
1. ## 시장 분석 — 현재 시장 상황과 구조적 배경을 수치와 함께 서술
2. ## 증권사별 관점 차이 — 각 증권사가 같은 현상을 어떻게 다르게 해석하는지 대조
3. ## 주요 논거 포인트 — 발췌문에서 가장 핵심적인 투자 근거를 항목별로 정리

**추가 섹션**은 질문 유형에 맞게 자유롭게 설계하세요.

**절대 원칙**
- 발췌문의 구체적 수치(목표주가, 성장률, EPS 등) 반드시 인용
- 일반론·상식 금지. 이 발췌문에만 있는 내용에 집중
- 발췌문에 없는 정보는 "리포트에서 확인되지 않음"으로 명시
- 참고한 증권사명과 날짜를 본문에 표기
- 마크다운으로 작성"""),

    # ── few-shot 1: broker_comparison ────────────────────────────────────
    ("human", """\
[구조 가이드] 두 증권사 투자의견·목표주가·핵심 논거를 대조하고, 이견의 근본 원인까지 분석
[질문] 하나증권과 키움증권의 3월 반도체 의견 차이를 설명해줘
[발췌]
### [하나증권] (리포트 날짜: 2026-03-10)
HBM3e 공급 부족이 2026년 하반기까지 지속. 삼성전자 목표주가 105,000원 유지, 투자의견 매수.
AI 서버 수요 폭발로 D램 업황 상저하고 전망. 2026년 영업이익 추정치 35조원.
### [키움증권] (리포트 날짜: 2026-03-15)
중국 레거시 DRAM 공급 과잉이 단기 가격 압박 요인. 삼성전자 목표주가 88,000원으로 하향.
상반기 실적 부진 후 하반기 반등 가능성 유지. 투자의견 중립. 2026년 영업이익 추정치 28조원."""),
    ("ai", """\
## 시장 분석

2026년 3월 반도체 시장은 AI 수요 호조와 레거시 DRAM 공급 과잉이 동시에 작용하는 이중 구도다.
하이퍼스케일러의 HBM·AI 서버 투자는 확대되는 반면, 중국발 레거시 DRAM 공급이 단기 가격을 압박하고 있어
증권사마다 어느 쪽에 더 무게를 두느냐에 따라 투자의견이 갈린다.

## 증권사별 관점 차이

| 구분 | 하나증권 (2026-03-10) | 키움증권 (2026-03-15) |
|------|----------------------|----------------------|
| 투자의견 | **매수** | **중립** |
| 삼성전자 목표주가 | **105,000원** | **88,000원** (하향) |
| 2026년 영업이익 추정 | **35조원** | **28조원** |
| 핵심 시각 | HBM 공급 부족 → 상저하고 | 레거시 공급 과잉 → 상반기 부진 |

**하나증권 — 강세론**
HBM3e 공급 부족이 하반기까지 지속된다고 보고, AI 서버 수요 폭발을 D램 업황 회복의 핵심 촉매로 평가한다.
영업이익 추정치 35조원은 HBM 믹스 개선에 따른 ASP 상승을 전제로 한다.

**키움증권 — 중립론**
중국 레거시 DRAM의 공급 과잉이 단기 가격을 끌어내리는 구조적 압박 요인이라고 본다.
상반기 실적 부진 가능성을 더 크게 가중하여 영업이익 추정치를 28조원으로 보수적으로 설정했다.

## 주요 논거 포인트

**① HBM 공급·수요 불균형 (하나증권)**
- HBM3e 공급 부족 → 2026년 하반기까지 공급자 우위 유지 전망
- AI 서버 Capex 확대가 HBM 수요를 직접 견인

**② 레거시 DRAM 가격 압박 (키움증권)**
- 중국산 레거시 DRAM 공급 과잉 → 단기 D램 가격 하락 압력
- 삼성전자 제품 믹스에서 레거시 비중이 여전히 상당 → 수익성 희석

**③ 이견의 핵심 — 단기 리스크 가중치**
두 증권사 모두 하반기 반등 가능성은 인정하나, 단기 하방 리스크를 얼마나 무겁게 보느냐가 목표주가 격차(105,000원 vs 88,000원)를 만든다."""),

    # ── few-shot 2: timeline ──────────────────────────────────────────────
    ("human", """\
[구조 가이드] 기간 내 의견 변화를 시간 순으로 정리하고, 변화를 유발한 시장 배경과 논거를 분석
[질문] 이번 달 반도체 섹터 투자의견 변화 알려줘
[발췌]
### [미래에셋증권] (리포트 날짜: 2026-04-03)
삼성전자 투자의견 중립 → 매수로 상향. 목표주가 82,000원 → 95,000원. HBM4 양산 일정 확인이 트리거.
### [KB증권] (리포트 날짜: 2026-04-18)
SK하이닉스 목표주가 210,000원 → 230,000원. 1Q26 영업이익 7.4조, 컨센서스 6.1조 대폭 상회."""),
    ("ai", """\
## 시장 분석

2026년 4월 반도체 시장은 HBM4 양산 가시화와 1분기 어닝 서프라이즈가 겹치며 업황 회복을 공식화하는 시점에 진입했다.
AI 서버 수요 확대가 예상보다 빠르게 실적으로 이어지면서 주요 증권사가 잇따라 의견을 상향 조정했다.

## 증권사별 관점 차이

| 날짜 | 증권사 | 종목 | 변화 | 트리거 |
|------|--------|------|------|--------|
| 2026-04-03 | 미래에셋증권 | 삼성전자 | 중립 → **매수**, 82,000 → **95,000원** | HBM4 양산 일정 확인 |
| 2026-04-18 | KB증권 | SK하이닉스 | 목표주가 210,000 → **230,000원** | 1Q26 영업이익 +21% 어닝 서프라이즈 |

**미래에셋증권**: 기술 이벤트(HBM4 양산 확인)를 트리거로 삼아 선제적 상향. 실적보다 기술 모멘텀에 무게.

**KB증권**: 1Q26 영업이익 7.4조(컨센서스 6.1조 대비 +21%)라는 수치 확인 후 상향. 실적 기반의 보수적 접근.

## 주요 논거 포인트

**① HBM4 양산 가시화 (미래에셋증권, 2026-04-03)**
- 삼성전자의 HBM4 양산 일정 확인 → HBM 경쟁력 재평가
- 기존 중립 의견의 핵심 우려였던 HBM 기술 격차가 해소 국면 진입

**② 어닝 서프라이즈 (KB증권, 2026-04-18)**
- SK하이닉스 1Q26 영업이익 **7.4조원** (예상 6.1조원 대비 +21%)
- AI 서버용 HBM 출하 증가가 예상보다 빠르게 실적에 반영

**③ 이달 변화의 함의**
두 건 모두 상향 방향이며, 기술·실적 양면에서 동시에 업사이드 요인이 확인됐다.
삼성전자 목표주가(95,000원)가 SK하이닉스 대비 여전히 보수적인 것은 HBM 점유율 격차 우려가 완전히 해소되지 않았음을 반영한다."""),

    # ── few-shot 3: valuation ─────────────────────────────────────────────
    ("human", """\
[구조 가이드] 증권사별 밸류에이션 방법론과 실적 추정치 변화를 비교하고, 방법론 차이의 함의 분석
[질문] 각 증권사 목표주가 상향 근거가 뭐야
[발췌]
### [대신증권] (리포트 날짜: 2026-03-08)
SK하이닉스 목표주가 200,000원 → 220,000원. 2026년 EPS 추정치 18,500원으로 상향. 12개월 Fwd PER 11.9배 적용.
### [메리츠증권] (리포트 날짜: 2026-03-14)
SK하이닉스 목표주가 195,000원 → 225,000원. HBM 매출 비중 2026년 45% 반영, EV/EBITDA 6.5배로 산출."""),
    ("ai", """\
## 시장 분석

2026년 3월 SK하이닉스 목표주가 상향은 HBM 매출 믹스 개선과 AI 수요 확대를 배경으로 한다.
증권사마다 밸류에이션 방법론이 달라 같은 실적 개선을 반영하더라도 목표주가 도달 경로가 다르게 설계된다.

## 증권사별 관점 차이

| 증권사 | 기존 | 변경 | 상향폭 | 방법론 | 핵심 변수 |
|--------|------|------|--------|--------|-----------|
| 대신증권 (2026-03-08) | 200,000원 | **220,000원** | +10% | PER | EPS 18,500원 |
| 메리츠증권 (2026-03-14) | 195,000원 | **225,000원** | +15.4% | EV/EBITDA | HBM 비중 45% |

**대신증권 — EPS 기반 PER 접근**
2026년 EPS 추정치를 18,500원으로 상향하고, 12개월 Fwd PER 11.9배를 적용해 220,000원을 도출했다.
이익 성장률 자체를 밸류에이션의 핵심 드라이버로 삼는 전통적 방법론이다.

**메리츠증권 — HBM 믹스 프리미엄 EV/EBITDA 접근**
HBM 매출 비중 45%를 별도로 반영하여 제품 믹스 개선에 따른 프리미엄을 EV/EBITDA 6.5배로 산출했다.
단순 이익 성장이 아닌, HBM 고마진 구조가 기업 가치에 직접 기여함을 강조하는 방법론이다.

## 주요 논거 포인트

**① 실적 추정치 상향 (대신증권)**
- 2026년 EPS **18,500원** — AI 서버 수요 반영한 D램 ASP 개선이 근거
- PER 11.9배는 과거 반도체 사이클 피크 밸류에이션 대비 할인된 수준 적용

**② HBM 믹스 프리미엄 (메리츠증권)**
- HBM 매출 비중 **45%** 반영 → 고마진 제품 비중 확대로 EBITDA 질 개선
- EV/EBITDA 6.5배는 메모리 피어 대비 프리미엄 부여 근거

**③ 방법론 차이의 함의**
메리츠의 상향폭(+15.4%)이 더 큰 이유는 HBM 비중 확대를 별도 프리미엄으로 계산했기 때문이다.
두 방법 모두 HBM 출하 지연 시 하향 조정 리스크를 공통으로 내포한다."""),

    # ── few-shot 4: risk ──────────────────────────────────────────────────
    ("human", """\
[구조 가이드] 리스크를 단기·구조적으로 분류하고, 발생 조건·영향도·증권사별 온도 차를 서술
[질문] 조선업에서 언급된 리스크 요인 정리해줘
[발췌]
### [삼성증권] (리포트 날짜: 2026-02-20)
후판 가격 상승이 2026년 영업이익률을 1.5~2.0%p 압박할 전망. LNG선 발주 지연 우려.
### [대신증권] (리포트 날짜: 2026-02-28)
중국 조선소 수주 경쟁력 강화로 국내 수주 점유율 하락 가능성. 원/달러 환율 하락 시 수익성 악화."""),
    ("ai", """\
## 시장 분석

2026년 조선업은 LNG선·컨테이너선 수주 호황이 이어지는 가운데, 원가 상승과 중국 경쟁 심화라는 두 가지 역풍이 수익성 개선 속도를 제약하고 있다.
삼성증권은 단기 원가 이슈에, 대신증권은 중장기 구조적 경쟁 리스크에 각각 방점을 찍는다.

## 증권사별 관점 차이

| 구분 | 삼성증권 (2026-02-20) | 대신증권 (2026-02-28) |
|------|----------------------|----------------------|
| 핵심 리스크 | 후판 가격 상승, LNG 발주 지연 | 중국 경쟁 심화, 환율 하락 |
| 영향 시계 | 단기 (2026년 상반기) | 중장기 (1년 이상) |
| 이익률 영향 | 영업이익률 **-1.5~2.0%p** | 수주 점유율·환율 복합 압박 |

## 주요 논거 포인트

**단기 리스크 (3~6개월)**

**① 후판 가격 상승 (삼성증권, 2026-02-20)**
- 후판 가격 상승 → 영업이익률 **1.5~2.0%p** 직접 압박
- 고정 계약 비중이 높아 비용 전가 제한적

**② LNG선 발주 지연 (삼성증권, 2026-02-20)**
- 글로벌 LNG 프로젝트 FID 지연 → 신규 수주 모멘텀 일시 둔화 가능

**구조적 리스크 (1년 이상)**

**③ 중국 조선소 경쟁 심화 (대신증권, 2026-02-28)**
- 중국의 가격 경쟁력 강화 → 국내 수주 점유율 하락 압력
- 벌크선·컨테이너선 부문 마진 경쟁 불가피

**④ 환율 리스크 (대신증권, 2026-02-28)**
- 수주는 달러 기준, 비용은 원화 기준 → 원/달러 하락 시 수익성 직접 악화
- 1,300원 하회 시 수익성 재평가 필요"""),

    # ── few-shot 5: consensus ─────────────────────────────────────────────
    ("human", """\
[구조 가이드] 공통 논거를 항목별로 수치와 함께 정리하고, 컨센서스 배경과 이견 영역을 구분
[질문] AI 인프라에 대해 증권사들이 공통으로 강조하는 게 뭐야
[발췌]
### [신한투자증권] (리포트 날짜: 2026-03-05)
빅테크 AI 설비투자(Capex) 합산 2026년 $3,200억 전망. 전력·냉각 인프라 부족이 데이터센터 확장 병목.
### [NH투자증권] (리포트 날짜: 2026-03-12)
글로벌 AI 서버 출하량 2026년 YoY +68% 전망. 전력 소비 급증으로 전력 인프라 수혜 부각.
### [한국투자증권] (리포트 날짜: 2026-03-20)
빅테크 Capex 가이던스 상향이 수주 증가로 연결. 전력망·냉각·네트워크 장비 동반 수혜."""),
    ("ai", """\
## 시장 분석

2026년 AI 인프라 시장은 빅테크의 대규모 Capex 집행이 현실화되면서 데이터센터 건설·전력·냉각·네트워크 전 밸류체인에 걸쳐 수혜가 확산되고 있다.
신한투자증권(2026-03-05) 기준 빅테크 AI 설비투자 합산이 2026년 $3,200억에 달할 전망이며,
NH투자증권(2026-03-12)은 AI 서버 출하량이 YoY +68% 증가할 것으로 추정한다.

## 증권사별 관점 차이

세 증권사 모두 AI 인프라 강세론에 동의하지만, 수혜 서브섹터 우선순위에서 온도 차가 있다.

| 구분 | 신한투자증권 | NH투자증권 | 한국투자증권 |
|------|------------|-----------|------------|
| 핵심 병목 | **전력·냉각 인프라** | **전력 소비 급증** | **전력망·냉각·네트워크 복합** |
| Capex 언급 | $3,200억 (2026년) | — | 가이던스 상향 |
| 서버 출하 | — | YoY +68% | — |

## 주요 논거 포인트

**① 빅테크 Capex 급증 (3/3 공통)**
- 신한투자증권: 2026년 빅테크 AI 설비투자 합산 **$3,200억** 전망
- NH투자증권: AI 서버 출하량 **YoY +68%** 예상
- 한국투자증권: Capex 가이던스 상향이 수주 증가로 직결

**② 전력·냉각 인프라 병목 → 관련 섹터 수혜 (3/3 공통)**
- 신한투자증권: 전력·냉각 부족이 데이터센터 확장의 실질 병목
- NH투자증권: 전력 소비 급증 → 전력 인프라 섹터 직접 수혜
- 한국투자증권: 전력망·냉각·네트워크 장비 동반 수혜 강조

**③ 컨센서스가 아직 형성되지 않은 영역**
전력망·냉각·네트워크 중 **어느 서브섹터의 수혜 강도가 더 클지**는 증권사별 시각이 다르다.
이 이견이 해소되는 방향에 따라 섹터 내 종목 선택이 갈릴 것으로 판단된다."""),

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


# ── 메인 진입점 ───────────────────────────────────────────────────────────────

def answer_question(
    retriever,
    question:    str,
    k_per_query: int  = 15,
    top_n:       int  = 12,
    output_dir:  str  = "./data/reports_output",
    save:        bool = True,
) -> dict:
    """
    자유형 질문 → 유형 분류 → 검색 → 리포트 생성

    모든 유형의 출력에 시장 분석 / 증권사별 관점 차이 / 주요 논거 포인트 포함.

    Returns:
        {
            "question":      원본 질문,
            "question_type": 분류된 유형,
            "answer":        생성된 리포트 (마크다운),
            "sources":       참고 증권사 목록,
            "chunk_count":   사용한 청크 수,
        }
    """
    print("\n" + "=" * 60)
    print(f"입력: {question}")
    print("=" * 60)

    print("\n[Step 1] 질문 유형 분류 중...")
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

    print("\n[Step 4] 리포트 생성 중... (gpt-4o)")
    answer = _generate_answer(
        question       = question,
        context        = context,
        structure_hint = intent.get("structure_hint", "질문에 맞게 자유롭게 구성"),
    )
    print("  -> 완료")

    if save:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        safe   = re.sub(r'[\\/:*?"<>|]', "_", question[:40])
        ts     = datetime.now().strftime("%Y%m%d_%H%M")
        path   = Path(output_dir) / f"freeform_{safe}_{ts}.md"
        header = f"# Q: {question}\n\n> 참고 증권사: {', '.join(sources)}\n\n---\n\n"
        path.write_text(header + answer, encoding="utf-8")
        print(f"  저장 완료: {path}")

    print("=" * 60)

    return {
        "question":      question,
        "question_type": intent["question_type"],
        "answer":        answer,
        "sources":       sources,
        "chunk_count":   len(docs),
    }
