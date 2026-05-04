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

    result = answer_question(retrievers, "하나증권과 키움증권의 3월 의견 차이를 설명해줘")
    print(result["answer"])
"""
from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path

from langchain.schema import Document, AIMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate

# ── 수정 1: router + reranker import ──────────────────────────────────────────
from src.retriever.router import retrieve as router_retrieve
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

    HumanMessage(content="하나증권과 키움증권의 3월 반도체 의견 차이를 설명해줘"),
    AIMessage(content=json.dumps({
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

    HumanMessage(content="이번 달 반도체 섹터 투자의견 변화 알려줘"),
    AIMessage(content=json.dumps({
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

    HumanMessage(content="각 증권사 목표주가 상향 근거가 뭐야"),
    AIMessage(content=json.dumps({
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

    HumanMessage(content="조선업에서 언급된 리스크 요인 정리해줘"),
    AIMessage(content=json.dumps({
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

    HumanMessage(content="AI 인프라에 대해 증권사들이 공통으로 강조하는 게 뭐야"),
    AIMessage(content=json.dumps({
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

# ── 수정 2: retriever → retrievers, retrieve_fn/rerank_fn 주입 ────────────────
def _collect_chunks(
    retrievers,
    queries:        list[str],
    target_brokers: list[str],
    retrieve_fn     = None,
    rerank_fn       = None,
    k_per_query:    int = 15,
    top_n:          int = 12,
) -> list[Document]:
    _retrieve = retrieve_fn if retrieve_fn else router_retrieve
    _rerank   = rerank_fn   if rerank_fn   else rerank

    all_candidates: list[Document] = []
    seen: set[tuple] = set()

    for q in queries:
        for doc in _retrieve(retrievers, q, k=k_per_query):
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

    return _rerank(queries[0], all_candidates, top_n=top_n)


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

## 증권사별 관점 차이

| 구분 | 하나증권 | 키움증권 |
|------|----------|----------|
| 투자의견 | **매수** | **중립** |
| 목표주가 | **105,000원** | **88,000원** |
| 영업이익 추정 | **35조원** | **28조원** |

## 주요 논거 포인트

**① HBM 공급 부족 (하나증권)** - HBM3e 공급 부족 지속
**② 레거시 DRAM 압박 (키움증권)** - 중국산 공급 과잉"""),

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

# ── 수정 3: retriever → retrievers, retrieve_fn/rerank_fn 추가 ────────────────
def answer_question(
    retrievers,
    question:    str,
    retrieve_fn  = None,
    rerank_fn    = None,
    k_per_query: int  = 15,
    top_n:       int  = 12,
    output_dir:  str  = "./data/reports_output",
    save:        bool = True,
) -> dict:
    """
    자유형 질문 → 유형 분류 → 검색 → 리포트 생성

    Args:
        retrievers:  ret_router.build_retriever()가 반환한 튜플
        question:    사용자 질문
        retrieve_fn: 커스텀 retrieve 함수 (None이면 router_retrieve 사용)
        rerank_fn:   커스텀 rerank 함수 (None이면 BGE rerank 사용)
        k_per_query: 쿼리당 검색 청크 수
        top_n:       reranker 최종 반환 수
        output_dir:  결과 저장 경로
        save:        파일 저장 여부
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
        retrievers,
        queries        = intent["search_queries"],
        target_brokers = intent["target_brokers"],
        retrieve_fn    = retrieve_fn,
        rerank_fn      = rerank_fn,
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
        base   = Path(output_dir) / f"freeform_{safe}_{ts}"

        header = f"# Q: {question}\n\n> 참고 증권사: {', '.join(sources)}\n\n---\n\n"
        (base.with_suffix(".md")).write_text(header + answer, encoding="utf-8")

        sources_data = [
            {"content": d.page_content, **d.metadata} for d in docs
        ]
        (base.with_name(base.name + "_sources.json")).write_text(
            json.dumps(sources_data, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"  저장 완료: {base.with_suffix('.md')}")

    print("=" * 60)

    return {
        "question":      question,
        "question_type": intent["question_type"],
        "answer":        answer,
        "sources":       sources,
        "chunk_count":   len(docs),
    }