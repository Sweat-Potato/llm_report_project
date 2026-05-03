"""
src/report_chain.py
멀티스텝 체인 리포트 생성 모듈

Chain:
  Step 1. 검색       → 관련 청크 수집 (k=20)
  Step 2. 요약       → 증권사별 핵심 내용 요약
  Step 3. 비교분석   → 컨센서스 & 차이점 추출
  Step 4. 인사이트   → 핵심 인사이트 도출
  Step 5. 리포트작성 → 최종 종합 리포트 생성
"""
import re
from datetime import datetime
from pathlib import Path

from langchain.schema import Document
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_community.vectorstores import Chroma
from langchain.retrievers import EnsembleRetriever

# retrieve_fn, rerank_fn 은 main.py에서 주입받음


# ── LLM 설정 ──────────────────────────────────────────────────────────────────

def get_llm_fast():
    return ChatOpenAI(model="gpt-4o-mini", temperature=0.1)

def get_llm_strong():
    return ChatOpenAI(model="gpt-4o", temperature=0.2)


# ── Step 1: 검색 (Hybrid + Reranker) ─────────────────────────────────────────

def step_retrieve(
    retriever:   EnsembleRetriever,
    topic:       str,
    retrieve_fn: callable,
    rerank_fn:   callable,
    k:           int = 20,
    top_n:       int = 10,
) -> list[Document]:
    """Step 1: Hybrid Search → Cross-Encoder Rerank"""
    print(f"\n[Step 1] '{topic}' 검색 중...")

    candidates = retrieve_fn(retriever, topic, k=k)
    print(f"  → Hybrid Search: {len(candidates)}개 후보")

    docs = rerank_fn(topic, candidates, top_n=top_n)
    print(f"  → Reranking 후: {len(docs)}개")

    broker_counts = {}
    for doc in docs:
        b = doc.metadata.get("broker", "unknown")
        broker_counts[b] = broker_counts.get(b, 0) + 1
    print(f"  → 증권사 분포: {dict(sorted(broker_counts.items(), key=lambda x: -x[1]))}")

    return docs


# ── Step 2: 증권사별 요약 ─────────────────────────────────────────────────────

SUMMARY_PROMPT = ChatPromptTemplate.from_template("""
당신은 15년 경력의 금융 리서치 전문 애널리스트입니다.

아래는 {broker}에서 발행한 리서치 리포트 원문입니다.
분석 주제: {topic}

[리포트 내용]
{content}

이 리포트에서 다음 항목을 분석하여 서술하세요. 리포트에 명시된 내용을 최대한 활용하고,
구체적인 수치·데이터·논거를 그대로 인용하세요. 추측이나 일반론은 쓰지 마세요.

**투자의견 및 목표주가**
리포트에서 명시한 투자의견(매수/중립/매도/비중확대 등)과 그 근거를 서술하세요.
목표주가나 밸류에이션 수치가 있으면 반드시 포함하세요.

**핵심 투자 논거**
이 증권사가 가장 강조하는 핵심 포인트를 구체적 수치·사례와 함께 서술하세요.
일반적인 산업 설명이 아니라, 이 리포트만의 고유한 분석 포인트에 집중하세요.

**성장 드라이버 및 촉매**
주가나 실적 상승을 이끌 구체적인 요인들을 서술하세요.

**리스크 요인**
리포트에서 언급한 리스크 요인들을 구체적으로 서술하세요.
언제, 어떤 조건에서 리스크가 현실화되는지 명시하세요.

**차별화된 시각**
다른 증권사와 다른 독자적인 분석 프레임이나 데이터 해석이 있다면 서술하세요.
""")

def step_summarize_by_broker(docs: list[Document], topic: str) -> dict[str, str]:
    print(f"\n[Step 2] 증권사별 요약 생성 중...")
    llm = get_llm_fast()

    broker_chunks: dict[str, list[str]] = {}
    broker_titles: dict[str, list[str]] = {}

    for doc in docs:
        broker = doc.metadata.get("broker", "unknown")
        title  = doc.metadata.get("title", "")
        broker_chunks.setdefault(broker, []).append(doc.page_content)
        if title and title not in broker_titles.get(broker, []):
            broker_titles.setdefault(broker, []).append(title)

    summaries = {}
    chain = SUMMARY_PROMPT | llm

    for broker, chunks in broker_chunks.items():
        print(f"  → {broker} 요약 중...")
        content = "\n\n---\n\n".join(chunks)[:6000]
        titles  = ", ".join(broker_titles.get(broker, [])[:3])
        response = chain.invoke({
            "broker":  broker,
            "topic":   topic,
            "content": f"[리포트 제목: {titles}]\n\n{content}",
        })
        summaries[broker] = response.content

    print(f"  → {len(summaries)}개 증권사 요약 완료")
    return summaries


# ── Step 3: 컨센서스 & 차이점 분석 ───────────────────────────────────────────

CONSENSUS_PROMPT = ChatPromptTemplate.from_template("""
당신은 시장 리서치 전문가입니다.

아래는 '{topic}'에 대한 여러 증권사 애널리스트들의 분석 요약입니다.
각 증권사의 주장을 면밀히 비교하여 아래 세 가지를 깊이 있게 서술하세요.

[증권사별 분석 요약]
{summaries}

---

**시장 컨센서스**
대다수 증권사가 동의하는 핵심 내용을 서술하세요.
단순 나열이 아니라, 왜 이런 컨센서스가 형성됐는지 구조적 배경까지 설명하세요.
구체적인 수치(목표주가 평균, 실적 예상치 등)가 있으면 종합해서 제시하세요.

**핵심 이견 및 논쟁점**
증권사 간 의견이 갈리는 지점을 구체적으로 분석하세요.
단순히 "A는 긍정, B는 부정"이 아니라, 왜 같은 데이터에서 다른 결론이 나오는지
분석 방법론·가정·중점 변수의 차이를 파고드세요.

**소수 의견 및 역발상 시각**
컨센서스와 결이 다른 시각, 혹은 시장이 충분히 주목하지 않는 분석을 소개하세요.
이 시각이 현실화될 조건과 그 파급효과도 서술하세요.
""")

DIFFERENCE_PROMPT = ChatPromptTemplate.from_template("""
당신은 시장 리서치 전문가입니다.

'{topic}'에 대한 증권사별 분석:
{summaries}

같은 현상이나 데이터를 서로 다르게 해석하는 부분을 심층 분석하세요.

**해석 차이의 구체적 비교**
각 쟁점마다 증권사들의 입장을 정확히 대조하세요.
어떤 데이터·지표를 근거로 삼는지, 그 해석이 왜 다른지를 중심으로 서술하세요.

**차이의 근본 원인**
분석 방법론, 매크로 환경 가정, 밸류에이션 프레임, 타임호라이즌 등
어디서 분기가 발생하는지 짚어주세요.

**투자자 관점에서의 시사점**
이 이견 구조 속에서 어떤 변수가 결정적인 판단 기준이 될지 서술하세요.
어떤 데이터 포인트가 나왔을 때 이견이 해소되거나 심화될지도 분석하세요.
""")

def step_analyze_consensus(summaries: dict[str, str], topic: str) -> tuple[str, str]:
    print(f"\n[Step 3] 컨센서스 & 차이점 분석 중...")
    llm = get_llm_fast()

    summaries_text = "\n\n".join([f"[{b}]\n{s}" for b, s in summaries.items()])

    consensus   = (CONSENSUS_PROMPT  | llm).invoke({"topic": topic, "summaries": summaries_text})
    differences = (DIFFERENCE_PROMPT | llm).invoke({"topic": topic, "summaries": summaries_text})

    print(f"  → 분석 완료")
    return consensus.content, differences.content


# ── Step 4: 인사이트 도출 ─────────────────────────────────────────────────────

INSIGHT_PROMPT = ChatPromptTemplate.from_template("""
당신은 20년 경력의 시니어 포트폴리오 매니저입니다.
수백억 규모의 포트폴리오를 운용하며 수많은 리서치를 검토해온 전문가로서 판단하세요.

주제: {topic}

[증권사별 분석 요약]
{summaries}

[컨센서스 분석]
{consensus}

[이견 분석]
{differences}

위 모든 자료를 바탕으로, 시장이 아직 충분히 인식하지 못한 인사이트를 도출하세요.
이미 알려진 사실의 반복이 아니라, 데이터 간 연결고리와 비선형적 함의에 집중하세요.

**Top 5 핵심 인사이트**
각각에 대해 근거를 충분히 서술하세요. "~할 것이다" 수준이 아니라
"왜 그렇게 판단하는가"의 논리 구조를 명확히 보여주세요.

**시장이 과소평가하는 요소**
현재 주가나 컨센서스에 충분히 반영되지 않은 변수를 짚어주세요.
그 변수가 언제, 어떤 경로로 표면화될지 시나리오를 그려주세요.

**단기 vs 중장기 전망의 괴리**
3~6개월 단기와 1~2년 중장기 사이에 전망이 엇갈리는 부분이 있다면
그 괴리의 원인과 투자 전략상 함의를 서술하세요.

**핵심 모니터링 지표 Top 5**
앞으로 가장 주목해야 할 데이터·이벤트·발표를 우선순위 순으로 서술하세요.
각각이 긍정/부정으로 나왔을 때의 시나리오도 간략히 제시하세요.

**포트폴리오 전략 시사점**
공격적 투자자와 보수적 투자자 각각의 관점에서 취할 전략을 구체적으로 서술하세요.
단순 매수/매도가 아니라 포지션 비중, 진입 조건, 익절/손절 기준까지 포함하세요.
""")

def step_extract_insights(
    summaries: dict[str, str],
    consensus: str,
    differences: str,
    topic: str,
) -> str:
    print(f"\n[Step 4] 핵심 인사이트 도출 중...")
    llm = get_llm_strong()

    summaries_text = "\n\n".join([f"[{b}]\n{s}" for b, s in summaries.items()])
    response = (INSIGHT_PROMPT | llm).invoke({
        "topic":       topic,
        "summaries":   summaries_text,
        "consensus":   consensus,
        "differences": differences,
    })
    print(f"  → 완료")
    return response.content


# ── Step 5: 최종 리포트 생성 ──────────────────────────────────────────────────

FINAL_REPORT_PROMPT = ChatPromptTemplate.from_template("""
당신은 최고 수준의 금융 리서치 전문가입니다.
국내외 주요 증권사 리포트를 종합 분석하여 기관 투자자 수준의 종합 리서치 리포트를 작성하세요.

**작성 원칙**
- 각 섹션을 충분한 깊이로 서술하세요. 내용이 많을수록 좋습니다.
- 리포트에서 언급된 구체적 수치(목표주가, 실적 예상치, 성장률, 밸류에이션 배수 등)를
  반드시 본문에 인용하세요. 숫자 없는 분석은 설득력이 없습니다.
- 각 증권사의 시각이 어떻게 다른지를 자연스럽게 본문에 녹여내세요.
- 일반론·상식·교과서 내용은 쓰지 마세요. 이 주제, 지금 이 시점에만 해당하는 내용을 쓰세요.
- 단정적 표현보다 조건부 분석("~할 경우 ~가 기대된다")이 전문성을 높입니다.

주제: {topic}
작성일: {date}
참고 증권사: {brokers}

[분석 자료]
=== 증권사별 핵심 분석 ===
{summaries}

=== 컨센서스 및 이견 ===
{consensus}

{differences}

=== 핵심 인사이트 ===
{insights}

---

위 분석을 바탕으로 아래 구조의 종합 리포트를 작성하세요.
각 섹션은 분량에 제한을 두지 말고, 담긴 내용을 충분히 풀어쓰세요.

---

# {topic} 종합 리서치 리포트
**작성일**: {date}
**참고 리포트**: {brokers}

---

## Executive Summary
이 리포트의 핵심 결론을 5~7문장으로 압축하세요.
투자판단에 가장 중요한 팩트 하나, 가장 큰 불확실성 하나, 그리고 컨센서스 대비 차별화된 시각 하나를 반드시 포함하세요.

---

## 1. 시장 현황 및 구조적 배경
### 1.1 현재 상황
지금 이 섹터/기업이 처한 상황을 구체적 수치와 함께 서술하세요.
최근 주가 흐름, 실적 추이, 업황 변화 등 현재 스냅샷을 그려주세요.

### 1.2 구조적 변화
단기 노이즈가 아니라 이 시점에 일어나고 있는 구조적 변화를 분석하세요.
공급망 재편, 기술 전환, 규제 변화, 수요 패턴 변화 등 장기적으로 유효한 변수에 집중하세요.

### 1.3 최근 주요 이슈 및 촉매
지난 1~3개월간 가장 중요한 이벤트와 그 영향을 서술하세요.
앞으로 예정된 주요 이벤트(실적발표, 정책 결정, 제품 출시 등)도 포함하세요.

---

## 2. 증권사별 분석 비교

### 2.1 투자의견 및 밸류에이션
각 증권사의 투자의견, 목표주가, 핵심 밸류에이션 근거를 비교하세요.
수치가 다른 경우 그 차이의 원인을 분석하세요.

### 2.2 낙관론 (Bull Case)
긍정적 시나리오를 지지하는 증권사들의 논거를 깊이 있게 서술하세요.
이 시나리오가 실현되려면 어떤 조건이 충족되어야 하는지도 명시하세요.

### 2.3 비관론 (Bear Case)
부정적 시나리오를 지지하는 논거를 깊이 있게 서술하세요.
현재 시장이 충분히 반영하고 있지 않은 리스크가 있다면 특히 강조하세요.

### 2.4 주목할 만한 독자적 시각
컨센서스와 다른 분석 프레임, 독특한 데이터 해석, 또는 시장의 사각지대를 짚는 시각이 있다면 소개하세요.

---

## 3. 핵심 투자 논거 심층 분석

### 3.1 성장 드라이버
이 섹터/기업의 성장을 이끄는 핵심 동력을 깊이 분석하세요.
단기 모멘텀과 구조적 성장 동력을 구분하여 서술하세요.

### 3.2 밸류에이션 분석
현재 밸류에이션 수준(PER, PBR, EV/EBITDA 등)이 역사적 맥락에서 어떤 의미인지 서술하세요.
상승 여력과 하방 리스크를 밸류에이션 관점에서 분석하세요.

### 3.3 경쟁 환경 및 포지셔닝
핵심 경쟁자 대비 이 기업/섹터의 차별화 포인트를 분석하세요.
시장 지위 변화 가능성, 진입장벽, 가격 결정력 등을 포함하세요.

### 3.4 실적 전망
주요 증권사들의 실적 추정치를 종합하고, 추정치 대비 상회/하회 가능성을 분석하세요.

---

## 4. 리스크 분석

### 4.1 단기 리스크 (3~6개월)
가장 즉각적으로 주가에 영향을 줄 수 있는 리스크를 서술하세요.
발생 확률과 영향 강도를 함께 평가하세요.

### 4.2 구조적·중장기 리스크 (1년 이상)
장기 투자 thesis를 훼손할 수 있는 구조적 위협을 분석하세요.
기술 전환 리스크, 경쟁 심화, 규제 리스크 등을 포함하세요.

### 4.3 시나리오 분석
- **Best Case**: 어떤 조건이 충족되면 이 시나리오가 실현되는가? 주가/실적 상단은?
- **Base Case**: 현재 컨센서스가 상정하는 기본 시나리오와 그 근거
- **Worst Case**: 어떤 충격이 오면 최악의 시나리오가 현실화되는가? 하단 추정치는?

---

## 5. 전망

### 5.1 단기 전망 (3~6개월)
가장 주목해야 할 근시일 내 이벤트와 그에 따른 주가 시나리오를 서술하세요.

### 5.2 중장기 전망 (1~2년)
이 시점의 구조적 변화가 1~2년 후 어떤 형태로 귀결될지 서술하세요.
현재 시장이 충분히 반영하지 못한 중장기 기회나 리스크를 강조하세요.

---

## 6. 핵심 모니터링 지표
앞으로 투자 판단에 결정적인 데이터·이벤트를 우선순위 순으로 서술하세요.
각 지표가 어떤 방향으로 나왔을 때 어떤 의미인지도 함께 설명하세요.

---

## 7. 투자 전략 제언
단계별 접근 방법을 구체적으로 서술하세요.
- 현재 포지션 취하기에 적절한 조건은 무엇인가?
- 추가 매수 또는 비중 축소를 검토할 트리거는?
- 공격적 투자자 vs 보수적 투자자의 접근 방법 차이

---

## ⚠️ 면책 조항
본 리포트는 여러 증권사의 리서치를 AI가 종합 분석한 자료입니다.
투자 결정의 최종 책임은 투자자 본인에게 있으며, 본 자료는 투자 권유가 아닙니다.
""")

def step_generate_final_report(
    topic:       str,
    summaries:   dict[str, str],
    consensus:   str,
    differences: str,
    insights:    str,
) -> str:
    print(f"\n[Step 5] 최종 리포트 생성 중... (gpt-4o)")
    llm = get_llm_strong()

    summaries_text = "\n\n".join([f"[{b}]\n{s}" for b, s in summaries.items()])
    brokers = ", ".join(summaries.keys())
    date    = datetime.now().strftime("%Y년 %m월 %d일")

    response = (FINAL_REPORT_PROMPT | llm).invoke({
        "topic":       topic,
        "date":        date,
        "brokers":     brokers,
        "summaries":   summaries_text,
        "consensus":   consensus,
        "differences": differences,
        "insights":    insights,
    })
    print(f"  → 완료")
    return response.content


# ── 전체 파이프라인 ────────────────────────────────────────────────────────────

def generate_report(
    retriever:   object,
    topic:       str,
    retrieve_fn: callable,
    rerank_fn:   callable,
    output_dir:  str = "./data/reports_output",
    k:           int = 20,
    top_n:       int = 10,
) -> str:
    """전체 리포트 생성 파이프라인"""
    print("\n" + "=" * 60)
    print(f"리포트 생성 시작: {topic}")
    print("=" * 60)

    docs = step_retrieve(retriever, topic, retrieve_fn=retrieve_fn, rerank_fn=rerank_fn, k=k, top_n=top_n)
    if not docs:
        return f"'{topic}'에 관련된 리포트를 찾을 수 없습니다."

    summaries               = step_summarize_by_broker(docs, topic)
    consensus, differences  = step_analyze_consensus(summaries, topic)
    insights                = step_extract_insights(summaries, consensus, differences, topic)
    final_report            = step_generate_final_report(topic, summaries, consensus, differences, insights)

    # 저장
    import json
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    safe_topic = re.sub(r'[\\/:*?"<>|]', "_", topic)
    base       = Path(output_dir) / f"report_{safe_topic}_{datetime.now().strftime('%Y%m%d_%H%M')}"

    base.with_suffix(".md").write_text(final_report, encoding="utf-8")

    sources_data = [
        {"content": d.page_content, **d.metadata} for d in docs
    ]
    base.with_name(base.name + "_sources.json").write_text(
        json.dumps(sources_data, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\n리포트 저장 완료: {base.with_suffix('.md')}")
    print("=" * 60)

    return final_report