"""
eval/04_eval_report.py
생성된 리포트 품질 평가 (규칙 기반 + LLM-as-judge + SemScore)

평가 기준 6가지:
  1. 구조 완성도   (규칙 기반): 필수 섹션이 모두 존재하는가?
  2. 전문 용어     (규칙 기반): 금융 전문 용어가 충분히 사용됐는가?
  3. 출처 충실도   (LLM 판정): 원본 리포트에 근거한 내용인가? (Hallucination 방지)
  4. 투자 실용성   (LLM 판정): 구체적 수치·조건이 있어 투자 판단에 도움이 되는가?
  5. 관점 균형성   (LLM 판정): 강세(Bull)·약세(Bear) 양쪽을 공정하게 다루는가?
  6. 의미 유사도   (SemScore): 생성 리포트가 원본 청크 내용을 의미적으로 얼마나 반영했는가?
                               임베딩 코사인 유사도 기반, 한국어에 최적, LLM 비용 없음

점수: 0.0 ~ 1.0  (LLM 점수는 0~10점 → /10 정규화)

★ 설정 블록 — 이곳만 수정 ★
  REPORT_DIR  : 평가할 리포트 파일이 있는 디렉터리
  MAX_REPORTS : 평가할 최대 리포트 수 (최신 파일 우선)

실행:
  uv run python eval/04_eval_report.py
"""

import json
import re
import sys
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(BASE_DIR))

from dotenv import load_dotenv
load_dotenv()

# ══════════════════════════════════════════════════════════════════════════════
# ★ 설정 블록
# ══════════════════════════════════════════════════════════════════════════════

REPORT_DIR  = BASE_DIR / "data" / "reports_output"
MAX_REPORTS = 1   # 최신 리포트 최대 N개 평가

# ══════════════════════════════════════════════════════════════════════════════

import numpy as np
import pandas as pd
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate

from eval.base import REPORT_EVAL_PATH, save_results, print_score_summary

# SemScore 용 임베딩 모델 (싱글톤)
_embeddings: OpenAIEmbeddings | None = None

def _get_embeddings() -> OpenAIEmbeddings:
    global _embeddings
    if _embeddings is None:
        _embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    return _embeddings


# ── 필수 섹션 목록 (freeform_chain.py 의 _ANSWER_PROMPT 공통 4섹션과 일치) ────
REQUIRED_SECTIONS = [
    "핵심 요약",
    "시장 분석",
    "증권사별 관점 차이",
    "주요 논거 포인트",
]

# ── 금융 전문 용어 (freeform 리포트에서 실제로 등장하는 전문 용어 기준) ──────────
# 너무 흔한 일반어("수요","공급","전망")는 제외 — 변별력 있는 전문 용어만 선정
FINANCE_TERMS = [
    # 투자의견 관련
    "투자의견", "매수", "비중확대", "목표주가", "컨센서스",
    # 실적·재무 용어
    "영업이익", "매출", "순이익", "밸류에이션", "실적",
    # 분석 전문 용어
    "업황", "밸류체인", "모멘텀", "이견", "논거",
]

# ── 규칙 기반: 답변 거부 표현 (langchain_v0_basics 04 노트북 패턴 참고) ─────────
# 리포트가 실질적 내용 없이 거부·회피하는 경우를 감지
_NEGATIVE_PHRASES = [
    "정보가 없", "확인할 수 없", "알 수 없",
    "답변할 수 없", "컨텍스트에 없", "제공되지 않",
]


# ── 규칙 기반 평가 ────────────────────────────────────────────────────────────

def _eval_structure(report_text: str) -> float:
    """
    필수 섹션 존재 여부.
    모든 섹션 있으면 1.0, 하나도 없으면 0.0.
    """
    found = sum(1 for section in REQUIRED_SECTIONS if section in report_text)
    return round(found / len(REQUIRED_SECTIONS), 3)


def _eval_finance_terms(report_text: str) -> float:
    """
    금융 전문 용어 밀도.
    전체 용어의 50% 이상 등장하면 1.0 으로 clamp.
    거부 표현이 있으면 0.5 패널티 적용 (langchain_v0_basics 04 노트북 패턴).
    """
    found = sum(1 for term in FINANCE_TERMS if term in report_text)
    ratio = found / len(FINANCE_TERMS)
    score = min(ratio * 2.0, 1.0)

    # 거부·회피 표현이 있으면 패널티
    has_negative = any(phrase in report_text for phrase in _NEGATIVE_PHRASES)
    if has_negative:
        score = round(score * 0.5, 3)

    return round(score, 3)


# ── LLM-as-Judge 평가 ────────────────────────────────────────────────────────
# 루브릭(rubric) 기반 채점: 각 점수 구간에 구체적 조건을 명시해
# LLM의 관대한 채점 경향(Leniency Bias)을 억제

_JUDGE_PROMPT = ChatPromptTemplate.from_template("""
당신은 금융 리서치 리포트 품질 평가 전문가입니다.

[채점 루브릭]
{criterion}

[평가할 리포트 내용]
{report_content}

위 루브릭의 조건을 하나씩 대조하여 해당하는 점수를 결정하세요.
점수는 0~100 사이 정수로 매기세요 (루브릭 구간 내에서 세부 조건 충족도에 따라 조정).
**정수 숫자 하나만** 출력하세요. 설명·이유·기호는 일절 쓰지 마세요.
""")

_FAITHFULNESS_PROMPT = ChatPromptTemplate.from_template("""
당신은 금융 리서치 리포트 품질 평가 전문가입니다.

[원본 증권사 청크]
{source_chunks}

[생성된 리포트]
{report_content}

아래 루브릭 조건을 원본 청크와 리포트를 대조하여 채점하세요.

[채점 루브릭]
0~20  : 원본에 없는 수치·주장이 다수 포함됨 (심각한 hallucination)
21~40 : 방향성은 맞으나 원본 수치·출처 대부분 누락됨
41~60 : 원본의 일부 수치·증권사 의견을 인용했으나 절반 이상 빠짐
61~80 : 원본 핵심 수치·증권사 의견을 대부분 반영하고 출처 명시됨
81~100: 원본 청크의 수치·날짜·증권사명을 빠짐없이 정확히 인용함

점수는 0~100 사이 정수로 매기세요 (구간 내에서 세부 충족도에 따라 조정).
**정수 숫자 하나만** 출력하세요. 설명·이유·기호는 일절 쓰지 마세요.
""")


def _llm_judge(llm, criterion: str, report_text: str) -> float:
    """
    LLM-as-judge: 0~100 점수를 받아 0~1 로 정규화.
    101단계 세분화로 0.73, 0.856 같은 세밀한 점수 가능.
    파싱 실패 시 0.0 반환 (평가 자체가 실패해도 전체가 멈추지 않도록).
    """
    try:
        chain    = _JUDGE_PROMPT | llm
        response = chain.invoke({
            "criterion"     : criterion,
            "report_content": report_text[:4000],
        })
        match = re.search(r'\b(\d{1,3})\b', response.content.strip())
        if not match:
            return 0.0
        score = float(match.group())
        return round(min(max(score / 100.0, 0.0), 1.0), 3)
    except Exception as e:
        print(f"      [경고] LLM 판정 실패: {e}")
        return 0.0


def _eval_faithfulness(llm, report_text: str, source_chunks: list[dict] | None) -> float:
    """출처 충실도: 리포트가 원본 청크에 근거하는가?"""
    if not source_chunks:
        # 소스 파일 없는 구버전 리포트 — 원본 없이 판단 (부정확)
        print("      [주의] _sources.json 없음 — 원본 없이 판단합니다")
        return _llm_judge(
            llm,
            "리포트가 실제 증권사 리서치 데이터에 충실하게 작성됐는가를 평가하세요. "
            "근거 없는 추측·일반론·교과서 내용이 많으면 낮게, "
            "구체적 수치와 증권사 견해를 충실히 반영했으면 높게 평가하세요.",
            report_text,
        )

    try:
        # 토큰 절약: 청크당 300자, 최대 10개
        chunk_texts = "\n\n---\n\n".join(
            c.get("content", "")[:300] for c in source_chunks[:10]
        )
        chain    = _FAITHFULNESS_PROMPT | llm
        response = chain.invoke({
            "source_chunks" : chunk_texts,
            "report_content": report_text[:3000],
        })
        match = re.search(r'\b(\d{1,3})\b', response.content.strip())
        if not match:
            return 0.0
        score = float(match.group())
        return round(min(max(score / 100.0, 0.0), 1.0), 3)
    except Exception as e:
        print(f"      [경고] faithfulness 판정 실패: {e}")
        return 0.0


def _eval_investment_utility(llm, report_text: str) -> float:
    """
    질문 맥락 적합성: 질문이 요구한 내용에 실질적으로 답하고 있는가?
    루브릭 기반으로 Leniency Bias 억제.
    """
    return _llm_judge(
        llm,
        "이 리포트가 제목(질문)에 얼마나 실질적으로 답하는지 아래 루브릭으로 채점하세요.\n\n"
        "0~20 : 질문과 무관하거나 '정보가 없다'는 식으로 회피함\n"
        "21~40: 질문 방향은 맞으나 증권사 견해 없이 일반론만 나열함\n"
        "41~60: 일부 증권사 견해를 인용했으나 수치·날짜 없이 추상적 서술에 그침\n"
        "61~80: 복수 증권사의 구체적 수치·날짜를 인용하며 질문에 실질적으로 답함\n"
        "81~100: 여러 증권사 논거를 수치·날짜와 함께 체계적으로 정리하고\n"
        "       질문의 핵심 포인트를 빠짐없이 다룸\n\n"
        "점수는 0~100 사이 정수로 매기세요 (구간 내에서 세부 충족도에 따라 조정).",
        report_text,
    )


def _eval_balance(llm, report_text: str) -> float:
    """
    다관점 균형성: 여러 증권사 시각을 균형 있게 다루고 컨센서스·이견을 명확히 구분했는가?
    루브릭 기반으로 Leniency Bias 억제.
    """
    return _llm_judge(
        llm,
        "이 리포트가 여러 증권사 관점을 균형 있게 다루는지 아래 루브릭으로 채점하세요.\n\n"
        "0~20 : 1개 증권사 의견만 있거나, 출처 표기가 전혀 없음\n"
        "21~40: 복수 증권사를 언급하지만 특정 증권사 의견에 치우침\n"
        "41~60: 여러 증권사를 인용했으나 컨센서스와 이견이 구분되지 않음\n"
        "61~80: 컨센서스와 이견이 명확히 구분되고 증권사명·날짜가 대부분 명시됨\n"
        "81~100: 3개 이상 증권사의 공통 논거와 차이점이 체계적으로 대조되고\n"
        "       모든 주장에 증권사명·날짜 출처가 정확히 표기됨\n\n"
        "점수는 0~100 사이 정수로 매기세요 (구간 내에서 세부 충족도에 따라 조정).",
        report_text,
    )


def _eval_semscore(report_text: str, source_chunks: list[dict] | None) -> float:
    """
    SemScore: 생성 리포트와 원본 청크 간 임베딩 코사인 유사도.

    - 원본 청크가 없으면 0.0 반환
    - 청크별 유사도를 계산해 평균값 사용
    - LLM 호출 없음, 한국어에 최적
    """
    if not source_chunks:
        print("      [주의] _sources.json 없음 — SemScore 건너뜁니다")
        return 0.0

    try:
        embeddings = _get_embeddings()

        # 리포트 임베딩 (앞 2000자만 사용해 토큰 절약)
        report_emb = np.array(embeddings.embed_query(report_text[:2000]))

        # 청크별 유사도 계산 (최대 10개)
        scores = []
        for chunk in source_chunks[:10]:
            content = chunk.get("content", "").strip()
            if not content:
                continue
            chunk_emb = np.array(embeddings.embed_query(content[:500]))
            # 코사인 유사도
            sim = float(
                np.dot(report_emb, chunk_emb)
                / (np.linalg.norm(report_emb) * np.linalg.norm(chunk_emb) + 1e-9)
            )
            scores.append(sim)

        if not scores:
            return 0.0

        return round(float(np.mean(scores)), 3)

    except Exception as e:
        print(f"      [경고] SemScore 계산 실패: {e}")
        return 0.0


# ── 리포트 로드 ───────────────────────────────────────────────────────────────

def _load_reports() -> list[tuple[str, str, list[dict] | None]]:
    """
    REPORT_DIR 에서 최신 .md 파일을 MAX_REPORTS 개만큼 로드.
    같은 이름의 _sources.json 이 있으면 같이 읽어 반환.
    Returns: [(파일명, 리포트텍스트, 소스청크리스트 or None), ...]
    """
    if not REPORT_DIR.exists():
        print(f"  ERROR: 리포트 디렉터리 없음 ({REPORT_DIR})")
        print(f"  먼저 app.py 에서 리포트를 생성하세요.")
        sys.exit(1)

    paths = sorted(
        REPORT_DIR.glob("*.md"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )[:MAX_REPORTS]

    if not paths:
        print(f"  ERROR: 평가할 .md 리포트 없음 ({REPORT_DIR})")
        sys.exit(1)

    loaded = []
    for p in paths:
        text         = p.read_text(encoding="utf-8")
        sources_path = p.with_name(p.stem + "_sources.json")
        sources      = json.loads(sources_path.read_text(encoding="utf-8")) if sources_path.exists() else None
        loaded.append((p.name, text, sources))

    has_sources = sum(1 for _, _, s in loaded if s is not None)
    print(f"  {len(loaded)}개 리포트 로드 완료 (소스 있음: {has_sources}개)")
    return loaded


# ── 메인 ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 65)
    print("리포트 품질 평가 (규칙 기반 + LLM-as-judge)")
    print("=" * 65)

    # 리포트 로드
    print("\n  리포트 로드 중...")
    reports = _load_reports()

    llm     = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    results = []

    for filename, text, sources in reports:
        print(f"\n  [{filename}] 평가 중...")

        record = {
            "filename"           : filename,
            "structure"          : _eval_structure(text),
            "finance_terms"      : _eval_finance_terms(text),
            "faithfulness"       : _eval_faithfulness(llm, text, sources),
            "investment_utility" : _eval_investment_utility(llm, text),
            "balance"            : _eval_balance(llm, text),
            "sem_score"          : _eval_semscore(text, sources),
        }

        metric_keys = [k for k in record if k != "filename"]
        record["average"] = round(sum(record[k] for k in metric_keys) / len(metric_keys), 3)
        results.append(record)

        # 개별 결과 출력
        print(f"    구조 완성도   : {record['structure']:.3f}")
        print(f"    전문 용어     : {record['finance_terms']:.3f}")
        print(f"    출처 충실도   : {record['faithfulness']:.3f}")
        print(f"    투자 실용성   : {record['investment_utility']:.3f}")
        print(f"    관점 균형성   : {record['balance']:.3f}")
        print(f"    의미 유사도   : {record['sem_score']:.3f}")
        print(f"    ──────────────────────────────────")
        print(f"    평균          : {record['average']:.3f}")

    # 전체 평균 출력 (sem_score 포함 — 버그 수정)
    df = pd.DataFrame(results)
    metric_cols = ["structure", "finance_terms", "faithfulness", "investment_utility", "balance", "sem_score"]
    overall_scores = {col: round(df[col].mean(), 3) for col in metric_cols}
    print_score_summary(overall_scores, label="전체 리포트 평균 점수")

    # 저장
    save_results(df, REPORT_EVAL_PATH, "리포트 평가 결과")

    print("\n" + "=" * 65)
    print("완료!")
    print("=" * 65)


if __name__ == "__main__":
    main()
