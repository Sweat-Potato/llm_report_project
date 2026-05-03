"""
eval/04_eval_report.py
생성된 리포트 품질 평가 (규칙 기반 + LLM-as-judge)

평가 기준 5가지:
  1. 구조 완성도   (규칙 기반): 필수 섹션이 모두 존재하는가?
  2. 전문 용어     (규칙 기반): 금융 전문 용어가 충분히 사용됐는가?
  3. 출처 충실도   (LLM 판정): 원본 리포트에 근거한 내용인가? (Hallucination 방지)
  4. 투자 실용성   (LLM 판정): 구체적 수치·조건이 있어 투자 판단에 도움이 되는가?
  5. 관점 균형성   (LLM 판정): 강세(Bull)·약세(Bear) 양쪽을 공정하게 다루는가?

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
MAX_REPORTS = 5   # 최신 리포트 최대 N개 평가

# ══════════════════════════════════════════════════════════════════════════════

import pandas as pd
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate

from eval.base import REPORT_EVAL_PATH, save_results, print_score_summary


# ── 필수 섹션 목록 (report_chain.py 의 FINAL_REPORT_PROMPT 구조와 일치) ────────
REQUIRED_SECTIONS = [
    "Executive Summary",
    "시장 현황",
    "증권사별 분석",
    "핵심 투자 논거",
    "리스크",
    "전망",
    "투자 전략",
]

# ── 금융 전문 용어 (이 중 절반 이상 존재하면 만점) ───────────────────────────
FINANCE_TERMS = [
    "목표주가", "투자의견", "매수", "중립", "비중확대",
    "밸류에이션", "PER", "PBR", "EV/EBITDA",
    "컨센서스", "어닝", "실적", "ROE", "FCF",
    "어닝 서프라이즈", "CAGR", "시가총액", "배당수익률",
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
    금융 용어 밀도.
    전체 용어의 50% 이상 등장하면 1.0 으로 clamp.
    """
    found = sum(1 for term in FINANCE_TERMS if term in report_text)
    ratio = found / len(FINANCE_TERMS)
    return round(min(ratio * 2.0, 1.0), 3)


# ── LLM-as-Judge 평가 ────────────────────────────────────────────────────────

_JUDGE_PROMPT = ChatPromptTemplate.from_template("""
당신은 금융 리서치 리포트 품질 평가 전문가입니다.

[평가 기준]
{criterion}

[평가할 리포트 내용]
{report_content}

위 기준으로 리포트를 평가해 0점(최하)~10점(최고) 사이 정수 점수를 매기세요.
**숫자만** 출력하세요. 다른 텍스트는 일절 쓰지 마세요.
""")

_FAITHFULNESS_PROMPT = ChatPromptTemplate.from_template("""
당신은 금융 리서치 리포트 품질 평가 전문가입니다.

[원본 증권사 청크]
{source_chunks}

[생성된 리포트]
{report_content}

위 원본 청크를 기준으로, 리포트가 원본에 충실하게 작성됐는지 평가하세요.
- 원본에 없는 수치·주장·결론이 있으면 낮게 평가하세요.
- 원본의 핵심 내용(수치, 증권사 의견, 리스크)을 충실히 반영했으면 높게 평가하세요.
0점(최하)~10점(최고) 사이 정수 점수를 매기세요.
**숫자만** 출력하세요. 다른 텍스트는 일절 쓰지 마세요.
""")


def _llm_judge(llm, criterion: str, report_text: str) -> float:
    """
    LLM-as-judge: 0~10 점수를 받아 0~1 로 정규화.
    파싱 실패 시 0.0 반환 (평가 자체가 실패해도 전체가 멈추지 않도록).
    """
    try:
        chain    = _JUDGE_PROMPT | llm
        response = chain.invoke({
            "criterion"     : criterion,
            "report_content": report_text[:4000],
        })
        match = re.search(r'\d+\.?\d*', response.content.strip())
        if not match:
            return 0.0
        score = float(match.group())
        return round(min(max(score / 10.0, 0.0), 1.0), 3)
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
        match = re.search(r'\d+\.?\d*', response.content.strip())
        if not match:
            return 0.0
        score = float(match.group())
        return round(min(max(score / 10.0, 0.0), 1.0), 3)
    except Exception as e:
        print(f"      [경고] faithfulness 판정 실패: {e}")
        return 0.0


def _eval_investment_utility(llm, report_text: str) -> float:
    """투자 실용성: 실제 투자 판단에 얼마나 도움이 되는가?"""
    return _llm_judge(
        llm,
        "이 리포트가 실제 투자 판단에 얼마나 유용한지 평가하세요. "
        "구체적 목표주가·밸류에이션·조건부 시나리오(Best/Base/Worst Case)가 있으면 높게, "
        "막연한 서술만 있으면 낮게 평가하세요.",
        report_text,
    )


def _eval_balance(llm, report_text: str) -> float:
    """관점 균형성: 강세/약세 관점을 공정하게 다루는가?"""
    return _llm_judge(
        llm,
        "리포트가 강세(Bull) 와 약세(Bear) 관점을 얼마나 균형 있게 다루는지 평가하세요. "
        "한쪽 의견만 있거나 리스크 분석이 빈약하면 낮게, "
        "양쪽 시나리오를 공정하게 분석했으면 높게 평가하세요.",
        report_text,
    )


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
        print(f"    ──────────────────────────────────")
        print(f"    평균          : {record['average']:.3f}")

    # 전체 평균 출력
    df = pd.DataFrame(results)
    metric_cols = ["structure", "finance_terms", "faithfulness", "investment_utility", "balance"]
    overall_scores = {col: round(df[col].mean(), 3) for col in metric_cols}
    print_score_summary(overall_scores, label="전체 리포트 평균 점수")

    # 저장
    save_results(df, REPORT_EVAL_PATH, "리포트 평가 결과")

    print("\n" + "=" * 65)
    print("완료!")
    print("=" * 65)


if __name__ == "__main__":
    main()
