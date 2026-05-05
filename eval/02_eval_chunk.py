"""
eval/02_eval_chunk.py
청크 품질 통계 평가 (LLM 불필요 — 순수 규칙 기반)

측정 지표:
  - chunk_count   : 총 청크 수
  - avg_size      : 평균 청크 크기 (자)
  - std_size      : 크기 표준편차 (편차가 클수록 불균일)
  - min_size      : 최소 청크 크기
  - max_size      : 최대 청크 크기
  - short_ratio   : 최종 생성된 청크 중 짧은 청크 비율 (< SHORT_CHUNK_THRESHOLD 자)
  - long_ratio    : 최종 생성된 청크 중 긴 청크 비율  (> LONG_CHUNK_THRESHOLD 자)
  - total_chars   : 전체 문자 수 (커버리지 기준)

★ 설정 블록 — 이곳만 수정 ★
  STRATEGIES      : 비교할 청킹 전략 리스트
  SHORT_CHUNK_THRESHOLD : 최종 청크 길이가 이 값 미만이면 "짧은 청크"로 분류
  LONG_CHUNK_THRESHOLD  : 최종 청크 길이가 이 값 초과이면 "긴 청크"로 분류

실행:
  uv run python eval/02_eval_chunk.py
"""

import sys
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(BASE_DIR))

from dotenv import load_dotenv
load_dotenv()

# ══════════════════════════════════════════════════════════════════════════════
# ★ 설정 블록 — 비교할 전략 추가/제거
# ══════════════════════════════════════════════════════════════════════════════

from src.processing.chunking import chunking_01_recursive as C1
from src.processing.chunking import chunking_03_hybrid    as C3
from src.processing.chunking import chunking_04_sentence  as C4

STRATEGIES      = [C1, C3, C4]   # 비교할 전략 리스트

# 주의: 아래 threshold는 "최종 생성된 청크 길이"를 평가하기 위한 기준입니다.
# 각 청킹 전략 내부의 SHORT_THRESHOLD / LONG_THRESHOLD(원문 길이 분기 기준)와는 별개입니다.
SHORT_CHUNK_THRESHOLD = 100       # 최종 청크가 이 값 미만이면 "짧은 청크" (정보 부족 우려)
LONG_CHUNK_THRESHOLD  = 2000      # 최종 청크가 이 값 초과이면 "긴 청크"  (LLM 컨텍스트 낭비 우려)

# ══════════════════════════════════════════════════════════════════════════════

import pandas as pd
from eval.base import CHUNK_EVAL_PATH, save_results


def _evaluate_strategy(strategy, reports: list[dict]) -> dict:
    """
    단일 청킹 전략의 품질 지표 계산.

    짧은 청크: 내용이 너무 적어 LLM 이 답변 생성에 활용하기 어려움.
    긴 청크  : 토큰을 낭비하고 관련 없는 내용이 섞일 위험이 있음.
    std_size : 균일할수록 retriever 가 일관된 품질로 검색 가능.
    """
    result = strategy.chunk_reports(reports)
    sizes  = [chunk.char_count for chunk in result.chunks]

    if not sizes:
        return {"strategy": strategy.STRATEGY_NAME}

    series = pd.Series(sizes)
    short_count = (series < SHORT_CHUNK_THRESHOLD).sum()
    long_count  = (series > LONG_CHUNK_THRESHOLD).sum()

    return {
        "strategy"    : strategy.STRATEGY_NAME,
        "chunk_count" : len(sizes),
        "avg_size"    : round(series.mean(), 1),
        "std_size"    : round(series.std(), 1),
        "min_size"    : series.min(),
        "max_size"    : series.max(),
        "short_ratio" : round(short_count / len(sizes), 3),
        "long_ratio"  : round(long_count  / len(sizes), 3),
        "total_chars" : series.sum(),
    }


def _print_strategy_stats(stats: dict) -> None:
    """단일 전략 결과 출력."""
    print(f"    청크 수   : {stats['chunk_count']:,}개")
    print(f"    평균 크기 : {stats['avg_size']}자  (±{stats['std_size']})")
    print(f"    범위      : {stats['min_size']}자 ~ {stats['max_size']}자")
    print(f"    짧은 청크 : {stats['short_ratio']*100:.1f}%  (<{SHORT_CHUNK_THRESHOLD}자)")
    print(f"    긴 청크   : {stats['long_ratio']*100:.1f}%   (>{LONG_CHUNK_THRESHOLD}자)")
    print(f"    전체 문자 : {stats['total_chars']:,}자")


def main():
    print("=" * 65)
    print("청크 품질 평가")
    print("=" * 65)

    # 리포트 로드
    cache_path = BASE_DIR / "data" / "loader_metadata" / "reports_cache.json"
    if not cache_path.exists():
        print(f"  ERROR: reports_cache.json 없음 ({cache_path})")
        print(f"  먼저 src/processing/Loader.py 를 실행하세요.")
        sys.exit(1)

    from src.processing.Loader import load_all_reports

    print("\n  리포트 로드 중...")
    reports = load_all_reports(cache_path=str(cache_path))
    print(f"  로드 완료: {len(reports)}개 리포트")

    # 각 전략 평가
    results = []
    for strategy in STRATEGIES:
        print(f"\n  [{strategy.STRATEGY_NAME}] 평가 중...")
        stats = _evaluate_strategy(strategy, reports)
        results.append(stats)
        _print_strategy_stats(stats)

    # 전략 비교 테이블 출력
    df = pd.DataFrame(results)
    print(f"\n{'='*65}")
    print("전략 비교 테이블")
    print(f"{'='*65}")
    print(df.to_string(index=False))

    # 추천 전략 선정 (짧은 청크 비율 + 긴 청크 비율이 낮고 std 균일한 전략)
    if len(df) > 1:
        df["_penalty"] = df["short_ratio"] + df["long_ratio"] + df["std_size"] / df["avg_size"]
        best = df.loc[df["_penalty"].idxmin(), "strategy"]
        print(f"\n  💡 추천: {best}")
        print(f"     (짧은/긴 청크 비율 + 크기 편차를 종합한 기준)")
        df = df.drop(columns=["_penalty"])

    # 저장
    save_results(df, CHUNK_EVAL_PATH, "청크 평가 결과")

    print("\n" + "=" * 65)
    print("완료!")
    print("=" * 65)


if __name__ == "__main__":
    main()
