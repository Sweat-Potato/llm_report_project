"""
compare.py
4가지 청킹 전략 비교 실행기

실행:
    python -m src.processing.chunking.compare            # 전략 1~4 모두
    python -m src.processing.chunking.compare --no-semantic  # API 비용 아낄 때

비교 지표:
  1. 청크 수 / 평균·최소·최대 크기 / 표준편차
  2. 문장 완결성   — 청크가 문장 중간에 끊기는 비율
  3. 금융 키워드 보존율 — 핵심 키워드가 청크 안에 온전히 들어있는 비율
  4. 오버랩 중복률   — 인접 청크 사이에 실제로 겹치는 텍스트 비율
  5. Parent-Child 전용 — 자식→부모 연결 성공률

결과:
  - 콘솔 출력 (표 형태)
  - data/chunks/compare_result.json 저장
"""

from __future__ import annotations
import json
import statistics
import re
from pathlib import Path

try:
    from .base import ChunkingResult, Chunk, load_reports_cache
    from . import chunking_01_recursive as s1
    from . import chunking_02_semantic  as s2
    from . import chunking_03_hybrid    as s3
    from . import chunking_04_sentence  as s4
    from src.processing.cleaner import clean_reports
except ImportError:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
    from src.processing.chunking.base import ChunkingResult, Chunk, load_reports_cache
    from src.processing.chunking import chunking_01_recursive as s1
    from src.processing.chunking import chunking_02_semantic  as s2
    from src.processing.chunking import chunking_03_hybrid    as s3
    from src.processing.chunking import chunking_04_sentence  as s4
    from src.processing.cleaner import clean_reports

# ─────────────────────────────────────────────────
# 금융 리포트 핵심 키워드
# ─────────────────────────────────────────────────
FINANCIAL_KEYWORDS = [
    "투자의견", "목표주가", "비중확대", "비중축소", "매수", "매도", "중립",
    "영업이익", "매출", "영업이익률", "EPS", "PER", "PBR", "ROE",
    "상향", "하향", "유지", "전망", "리스크", "밸류에이션",
]

# 문장 끝 패턴 (이걸로 끝나면 완결)
_SENTENCE_END = re.compile(r"(다|합니다|합니다|ND|%)[\.\s]*$")


# ─────────────────────────────────────────────────
# 지표별 측정 함수
# ─────────────────────────────────────────────────

def measure_size_stats(chunks: list[Chunk]) -> dict:
    """청크 크기 분포 통계"""
    sizes = [c.char_count for c in chunks]
    if not sizes:
        return {}
    return {
        "count":  len(sizes),
        "mean":   round(statistics.mean(sizes), 1),
        "median": round(statistics.median(sizes), 1),
        "stdev":  round(statistics.stdev(sizes), 1) if len(sizes) > 1 else 0.0,
        "min":    min(sizes),
        "max":    max(sizes),
    }


def measure_sentence_completeness(chunks: list[Chunk]) -> dict:
    """
    청크 끝이 완전한 문장으로 끝나는 비율.
    금융 리포트에서 문맥 절단을 얼마나 방지했는지 측정.
    """
    complete = 0
    for c in chunks:
        text = c.text.strip()
        if _SENTENCE_END.search(text[-5:] if len(text) >= 5 else text):
            complete += 1
    ratio = complete / len(chunks) if chunks else 0
    return {
        "complete_count": complete,
        "total_count":    len(chunks),
        "completeness":   round(ratio * 100, 1),  # %
    }


def measure_keyword_preservation(chunks: list[Chunk]) -> dict:
    """
    금융 핵심 키워드가 한 청크 안에 온전히 들어있는 비율.
    키워드가 청크 경계에서 잘리면 검색 품질이 떨어짐.
    """
    keyword_in_chunk = 0
    keyword_total    = 0

    full_text = " ".join(c.text for c in chunks)
    for kw in FINANCIAL_KEYWORDS:
        total_occurrences = full_text.count(kw)
        keyword_total += total_occurrences

        for c in chunks:
            keyword_in_chunk += c.text.count(kw)

    ratio = keyword_in_chunk / keyword_total if keyword_total else 1.0
    return {
        "keyword_occurrences_in_chunks": keyword_in_chunk,
        "keyword_occurrences_total":     keyword_total,
        "preservation_rate":             round(ratio * 100, 1),  # %
    }


def measure_overlap_efficiency(chunks: list[Chunk]) -> dict:
    """
    인접 청크 사이 실제 중복 텍스트 길이 평균.
    overlap이 너무 크면 중복 저장 낭비, 너무 작으면 문맥 단절.
    """
    from itertools import groupby

    def group_key(c: Chunk):
        return (c.source_firm, c.report_date or "")

    sorted_chunks = sorted(chunks, key=lambda c: (group_key(c), c.chunk_index))
    overlaps = []

    for _, group in groupby(sorted_chunks, key=group_key):
        g = list(group)
        for i in range(len(g) - 1):
            a, b = g[i].text, g[i + 1].text
            min_len = min(len(a), len(b), 100)
            overlap_len = 0
            for l in range(min_len, 0, -1):
                if a.endswith(b[:l]):
                    overlap_len = l
                    break
            overlaps.append(overlap_len)

    if not overlaps:
        return {"avg_overlap_chars": 0, "max_overlap_chars": 0}
    return {
        "avg_overlap_chars": round(statistics.mean(overlaps), 1),
        "max_overlap_chars": max(overlaps),
    }


def measure_parent_child_linkage(result: ChunkingResult) -> dict | None:
    """전략 3 전용: 자식→부모 연결 성공률"""
    if result.strategy != "parent_child":
        return None

    parents  = {c.chunk_id for c in result.chunks if c.chunk_level == "parent"}
    children = [c for c in result.chunks if c.chunk_level == "child"]

    if not children:
        return {"parent_count": 0, "child_count": 0, "link_rate": 0}

    linked = sum(1 for c in children if c.parent_id in parents)
    return {
        "parent_count": len(parents),
        "child_count":  len(children),
        "link_rate":    round(linked / len(children) * 100, 1),
    }


# ─────────────────────────────────────────────────
# 전략별 측정 통합
# ─────────────────────────────────────────────────

def evaluate(result: ChunkingResult) -> dict:
    """단일 전략 전체 평가"""
    if result.strategy == "parent_child":
        search_chunks  = [c for c in result.chunks if c.chunk_level == "child"]
        context_chunks = [c for c in result.chunks if c.chunk_level == "parent"]
    else:
        search_chunks  = result.chunks
        context_chunks = result.chunks

    return {
        "strategy":    result.strategy,
        "report_count": result.report_count,
        "size_stats":        measure_size_stats(search_chunks),
        "sentence_completeness": measure_sentence_completeness(search_chunks),
        "keyword_preservation": measure_keyword_preservation(context_chunks),
        "overlap_efficiency": measure_overlap_efficiency(search_chunks),
        "parent_child_linkage": measure_parent_child_linkage(result),
    }


# ─────────────────────────────────────────────────
# 출력 포맷
# ─────────────────────────────────────────────────

def print_comparison(evaluations: list[dict]) -> None:
    """비교표 콘솔 출력"""
    COL = 26

    def row(label: str, *values):
        cells = f"{label:<22}" + "".join(f"{str(v):<{COL}}" for v in values)
        print(cells)

    names = [e["strategy"] for e in evaluations]

    print("\n" + "=" * (22 + COL * len(names)))
    print("청킹 전략 비교 결과")
    print("=" * (22 + COL * len(names)))
    row("", *names)
    print("-" * (22 + COL * len(names)))

    # 크기 통계
    print("\n[ 크기 통계 (검색 청크 기준) ]")
    row("청크 수",       *[e["size_stats"].get("count","-") for e in evaluations])
    row("평균 크기 (자)", *[e["size_stats"].get("mean","-")  for e in evaluations])
    row("중앙값 (자)",   *[e["size_stats"].get("median","-") for e in evaluations])
    row("표준편차",      *[e["size_stats"].get("stdev","-")  for e in evaluations])
    row("최소 크기",     *[e["size_stats"].get("min","-")    for e in evaluations])
    row("최대 크기",     *[e["size_stats"].get("max","-")    for e in evaluations])

    # 문장 완결성
    print("\n[ 문장 완결성 ]")
    row("완결 청크 수",  *[e["sentence_completeness"].get("complete_count","-") for e in evaluations])
    row("완결률 (%)",   *[e["sentence_completeness"].get("completeness","-")   for e in evaluations])

    # 키워드 보존율
    print("\n[ 금융 키워드 보존율 ]")
    row("보존율 (%)",   *[e["keyword_preservation"].get("preservation_rate","-") for e in evaluations])

    # 오버랩
    print("\n[ 오버랩 효율 ]")
    row("평균 중복 (자)", *[e["overlap_efficiency"].get("avg_overlap_chars","-") for e in evaluations])

    # Parent-Child 전용
    pc_evals = [e for e in evaluations if e.get("parent_child_linkage")]
    if pc_evals:
        print("\n[ Parent-Child 연결 전용 ]")
        for e in pc_evals:
            pc = e["parent_child_linkage"]
            print(f"  부모 {pc['parent_count']}개 / 자식 {pc['child_count']}개"
                  f" / 연결 성공률 {pc['link_rate']}%")

    print("=" * (22 + COL * len(names)))


# ─────────────────────────────────────────────────
# 메인 실행
# ─────────────────────────────────────────────────

def run_compare(cache_path: str, out_dir: str, skip_semantic: bool = False) -> None:
    out_dir_p = Path(out_dir)
    out_dir_p.mkdir(parents=True, exist_ok=True)

    print(f"📂 캐시 로드: {cache_path}")
    reports = load_reports_cache(cache_path)
    print(f"   리포트 {len(reports)}개\n")

    print("── 클리닝 ────────────────────────────────────────────────")
    reports = clean_reports(reports, verbose=True)

    evals = []

    # ── 전략 1: Recursive ─────────────────────────
    print("\n── 전략 1: RecursiveCharacterTextSplitter ──────────────")
    r1 = s1.chunk_reports(reports)
    r1.save(str(out_dir_p / "chunking_01_recursive.json"))
    evals.append(evaluate(r1))
    print(f"   완료: {r1.chunk_count}개 청크")

    # ── 전략 2: Semantic ──────────────────────────
    if skip_semantic:
        print("\n── 전략 2: SemanticChunker ── 스킵 (--no-semantic)")
    else:
        print("\n── 전략 2: SemanticChunker (OpenAI API 호출) ───────────")
        print("   ⚠️  API 비용 발생. 중단: Ctrl+C")
        try:
            r2 = s2.chunk_reports(reports)
            r2.save(str(out_dir_p / "chunking_02_semantic.json"))
            evals.append(evaluate(r2))
            print(f"   완료: {r2.chunk_count}개 청크")
        except KeyboardInterrupt:
            print("\n   ⏹ 전략 2 중단됨")
        except Exception as e:
            print(f"\n   ❌ 전략 2 실패: {e}")

    # ── 전략 3: Hybrid (sentence + semantic + recursive) ──
    if skip_semantic:
        print("\n── 전략 3: Hybrid ── 스킵 (--no-semantic)")
    else:
        print("\n── 전략 3: Hybrid (sentence/semantic/recursive) ───────")
        print("   ⚠️  API 비용 발생 (semantic 구간). 중단: Ctrl+C")
        try:
            r3 = s3.chunk_reports(reports)
            r3.save(str(out_dir_p / "chunking_03_hybrid.json"))
            evals.append(evaluate(r3))
            print(f"   완료: {r3.chunk_count}개 청크")
        except KeyboardInterrupt:
            print("\n   ⏹ 전략 3 중단됨")
        except Exception as e:
            print(f"\n   ❌ 전략 3 실패: {e}")

    # ── 전략 4: Paragraph ─────────────────────────
    print("\n── 전략 4: Paragraph (문단 기준) ──────────────────────")
    r4 = s4.chunk_reports(reports)
    r4.save(str(out_dir_p / "chunking_04_sentence.json"))
    evals.append(evaluate(r4))
    print(f"   완료: {r4.chunk_count}개 청크")

    # ── 비교표 출력 ───────────────────────────────
    print_comparison(evals)

    # JSON 저장
    compare_out = out_dir_p / "compare_result.json"
    with open(compare_out, "w", encoding="utf-8") as f:
        json.dump(evals, f, ensure_ascii=False, indent=2)
    print(f"\n[비교 결과 저장] {compare_out}")


if __name__ == "__main__":
    import sys
    from pathlib import Path

    BASE_DIR   = Path(__file__).parent.parent.parent.parent
    CACHE_PATH = str(BASE_DIR / "data" / "loader_metadata" / "reports_cache.json")
    OUT_DIR    = str(BASE_DIR / "data" / "chunks")

    skip_semantic = "--no-semantic" in sys.argv

    run_compare(CACHE_PATH, OUT_DIR, skip_semantic=skip_semantic)