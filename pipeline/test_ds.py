"""
test_ds.py
DS투자증권 리포트만 대상으로 청킹 파이프라인 테스트

실행: uv run python pipeline/test_ds.py
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.processing.Loader  import load_all_reports
from src.processing.cleaner import clean_reports
from src.processing.chunking import chunking_01_recursive as s1
from src.processing.chunking import chunking_02_semantic  as s2

# ===========================================
#   설정 — 실행할 항목만 남기고 나머지 주석처리
# ===========================================

# ── 데이터 경로 ──────────────────────────────
DS_DIR     = PROJECT_ROOT / "data" / "reports" / "reports_naver_industry" / "DS투자증권"
TEST_DIR   = PROJECT_ROOT / "data" / "test"
CACHE_PATH = TEST_DIR / "ds_cache.json"

# ── 청킹 전략 (실행할 것만 남기기) ──────────
STRATEGIES = [
    (s1, "ds_chunking_01_recursive.json"),  # 전략 1: RecursiveCharacterTextSplitter
    # (s2, "ds_chunking_02_semantic.json"),   # 전략 2: SemanticChunker (OpenAI 비용)
]

# ===========================================


def print_chunks(result, max_chunks: int = None) -> None:
    """청킹 결과 전체 출력"""
    chunks = result.chunks
    total  = len(chunks)

    print(f"\n{'━' * 70}")
    print(f"  전략: {result.strategy}  |  총 {total}개 청크  |  평균 {result.avg_chunk_size:.0f}자")
    print(f"{'━' * 70}")

    limit = max_chunks if max_chunks else total

    for i, chunk in enumerate(chunks[:limit]):
        level = f" [{chunk.chunk_level}]" if chunk.chunk_level else ""
        print(f"\n┌─ [{i+1}/{total}] {chunk.chunk_id}{level}")
        print(f"│  리포트 : {chunk.source_firm} | {chunk.sector} | {chunk.report_date}")
        print(f"│  크기   : {chunk.char_count}자  "
              f"(청크 {chunk.chunk_index+1}/{chunk.total_chunks})")
        if chunk.parent_id:
            print(f"│  부모ID : {chunk.parent_id}")
        print(f"│")
        text = chunk.text.replace("\n", " ").strip()
        for j in range(0, len(text), 100):
            print(f"│  {text[j:j+100]}")
        print(f"└{'─' * 68}")

    if max_chunks and total > max_chunks:
        print(f"\n  ... 외 {total - max_chunks}개 청크 생략 (총 {total}개)")


def main():
    TEST_DIR.mkdir(parents=True, exist_ok=True)

    # 1. LOAD
    print("\n[ STEP 1 ] LOAD  (DS투자증권)")
    print(f"경로: {DS_DIR}")
    reports = load_all_reports(str(DS_DIR), str(CACHE_PATH))
    if not reports:
        print("리포트 없음. 경로를 확인하세요.")
        sys.exit(1)
    print(f"  → {len(reports)}개 리포트 로드 완료")

    # 2. CLEAN
    print("\n[ STEP 2 ] CLEAN")
    reports = clean_reports(reports, verbose=True)

    # 3. CHUNK
    print("\n[ STEP 3 ] CHUNK")

    results = []
    for module, filename in STRATEGIES:
        out_path = TEST_DIR / filename
        print(f"\n-- {filename}")
        try:
            result = module.chunk_reports(reports)
            result.save(str(out_path))
            sizes = [c.char_count for c in result.chunks]
            print(f"   청크 수: {result.chunk_count} | 평균: {result.avg_chunk_size:.0f}자 "
                  f"| 최소: {min(sizes)}자 | 최대: {max(sizes)}자")
            print(f"   저장: {out_path}")
            print_chunks(result)
            results.append(result)
        except Exception as e:
            print(f"   실패: {e}")

    if len(results) > 1:
        print("\n[ 전략 비교 ]")
        for r in results:
            sizes = [c.char_count for c in r.chunks]
            print(f"  {r.strategy:<20} 청크:{r.chunk_count:>5} | 평균:{r.avg_chunk_size:>7.0f}자")

    print("\n테스트 완료")
    print(f"저장 위치: {TEST_DIR}")


if __name__ == "__main__":
    main()
