"""
chunking_04_paragraph.py
전략 4: 문단 기준 청킹 (Paragraph-based)

특징:
  - 빈 줄(\n\n)을 기준으로 문단 단위 분리
  - 너무 짧은 문단은 다음 문단과 병합 (min_size)
  - 너무 긴 문단은 문장 단위로 재분할 (max_size)
  - API 호출 없음 → 빠르고 무료
  - clean_text 우선 사용 (Cleaner 출력), 없으면 full_text fallback
"""

import re

try:
    from .base import Chunk, ChunkingResult, load_reports_cache, make_chunk_id, extract_meta
except ImportError:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
    from src.processing.chunking.base import Chunk, ChunkingResult, load_reports_cache, make_chunk_id, extract_meta

STRATEGY_NAME = "chunking_04_sentence"

# ── 파라미터 ──────────────────────────────────────
MIN_CHUNK_SIZE = 50    # 이하: 앞 청크에 병합
MAX_CHUNK_SIZE = 1000  # 초과: 문장 단위로 재분할


# ─────────────────────────────────────
# 문단 분리
# ─────────────────────────────────────
def _split_paragraphs(text: str) -> list[str]:
    """빈 줄 기준으로 문단 분리"""
    paragraphs = re.split(r"\n{2,}", text)
    return [p.strip() for p in paragraphs if p.strip()]


def _split_sentences(text: str) -> list[str]:
    """
    문단이 너무 길 때 문장 단위로 재분할
    마침표/다/요/임/됨 으로 끝나는 지점에서 분리
    """
    lines  = text.split("\n")
    merged = []
    buffer = ""

    for line in lines:
        line = line.strip()
        if not line:
            if buffer:
                merged.append(buffer)
                buffer = ""
            continue
        buffer = (buffer + " " + line).strip() if buffer else line
        if re.search(r"[.。]\s*$|[다요임됨]\s*\.?\s*$", buffer):
            merged.append(buffer)
            buffer = ""

    if buffer:
        merged.append(buffer)

    return [m for m in merged if len(m) > MIN_CHUNK_SIZE]


# ─────────────────────────────────────
# 단일 리포트 청킹
# ─────────────────────────────────────
def _chunk_single(text: str) -> list[str]:
    """
    텍스트를 문단 기준으로 청킹
    1. 빈 줄 기준 문단 분리
    2. 너무 짧은 문단 → 앞 문단에 병합
    3. 너무 긴 문단 → 문장 단위로 재분할
    """
    paragraphs = _split_paragraphs(text)

    # 짧은 문단 병합
    merged = []
    buffer = ""

    for para in paragraphs:
        if not buffer:
            buffer = para
            continue

        if len(para) < MIN_CHUNK_SIZE:
            # 짧은 문단은 앞에 붙임
            buffer = buffer + "\n\n" + para
        else:
            merged.append(buffer)
            buffer = para

    if buffer:
        merged.append(buffer)

    # 긴 문단 재분할
    result = []
    for para in merged:
        if len(para) > MAX_CHUNK_SIZE:
            # 문장 단위로 재분할
            sentences = _split_sentences(para)
            result.extend(sentences)
        else:
            result.append(para)

    return [r for r in result if len(r) > MIN_CHUNK_SIZE]


# ─────────────────────────────────────
# 메인 함수 (ingest.py 인터페이스)
# ─────────────────────────────────────
def chunk_reports(reports: list[dict]) -> ChunkingResult:
    """
    reports_cache.json 의 리포트 리스트를 받아
    전략 4(문단 기준)로 청킹.

    Args:
        reports: load_reports_cache() 또는 clean_reports()의 반환값

    Returns:
        ChunkingResult
    """
    all_chunks:  list[Chunk] = []
    total_chars: int         = 0
    global_idx:  int         = 0

    for report in reports:
        # clean_text 우선, 없으면 full_text fallback
        text_src = report.get("clean_text") or report.get("full_text", "")
        text_src = text_src.strip()
        if not text_src:
            continue

        total_chars += len(text_src)
        meta = extract_meta(report)

        try:
            texts = _chunk_single(text_src)
        except Exception as e:
            print(f"  ⚠️  청킹 실패 ({report.get('filename', '')}): {e}")
            continue

        for local_idx, text in enumerate(texts):
            if not text.strip():
                continue
            chunk = Chunk(
                chunk_id     = make_chunk_id(
                    meta["source_firm"], meta["report_date"], global_idx
                ),
                text         = text,
                char_count   = len(text),
                chunk_index  = local_idx,
                total_chunks = len(texts),
                strategy     = STRATEGY_NAME,
                **meta,
            )
            all_chunks.append(chunk)
            global_idx += 1

    return ChunkingResult(
        strategy     = STRATEGY_NAME,
        chunks       = all_chunks,
        report_count = len(reports),
        total_chars  = total_chars,
    )


# ── 단독 실행 ─────────────────────────────────────
if __name__ == "__main__":
    from pathlib import Path
    import sys

    BASE_DIR   = Path(__file__).parent.parent.parent.parent
    CACHE_PATH = BASE_DIR / "data" / "loader_metadata" / "reports_cache.json"
    OUT_PATH   = BASE_DIR / "data" / "chunks" / "chunking_04_paragraph.json"

    if not CACHE_PATH.exists():
        print(f"❌ 캐시 없음: {CACHE_PATH}")
        sys.exit(1)

    reports = load_reports_cache(str(CACHE_PATH))
    print(f"📂 리포트 {len(reports)}개 로드")

    result = chunk_reports(reports)

    print(f"\n[전략 4: {STRATEGY_NAME}]")
    print(f"  청크 수     : {result.chunk_count}")
    print(f"  평균 크기   : {result.avg_chunk_size:.0f}자")
    if result.chunks:
        sizes = [c.char_count for c in result.chunks]
        print(f"  최소 / 최대 : {min(sizes)}자 / {max(sizes)}자")

    result.save(str(OUT_PATH))