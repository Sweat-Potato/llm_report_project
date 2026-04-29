"""
chunking_01_recursive.py
전략 1: RecursiveCharacterTextSplitter (베이스라인)

langchain_v0_basics 6-2_Chunking/02-RecursiveCharacterTextSplitter 기반

특징:
  - 구분자 우선순위: \n\n → \n → ". " → 한국어 서술체 → " " → ""
  - 고정 크기(chunk_size) 보장, chunk_overlap으로 문맥 연결
  - 가장 빠르고 안정적인 베이스라인

금융 리포트 최적화 포인트:
  - 한국어 문장 끝 패턴("다.", "한다.", "습니다.") 구분자 추가
  - chunk_size=500으로 리포트 문단 단위와 맞춤
  - 메타데이터(증권사, 섹터, 날짜 등) 각 청크에 첨부
"""

from langchain_text_splitters import RecursiveCharacterTextSplitter

try:
    from .base import Chunk, ChunkingResult, load_reports_cache, make_chunk_id, extract_meta
except ImportError:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
    from src.processing.chunking.base import Chunk, ChunkingResult, load_reports_cache, make_chunk_id, extract_meta

STRATEGY_NAME = "chunking_01_recursive"

# ── 파라미터 ──────────────────────────────────────
CHUNK_SIZE    = 500   # 청크 최대 문자 수
CHUNK_OVERLAP = 50    # 청크 간 중복 문자 수

# 구분자 우선순위 (langchain 노트북 + 금융 리포트 특화)
SEPARATORS = [
    "\n\n\n",       # 페이지/섹션 강 구분
    "\n\n",         # 단락 구분
    "\n",           # 줄 구분
    "다.\n",        # 한국어 서술체 문장 끝 + 줄바꿈
    "습니다.\n",    # 정중체 문장 끝 + 줄바꿈
    "한다.\n",      # 평서문 끝 + 줄바꿈
    ". ",           # 영어/숫자 문장 끝
    "다. ",         # 한국어 서술체 문장 끝
    "습니다. ",     # 정중체 문장 끝
    " ",            # 단어 경계
    "",             # 문자 단위 (마지막 수단)
]


def chunk_reports(reports: list[dict]) -> ChunkingResult:
    """
    reports_cache.json 의 리포트 리스트를 받아
    전략 1(RecursiveCharacterTextSplitter)로 청킹.

    Args:
        reports: load_reports_cache()의 반환값

    Returns:
        ChunkingResult
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=SEPARATORS,
        length_function=len,
        is_separator_regex=False,
    )

    all_chunks: list[Chunk] = []
    total_chars = 0
    global_idx  = 0  # 전체 청크 번호

    for report in reports:
        # clean_text 우선, 없으면 full_text 사용 (cleaner.py 미적용 시 fallback)
        text_src = report.get("clean_text") or report.get("full_text", "")
        text_src = text_src.strip()
        if not text_src:
            continue

        total_chars += len(text_src)
        meta = extract_meta(report)

        # 분할
        texts = splitter.split_text(text_src)

        for local_idx, text in enumerate(texts):
            chunk = Chunk(
                chunk_id    = make_chunk_id(
                    meta["source_firm"], meta["report_date"], global_idx
                ),
                text        = text,
                char_count  = len(text),
                chunk_index = local_idx,
                total_chunks= len(texts),
                strategy    = STRATEGY_NAME,
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
    OUT_PATH   = BASE_DIR / "data" / "chunks" / "chunking_01_recursive.json"

    if not CACHE_PATH.exists():
        print(f"❌ 캐시 없음: {CACHE_PATH}")
        sys.exit(1)

    reports = load_reports_cache(str(CACHE_PATH))
    print(f"📂 리포트 {len(reports)}개 로드")

    result = chunk_reports(reports)

    print(f"\n[전략 1: {STRATEGY_NAME}]")
    print(f"  청크 수     : {result.chunk_count}")
    print(f"  평균 크기   : {result.avg_chunk_size:.0f}자")
    sizes = [c.char_count for c in result.chunks]
    print(f"  최소 / 최대 : {min(sizes)}자 / {max(sizes)}자")


    total = result.chunk_count

    for chunk in result.chunks:
        if chunk.sector == "반도체":
            print(f"\n--- {chunk.chunk_id} ---")
            print(f"\n[{chunk.chunk_index + 1}/{total}] 번째 청크")
            print(chunk.text)

    result.save(str(OUT_PATH))
