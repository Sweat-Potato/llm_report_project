"""
chunking_02_semantic.py
전략 2: SemanticChunker (의미 기반 분할)

langchain_v0_basics 6-2_Chunking/04-SemanticChunker 기반

원리:
  1. 텍스트를 문장 단위로 분리
  2. 각 문장의 임베딩 벡터 생성
  3. 인접 문장 간 코사인 유사도 계산
  4. 유사도가 급격히 떨어지는 지점(breakpoint)에서 분할

전략 1(크기 기반)과의 차이:
  ✅ 의미가 바뀌는 지점에서만 분할 → 문맥 보존 최우수
  ✅ 금융 리포트의 섹션 전환(시황→리스크→종목분석)을 자연스럽게 포착
  ❌ OpenAI Embedding API 호출 → 비용 발생, 속도 느림
  ❌ 청크 크기가 가변적 (LLM 컨텍스트 제한 주의)

Breakpoint 방식 (breakpoint_threshold_type):
  percentile       : 유사도 차이 상위 N%에서 분할 (기본값, 분할 수 조정 쉬움)
  standard_deviation: 통계적으로 명확한 변화 지점 분할
  interquartile    : 분포가 고르지 않은 문서에 안정적

금융 리포트 권장: percentile, threshold_amount=85
  → 상위 15%에서만 분할 → 섹션 단위 큰 청크 생성
"""

from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai import OpenAIEmbeddings

try:
    from .base import Chunk, ChunkingResult, load_reports_cache, make_chunk_id, extract_meta
except ImportError:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
    from src.processing.chunking.base import Chunk, ChunkingResult, load_reports_cache, make_chunk_id, extract_meta

STRATEGY_NAME = "chunking_02_semantic"

# ── 파라미터 ──────────────────────────────────────
BREAKPOINT_TYPE   = "percentile"  # "percentile" | "standard_deviation" | "interquartile"
BREAKPOINT_AMOUNT = 85            # percentile 기준: 상위 15%에서만 분할 (클수록 청크 크고 개수 적음)
LONG_THRESHOLD    = 20000         # 초과: 스킵 (Semantic 청킹 시간 과다 소요)
#
# 조정 가이드:
#   threshold_amount 낮춤 → 더 자주 분할 → 작은 청크 多
#   threshold_amount 높임 → 드물게 분할 → 큰 청크 少
#   금융 섹션 단위(1000~2000자)를 원하면 85~90 권장


def _make_splitter() -> SemanticChunker:
    """SemanticChunker 인스턴스 생성 (API 키는 .env에서 로드)"""
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    return SemanticChunker(
        embeddings,
        breakpoint_threshold_type=BREAKPOINT_TYPE,
        breakpoint_threshold_amount=BREAKPOINT_AMOUNT,
    )


def chunk_reports(reports: list[dict]) -> ChunkingResult:
    """
    전략 2: SemanticChunker로 의미 기반 청킹.
    20000자 초과 리포트는 스킵.

    ⚠️  각 리포트 처리 시 OpenAI Embedding API를 호출합니다.
        비용 발생 주의.

    Args:
        reports: load_reports_cache() 또는 clean_reports() 반환값

    Returns:
        ChunkingResult
    """
    splitter = _make_splitter()

    all_chunks: list[Chunk] = []
    total_chars = 0
    global_idx  = 0
    skipped     = 0

    for report in reports:
        text_src = report.get("clean_text") or report.get("full_text", "")
        text_src = text_src.strip()
        if not text_src:
            continue

        # 20000자 초과 리포트 스킵
        if len(text_src) > LONG_THRESHOLD:
            skipped += 1
            print(f"  ⏭️  스킵 ({len(text_src):,}자 > {LONG_THRESHOLD:,}자): {report.get('filename', '')[:50]}")
            continue

        total_chars += len(text_src)
        meta = extract_meta(report)

        print(f"  🔍 임베딩 중: {meta['source_firm']} / {meta['sector']} ({len(text_src):,}자) ...")

        try:
            texts = splitter.split_text(text_src)
        except Exception as e:
            print(f"  ⚠️  SemanticChunker 실패 ({meta['filename']}): {e}")
            continue

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

        print(f"     → {len(texts)}개 청크  (평균 {sum(len(t) for t in texts)//len(texts) if texts else 0}자)")

    if skipped:
        print(f"\n  📊 스킵된 리포트: {skipped}개 ({LONG_THRESHOLD:,}자 초과)")

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
    from dotenv import load_dotenv

    load_dotenv()

    BASE_DIR   = Path(__file__).parent.parent.parent.parent
    CACHE_PATH = BASE_DIR / "data" / "loader_metadata" / "reports_cache.json"
    OUT_PATH   = BASE_DIR / "data" / "chunks" / "chunking_02_semantic.json"

    if not CACHE_PATH.exists():
        print(f"❌ 캐시 없음: {CACHE_PATH}")
        sys.exit(1)

    reports = load_reports_cache(str(CACHE_PATH))
    print(f"📂 리포트 {len(reports)}개 로드")
    print(f"⚠️  OpenAI Embedding API 호출 시작 (비용 발생)\n")

    result = chunk_reports(reports)

    print(f"\n[전략 2: {STRATEGY_NAME}]")
    print(f"  청크 수     : {result.chunk_count}")
    sizes = [c.char_count for c in result.chunks]
    if sizes:
        print(f"  평균 크기   : {result.avg_chunk_size:.0f}자")
        print(f"  최소 / 최대 : {min(sizes)}자 / {max(sizes)}자")

    result.save(str(OUT_PATH))
