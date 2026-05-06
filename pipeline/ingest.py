"""
ingest.py
Loader -> Cleaner -> Chunker -> Embedder -> VectorStore 전체 파이프라인

실행: uv run python pipeline/ingest.py
"""

from __future__ import annotations
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.processing.Loader  import load_all_reports
from src.processing.cleaner import clean_reports
from src.processing.chunking import (
    chunking_01_recursive as c1,
    chunking_02_semantic  as c2,
    chunking_03_hybrid    as c3,
    chunking_04_sentence  as c4,
)
from src.embedding  import embedding_01_openai    as emb1
from src.vectorstore import vectorstore_01_chroma as vs1

# ===========================================
#   설정 — 실행할 항목만 남기고 나머지 주석처리
# ===========================================

# ── 데이터 경로 ──────────────────────────────
REPORTS_DIR   = PROJECT_ROOT / "data" / "reports" / "reports_naver_industry"

CACHE_PATH  = PROJECT_ROOT / "data" / "loader_metadata" / "reports_cache.json"
CHUNKS_DIR  = PROJECT_ROOT / "data" / "chunks"
VS_BASE_DIR = PROJECT_ROOT / "data" / "vectorstore"

# ── 캐시 초기화 여부 ─────────────────────────
FORCE_RELOAD = False
# FORCE_RELOAD = True  # 전체 재파싱

# ── 청킹 전략 (하나만 선택) ─────────────────
#CHUNKING = c1    # 전략 1: RecursiveCharacterTextSplitter
#CHUNKING = c2  # 전략 2: SemanticChunker (OpenAI 비용)
CHUNKING = c3  # 전략 3: 길이별 자동 분기 (OpenAI 비용)
#CHUNKING = c4  # 전략 4: 문단 기준 청킹

# ── 임베딩 전략 (하나만 선택) ────────────────
EMBEDDING = emb1    # 전략 1: OpenAI text-embedding-3-small

# ── 벡터스토어 전략 (하나만 선택) ───────────
VECTORSTORE = vs1    # 전략 1: ChromaDB

# ── STEP 4 실행 여부 ─────────────────────────
RUN_EMBEDDING = True

# ── 임베딩 재시도 설정 ───────────────────────
MAX_RETRIES   = 5    # 최대 재시도 횟수
RETRY_WAIT    = 60   # 재시도 대기 시간 (초)

# ===========================================


def _embed_with_retry(docs, embeddings, db_path):
    """임베딩 + 벡터스토어 저장 (429 Rate limit 시 자동 재시도)"""
    for attempt in range(MAX_RETRIES):
        try:
            VECTORSTORE.build(docs, embeddings, db_path)
            return
        except Exception as e:
            if "429" in str(e) and attempt < MAX_RETRIES - 1:
                wait = RETRY_WAIT * (attempt + 1)
                print(f"   ⚠️  Rate limit → {wait}초 대기 후 재시도... ({attempt+1}/{MAX_RETRIES})")
                time.sleep(wait)
            else:
                raise e


def main():
    # 1. LOAD
    print("\n[ STEP 1 ] LOAD")
    if FORCE_RELOAD and CACHE_PATH.exists():
        print("캐시 삭제 후 전체 재파싱")
        CACHE_PATH.unlink()
    reports = load_all_reports(str(REPORTS_DIR), str(CACHE_PATH))
    if not reports:
        print("리포트 없음. 경로를 확인하세요.")
        sys.exit(1)

    # 2. CLEAN
    print("\n[ STEP 2 ] CLEAN")
    reports = clean_reports(reports, verbose=True)

    # 3. CHUNK
    print("\n[ STEP 3 ] CHUNK")
    CHUNKS_DIR.mkdir(parents=True, exist_ok=True)

    out_path = CHUNKS_DIR / f"{CHUNKING.STRATEGY_NAME}.json"
    print(f"\n-- {CHUNKING.STRATEGY_NAME}")

    # ← 청크 파일이 이미 있으면 스킵
    if out_path.exists():
        print(f"   청크 파일 존재 → 스킵: {out_path}")
        from src.processing.chunking.base import ChunkingResult
        result = ChunkingResult.load(str(out_path))
        sizes = [c.char_count for c in result.chunks]
        print(f"   청크 수: {result.chunk_count} | 평균: {result.avg_chunk_size:.0f}자 "
              f"| 최소: {min(sizes)}자 | 최대: {max(sizes)}자")
    else:
        try:
            result = CHUNKING.chunk_reports(reports)
            result.save(str(out_path))
            sizes = [c.char_count for c in result.chunks]
            print(f"   청크 수: {result.chunk_count} | 평균: {result.avg_chunk_size:.0f}자 "
                  f"| 최소: {min(sizes)}자 | 최대: {max(sizes)}자")
        except Exception as e:
            print(f"   실패: {e}")
            return

    # 4. EMBED + VECTORSTORE
    if not RUN_EMBEDDING:
        print("\n[ STEP 4 ] SKIP (RUN_EMBEDDING = False)")
        print("\n파이프라인 완료")
        return

    print("\n[ STEP 4 ] EMBED + VECTORSTORE")
    embeddings = EMBEDDING.get_embeddings()

    db_path = str(VS_BASE_DIR / VECTORSTORE.STRATEGY_NAME / EMBEDDING.STRATEGY_NAME / CHUNKING.STRATEGY_NAME)
    print(f"\n-- {CHUNKING.STRATEGY_NAME}  →  {EMBEDDING.STRATEGY_NAME}  →  {VECTORSTORE.STRATEGY_NAME}")
    try:
        docs = EMBEDDING.chunks_to_documents(result.chunks)
        _embed_with_retry(docs, embeddings, db_path)  # ← 재시도 로직 적용
    except Exception as e:
        print(f"   실패: {e}")

    print("\n파이프라인 완료")


if __name__ == "__main__":
    main()