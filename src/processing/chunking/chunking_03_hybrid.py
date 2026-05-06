"""
chunking_03_hybrid.py
전략 3: 길이별 3단계 청킹 (sentence / semantic / recursive)

특징:
  - 텍스트 길이에 따라 청킹 전략 자동 분기
    · 1000자 미만  : 문장 단위 청킹 (sentence)
    · 1000~20000자 : Semantic Chunking (OpenAI 임베딩 기반)
    · 20000자 초과: RecursiveCharacterTextSplitter (chunk_size=600)
  - Semantic: 문장 임베딩 유사도 기반 breakpoint 감지
  - threshold를 텍스트 길이에 따라 자동 조정 (60~65)
  - clean_text 우선 사용 (Cleaner 출력), 없으면 full_text fallback
  - 임베딩 모델: text-embedding-3-small
"""

import re
import os
from dotenv import load_dotenv
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

try:
    from .base import Chunk, ChunkingResult, load_reports_cache, make_chunk_id, extract_meta
except ImportError:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
    from src.processing.chunking.base import Chunk, ChunkingResult, load_reports_cache, make_chunk_id, extract_meta

STRATEGY_NAME = "chunking_03_hybrid"

# ── 파라미터 ──────────────────────────────────────
SHORT_THRESHOLD  = 1000    # 이하: 문장 단위 청킹
LONG_THRESHOLD   = 20000   # 초과: recursive fallback
MIN_CHUNK_SIZE   = 100      # 이하 청크 제외


RECURSIVE_SEPARATORS = [
    "\n\n\n", "\n\n", "\n",
    "다.\n", "습니다.\n", "한다.\n",
    ". ", "다. ", "습니다. ",
    " ", "",
]

def _get_recursive_params(text_length: int) -> tuple[int, int]:
    return 600,90


# ─────────────────────────────────────
# Semantic Chunker
# ─────────────────────────────────────
def _get_threshold(text_length: int) -> int:
    if text_length < 5000:
        return 50
    elif text_length < 20000:
        return 55
    else:
        return 60


def _build_semantic_chunker(text_length: int) -> SemanticChunker:
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        api_key=os.getenv("OPENAI_API_KEY")
    )
    threshold = _get_threshold(text_length)
    return SemanticChunker(
        embeddings=embeddings,
        breakpoint_threshold_type="percentile",
        breakpoint_threshold_amount=threshold,
    )


# ─────────────────────────────────────
# 단일 리포트 청킹
# ─────────────────────────────────────
def _chunk_single(text: str) -> tuple[list[str], str]:
    """
    텍스트를 청킹하여 (텍스트 리스트, 방법명) 반환
    """

    MAX_CHUNK_SIZE = 700 

    if len(text) < SHORT_THRESHOLD:
        # 문장 단위 청킹
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

        raw_texts = [m for m in merged if len(m) > MIN_CHUNK_SIZE]
        return raw_texts, "sentence"

    elif len(text) > LONG_THRESHOLD:
        # 긴 리포트: RecursiveCharacterTextSplitter fallback
        # 길이에 따라 chunk_size 자동 조정
        chunk_size, chunk_overlap = _get_recursive_params(len(text))
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=RECURSIVE_SEPARATORS,
            length_function=len,
            is_separator_regex=False,
        )
        raw_texts = splitter.split_text(text)
        raw_texts = [t for t in raw_texts if len(t) > MIN_CHUNK_SIZE]
        return raw_texts, "recursive_fallback"

    else:
        chunker   = _build_semantic_chunker(len(text))
        raw_texts = chunker.split_text(text)
    
        # ← 추가: 너무 긴 청크는 recursive로 재분할
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=600,
            chunk_overlap=90,
            separators=RECURSIVE_SEPARATORS,
            length_function=len,
            is_separator_regex=False,
    )
        final_texts = []
        for t in raw_texts:
            if len(t) > MAX_CHUNK_SIZE:
                final_texts.extend(splitter.split_text(t))
            else:
                final_texts.append(t)
        raw_texts = final_texts
        return raw_texts, "semantic_v2"


# ─────────────────────────────────────
# 메인 함수 (ingest.py 인터페이스)
# ─────────────────────────────────────
def chunk_reports(reports: list[dict]) -> ChunkingResult:
    """
    reports_cache.json 의 리포트 리스트를 받아
    전략 3(sentence / semantic / recursive)으로 청킹.

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
            texts, method = _chunk_single(text_src)
        except Exception as e:
            print(f"  ⚠️  청킹 실패 ({report.get('filename', '')}): {e}")
            continue

        for local_idx, text in enumerate(texts):
            if not text.strip():
                continue
            if len(text) < MIN_CHUNK_SIZE:  # ← 추가
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
    from src.processing.cleaner import clean_reports  # ← 추가

    from pathlib import Path
    import sys

    BASE_DIR   = Path(__file__).parent.parent.parent.parent
    CACHE_PATH = BASE_DIR / "data" / "loader_metadata" / "reports_cache.json"
    OUT_PATH   = BASE_DIR / "data" / "chunks" / "chunking_03_hybrid.json"

    if not CACHE_PATH.exists():
        print(f"❌ 캐시 없음: {CACHE_PATH}")
        sys.exit(1)

    reports = load_reports_cache(str(CACHE_PATH))
    print(f"📂 리포트 {len(reports)}개 로드")

    cleaned = clean_reports(reports, verbose=False)   # ← 추가
    result = chunk_reports(cleaned)

    print(f"\n[전략 3: {STRATEGY_NAME}]")
    print(f"  청크 수     : {result.chunk_count}")
    print(f"  평균 크기   : {result.avg_chunk_size:.0f}자")
    if result.chunks:
        sizes = [c.char_count for c in result.chunks]
        print(f"  최소 / 최대 : {min(sizes)}자 / {max(sizes)}자")

    result.save(str(OUT_PATH))