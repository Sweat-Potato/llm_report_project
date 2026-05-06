"""
embedding_01_openai.py
OpenAI text-embedding-3-small 임베딩 전략

제공:
  - get_embeddings()        → OpenAIEmbeddings 객체 반환
  - chunks_to_documents()   → Chunk 리스트 → LangChain Document 리스트 변환

실행 비용: OpenAI Embedding API 호출 (토큰당 과금)
"""

from __future__ import annotations

from langchain_openai import OpenAIEmbeddings
from langchain.schema import Document

from src.processing.chunking.base import Chunk

# ── 설정 ────────────────────────────────────────
STRATEGY_NAME = "openai_text-embedding-3-small"
EMBED_MODEL   = "text-embedding-3-small"

# 한 번에 보낼 텍스트 수 제한 (TPM 초과 방지)
# OpenAI SDK가 429 발생 시 max_retries 횟수만큼 지수 백오프로 자동 재시도
CHUNK_SIZE  = 50
MAX_RETRIES = 6


def get_embeddings() -> OpenAIEmbeddings:
    """OpenAI 임베딩 모델 반환 (TPM 제한 대응: chunk_size + 자동 재시도)"""
    return OpenAIEmbeddings(
        model       = EMBED_MODEL,
        chunk_size  = CHUNK_SIZE,   # API 호출당 최대 텍스트 수
        max_retries = MAX_RETRIES,  # 429 발생 시 지수 백오프 재시도
    )


def chunks_to_documents(chunks: list[Chunk]) -> list[Document]:
    """
    Chunk 리스트 → LangChain Document 리스트 변환

    Chunk의 모든 메타데이터를 Document.metadata로 전달.
    ChromaDB는 None 값을 허용하지 않으므로 빈 문자열/0으로 대체.
    """
    docs = []
    for chunk in chunks:
        metadata = {
            "chunk_id":     chunk.chunk_id,
            "source_firm":  chunk.source_firm,
            "report_date":  chunk.report_date  or "",
            "sector":       chunk.sector       or "",
            "title":        chunk.title        or "",
            "report_type":  chunk.report_type  or "",
            "analyst":      chunk.analyst      or "",
            "rating":       chunk.rating       or "",
            "target_price": chunk.target_price or 0,
            "filename":     chunk.filename,
            "chunk_index":  chunk.chunk_index,
            "total_chunks": chunk.total_chunks,
            "strategy":     chunk.strategy,
            "char_count":   chunk.char_count,
        }
        if chunk.parent_id:
            metadata["parent_id"]   = chunk.parent_id
        if chunk.chunk_level:
            metadata["chunk_level"] = chunk.chunk_level

        docs.append(Document(page_content=chunk.text, metadata=metadata))

    return docs
