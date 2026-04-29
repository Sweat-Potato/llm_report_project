"""
vectorstore_01_chroma.py
ChromaDB 벡터스토어 전략 (영구 저장)

제공:
  - build()   → Documents를 임베딩하여 ChromaDB에 저장
  - load()    → 기존 ChromaDB 로드 (임베딩 재생성 없음)
  - exists()  → DB 존재 여부 확인
"""

from __future__ import annotations

import shutil
from pathlib import Path

from langchain.schema import Document
from langchain_community.vectorstores import Chroma

# ── 설정 ────────────────────────────────────────
STRATEGY_NAME   = "chroma"
COLLECTION_NAME = "research_reports"
BATCH_SIZE      = 100   # OpenAI API 토큰 한도 대응


def build(docs: list[Document], embeddings, db_path: str) -> Chroma:
    """
    ChromaDB 벡터스토어 생성 (배치 처리, 덮어쓰기)
    - DB가 이미 존재하면 컬렉션 삭제 후 새로 생성
    - 중복 저장 방지
    """
    # 기존 디렉토리 통째로 삭제 (덮어쓰기)
    if Path(db_path).exists():
        shutil.rmtree(db_path)
        print(f"  기존 DB 삭제 완료")
    Path(db_path).mkdir(parents=True, exist_ok=True)

    print(f"  임베딩 + ChromaDB 저장 중... ({len(docs)}개 청크)")
    print(f"  저장 위치: {Path(db_path).absolute()}")

    vectorstore = None

    for i in range(0, len(docs), BATCH_SIZE):
        batch     = docs[i : i + BATCH_SIZE]
        batch_num = i // BATCH_SIZE + 1
        total_bat = (len(docs) - 1) // BATCH_SIZE + 1
        print(f"  배치 {batch_num}/{total_bat} ({len(batch)}청크)")

        if vectorstore is None:
            vectorstore = Chroma.from_documents(
                documents         = batch,
                embedding         = embeddings,
                persist_directory = db_path,
                collection_name   = COLLECTION_NAME,
            )
        else:
            vectorstore.add_documents(batch)

    count = vectorstore._collection.count()
    print(f"  ChromaDB 저장 완료 (총 {count}개 청크)")
    return vectorstore


def load(db_path: str, embeddings) -> Chroma:
    """기존 ChromaDB 로드 (임베딩 재생성 없이 빠르게 로드)"""
    if not exists(db_path):
        raise FileNotFoundError(
            f"ChromaDB가 없습니다: {db_path}\n"
            f"먼저 pipeline/ingest.py를 실행하세요."
        )

    vectorstore = Chroma(
        persist_directory  = db_path,
        embedding_function = embeddings,
        collection_name    = COLLECTION_NAME,
    )

    count = vectorstore._collection.count()
    print(f"  ChromaDB 로드 완료: {count}개 청크 ({Path(db_path).absolute()})")
    return vectorstore


def exists(db_path: str) -> bool:
    """DB가 이미 존재하고 데이터가 있는지 확인"""
    p = Path(db_path)
    return p.exists() and any(p.iterdir())