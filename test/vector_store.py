"""
src/vector_store.py
ChromaDB 벡터스토어 모듈 (영구 저장)
"""
import os
from pathlib import Path
from langchain.schema import Document
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma


CHROMA_DB_PATH  = "./data/chroma_db"
COLLECTION_NAME = "research_reports"
EMBED_MODEL     = "text-embedding-3-small"
BATCH_SIZE      = 100   # OpenAI API 토큰 한도 대응


def get_embeddings() -> OpenAIEmbeddings:
    return OpenAIEmbeddings(model=EMBED_MODEL)


def build_vectorstore(docs: list[Document], db_path: str = CHROMA_DB_PATH) -> Chroma:
    """
    ChromaDB 벡터스토어 생성 (배치 처리)
    - 이미 DB가 있으면 기존 데이터 유지하고 추가
    - 없으면 새로 생성
    """
    embeddings = get_embeddings()
    Path(db_path).mkdir(parents=True, exist_ok=True)

    print(f"임베딩 생성 + ChromaDB 저장 중... ({len(docs)}개 청크)")
    print(f"저장 위치: {Path(db_path).absolute()}")

    vectorstore = None

    for i in range(0, len(docs), BATCH_SIZE):
        batch = docs[i:i + BATCH_SIZE]
        print(f"  배치 {i // BATCH_SIZE + 1}/{(len(docs) - 1) // BATCH_SIZE + 1} ({len(batch)}청크)")

        if vectorstore is None:
            vectorstore = Chroma.from_documents(
                documents       = batch,
                embedding       = embeddings,
                persist_directory = db_path,
                collection_name = COLLECTION_NAME,
            )
        else:
            vectorstore.add_documents(batch)

    print(f"ChromaDB 저장 완료 (총 {vectorstore._collection.count()}개 청크)\n")
    return vectorstore


def load_vectorstore(db_path: str = CHROMA_DB_PATH) -> Chroma:
    """
    기존 ChromaDB 로드 (임베딩 재생성 없이 빠르게 로드)
    """
    if not Path(db_path).exists():
        raise FileNotFoundError(
            f"ChromaDB가 없습니다: {db_path}\n"
            f"먼저 pipeline/ingest.py를 실행하세요."
        )

    embeddings  = get_embeddings()
    vectorstore = Chroma(
        persist_directory = db_path,
        embedding_function = embeddings,
        collection_name   = COLLECTION_NAME,
    )

    count = vectorstore._collection.count()
    print(f"ChromaDB 로드 완료: {count}개 청크 ({Path(db_path).absolute()})")
    return vectorstore


def is_db_exists(db_path: str = CHROMA_DB_PATH) -> bool:
    """DB가 이미 존재하고 데이터가 있는지 확인"""
    chroma_dir = Path(db_path)
    return chroma_dir.exists() and any(chroma_dir.iterdir())
