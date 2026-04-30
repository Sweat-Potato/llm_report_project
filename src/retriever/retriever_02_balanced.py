"""
retriever_02_balanced.py
Hybrid Search + 증권사별 균등 샘플링

동작 흐름:
  BM25 + Vector Ensemble으로 후보 검색
  → 관련 청크 안에서 증권사당 최대 N개씩 선택
  → 관련없는 증권사는 강제 포함 안 함
  → 최종 k개 반환

기존 retriever_01(앙상블)과 차이:
  retriever_01: 점수 높은 순서대로 k개 반환
              → 특정 증권사에 편향될 수 있음
  retriever_02: 증권사별 균등 샘플링
              → 다양한 증권사 관점 확보
              → 관련없는 증권사 청크 강제 포함 안 함
"""

from collections import defaultdict
from langchain.schema import Document
from langchain_community.vectorstores import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever


# ── 한국어 토크나이저 ──────────────────────────────────────────────────────────

_STOP_TAGS = {"JX", "JC", "JKS", "JKO", "JKG", "JKB", "JKV", "JKQ",
              "EF", "EC", "ETM", "ETN", "XSV", "XSA", "XSN", "SF", "SP", "SS"}

_kiwi = None

def _get_kiwi():
    global _kiwi
    if _kiwi is None:
        from kiwipiepy import Kiwi
        _kiwi = Kiwi()
    return _kiwi


def korean_tokenizer(text: str) -> list[str]:
    try:
        kiwi   = _get_kiwi()
        tokens = kiwi.tokenize(text)
        return [t.form for t in tokens if t.tag not in _STOP_TAGS and len(t.form) > 1]
    except Exception:
        return text.split()


# ── 설정 ──────────────────────────────────────────────────────────────────────

BM25_WEIGHT   = 0.3   # 벡터 검색 비중 높임 (의미 기반 강화)
MAX_PER_FIRM  = 2     # 증권사당 최대 반환 청크 수
STRATEGY_NAME = "ensemble_balanced"


# ── Retriever 빌더 ────────────────────────────────────────────────────────────

def build_retriever(
    vectorstore: Chroma,
    docs:        list[Document],
    k:           int   = 20,
    bm25_weight: float = BM25_WEIGHT,
) -> EnsembleRetriever:
    bm25_retriever = BM25Retriever.from_documents(
        docs,
        k=k,
        preprocess_func=korean_tokenizer,
    )

    vector_retriever = vectorstore.as_retriever(
        search_type   = "similarity",
        search_kwargs = {"k": k},
    )

    ensemble = EnsembleRetriever(
        retrievers = [bm25_retriever, vector_retriever],
        weights    = [bm25_weight, 1 - bm25_weight],
    )

    print(f"Hybrid Retriever 생성 완료 (BM25 {bm25_weight:.0%} / Vector {1-bm25_weight:.0%}, k={k})")
    return ensemble


# ── 균등 샘플링 검색 ──────────────────────────────────────────────────────────

def retrieve(
    retriever:    EnsembleRetriever,
    query:        str,
    k:            int = 20,
    max_per_firm: int = MAX_PER_FIRM,
) -> list[Document]:
    """
    Hybrid 검색 + 증권사별 균등 샘플링

    1. 후보 검색
    2. 중복 제거
    3. 점수 높은 순서 유지하며 증권사당 최대 max_per_firm개씩 선택
       (관련없는 증권사는 강제 포함 안 함)
    4. 최종 k개 반환
    """
    raw = retriever.invoke(query)

    # 중복 제거
    seen       = set()
    candidates = []
    for doc in raw:
        key = (
            doc.metadata.get("filename", "") or doc.metadata.get("pdf_path", ""),
            doc.metadata.get("chunk_index", ""),
        )
        if key not in seen:
            seen.add(key)
            candidates.append(doc)

    # 증권사당 max_per_firm개 제한
    # 점수 높은 순서 유지, 관련없는 증권사는 강제 포함 안 함
    result:      list[Document] = []
    firm_counts: dict[str, int] = defaultdict(int)

    for doc in candidates:
        if len(result) >= k:
            break
        firm = (
            doc.metadata.get("source_firm") or
            doc.metadata.get("broker") or
            "기타"
        )
        if firm_counts[firm] < max_per_firm:
            result.append(doc)
            firm_counts[firm] += 1

    print(f"\n균등 샘플링 결과: {len(candidates)}개 후보 → {len(result)}개 반환")
    print(f"증권사별 청크 수:")
    for firm, count in sorted(firm_counts.items()):
        print(f"  {firm:15}: {count}개")

    return result


# ── 단독 실행 (테스트) ────────────────────────────────────────────────────────

if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()

    from pathlib import Path
    from src.embedding.embedding_01_openai import get_embeddings
    from src.vectorstore.vectorstore_01_chroma import load

    PROJECT_ROOT = Path(__file__).parent.parent.parent
    DB_PATH      = str(PROJECT_ROOT / "data" / "vectorstore" / "chroma" / "openai_text-embedding-3-small" / "chunking_01_recursive")

    embeddings  = get_embeddings()
    vectorstore = load(DB_PATH, embeddings)

    from langchain.schema import Document as LC_Document
    results  = vectorstore.get(include=["documents", "metadatas"])
    all_docs = [
        LC_Document(page_content=text, metadata=meta)
        for text, meta in zip(results["documents"], results["metadatas"])
    ]

    retriever = build_retriever(vectorstore, all_docs, k=20)

    query = "2차전지 업황 전망"
    docs  = retrieve(retriever, query, k=10, max_per_firm=2)

    print(f"\n[테스트 결과] '{query}'")
    for i, doc in enumerate(docs, 1):
        firm = doc.metadata.get("source_firm", "-")
        date = doc.metadata.get("report_date", "-")
        print(f"\n[{i}] {firm} | {date}")
        print(f"    {doc.page_content[:150]}...")