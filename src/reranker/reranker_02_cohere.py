"""
reranker_02_cohere.py
Cohere Rerank API 기반 Reranker

Retriever가 가져온 후보 문서들을
Cohere Rerank API로 재점수화해서 진짜 관련있는 문서만 추림

사전 설치:
    pip install cohere

모델:
    rerank-multilingual-v3.0  → 다국어 지원 (한국어 포함), 권장
    rerank-english-v3.0       → 영어 전용, 더 빠름
"""
STRATEGY_NAME = "reranker_02_cohere"

import os
from langchain.schema import Document


# ── 설정 ──────────────────────────────────────────────────────────────────────

DEFAULT_MODEL   = "rerank-multilingual-v3.0"  # 한국어 지원
DEFAULT_TOP_N   = 10
SCORE_THRESHOLD = 0.15   # 이 점수 이하는 완전히 관련없는 문서로 제거


# ── Reranker ──────────────────────────────────────────────────────────────────

class CohereReranker:
    """
    Cohere Rerank API Reranker
    - 쿼리 + 문서를 API로 전송해서 관련도 점수 계산
    - Cross-Encoder 방식 (BGE와 동일한 원리, 클라우드 실행)
    - GPU 불필요, 한국어 금융 용어 처리 우수
    - 최초 1회 클라이언트 초기화 후 재사용 (캐싱)
    """

    def __init__(self, model_name: str = DEFAULT_MODEL):
        self.model_name = model_name
        self._client    = None   # lazy loading

    def _load_client(self):
        """최초 사용 시 Cohere 클라이언트 초기화"""
        if self._client is None:
            import cohere
            api_key = os.getenv("COHERE_API_KEY")
            if not api_key:
                raise ValueError(
                    "COHERE_API_KEY가 없습니다. "
                    ".env 파일에 COHERE_API_KEY=... 설정하세요."
                )
            print(f"Cohere Reranker 초기화: {self.model_name}")
            self._client = cohere.Client(api_key=api_key)
            print("Cohere Reranker 초기화 완료")
        return self._client

    def rerank(
        self,
        query:     str,
        docs:      list[Document],
        top_n:     int   = DEFAULT_TOP_N,
        threshold: float = SCORE_THRESHOLD,
    ) -> list[Document]:
        """
        문서 재순위화
        query:     검색 쿼리
        docs:      Retriever가 가져온 후보 문서들
        top_n:     최종 반환할 문서 수
        threshold: 최소 관련도 점수 (이하 제거)

        Returns: 재순위화된 Document 리스트 (score 메타데이터 포함)
        """
        if not docs:
            return []

        client = self._load_client()

        # Cohere API 호출
        response = client.rerank(
            model     = self.model_name,
            query     = query,
            documents = [doc.page_content for doc in docs],
            top_n     = len(docs),   # 전체 점수 받고 직접 필터링
        )

        # threshold 필터링 + top_n 적용
        results = []
        for item in response.results:
            score = item.relevance_score
            if score < threshold:
                continue
            doc = docs[item.index]
            doc.metadata["rerank_score"] = round(float(score), 4)
            results.append(doc)
            if len(results) >= top_n:
                break

        return results


# ── 싱글턴 인스턴스 (클라이언트 재초기화 방지) ───────────────────────────────────

_reranker_instance: CohereReranker | None = None

def get_reranker(model_name: str = DEFAULT_MODEL) -> CohereReranker:
    """싱글턴 Reranker 반환 (클라이언트를 한 번만 초기화)"""
    global _reranker_instance
    if _reranker_instance is None or _reranker_instance.model_name != model_name:
        _reranker_instance = CohereReranker(model_name)
    return _reranker_instance


def rerank(
    query:      str,
    docs:       list[Document],
    top_n:      int   = DEFAULT_TOP_N,
    threshold:  float = SCORE_THRESHOLD,
    model_name: str   = DEFAULT_MODEL,
) -> list[Document]:
    """편의 함수: reranker 인스턴스 없이 바로 호출"""
    return get_reranker(model_name).rerank(query, docs, top_n, threshold)



# reranker_02_cohere.py 맨 아래에 추가
if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()

    from pathlib import Path
    from src.embedding.embedding_01_openai import get_embeddings
    from src.vectorstore.vectorstore_01_chroma import load
    from src.retriever.retriever_02_balanced import build_retriever, retrieve

    PROJECT_ROOT = Path(__file__).parent.parent.parent
    DB_PATH      = str(PROJECT_ROOT / "data" / "vectorstore" / "chroma" / "openai_text-embedding-3-small" / "chunking_03_hybrid")

    # 벡터스토어 로드
    embeddings  = get_embeddings()
    vectorstore = load(DB_PATH, embeddings)

    from langchain.schema import Document as LC_Document
    results  = vectorstore.get(include=["documents", "metadatas"])
    all_docs = [
        LC_Document(page_content=text, metadata=meta)
        for text, meta in zip(results["documents"], results["metadatas"])
    ]

    # 리트리버 생성
    retriever = build_retriever(vectorstore, all_docs, k=40)

    # 후보 검색
    query      = "2차전지 업황 전망"
    candidates = retrieve(retriever, query, k=40, max_per_firm=3)

    # Cohere rerank
    docs = rerank(query, candidates, top_n=10)

    print(f"\n[최종 결과] '{query}' → rerank 후 {len(docs)}개")
    for i, doc in enumerate(docs, 1):
        firm  = doc.metadata.get("source_firm", "-")
        date  = doc.metadata.get("report_date", "-")
        score = doc.metadata.get("rerank_score", "-")
        print(f"\n[{i}] {firm} | {date} | score: {score}")
        print(f"    {doc.page_content[:150]}...")