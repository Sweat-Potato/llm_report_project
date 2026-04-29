"""
src/reranker.py
Cross-Encoder Reranker 모듈 (BGE Reranker)

Retriever가 가져온 후보 문서들을
Cross-Encoder로 재점수화해서 진짜 관련있는 문서만 추림

사전 설치:
    pip install sentence-transformers

모델:
    BAAI/bge-reranker-v2-m3  → 다국어 지원 (한국어 포함), 무료, 로컬 실행
    BAAI/bge-reranker-large  → 영어 위주, 더 빠름
"""
STRATEGY_NAME = "reranker_01_crossencoder"

from langchain.schema import Document


# ── 설정 ──────────────────────────────────────────────────────────────────────

DEFAULT_MODEL  = "BAAI/bge-reranker-v2-m3"
DEFAULT_TOP_N  = 8
SCORE_THRESHOLD = 0.1   # 이 점수 이하는 완전히 관련없는 문서로 제거


# ── Reranker ──────────────────────────────────────────────────────────────────

class BGEReranker:
    """
    BGE Cross-Encoder Reranker
    - 쿼리 + 문서를 동시에 입력해서 관련도 점수 계산
    - Bi-Encoder(벡터 유사도)보다 훨씬 정확
    - 최초 1회 모델 로딩 후 재사용 (캐싱)
    """

    def __init__(self, model_name: str = DEFAULT_MODEL):
        self.model_name = model_name
        self._model     = None   # lazy loading

    def _load_model(self):
        """최초 사용 시 모델 로딩 (이후 캐싱)"""
        if self._model is None:
            from sentence_transformers import CrossEncoder
            print(f"Reranker 모델 로딩 중: {self.model_name}")
            self._model = CrossEncoder(self.model_name)
            print("Reranker 모델 로딩 완료")
        return self._model

    def rerank(
        self,
        query:     str,
        docs:      list[Document],
        top_n:     int   = DEFAULT_TOP_N,
        threshold: float = SCORE_THRESHOLD,
    ) -> list[Document]:
        """
        문서 재순위화
        query: 검색 쿼리
        docs:  Retriever가 가져온 후보 문서들
        top_n: 최종 반환할 문서 수
        threshold: 최소 관련도 점수 (이하 제거)

        Returns: 재순위화된 Document 리스트 (score 메타데이터 포함)
        """
        if not docs:
            return []

        model = self._load_model()

        # Cross-Encoder 입력: [(query, doc_text), ...]
        pairs  = [(query, doc.page_content) for doc in docs]
        scores = model.predict(pairs)

        # 점수 기준 정렬
        scored = sorted(
            zip(scores, docs),
            key    = lambda x: x[0],
            reverse= True,
        )

        # threshold 필터링 + top_n 적용
        results = []
        for score, doc in scored:
            if score < threshold:
                continue
            # 점수를 메타데이터에 추가
            doc.metadata["rerank_score"] = round(float(score), 4)
            results.append(doc)
            if len(results) >= top_n:
                break

        return results


# ── 싱글턴 인스턴스 (모델 재로딩 방지) ──────────────────────────────────────────

_reranker_instance: BGEReranker | None = None

def get_reranker(model_name: str = DEFAULT_MODEL) -> BGEReranker:
    """싱글턴 Reranker 반환 (모델을 한 번만 로딩)"""
    global _reranker_instance
    if _reranker_instance is None or _reranker_instance.model_name != model_name:
        _reranker_instance = BGEReranker(model_name)
    return _reranker_instance


def rerank(
    query:     str,
    docs:      list[Document],
    top_n:     int   = DEFAULT_TOP_N,
    threshold: float = SCORE_THRESHOLD,
    model_name: str  = DEFAULT_MODEL,
) -> list[Document]:
    """편의 함수: reranker 인스턴스 없이 바로 호출"""
    return get_reranker(model_name).rerank(query, docs, top_n, threshold)
