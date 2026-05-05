"""
src/reranker/reranker_01_crossencoder.py
FlashRank 기반 Reranker (ONNX, PyTorch 불필요)

Windows에서 sentence-transformers CrossEncoder 사용 시 PyTorch/OpenMP 충돌로
Segmentation Fault 가 발생해, 동일 품질의 FlashRank(ONNX)로 교체.

모델:
    ms-marco-MultiBERT-L-12  → 다국어 지원 (한국어 포함), ONNX 실행
"""
STRATEGY_NAME = "reranker_01_crossencoder"

from langchain.schema import Document

DEFAULT_MODEL   = "ms-marco-MultiBERT-L-12"
DEFAULT_TOP_N   = 8
SCORE_THRESHOLD = 0.0   # flashrank 점수 범위가 달라 threshold 완화


class FlashReranker:
    def __init__(self, model_name: str = DEFAULT_MODEL):
        self.model_name = model_name
        self._ranker    = None

    def _load_model(self):
        if self._ranker is None:
            from flashrank import Ranker
            print(f"Reranker 모델 로딩 중: {self.model_name}")
            self._ranker = Ranker(model_name=self.model_name)
            print("Reranker 모델 로딩 완료")
        return self._ranker

    def rerank(
        self,
        query:     str,
        docs:      list[Document],
        top_n:     int   = DEFAULT_TOP_N,
        threshold: float = SCORE_THRESHOLD,
    ) -> list[Document]:
        if not docs:
            return []

        from flashrank import RerankRequest

        ranker   = self._load_model()
        passages = [
            {"id": i, "text": doc.page_content}
            for i, doc in enumerate(docs)
        ]
        req     = RerankRequest(query=query, passages=passages)
        results = ranker.rerank(req)

        out = []
        for item in results:
            score = float(item.get("score", 0.0))
            if score < threshold:
                continue
            doc = docs[item["id"]]
            doc.metadata["rerank_score"] = round(score, 4)
            out.append(doc)
            if len(out) >= top_n:
                break

        return out


_reranker_instance: FlashReranker | None = None

def get_reranker(model_name: str = DEFAULT_MODEL) -> FlashReranker:
    global _reranker_instance
    if _reranker_instance is None or _reranker_instance.model_name != model_name:
        _reranker_instance = FlashReranker(model_name)
    return _reranker_instance


def rerank(
    query:      str,
    docs:       list[Document],
    top_n:      int   = DEFAULT_TOP_N,
    threshold:  float = SCORE_THRESHOLD,
    model_name: str   = DEFAULT_MODEL,
) -> list[Document]:
    return get_reranker(model_name).rerank(query, docs, top_n, threshold)
