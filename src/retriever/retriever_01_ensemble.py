"""
src/retriever.py
Hybrid Search 모듈 (BM25 + Vector Ensemble)

BM25  → 키워드 기반 검색 (HBM3e, 비중확대 같은 금융 전문용어에 강함)
Vector → 의미 기반 검색 (유사 의미 문서 검색에 강함)
Ensemble → 두 결과를 가중치로 합산

한국어 형태소 분석(kiwipiepy)으로 BM25 토크나이징 품질 개선:
- "반도체업황"→["반도체","업황"], 조사·어미 제거
- 금융 범용 단어(업황, 전망, 분석 등) 제외 → 노이즈 감소
"""
STRATEGY_NAME = "retriever_01_ensemble"

from langchain.schema import Document
from langchain_community.vectorstores import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever


# ── 한국어 토크나이저 ──────────────────────────────────────────────────────────

# 조사(JX,JC,JK*), 어미(EF,EC,ETM), 접미사(XS*) 제거 → 의미 있는 형태소만 남김
_STOP_TAGS = {"JX", "JC", "JKS", "JKO", "JKG", "JKB", "JKV", "JKQ",
              "EF", "EC", "ETM", "ETN", "XSV", "XSA", "XSN", "SF", "SP", "SS"}

# 금융 리포트에서 너무 범용적으로 사용되어 노이즈가 되는 단어
# → BM25 검색 시 제외하여 섹터 무관 청크 유입 방지
FINANCIAL_STOP_WORDS = {
    # 상태/방향
    "업황", "전망", "분석", "투자", "의견",
    "시장", "성장", "실적", "수익", "이익",
    "예상", "전년", "대비", "기준", "수준",
    "증가", "감소", "상승", "하락", "유지",
    "긍정", "부정", "중립", "판단", "추정",
    # 보고서 공통 표현
    "리포트", "보고서", "리서치", "센터", "증권",
    "투자자", "주가", "목표", "영업", "매출",
    "분기", "연간", "반기", "전분기", "전년도",
}

_kiwi = None

def _get_kiwi():
    global _kiwi
    if _kiwi is None:
        from kiwipiepy import Kiwi
        _kiwi = Kiwi()
    return _kiwi


def korean_tokenizer(text: str) -> list[str]:
    """한국어 형태소 분석 기반 토크나이저 (BM25용)"""
    try:
        kiwi   = _get_kiwi()
        tokens = kiwi.tokenize(text)
        return [
            t.form for t in tokens
            if t.tag not in _STOP_TAGS
            and len(t.form) > 1
            and t.form not in FINANCIAL_STOP_WORDS  # 금융 범용 단어 제외
        ]
    except Exception:
        return [
            w for w in text.split()
            if w not in FINANCIAL_STOP_WORDS
        ]


# ── 설정 ──────────────────────────────────────────────────────────────────────

# 금융 전문용어·고유명사 매칭을 위해 BM25 비중을 높임
BM25_WEIGHT = 0.3


# ── Retriever 빌더 ────────────────────────────────────────────────────────────

def build_retriever(
    vectorstore: Chroma,
    docs:        list[Document],
    k:           int   = 20,
    bm25_weight: float = BM25_WEIGHT,
) -> EnsembleRetriever:
    """
    Hybrid Retriever 생성
    vectorstore: 이미 생성된 ChromaDB 인스턴스
    docs:        BM25용 Document 리스트 (청크 전체)
    k:           각 retriever가 가져올 문서 수
    """
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


def retrieve(
    retriever: EnsembleRetriever,
    query:     str,
    k:         int = 20,
) -> list[Document]:
    """
    Hybrid 검색 실행
    Returns: 관련 Document 리스트
    """
    docs = retriever.invoke(query)

    # 중복 제거 (같은 pdf_path + chunk_index)
    seen   = set()
    unique = []
    for doc in docs:
        key = (
            doc.metadata.get("pdf_path", ""),
            doc.metadata.get("chunk_index", ""),
        )
        if key not in seen:
            seen.add(key)
            unique.append(doc)

    return unique[:k]