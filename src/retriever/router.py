"""
src/retriever/router.py
쿼리 의도에 따라 리트리버 자동 선택

동작 흐름:
  쿼리 입력
  → LLM이 증권사명 추출 (특정 증권사 언급 시)
  → 증권사명 정규화 (대신 → 대신증권 등)
  → LLM이 의도 분류 (비교 vs 일반)
  → 특정 증권사 언급: 해당 증권사 청크만 필터링
  → 비교 쿼리: retriever_02_balanced (증권사별 균등)
  → 일반 쿼리: retriever_01_ensemble (점수 기반)
  → 선택된 리트리버로 검색 실행

분류 기준:
  balanced: 증권사별 비교/다양한 관점이 필요한 쿼리
  ensemble: 특정 사실/수치/전망을 찾는 일반 쿼리
"""

import json
from langchain.schema import Document
from langchain_openai import ChatOpenAI
from langchain.retrievers import EnsembleRetriever

try:
    from . import retriever_01_ensemble as ret1
    from . import retriever_02_balanced as ret2
except ImportError:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from src.retriever import retriever_01_ensemble as ret1
    from src.retriever import retriever_02_balanced as ret2


# ── 설정 ──────────────────────────────────────────────────────────────────────

CLASSIFIER_MODEL = "gpt-4o-mini"

_CLASSIFY_PROMPT = """다음 증권사 리포트 검색 쿼리를 분석하세요.

쿼리: "{query}"

다음 두 가지를 JSON으로 답하세요:

1. intent: 쿼리 의도
   - "balanced": 증권사별 비교, 다양한 관점이 필요한 쿼리
     예시: "증권사별 반도체 의견", "각 증권사 2차전지 전망 비교", "다양한 관점"
   - "ensemble": 특정 사실/수치/전망을 찾는 일반 쿼리
     예시: "반도체 업황 전망", "삼성전자 목표주가", "HBM 수요 현황"

2. firms: 쿼리에서 언급된 특정 증권사 이름 목록 (없으면 빈 리스트)
   - 증권사명만 추출 (예: "하나증권", "대신증권", "키움증권")
   - 언급이 없으면 []

반드시 아래 JSON 형식으로만 답하세요:
{{"intent": "balanced" 또는 "ensemble", "firms": ["증권사1", "증권사2"]}}"""


# ── 증권사 이름 정규화 ────────────────────────────────────────────────────────
# 사용자 입력 → 메타데이터 실제 이름 (source_firm)

FIRM_ALIASES = {
    # DS투자증권
    "DS투자증권": "DS투자증권",
    "DS증권": "DS투자증권",
    "ds투자증권": "DS투자증권",
    "ds증권": "DS투자증권",

    # IBK투자증권
    "IBK투자증권": "IBK투자증권",
    "IBK증권": "IBK투자증권",
    "ibk투자증권": "IBK투자증권",
    "ibk증권": "IBK투자증권",
    "IBK": "IBK투자증권",

    # iM증권
    "iM증권": "iM증권",
    "IM증권": "iM증권",
    "im증권": "iM증권",
    "iM": "iM증권",

    # SK증권
    "SK증권": "SK증권",
    "sk증권": "SK증권",
    "SK": "SK증권",

    # 교보증권
    "교보증권": "교보증권",
    "교보": "교보증권",

    # 대신증권
    "대신증권": "대신증권",
    "대신": "대신증권",

    # 유안타증권
    "유안타증권": "유안타증권",
    "유안타": "유안타증권",

    # 유진투자증권
    "유진투자증권": "유진투자증권",
    "유진증권": "유진투자증권",
    "유진": "유진투자증권",

    # 키움증권
    "키움증권": "키움증권",
    "키움": "키움증권",

    # 하나증권
    "하나증권": "하나증권",
    "하나": "하나증권",

    # 한국IR협의회
    "한국IR협의회": "한국IR협의회",
    "한국IR": "한국IR협의회",
    "IR협의회": "한국IR협의회",

    # 한화투자증권
    "한화투자증권": "한화투자증권",
    "한화증권": "한화투자증권",
    "한화": "한화투자증권",
}


def normalize_firms(firms: list[str]) -> list[str]:
    """LLM이 추출한 증권사명을 메타데이터 실제 이름으로 정규화"""
    result = []
    for firm in firms:
        normalized = FIRM_ALIASES.get(firm)
        if normalized:
            result.append(normalized)
        else:
            # 부분 매칭 시도
            matched = False
            for alias, canonical in FIRM_ALIASES.items():
                if alias in firm or firm in alias:
                    result.append(canonical)
                    matched = True
                    break
            if not matched:
                print(f"  ⚠️  알 수 없는 증권사: {firm} → 그대로 사용")
                result.append(firm)

    return list(set(result))  # 중복 제거


# ── LLM 분류기 ────────────────────────────────────────────────────────────────

_llm = None

def _get_llm() -> ChatOpenAI:
    global _llm
    if _llm is None:
        _llm = ChatOpenAI(
            model=CLASSIFIER_MODEL,
            temperature=0,
        )
    return _llm


def analyze_query(query: str) -> tuple[str, list[str]]:
    """
    쿼리 분석: 의도 분류 + 증권사명 추출 + 정규화
    Returns: (intent, firms)
      intent: "balanced" | "ensemble"
      firms:  정규화된 증권사 목록 (없으면 [])
    """
    prompt = _CLASSIFY_PROMPT.format(query=query)
    result = _get_llm().invoke(prompt).content.strip()

    try:
        parsed = json.loads(result)
        intent = parsed.get("intent", "ensemble")
        firms  = parsed.get("firms", [])

        if intent not in ("balanced", "ensemble"):
            intent = "ensemble"

        # 증권사명 정규화
        firms = normalize_firms(firms)

        return intent, firms

    except Exception:
        print(f"  ⚠️  분류 실패 → ensemble, 전체 검색")
        return "ensemble", []


def _filter_by_firms(docs: list[Document], firms: list[str]) -> list[Document]:
    """
    특정 증권사 청크만 필터링
    firms가 비어있으면 전체 반환
    증권사별로 리포트 보유 여부 확인 후 없으면 안내
    """
    if not firms:
        return docs

    # 증권사별 보유 여부 확인
    available_firms = set(
        doc.metadata.get("source_firm", "")
        for doc in docs
    )
    for firm in firms:
        if firm not in available_firms:
            print(f"  ❌ '{firm}'의 리포트가 없습니다.")
        else:
            firm_count = sum(
                1 for doc in docs
                if doc.metadata.get("source_firm") == firm
            )
            print(f"  ✅ '{firm}': {firm_count}개 청크 보유")

    filtered = [
        doc for doc in docs
        if any(
            firm in (doc.metadata.get("source_firm") or "")
            for firm in firms
        )
    ]

    if not filtered:
        print(f"  ⚠️  검색 가능한 리포트 없음 → 전체 검색으로 fallback")
        return docs

    return filtered


# ── 라우터 ────────────────────────────────────────────────────────────────────

def build_retriever(
    vectorstore,
    all_docs: list[Document],
    k:        int = 40,
) -> tuple:
    """
    두 리트리버 모두 준비
    retrieve() 호출 시 쿼리에 따라 선택

    Returns: (ret1_instance, ret2_instance, all_docs)
    """
    ret1_instance = ret1.build_retriever(vectorstore, all_docs, k=k)
    ret2_instance = ret2.build_retriever(vectorstore, all_docs, k=k)
    return ret1_instance, ret2_instance, all_docs


def retrieve(
    retrievers: tuple,
    query:      str,
    k:          int = 40,
) -> list[Document]:
    """
    쿼리 의도 + 증권사 언급에 따라 자동 검색

    동작:
    1. LLM으로 의도 분류 + 증권사명 추출 + 정규화
    2. 특정 증권사 언급 시 → 해당 증권사 청크만 검색
    3. 비교 쿼리 → retriever_02_balanced
       일반 쿼리 → retriever_01_ensemble
    """
    ret1_instance, ret2_instance, all_docs = retrievers

    # 1. 쿼리 분석
    intent, firms = analyze_query(query)
    print(f"  쿼리 의도: {intent}")
    if firms:
        print(f"  언급된 증권사: {firms}")

    # 2. 특정 증권사 언급 시 → 해당 증권사 문서만 필터링 후 검색
    if firms:
        print(f"  → 메타데이터 필터링: {firms}")
        firm_docs = _filter_by_firms(all_docs, firms)
        print(f"  → 필터링 결과: {len(firm_docs)}개 청크")

        from langchain_community.retrievers import BM25Retriever
        from collections import defaultdict

        bm25 = BM25Retriever.from_documents(
            firm_docs,
            k=k,
            preprocess_func=ret1.korean_tokenizer,
        )

        raw = bm25.invoke(query)

        # 중복 제거
        seen = set()
        result = []
        for doc in raw:
            key = (
                doc.metadata.get("filename", "") or doc.metadata.get("pdf_path", ""),
                doc.metadata.get("chunk_index", ""),
            )
            if key not in seen:
                seen.add(key)
                result.append(doc)

        # 비교 쿼리면 균등 샘플링 적용
        if intent == "balanced":
            firm_counts = defaultdict(int)
            balanced = []
            for doc in result:
                firm = doc.metadata.get("source_firm", "기타")
                if firm_counts[firm] < ret2.MAX_PER_FIRM:
                    balanced.append(doc)
                    firm_counts[firm] += 1
                if len(balanced) >= k:
                    break
            return balanced

        return result[:k]

    # 3. 증권사 미언급 → 일반 리트리버 선택
    if intent == "balanced":
        print(f"  → retriever_02_balanced 사용 (증권사별 균등 샘플링)")
        return ret2.retrieve(ret2_instance, query, k=k)
    else:
        print(f"  → retriever_01_ensemble 사용 (점수 기반)")
        return ret1.retrieve(ret1_instance, query, k=k)


# ── 단독 실행 (테스트) ────────────────────────────────────────────────────────

if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()

    # ── 실제 검색 테스트 ──────────────────────────────
    print("\n" + "=" * 65)
    print("실제 검색 테스트")
    print("=" * 65)

    from pathlib import Path
    from src.embedding.embedding_01_openai import get_embeddings
    from src.vectorstore.vectorstore_01_chroma import load

    PROJECT_ROOT = Path(__file__).parent.parent.parent
    DB_PATH = str(PROJECT_ROOT / "data" / "vectorstore" / "chroma" / "openai_text-embedding-3-small" / "chunking_03_hybrid")

    embeddings  = get_embeddings()
    vectorstore = load(DB_PATH, embeddings)

    from langchain.schema import Document as LC_Document
    results  = vectorstore.get(include=["documents", "metadatas"])
    all_docs = [
        LC_Document(page_content=text, metadata=meta)
        for text, meta in zip(results["documents"], results["metadatas"])
    ]

    retrievers = build_retriever(vectorstore, all_docs, k=40)

    query = "대신증권 한화증권의 반도체 업황 비교해줘"
    docs  = retrieve(retrievers, query, k=20)

    print(f"\n[검색 결과] '{query}' → {len(docs)}개")
    for i, doc in enumerate(docs, 1):
        firm = doc.metadata.get("source_firm", "-")
        date = doc.metadata.get("report_date", "-")
        title = doc.metadata.get("title", "")[:40]
        print(f"\n[{i}] {firm} | {date} | {title}")
        print(f"    {doc.page_content[:150]}...")