"""
main.py
리서치 리포트 RAG 시스템 진입점

사전 준비:
    1. .env 파일에 OPENAI_API_KEY=sk-... 설정
    2. uv run python pipeline/pipeline.py  ← 최초 1회 실행 (ChromaDB 생성)

실행:
    uv run python pipeline/main.py                       # 인터랙티브 모드
    uv run python pipeline/main.py --query "반도체 업황" # 바로 검색
    uv run python pipeline/main.py --report "AI 반도체"  # 바로 리포트 생성
"""
import sys
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv()

from src.processing.chunking import chunking_01_recursive as c1
from src.processing.chunking import chunking_02_semantic  as c2
from src.embedding   import embedding_01_openai    as emb1
from src.vectorstore import vectorstore_01_chroma  as vs1
from src.retriever   import retriever_01_ensemble  as ret1
from src.reranker    import reranker_01_crossencoder as rer1

from src.reportcreator.report_chain import generate_report

# ===========================================
#   설정 — 실행할 항목만 남기고 나머지 주석처리
# ===========================================

VS_BASE_DIR = PROJECT_ROOT / "data" / "vectorstore"

# ── 임베딩 전략 (pipeline.py와 동일하게) ─────
EMBEDDING = emb1    # 전략 1: OpenAI text-embedding-3-small
# EMBEDDING = emb2  # 전략 2: (추후 추가)

# ── 벡터스토어 전략 (pipeline.py와 동일하게) ─
VECTORSTORE = vs1    # 전략 1: ChromaDB
# VECTORSTORE = vs2  # 전략 2: (추후 추가)

# ── 청킹 전략 (pipeline.py에서 사용한 것으로) ─
CHUNKING = c1    # 전략 1: RecursiveCharacterTextSplitter
# CHUNKING = c2  # 전략 2: SemanticChunker (OpenAI 비용)

# ── 리트리버 전략 (하나만 선택) ──────────────
RETRIEVER = ret1    # 전략 1: BM25 + Vector Ensemble
# RETRIEVER = ret2  # 전략 2: (추후 추가)

# ── 리랭커 전략 (하나만 선택) ────────────────
RERANKER = rer1     # 전략 1: BGE Cross-Encoder
# RERANKER = rer2   # 전략 2: (추후 추가)

# ===========================================

# 전략 조합으로 DB 경로 자동 결정
DB_PATH = str(VS_BASE_DIR / VECTORSTORE.STRATEGY_NAME / EMBEDDING.STRATEGY_NAME / CHUNKING.STRATEGY_NAME)


# ── 검색 ──────────────────────────────────────────────────────────────────────

def search(retriever, query: str, top_n: int = 5):
    """Hybrid Search + Rerank 후 결과 출력"""
    candidates = RETRIEVER.retrieve(retriever, query, k=20)
    docs       = RERANKER.rerank(query, candidates, top_n=top_n)

    print(f"\n검색어: '{query}'")
    print(f"후보 {len(candidates)}개 → Rerank 후 {len(docs)}개")
    print("=" * 60)

    for i, doc in enumerate(docs, 1):
        score  = doc.metadata.get("rerank_score", "-")
        broker = doc.metadata.get("source_firm",  "-")
        date   = doc.metadata.get("report_date",  "-")
        sector = doc.metadata.get("sector",        "-")
        title  = doc.metadata.get("title",         "")[:50]

        print(f"\n[{i}] rerank_score: {score}")
        print(f"    증권사: {broker} | 날짜: {date} | 섹터: {sector}")
        print(f"    제목:   {title}")
        print(f"    내용:   {doc.page_content[:200]}...")


# ── 인터랙티브 모드 ───────────────────────────────────────────────────────────

def interactive_mode(retriever):
    print("\n" + "=" * 60)
    print("리서치 RAG 시스템 (종료: q)")
    print("-" * 60)
    print("명령어:")
    print("  search <키워드>   → Hybrid Search + Rerank")
    print("  report <주제>     → 종합 리포트 생성")
    print("  q                 → 종료")
    print("=" * 60)

    while True:
        try:
            user_input = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n종료합니다.")
            break

        if not user_input or user_input.lower() == "q":
            print("종료합니다.")
            break

        parts = user_input.split(" ", 1)
        cmd   = parts[0].lower()
        query = parts[1].strip() if len(parts) > 1 else ""

        if not query:
            print("키워드를 입력해주세요.")
            continue

        if cmd == "search":
            search(retriever, query)
        elif cmd == "report":
            report = generate_report(
                retriever,
                query,
                retrieve_fn = RETRIEVER.retrieve,
                rerank_fn   = RERANKER.rerank,
            )
            print("\n" + "=" * 60)
            print(report[:1000])
            print("\n...(전체 내용은 data/reports_output/ 폴더를 확인하세요)")
        else:
            search(retriever, user_input)


# ── 메인 ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="리서치 리포트 RAG 시스템")
    parser.add_argument("--query",  type=str, default=None, help="바로 검색")
    parser.add_argument("--report", type=str, default=None, help="바로 리포트 생성")
    parser.add_argument("--k",      type=int, default=20,   help="Hybrid Search 후보 수")
    parser.add_argument("--top-n",  type=int, default=8,    help="Reranker 최종 반환 수")
    args = parser.parse_args()

    print("=" * 60)
    print("리서치 리포트 RAG 시스템")
    print(f"DB       : {DB_PATH}")
    print(f"Retriever: {RETRIEVER.STRATEGY_NAME}")
    print(f"Reranker : {RERANKER.STRATEGY_NAME}")
    print("=" * 60)

    # DB 존재 여부 확인
    if not VECTORSTORE.exists(DB_PATH):
        print(f"\nChromaDB가 없습니다: {DB_PATH}")
        print("먼저 아래 명령어로 데이터를 수집하세요:\n")
        print("  uv run python pipeline/pipeline.py")
        sys.exit(1)

    print(f"\n임베딩 로드 중... ({EMBEDDING.STRATEGY_NAME})")
    embeddings  = EMBEDDING.get_embeddings()
    print(f"벡터스토어 로드 중... ({VECTORSTORE.STRATEGY_NAME})")
    vectorstore = VECTORSTORE.load(DB_PATH, embeddings)

    print(f"리트리버 인덱스 구성 중... ({RETRIEVER.STRATEGY_NAME})")
    results  = vectorstore.get(include=["documents", "metadatas"])
    from langchain.schema import Document
    all_docs = [
        Document(page_content=text, metadata=meta)
        for text, meta in zip(results["documents"], results["metadatas"])
    ]
    print(f"총 {len(all_docs)}개 청크 로드 완료")

    retriever = RETRIEVER.build_retriever(vectorstore, all_docs, k=args.k)

    if args.query:
        search(retriever, args.query, top_n=args.top_n)
    elif args.report:
        report = generate_report(
            retriever,
            args.report,
            retrieve_fn = RETRIEVER.retrieve,
            rerank_fn   = RERANKER.rerank,
            k           = args.k,
            top_n       = args.top_n,
        )
        print("\n" + report)
    else:
        interactive_mode(retriever)


if __name__ == "__main__":
    main()
