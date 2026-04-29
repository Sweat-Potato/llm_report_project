"""
main.py
리서치 리포트 RAG 시스템 진입점

사전 준비:
    1. pip install -r requirements.txt
    2. .env 파일에 OPENAI_API_KEY=sk-... 설정
    3. data/reports/ 폴더에 PDF 저장
    4. python pipeline/ingest.py   ← 최초 1회 실행 (ChromaDB 생성)

실행:
    python main.py                            # 인터랙티브 모드
    python main.py --query "반도체 업황"      # 바로 검색
    python main.py --report "AI 반도체 전망"  # 바로 리포트 생성
"""
import sys
import argparse
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

from src.vector_store import load_vectorstore, is_db_exists, CHROMA_DB_PATH
from src.retriever.retriever    import build_retriever
from src.reranker.reranker     import rerank
from src.report_chain import generate_report, step_retrieve


# ── 검색 ──────────────────────────────────────────────────────────────────────

def search(retriever, query: str, top_n: int = 5):
    """Hybrid Search + Rerank 후 결과 출력"""
    from src.retriever import retrieve
    candidates = retrieve(retriever, query, k=20)
    docs       = rerank(query, candidates, top_n=top_n)

    print(f"\n검색어: '{query}'")
    print(f"후보 {len(candidates)}개 → Rerank 후 {len(docs)}개")
    print("=" * 60)

    for i, doc in enumerate(docs, 1):
        score  = doc.metadata.get("rerank_score", "-")
        broker = doc.metadata.get("broker",  "-")
        date   = doc.metadata.get("date",    "-")
        sector = doc.metadata.get("sector",  "-")
        section= doc.metadata.get("section", "-")
        title  = doc.metadata.get("title",   "")[:50]

        print(f"\n[{i}] rerank_score: {score}")
        print(f"    증권사: {broker} | 날짜: {date} | 섹터: {sector}")
        print(f"    섹션:   {section}")
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
            report = generate_report(retriever, query)
            print("\n" + "=" * 60)
            print(report[:1000])
            print("\n...(전체 내용은 data/reports_output/ 폴더를 확인하세요)")

        else:
            search(retriever, user_input)


# ── 메인 ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="리서치 리포트 RAG 시스템")
    parser.add_argument("--query",   type=str, default=None, help="바로 검색")
    parser.add_argument("--report",  type=str, default=None, help="바로 리포트 생성")
    parser.add_argument("--db-path", type=str, default=CHROMA_DB_PATH, help="ChromaDB 경로")
    parser.add_argument("--k",       type=int, default=20,   help="Hybrid Search 후보 수")
    parser.add_argument("--top-n",   type=int, default=8,    help="Reranker 최종 반환 수")
    args = parser.parse_args()

    print("=" * 60)
    print("리서치 리포트 RAG 시스템")
    print("=" * 60)

    # DB 존재 여부 확인
    if not is_db_exists(args.db_path):
        print(f"\n❌ ChromaDB가 없습니다: {args.db_path}")
        print("먼저 아래 명령어로 데이터를 수집하세요:\n")
        print("  python pipeline/ingest.py --pdf-dir data/reports/reports_naver_industry")
        sys.exit(1)

    # ChromaDB 로드
    vectorstore = load_vectorstore(args.db_path)

    # BM25용 청크 로드
    print("BM25 인덱스 구성 중...")
    results = vectorstore.get(include=["documents", "metadatas"])
    from langchain.schema import Document
    all_docs = [
        Document(page_content=text, metadata=meta)
        for text, meta in zip(results["documents"], results["metadatas"])
    ]
    print(f"총 {len(all_docs)}개 청크 로드 완료")

    # Hybrid Retriever 생성
    retriever = build_retriever(vectorstore, all_docs, k=args.k)

    # 실행 모드
    if args.query:
        search(retriever, args.query, top_n=args.top_n)

    elif args.report:
        report = generate_report(retriever, args.report, k=args.k, top_n=args.top_n)
        print("\n" + report)

    else:
        interactive_mode(retriever)


if __name__ == "__main__":
    main()
