"""
pipeline/inspect_chunks.py
PDF별 청킹 결과를 파일로 저장해서 한눈에 확인하는 스크립트

실행:
    python pipeline/inspect_chunks.py
    python pipeline/inspect_chunks.py --pdf-dir data/reports/reports_naver_industry
    python pipeline/inspect_chunks.py --broker DS투자증권   # 특정 증권사만
    python pipeline/inspect_chunks.py --limit 5            # PDF 5개만

출력:
    data/inspect/
      summary.txt                          ← 전체 요약
      DS투자증권/
        260414_DS투자증권_반도체_낸NAND붐온.txt
        260416_DS투자증권_반도체_...txt
      하나증권/
        ...
"""
import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

from src.loader  import load_pdf, parse_filename_meta
from src.cleaner import clean_documents
from src.chunker import chunk_documents


# ── 설정 ──────────────────────────────────────────────────────────────────────

OUTPUT_DIR = "./data/inspect"


# ── 청크 파일 포맷 ────────────────────────────────────────────────────────────

def format_chunk_file(pdf_path: Path, pages_raw, pages_clean, chunks) -> str:
    """단일 PDF의 청킹 결과를 보기 좋게 포맷"""
    meta = parse_filename_meta(pdf_path)
    lines = []

    lines.append("=" * 70)
    lines.append(f"파일:   {pdf_path.name}")
    lines.append(f"증권사: {meta['broker']}")
    lines.append(f"섹터:   {meta['sector']}")
    lines.append(f"날짜:   {meta['date']}")
    lines.append(f"제목:   {meta['title']}")
    lines.append("=" * 70)
    lines.append(f"원본 페이지 수:  {len(pages_raw)}")
    lines.append(f"정제 후 페이지: {len(pages_clean)}")
    lines.append(f"총 청크 수:     {len(chunks)}")

    # 섹션 분포
    section_counts: dict[str, int] = {}
    for c in chunks:
        sec = c.metadata.get("section", "개요")
        section_counts[sec] = section_counts.get(sec, 0) + 1
    lines.append(f"\n섹션 분포:")
    for sec, cnt in section_counts.items():
        lines.append(f"  [{cnt}개] {sec[:50]}")

    lines.append("\n" + "─" * 70)

    # 청크별 내용
    for i, chunk in enumerate(chunks):
        sec    = chunk.metadata.get("section", "개요")
        length = len(chunk.page_content)

        lines.append(f"\n【청크 {i+1:02d} / {len(chunks)}】  섹션: {sec[:40]}  ({length}자)")
        lines.append("┄" * 50)
        lines.append(chunk.page_content)
        lines.append("")

    return "\n".join(lines)


def format_summary(results: list[dict]) -> str:
    """전체 요약 파일 포맷"""
    lines = []
    lines.append("=" * 70)
    lines.append("청킹 결과 전체 요약")
    lines.append("=" * 70)

    total_pdfs   = len(results)
    total_pages  = sum(r["pages_raw"]   for r in results)
    total_clean  = sum(r["pages_clean"] for r in results)
    total_chunks = sum(r["chunks"]      for r in results)
    removed      = total_pages - total_clean

    lines.append(f"총 PDF 수:        {total_pdfs}개")
    lines.append(f"총 원본 페이지:   {total_pages}개")
    lines.append(f"정제 후 페이지:   {total_clean}개 (제거: {removed}개, {removed/max(total_pages,1)*100:.1f}%)")
    lines.append(f"총 청크 수:       {total_chunks}개")
    lines.append(f"PDF당 평균 청크:  {total_chunks/max(total_pdfs,1):.1f}개")
    lines.append("")

    # 증권사별 요약
    broker_stats: dict[str, dict] = {}
    for r in results:
        b = r["broker"]
        if b not in broker_stats:
            broker_stats[b] = {"pdfs": 0, "chunks": 0, "pages": 0}
        broker_stats[b]["pdfs"]   += 1
        broker_stats[b]["chunks"] += r["chunks"]
        broker_stats[b]["pages"]  += r["pages_clean"]

    lines.append("─" * 70)
    lines.append(f"{'증권사':<15} {'PDF':>5} {'페이지':>7} {'청크':>7} {'PDF당 청크':>10}")
    lines.append("─" * 70)
    for b, s in sorted(broker_stats.items(), key=lambda x: -x[1]["chunks"]):
        avg = s["chunks"] / max(s["pdfs"], 1)
        lines.append(f"{b:<15} {s['pdfs']:>5} {s['pages']:>7} {s['chunks']:>7} {avg:>10.1f}")
    lines.append("─" * 70)
    lines.append("")

    # PDF별 상세
    lines.append("=" * 70)
    lines.append("PDF별 상세")
    lines.append("=" * 70)
    lines.append(f"{'파일명':<55} {'원본':>5} {'정제':>5} {'청크':>5}")
    lines.append("─" * 70)

    for r in sorted(results, key=lambda x: (x["broker"], x["date"])):
        fname = r["filename"][:53]
        lines.append(
            f"{fname:<55} {r['pages_raw']:>5} {r['pages_clean']:>5} {r['chunks']:>5}"
        )

    return "\n".join(lines)


# ── 메인 ──────────────────────────────────────────────────────────────────────

def run(pdf_dir: str, output_dir: str, broker_filter: str = "", limit: int = 0):
    pdf_files = list(Path(pdf_dir).rglob("*.pdf"))

    # 증권사 필터
    if broker_filter:
        pdf_files = [f for f in pdf_files if broker_filter in str(f)]
        print(f"증권사 필터 '{broker_filter}': {len(pdf_files)}개")

    # 개수 제한
    if limit:
        pdf_files = pdf_files[:limit]
        print(f"처음 {limit}개만 처리")

    print(f"\n총 {len(pdf_files)}개 PDF 처리 시작\n")

    out_root = Path(output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    results = []

    for i, pdf_path in enumerate(pdf_files, 1):
        meta = parse_filename_meta(pdf_path)
        print(f"[{i:03d}/{len(pdf_files)}] {meta['broker']} | {meta['title'][:35]}")

        # 1. 로딩
        pages_raw = load_pdf(pdf_path)
        if not pages_raw:
            print(f"  ✗ 로딩 실패, 스킵")
            continue

        for page in pages_raw:
            page.metadata.update(meta)

        # 2. 정제
        pages_clean = clean_documents(list(pages_raw))  # 복사본 전달

        # 3. 청킹
        chunks = chunk_documents(pages_clean)

        print(f"  → 원본 {len(pages_raw)}p / 정제 {len(pages_clean)}p / {len(chunks)}청크")

        # 4. 파일 저장
        broker_dir = out_root / meta["broker"]
        broker_dir.mkdir(exist_ok=True)

        safe_name = "".join(
            c if c not in r'\/:*?"<>|' else "_"
            for c in pdf_path.stem
        )[:80]
        out_file = broker_dir / f"{safe_name}.txt"

        content = format_chunk_file(pdf_path, pages_raw, pages_clean, chunks)
        out_file.write_text(content, encoding="utf-8")

        results.append({
            "filename":    pdf_path.name,
            "broker":      meta["broker"],
            "sector":      meta["sector"],
            "date":        meta["date"],
            "pages_raw":   len(pages_raw),
            "pages_clean": len(pages_clean),
            "chunks":      len(chunks),
            "out_file":    str(out_file),
        })

    # 5. 전체 요약 저장
    summary_path = out_root / "summary.txt"
    summary_path.write_text(format_summary(results), encoding="utf-8")

    print(f"\n{'=' * 50}")
    print(f"완료! 결과 저장: {out_root.absolute()}")
    print(f"  PDF {len(results)}개 처리")
    print(f"  총 청크: {sum(r['chunks'] for r in results)}개")
    print(f"  요약 파일: {summary_path}")
    print(f"{'=' * 50}")


def main():
    parser = argparse.ArgumentParser(description="PDF별 청킹 결과 저장")
    parser.add_argument("--pdf-dir",  default="./data/reports", help="PDF 폴더 경로")
    parser.add_argument("--output",   default=OUTPUT_DIR,       help="결과 저장 폴더")
    parser.add_argument("--broker",   default="",               help="특정 증권사만 (예: DS투자증권)")
    parser.add_argument("--limit",    type=int, default=0,      help="처리할 PDF 최대 개수 (0=전체)")
    args = parser.parse_args()

    run(args.pdf_dir, args.output, args.broker, args.limit)


if __name__ == "__main__":
    main()
