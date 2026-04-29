"""
src/loader.py
PDF 로딩 전용 모듈 (pymupdf4llm 기반)

pymupdf4llm은 페이지 레이아웃을 분석해서 마크다운으로 변환한다.
- 2단 컬럼: 좌/우 컬럼을 순서대로 읽음 (기존 fitz는 섞임)
- 표: | col | col | 형식으로 변환 → cleaner.py에서 제거
- 이미지/그래프 블록: 자동 스킵
"""
import pymupdf4llm
from pathlib import Path
from langchain.schema import Document


def parse_filename_meta(pdf_path: Path) -> dict:
    """
    파일명에서 메타데이터 파싱
    예: 260414_DS투자증권_반도체_낸(NAND)붐온.pdf
        → date, broker, sector, title
    """
    parts = pdf_path.stem.split("_", 3)
    return {
        "date":     parts[0] if len(parts) > 0 else "",
        "broker":   parts[1] if len(parts) > 1 else pdf_path.parent.name,
        "sector":   parts[2] if len(parts) > 2 else "",
        "title":    parts[3] if len(parts) > 3 else pdf_path.stem,
        "pdf_path": str(pdf_path),
        "source":   "naver_industry",
    }


def load_pdf(pdf_path: str | Path) -> list[Document]:
    """
    단일 PDF 로딩 (pymupdf4llm)

    page_chunks=True → 페이지별 dict 리스트 반환
    각 dict: {"metadata": {"page": int, ...}, "text": "마크다운 텍스트"}
    """
    try:
        pages_data = pymupdf4llm.to_markdown(
            str(pdf_path),
            page_chunks=True,
            show_progress=False,
        )

        pages = []
        for page_data in pages_data:
            text = page_data.get("text", "").strip()
            if not text:
                continue

            page_meta = page_data.get("metadata", {})
            pages.append(Document(
                page_content=text,
                metadata={
                    "page":        page_meta.get("page", 0),
                    "total_pages": len(pages_data),
                    "source":      str(pdf_path),
                }
            ))

        return pages

    except Exception as e:
        print(f"  로딩 실패 ({Path(pdf_path).name}): {e}")
        return []


def load_all_pdfs(pdf_dir: str | Path) -> list[Document]:
    """
    폴더 전체 PDF 로딩 → 메타데이터 포함된 Document 리스트 반환
    정제(clean)는 cleaner.py에서 별도 처리
    """
    pdf_files = list(Path(pdf_dir).rglob("*.pdf"))
    print(f"PDF 발견: {len(pdf_files)}개")

    all_pages = []

    for pdf_path in pdf_files:
        pages = load_pdf(pdf_path)
        if not pages:
            continue

        meta = parse_filename_meta(pdf_path)
        for page in pages:
            page.metadata.update(meta)

        all_pages.extend(pages)
        print(f"  ✓ [{meta['broker']}] [{meta['sector']}] {meta['title'][:30]} ({len(pages)}p)")

    print(f"\n총 {len(pdf_files)}개 PDF / {len(all_pages)}페이지 로딩 완료\n")
    return all_pages
