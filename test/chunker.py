"""
src/chunker.py
구조 기반 청킹 모듈
"""
import re
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter


# ── 섹션 헤더 패턴 ────────────────────────────────────────────────────────────
# pymupdf4llm은 마크다운 헤더(##, ###)를 자동 생성하므로 최우선 패턴으로 추가

SECTION_PATTERNS = [
    r"^#{1,4}\s+.+",           # pymupdf4llm 마크다운 헤더 (## ~ #### 수준까지)
    r"^[■◆▶●]\s*.+",
    r"^[Ⅰ-Ⅹ]+[\.\s].+",
    r"^\d+[\.\)]\s*[가-힣A-Z].+",
    r"^[①-⑩]\s*.+",
]

_SECTION_RES = [re.compile(p) for p in SECTION_PATTERNS]


def is_section_header(line: str) -> bool:
    line = line.strip()
    if not line:
        return False
    return any(p.match(line) for p in _SECTION_RES)


# ── 구조 기반 분할 ────────────────────────────────────────────────────────────

def split_by_structure(text: str, max_chunk_size: int = 600) -> list[dict]:
    """
    섹션 헤더를 기준으로 텍스트 분할
    → 헤더가 없으면 RecursiveCharacterTextSplitter로 폴백
    """
    lines           = text.split("\n")
    chunks          = []
    current_section = "개요"
    current_lines   = []

    for line in lines:
        if is_section_header(line):
            if current_lines:
                content = "\n".join(current_lines).strip()
                if content:
                    chunks.append({"section": current_section, "content": content})
            current_section = line.strip()
            current_lines   = []
        else:
            current_lines.append(line)

    if current_lines:
        content = "\n".join(current_lines).strip()
        if content:
            chunks.append({"section": current_section, "content": content})

    # 섹션이 너무 길면 문단 단위로 추가 분할
    final_chunks = []
    for chunk in chunks:
        content = chunk["content"]
        if len(content) <= max_chunk_size:
            final_chunks.append(chunk)
        else:
            paragraphs  = content.split("\n\n")
            sub_content = ""
            for para in paragraphs:
                if len(sub_content) + len(para) > max_chunk_size and sub_content:
                    final_chunks.append({"section": chunk["section"], "content": sub_content.strip()})
                    sub_content = para
                else:
                    sub_content += "\n\n" + para
            if sub_content.strip():
                final_chunks.append({"section": chunk["section"], "content": sub_content.strip()})

    return final_chunks


# ── Document 청킹 ─────────────────────────────────────────────────────────────

def chunk_documents(
    pages:          list[Document],
    max_chunk_size: int = 600,
    chunk_overlap:  int = 60,
) -> list[Document]:
    """
    페이지 목록 → 구조 기반 청킹 → Document 리스트 반환
    - 페이지 경계 무시하고 문서 단위로 합쳐서 처리
    - 섹션 헤더 감지 (## 마크다운 포함) → 없으면 RecursiveCharacterTextSplitter 폴백

    max_chunk_size: 청크 최대 글자 수 (기본 1000)
    chunk_overlap:  RecursiveTextSplitter 폴백 시 오버랩 글자 수 (기본 150)
    """
    if not pages:
        return []

    # 같은 PDF의 페이지들을 합치기 (pdf_path 기준)
    pdf_groups: dict[str, list[Document]] = {}
    for page in pages:
        key = page.metadata.get("pdf_path", "unknown")
        pdf_groups.setdefault(key, []).append(page)

    all_chunks = []

    for pdf_path, pdf_pages in pdf_groups.items():
        full_text = "\n\n".join(
            p.page_content for p in pdf_pages if p.page_content.strip()
        )
        base_meta = pdf_pages[0].metadata.copy()

        # 구조 기반 청킹 시도
        sections = split_by_structure(full_text, max_chunk_size)

        # 섹션이 전부 "개요"면 헤더가 없는 문서 → RecursiveCharacterTextSplitter 폴백
        unique_sections = set(s["section"] for s in sections)
        if unique_sections == {"개요"} or len(sections) <= 2:
            splitter = RecursiveCharacterTextSplitter(
                chunk_size    = max_chunk_size,
                chunk_overlap = chunk_overlap,
                separators    = ["\n\n", "\n", ".", " ", ""],
            )
            raw_chunks = splitter.split_text(full_text)
            sections   = [{"section": "개요", "content": c} for c in raw_chunks]

        # Document 생성
        for i, sec in enumerate(sections):
            if not sec["content"].strip():
                continue
            chunk_meta = {
                **base_meta,
                "section":      sec["section"],
                "chunk_index":  i,
                "total_chunks": len(sections),
            }
            all_chunks.append(Document(
                page_content=sec["content"],
                metadata=chunk_meta,
            ))

    return all_chunks
