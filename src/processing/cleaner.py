"""
cleaner.py
증권사 리포트 텍스트 정제 모듈

처리 순서:
  0. HTML 태그 제거
  1. 머리글 / 바닥글 제거
  2. 목차 블록 제거
  3. 그림/표 캡션 + 자료 출처 제거
  4. 표 / 차트 / 그래프 데이터 제거
  5. 법적 고지 문구 제거
  6. 페이지 번호 제거
  7. 공백 / 줄바꿈 정리

사용법:
    # 페이지 단위 Document 리스트 정제 (Loader와 연동 시)
    from src.processing.cleaner import clean_documents
    cleaned_pages = clean_documents(pages)

    # reports_cache.json 리스트 일괄 정제 (chunking 파이프라인 연동 시)
    from src.processing.cleaner import clean_reports
    cleaned_reports = clean_reports(reports)
"""

import re
import json
from pathlib import Path
from langchain_core.documents import Document

BASE_DIR        = Path(__file__).parent.parent.parent
CLEAN_CACHE_PATH = BASE_DIR / "data" / "clean_text" / "clean_text.json"


# ══════════════════════════════════════════════════════
# 0. HTML 태그 제거
# ══════════════════════════════════════════════════════

def _remove_html_tags(text: str) -> str:
    """PDF 파싱 결과물에 섞인 HTML 태그 제거"""
    text = re.sub(r"<[^>]+>[^<]*<\/[^>]+>", "", text)
    text = re.sub(r"<[^>]+>", "", text)
    return text


# ══════════════════════════════════════════════════════
# 1. 머리글 / 바닥글
# ══════════════════════════════════════════════════════

_HEADER_FOOTER_PATTERNS = [
    r"^.{0,30}증권\s*\d{4}\.\d{2}\.\d{2}.*$",
    r"^(Company|Industry|Sector|Economy|Sector\s+Report)\s*(Report|Analysis)?\s*$",
    r"^(기업분석|산업분석|경제분석|투자전략)\s*$",
    r"^[A-Z\s&]{3,30}SECURITIES?\s*$",
    r"^DS\s+INVESTMENT\s*&?\s*SECURITIES?\s*$",
    r"^(하나|미래에셋|삼성|신한|키움|대신|교보|한화|SK|IBK|유진|유안타|메리츠).{0,5}증권\s*$",
    r"^[■▶]?\s*(담당\s*)?애널리스트.*$",
    r"^\s*\w+\s+\w+\s+\d{2,3}-\d{3,4}-\d{4}\s*$",
    r"^\s*\d{4}[.\-/]\d{2}[.\-/]\d{2}\s*$",
    r"^\s*\d{2}[.\-/]\d{2}[.\-/]\d{2}\s*$",
    # 섹터 레이블 단독 줄 (예: "반도체/장비", "건설/부동산")
    r"^[가-힣]{2,6}[\/·][가-힣]{2,6}\s*$",
]
_HEADER_FOOTER_RE = [re.compile(p, re.IGNORECASE) for p in _HEADER_FOOTER_PATTERNS]

def _is_header_footer(line: str) -> bool:
    s = line.strip()
    if not s:
        return False
    return any(p.match(s) for p in _HEADER_FOOTER_RE)


# ══════════════════════════════════════════════════════
# 2. 목차
# ══════════════════════════════════════════════════════

# 명시적 목차 헤더
_TOC_START_RE = re.compile(
    r"^(목\s*차|Contents?|Table\s+of\s+Contents?|INDEX|차\s*례)\s*$",
    re.IGNORECASE,
)

# 목차 항목: 텍스트 + 공백 2개 이상 + 페이지번호 로 끝나는 줄
# 예: "AI 패권전쟁과 중국의...   06"
_TOC_ITEM_WITH_PAGE_RE = re.compile(
    r"^.{2,60}\s{2,}\d{1,3}\s*$"
)

# 목차 항목: 점선/대시로 연결
# 예: "제목.......10"
_TOC_ITEM_DOTTED_RE = re.compile(
    r"^.{2,40}(\.{3,}|─{3,}|\-{3,})\s*\d{1,3}\s*$"
)


def _is_toc_item(line: str) -> bool:
    s = line.strip()
    if not s:
        return True
    if _TOC_ITEM_WITH_PAGE_RE.match(s):
        return True
    if _TOC_ITEM_DOTTED_RE.match(s):
        return True
    return False


def _remove_toc(text: str) -> str:
    """
    목차 블록 제거
    1. 명시적 헤더(목차/Contents) 이후 항목 제거
    2. 페이지 번호로 끝나는 줄이 3개 이상 연속이면 묵시적 목차로 판단
    """
    lines  = text.split("\n")
    result = []
    in_toc = False
    i      = 0

    while i < len(lines):
        line = lines[i]
        s    = line.strip()

        # 명시적 목차 헤더
        if _TOC_START_RE.match(s):
            in_toc = True
            i += 1
            continue

        if in_toc:
            if _is_toc_item(line):
                i += 1
                continue
            else:
                in_toc = False

        # 묵시적 목차 감지: 페이지 번호로 끝나는 줄이 3개 이상 연속
        if _TOC_ITEM_WITH_PAGE_RE.match(s):
            j = i + 1
            while j < len(lines) and (
                not lines[j].strip() or _TOC_ITEM_WITH_PAGE_RE.match(lines[j].strip())
            ):
                j += 1
            if j - i >= 3:
                i = j
                continue

        result.append(line)
        i += 1

    return "\n".join(result)


# ══════════════════════════════════════════════════════
# 3. 그림/표 캡션 + 자료 출처
# ══════════════════════════════════════════════════════

# 그림/표 캡션
_CAPTION_RE = re.compile(
    r"^(그림|표|Figure|Fig\.?|Table|Chart|도표)\s*\d+[\.\-\s].*$",
    re.IGNORECASE,
)

# 자료 출처
_SOURCE_RE = re.compile(
    r"^(자료|출처|Source|주|Note|주석)\s*[:：].*$",
    re.IGNORECASE,
)

# 표 컬럼 헤더 레이블
_TABLE_HEADER_RE = re.compile(
    r"^(업체명|종목명|회사명|종목코드|티커|Ticker|구분|항목|내용|비고|단위"
    r"|시가총액|종가|매출액|영업이익|순이익|PER|PBR|EPS|ROE|배당수익률"
    r"|DRAM|NAND|HBM|매출|이익|지배|비지배|연결|별도)\s*$",
    re.IGNORECASE,
)

# [도표N] 형식 캡션
_BRACKET_CAPTION_RE = re.compile(
    r"^[\[\(]\s*(그림|표|도표|Figure|Fig\.?|Table|Chart)\s*\d+[\]\)\s].*$",
    re.IGNORECASE,
)


def _remove_captions_and_sources(text: str) -> str:
    """그림/표 캡션 + 자료 출처 + 표 헤더 레이블 제거"""
    lines  = text.split("\n")
    result = []
    for line in lines:
        s = line.strip()
        if not s:
            result.append(line)
            continue
        if _CAPTION_RE.match(s):
            continue
        if _BRACKET_CAPTION_RE.match(s):
            continue
        if _SOURCE_RE.match(s):
            continue
        if _TABLE_HEADER_RE.match(s):
            continue
        # 단위 표시 줄: "(단위: 십억원)" 등
        if re.match(r"^\(단위\s*[:：][^)]{1,20}\)\s*$", s):
            continue
        # 단독 연도/분기 코드: "2026E", "1Q25", "2027F" 등
        if re.fullmatch(r"\d{4}[EF]?|\d[QH]\d{2}", s):
            continue
        result.append(line)
    return "\n".join(result)


# ══════════════════════════════════════════════════════
# 4. 표 / 차트 / 그래프 데이터
# ══════════════════════════════════════════════════════

_NUM_TOKEN_RE = re.compile(
    r"^("
    r"\d{1,4}[QHF]?\d{0,2}E?"
    r"|\d{1,3}(,\d{3})+"
    r"|\-?\d+\.?\d*%?"
    r"|[A-Z]{1,6}\d{0,4}"
    r"|\d+배|\d+(원|억|조|pt|%)"
    r"|N/A|n\.a\.|na"
    r")$",
    re.IGNORECASE,
)

def _is_table_chart_line(line: str) -> bool:
    s = line.strip()
    if not s:
        return False
    # 숫자·기호만
    if re.fullmatch(r"[\d\s,.\-\+%\(\)\/×÷±≈~→←↑↓]+", s):
        return True
    # 파이프 표
    if s.count("|") >= 2:
        return True
    # 구분선
    if re.fullmatch(r"[\-─=\+\|\s]{3,}", s):
        return True
    # 탭 구분 숫자 나열
    if "\t" in s and len(re.findall(r"\d", s)) > len(s) * 0.3:
        return True
    # 숫자/금융약어 60% 이상
    tokens = s.split()
    if 2 <= len(tokens) <= 12:
        num_count = sum(1 for t in tokens if _NUM_TOKEN_RE.match(t))
        if num_count >= len(tokens) * 0.6:
            return True
    # 단위 레이블
    if len(s) <= 40 and re.fullmatch(r"[\(\)a-zA-Z가-힣%,\s·/]+", s):
        if re.search(r"\(.{1,15}\)", s):
            return True
    return False


def _is_entity_name_block(segment: list[str]) -> bool:
    """
    연속 줄이 고유명사(업체명, 브랜드명) 나열인지 판단
    예: Synopsys / Cadence / Siemens / Empyrean
    """
    if len(segment) < 2:
        return False
    entity_count = 0
    for s in segment:
        tokens = s.split()
        if (
            1 <= len(tokens) <= 4
            and len(s) <= 50
            and not re.search(r"[.!?:;]", s)
            and not re.search(r"\d{4,}", s)
        ):
            entity_count += 1
    return entity_count >= len(segment) * 0.8


def _remove_tables_charts(text: str) -> str:
    """
    표/차트 블록 제거
    - 연속 2줄 이상 → 통째 제거
    - 단독 1줄     → 앞뒤 문맥 보고 제거
    - 업체명 나열 블록 → 표 내부로 판단하고 제거
    """
    lines  = text.split("\n")
    result = []
    i      = 0

    while i < len(lines):
        line = lines[i]

        if not line.strip():
            result.append(line)
            i += 1
            continue

        if _is_table_chart_line(line):
            # 연속 범위 파악
            j = i + 1
            while j < len(lines) and (not lines[j].strip() or _is_table_chart_line(lines[j])):
                j += 1

            if j - i >= 2:
                # 연속 블록 → 통째 제거
                i = j
                continue
            else:
                # 단독 줄 → 앞뒤 문맥 확인
                prev_ok = bool(result and result[-1].strip() and not _is_table_chart_line(result[-1]))
                next_ok = j < len(lines) and lines[j].strip() and not _is_table_chart_line(lines[j])
                if prev_ok or next_ok:
                    i += 1
                    continue

        # 고유명사 나열 블록 감지 (업체명 목록)
        else:
            j = i + 1
            while j < len(lines) and lines[j].strip() and not _is_table_chart_line(lines[j]):
                s = lines[j].strip()
                tokens = s.split()
                if len(tokens) <= 4 and len(s) <= 50 and not re.search(r"[가-힣]{5,}", s):
                    j += 1
                else:
                    break

            segment = [lines[k].strip() for k in range(i, j) if lines[k].strip()]
            if len(segment) >= 3 and _is_entity_name_block(segment):
                i = j
                continue

        result.append(line)
        i += 1

    return "\n".join(result)


# ══════════════════════════════════════════════════════
# 5. 법적 고지 / 면책 문구
# ══════════════════════════════════════════════════════

_DISCLAIMER_PATTERNS = [
    r"본 자료는 투자 참고용.*?(\n\n|\Z)",
    r"당사는 이 자료.*?(\n\n|\Z)",
    r"이 자료에 게재된.*?(\n\n|\Z)",
    r"본 조사자료는.*?(\n\n|\Z)",
    r"Compliance\s*(Notice|Rule).*?(\n\n|\Z)",
    r"본 보고서는 고객의 투자를.*?(\n\n|\Z)",
    r"투자등급\s*및\s*적용\s*기준.*?(\n\n|\Z)",
    r"Analyst\s+Certification.*?(\n\n|\Z)",
    r"IMPORTANT\s+DISCLOSURES.*?(\n\n|\Z)",
    r"본 자료에\s*기재된\s*내용들은.*?(\n\n|\Z)",
]

def _remove_disclaimers(text: str) -> str:
    for p in _DISCLAIMER_PATTERNS:
        text = re.sub(p, "\n", text, flags=re.DOTALL | re.IGNORECASE)
    return text


# ══════════════════════════════════════════════════════
# 6. 페이지 번호
# ══════════════════════════════════════════════════════

def _remove_page_numbers(text: str) -> str:
    text = re.sub(r"^\s*-\s*\d+\s*-\s*$",               "", text, flags=re.MULTILINE)
    text = re.sub(r"^\s*\d+\s*/\s*\d+\s*$",              "", text, flags=re.MULTILINE)
    text = re.sub(r"^\s*\d{1,3}\s*$",                    "", text, flags=re.MULTILINE)
    text = re.sub(r"^\s*Page\s*\d+\s*(of\s*\d+)?\s*$",   "", text, flags=re.MULTILINE | re.IGNORECASE)
    # pymupdf4llm 마크다운 페이지번호: "###### 04", "####### 12" 등
    text = re.sub(r"^#{4,6}\s*\d{1,3}\s*$",              "", text, flags=re.MULTILINE)
    return text


# ══════════════════════════════════════════════════════
# 메인 정제 함수
# ══════════════════════════════════════════════════════

def clean_page(text: str) -> str:
    """
    단일 텍스트 정제 (raw 텍스트 입력 → 정제 텍스트 반환)
    페이지 단위 또는 full_text 전체 모두 적용 가능
    """
    # 0. HTML 태그 제거
    text = _remove_html_tags(text)

    # 1. 머리글/바닥글
    lines = text.split("\n")
    lines = [l for l in lines if not _is_header_footer(l)]
    text  = "\n".join(lines)

    # 2. 목차
    text = _remove_toc(text)

    # 3. 그림/표 캡션 + 자료 출처
    text = _remove_captions_and_sources(text)

    # 4. 표/차트/그래프 데이터
    text = _remove_tables_charts(text)

    # 5. 법적 고지
    text = _remove_disclaimers(text)

    # 6. 페이지 번호
    text = _remove_page_numbers(text)

    # 7. 공백/줄바꿈 정리
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    
    return text.strip()


def clean_documents(pages: list[Document]) -> list[Document]:
    """
    Document 리스트 전체 정제 (Loader와 연동 시 사용)
    정제 후 50자 미만 페이지는 노이즈로 제거
    """
    cleaned = []
    removed = 0

    for page in pages:
        page.page_content = clean_page(page.page_content)
        if len(page.page_content) < 50:
            removed += 1
            continue
        cleaned.append(page)

    print(f"정제 완료: {len(cleaned)}페이지 유지 / {removed}페이지 제거")
    return cleaned


# ══════════════════════════════════════════════════════
# 파이프라인 연동 — reports_cache.json 일괄 처리
# ══════════════════════════════════════════════════════

def clean_reports(
    reports: list[dict],
    verbose: bool = True,
) -> list[dict]:
    """
    reports_cache.json 로드 결과를 일괄 정제.
    각 report의 full_text → clean_page() → clean_text 필드 추가.

    chunking 전 단계에서 호출:
        reports = load_cache(...)
        reports = clean_reports(reports)
        result  = chunk_reports(reports)  # clean_text 우선 사용
    """
    if verbose:
        print(f"🧹 클리닝 시작 ({len(reports)}개 리포트)\n")

    results = []
    for report in reports:
        original    = report.get("full_text", "")
        cleaned     = clean_page(original)

        removed     = len(original) - len(cleaned)
        ratio       = removed / len(original) * 100 if original else 0

        if verbose:
            print(f"  📄 {report.get('filename', '')[:55]}")
            print(f"     {len(original):,}자 → {len(cleaned):,}자  "
                  f"(-{removed:,}자, {ratio:.1f}% 제거)")

        result = dict(report)
        result["clean_text"]        = cleaned
        result["clean_char_count"]  = len(cleaned)
        result["noise_removed_pct"] = round(ratio, 1)
        results.append(result)

    if verbose:
        total_orig    = sum(len(r.get("full_text", ""))   for r in reports)
        total_cleaned = sum(r.get("clean_char_count", 0)  for r in results)
        removed_total = total_orig - total_cleaned
        print(f"\n✅ 클리닝 완료")
        print(f"   원본 합계  : {total_orig:,}자")
        print(f"   정제 합계  : {total_cleaned:,}자")
        print(f"   제거 합계  : {removed_total:,}자 ({removed_total / total_orig * 100:.1f}%)")

    save_clean_texts(results)
    return results


def save_clean_texts(
    reports: list[dict],
    cache_path: str = str(CLEAN_CACHE_PATH),
) -> None:
    out = Path(cache_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    data = [{"filename": r["filename"], "clean_text": r["clean_text"]} for r in reports]
    with open(out, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"   저장: {out} ({len(data)}개)")


# ══════════════════════════════════════════════════════
# 단독 실행 (테스트)
# ══════════════════════════════════════════════════════

if __name__ == "__main__":
    import json
    from pathlib import Path

    BASE_DIR   = Path(__file__).parent.parent.parent
    CACHE_PATH = BASE_DIR / "data" / "loader_metadata" / "reports_cache.json"

    with open(CACHE_PATH, encoding="utf-8") as f:
        reports = json.load(f)

    cleaned = clean_reports(reports, verbose=True)

    # 첫 번째 리포트 전후 비교
    orig         = reports[0]["full_text"]
    cleaned_text = cleaned[0]["clean_text"]

    print("\n" + "=" * 60)
    print("원본 텍스트 (앞 300자)")
    print("=" * 60)
    print(repr(orig[:300]))

    print("\n" + "=" * 60)
    print("정제 텍스트 (앞 300자)")
    print("=" * 60)
    print(repr(cleaned_text[:300]))

    print("\n" + "=" * 60)
    print("정제 텍스트 (마지막 200자) — 면책조항 제거 확인")
    print("=" * 60)
    print(repr(cleaned_text[-200:]))
