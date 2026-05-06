"""
cleaner.py
증권사 리포트 텍스트 정제 모듈

처리 순서:
  0. HTML 태그 제거
  1. 머리글 / 바닥글 제거 (마크다운 헤딩/볼드 포함)
  2. 목차 블록 제거
  3. 그림/표 캡션 + 자료 출처 제거 (마크다운 볼드 포함)
  4. 표 / 차트 / 그래프 데이터 제거
  5. 법적 고지 문구 제거
  6. 페이지 번호 제거
  7. 마크다운 문법 기호 제거 (#, **, *, ---)
  8. 공백 / 줄바꿈 정리
  9. 너무 짧은 줄 제거 (차트 레이블, 날짜 조각 등 노이즈)

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
# 마크다운 인라인 기호 제거 헬퍼 (패턴 매칭 전처리용)
# ══════════════════════════════════════════════════════

def _strip_md_inline(s: str) -> str:
    """헤딩/볼드/이탤릭 마커를 벗겨낸 plain text 반환 (패턴 매칭용)."""
    s = re.sub(r"^#{1,6}\s*", "", s)
    s = re.sub(r"\*{1,3}([^*]*)\*{1,3}", r"\1", s)
    s = re.sub(r"_{1,2}([^_]*)_{1,2}", r"\1", s)
    return s.strip()


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
    if any(p.match(s) for p in _HEADER_FOOTER_RE):
        return True
    # 마크다운 헤딩/볼드로 감싼 경우도 체크 (예: ## 기업분석, # **하나증권**)
    s_plain = _strip_md_inline(s)
    if s_plain != s:
        return any(p.match(s_plain) for p in _HEADER_FOOTER_RE)
    return False


# ══════════════════════════════════════════════════════
# 2. 목차
# ══════════════════════════════════════════════════════

_TOC_START_RE = re.compile(
    r"^(목\s*차|Contents?|Table\s+of\s+Contents?|INDEX|차\s*례)\s*$"
    r"|^#+\s*\*{0,2}(목\s*차|Contents?|Table\s+of\s+Contents?|INDEX)\*{0,2}\s*$",
    re.IGNORECASE,
)

_TOC_ITEM_WITH_PAGE_RE = re.compile(
    r"^.{2,60}\s{2,}\d{1,3}\s*$"
)

_TOC_ITEM_DOTTED_RE = re.compile(
    r"^.{2,40}(\.{3,}|─{3,}|\-{3,})\s*\d{1,3}\s*$"
)

_TOC_ITEM_MD_RE = re.compile(
    r"^\*{0,2}[·•\s]*\d{1,3}(pg|p|페이지)?\*{0,2}\s*$",
    re.IGNORECASE,
)

_TOC_ITEM_ROMAN_RE = re.compile(
    r"^[ⅠⅡⅢⅣⅤⅥⅦⅧⅨⅩⅰⅱⅲⅳⅴ]+[\.]?\s*.{1,40}(\s+\d{1,3})?\s*$"
)

_TOC_SECTION_RE = re.compile(
    r"^\[.{2,10}\]\s*$"
)


def _is_toc_item(line: str) -> bool:
    s = line.strip()
    if not s:
        return True
    if _TOC_ITEM_WITH_PAGE_RE.match(s):
        return True
    if _TOC_ITEM_DOTTED_RE.match(s):
        return True
    if _TOC_ITEM_MD_RE.match(s):
        return True
    if _TOC_ITEM_ROMAN_RE.match(s):
        return True
    if _TOC_SECTION_RE.match(s):
        return True
    return False


def _remove_toc(text: str) -> str:
    lines  = text.split("\n")
    result = []
    in_toc = False
    i      = 0

    while i < len(lines):
        line = lines[i]
        s    = line.strip()

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

_CAPTION_RE = re.compile(
    r"^(그림|표|Figure|Fig\.?|Table|Chart|도표)\s*\d+[\.\-\s].*$",
    re.IGNORECASE,
)

_SOURCE_RE = re.compile(
    r"^(자료|출처|Source|주|Note|주석)\s*[:：].*$",
    re.IGNORECASE,
)

_TABLE_HEADER_RE = re.compile(
    r"^(업체명|종목명|회사명|종목코드|티커|Ticker|구분|항목|내용|비고|단위"
    r"|시가총액|종가|매출액|영업이익|순이익|PER|PBR|EPS|ROE|배당수익률"
    r"|DRAM|NAND|HBM|매출|이익|지배|비지배|연결|별도)\s*$",
    re.IGNORECASE,
)

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
        s_plain = _strip_md_inline(s)
        if _CAPTION_RE.match(s) or _CAPTION_RE.match(s_plain):
            continue
        if _BRACKET_CAPTION_RE.match(s) or _BRACKET_CAPTION_RE.match(s_plain):
            continue
        if _SOURCE_RE.match(s) or _SOURCE_RE.match(s_plain):
            continue
        if _TABLE_HEADER_RE.match(s) or _TABLE_HEADER_RE.match(s_plain):
            continue
        if re.match(r"^\(단위\s*[:：][^)]{1,20}\)\s*$", s) or re.match(r"^\(단위\s*[:：][^)]{1,20}\)\s*$", s_plain):
            continue
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

# 재무제표 항목명 패턴
_FINANCIAL_STMT_RE = re.compile(
    r"^(매출채권|재고자산|부채총계|자본총계|영업이익|당기순이익|매입채무"
    r"|유형자산|무형자산|이익잉여금|자본금|자본잉여금|비지배주주지분"
    r"|현금및현금성자산|단기금융자산|장기금융자산|관계기업|투자부동산"
    r"|차입금|사채|충당부채|기타유동|기타비유동|법인세|영업활동|투자활동"
    r"|재무활동|CAPEX|FCF|배당금|EPS|BPS|DPS|PER|PBR|EV|EBITDA|ROE|ROA|ROIC)",
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

    # 날짜 범위 패턴 ('22. 3~ '23. 5 등 차트 축 레이블)
    if re.fullmatch(r"[\d'\~\.\s/QHFE,:\-]+", s) and len(s) < 50:
        return True

    # 연도/월 축 레이블 ('23/1 '24/1 '25/1 등)
    if re.fullmatch(r"(\'?\d{2}[/\.\s]\d{0,2}\s*)+", s):
        return True

    # 재무제표 행: 항목명 + 숫자 3개 이상
    if _FINANCIAL_STMT_RE.match(s):
        nums = re.findall(r"-?\d[\d,\.]*", s)
        if len(nums) >= 2:
            return True

    return False


def _is_entity_name_block(segment: list[str]) -> bool:
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


# 재무제표 블록 감지 패턴
_FINANCIAL_BLOCK_START_RE = re.compile(
    r"^(재무상태표|손익계산서|현금흐름표|포괄손익계산서|Financial\s*(Data|Statement)|"
    r"주요\s*재무\s*지표|재무\s*요약|Financial\s*Summary|Income\s*Statement|"
    r"Balance\s*Sheet|Cash\s*Flow)",
    re.IGNORECASE,
)


def _remove_financial_statements(text: str) -> str:
    """
    재무제표 블록 통째 제거
    - 재무상태표/손익계산서/현금흐름표 헤더 감지 후 블록 제거
    - 숫자가 밀집된 줄이 5줄 이상 연속이면 재무제표로 판단
    """
    lines  = text.split("\n")
    result = []
    i      = 0

    while i < len(lines):
        line = lines[i]
        s    = line.strip()

        # 재무제표 헤더 감지
        if _FINANCIAL_BLOCK_START_RE.match(s):
            # 헤더 이후 숫자가 많은 줄들을 스킵
            j = i + 1
            while j < len(lines):
                next_s = lines[j].strip()
                if not next_s:
                    j += 1
                    continue
                # 숫자 비중 50% 이상이면 재무제표 내부
                num_ratio = len(re.findall(r"\d", next_s)) / max(len(next_s), 1)
                if num_ratio >= 0.3 or _is_table_chart_line(lines[j]):
                    j += 1
                else:
                    # 3줄 이상 제거됐으면 재무제표 블록으로 확정
                    if j - i >= 3:
                        break
                    else:
                        # 짧으면 헤더만 제거하고 나머지 유지
                        j = i + 1
                        break
            i = j
            continue

        # 숫자 밀집 줄이 5줄 이상 연속이면 재무제표로 판단
        if _is_table_chart_line(line):
            j = i + 1
            while j < len(lines) and (not lines[j].strip() or _is_table_chart_line(lines[j])):
                j += 1

            if j - i >= 5:
                i = j
                continue
            elif j - i >= 2:
                i = j
                continue
            else:
                prev_ok = bool(result and result[-1].strip() and not _is_table_chart_line(result[-1]))
                next_ok = j < len(lines) and lines[j].strip() and not _is_table_chart_line(lines[j])
                if prev_ok or next_ok:
                    i += 1
                    continue

        # 고유명사 나열 블록 감지
        else:
            j = i + 1
            while j < len(lines) and lines[j].strip() and not _is_table_chart_line(lines[j]):
                ns = lines[j].strip()
                tokens = ns.split()
                if len(tokens) <= 4 and len(ns) <= 50 and not re.search(r"[가-힣]{5,}", ns):
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
            j = i + 1
            while j < len(lines) and (not lines[j].strip() or _is_table_chart_line(lines[j])):
                j += 1

            if j - i >= 2:
                i = j
                continue
            else:
                prev_ok = bool(result and result[-1].strip() and not _is_table_chart_line(result[-1]))
                next_ok = j < len(lines) and lines[j].strip() and not _is_table_chart_line(lines[j])
                if prev_ok or next_ok:
                    i += 1
                    continue

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
    r"제공시점\s*현재\s*기관투자가.*?(\n\n|\Z)",
    r"동\s*자료의\s*추천종목.*?(\n\n|\Z)",
    r"[\uf0a0].*?(\n\n|\Z)",
    r"당사는\s*자료\s*작성일.*?(\n\n|\Z)",
    r"동\s*자료는\s*당사의\s*제작물.*?(\n\n|\Z)",
    r"당사는\s*본\s*자료\s*발간일.*?(\n\n|\Z)",
    r"제공시점\s*현재\s*기관투자가.+",
]

def _remove_disclaimers(text: str) -> str:
    for p in _DISCLAIMER_PATTERNS:
        text = re.sub(p, "\n", text, flags=re.DOTALL | re.IGNORECASE)

    # 면책조항 키워드 이후 전체 제거
    DISCLAIMER_TRIGGERS = [
        "Compliance",
        "제공시점 현재 기관투자가",
        "동 자료의 추천종목",
        "당사는 자료 작성일",
        "당사는 본 자료 발간일",
        "\uf0a0",
        "본 자료는 고객의 증권투자",
        "본 조사분석자료는",
        "투자의견 및 적용기준",
        "투자의견 비율 기준일",
        "매수 + 10%",
        "투자의견 및 목표주가 변동추이",
    ]

    lines = text.split("\n")
    result = []
    skip = False
    for line in lines:
        if "Compliance" in line:
            print(f"[DEBUG] Compliance 발견: {repr(line[:80])}")
            break
        if not skip and (
            any(trigger in line for trigger in DISCLAIMER_TRIGGERS)
            or ("당사는" in line and ("보유하고 있지 않습니다" in line or "사전 제공한 사실이 없습니다" in line))
            or ("동 자료" in line and ("분석사는" in line or "게시된 내용들은" in line or "금융투자분석사는" in line))
            or ("본인의 의견을 정확하게 반영" in line)
            or ("외부의 부당한 압력이나 간섭없이" in line)
        ):
            skip = True
        if not skip:
            result.append(line)
    return "\n".join(result)


# ══════════════════════════════════════════════════════
# 6. 페이지 번호
# ══════════════════════════════════════════════════════

def _remove_page_numbers(text: str) -> str:
    text = re.sub(r"^\s*-\s*\d+\s*-\s*$",               "", text, flags=re.MULTILINE)
    text = re.sub(r"^\s*\d+\s*/\s*\d+\s*$",              "", text, flags=re.MULTILINE)
    text = re.sub(r"^\s*\d{1,3}\s*$",                    "", text, flags=re.MULTILINE)
    text = re.sub(r"^\s*Page\s*\d+\s*(of\s*\d+)?\s*$",   "", text, flags=re.MULTILINE | re.IGNORECASE)
    text = re.sub(r"^#{4,6}\s*\d{1,3}\s*$",              "", text, flags=re.MULTILINE)
    return text


# ══════════════════════════════════════════════════════
# 7. 마크다운 문법 기호 제거 (pymupdf4llm 출력 정제)
# ══════════════════════════════════════════════════════

def _strip_markdown_syntax(text: str) -> str:
    text = re.sub(r"^#{1,6}\s+", "", text, flags=re.MULTILINE)
    text = re.sub(r"\*{3}([^*\n]*)\*{3}", r"\1", text)
    text = re.sub(r"\*{2}([^*\n]*)\*{2}", r"\1", text)
    text = re.sub(r"\*([^*\n]+)\*",       r"\1", text)
    text = re.sub(r"^[-*_]{3,}\s*$", "", text, flags=re.MULTILINE)
    text = re.sub(r"`([^`\n]+)`", r"\1", text)
    return text


# ══════════════════════════════════════════════════════
# 메인 정제 함수
# ══════════════════════════════════════════════════════

def clean_page(text: str) -> str:
    """
    단일 텍스트 정제 (raw 텍스트 입력 → 정제 텍스트 반환)
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

    # 4-1. 재무제표 블록 제거 (표/차트 제거보다 먼저)
    text = _remove_financial_statements(text)

    # 4-2. 표/차트/그래프 데이터
    text = _remove_tables_charts(text)

    # 5. 법적 고지
    text = _remove_disclaimers(text)

    # 6. 페이지 번호
    text = _remove_page_numbers(text)

    # 7. 마크다운 문법 기호 제거
    text = _strip_markdown_syntax(text)

    # 8. 공백/줄바꿈 정리
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)

    # 9. 너무 짧은 줄 제거 (차트 레이블, 날짜 조각 등 노이즈)
    lines = text.split("\n")
    lines = [l for l in lines if len(l.strip()) == 0 or len(l.strip()) >= 15]
    text = "\n".join(lines)

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