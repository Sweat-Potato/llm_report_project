"""
네이버 증권 산업분석 리포트 크롤러
=====================================
사용법:
    pip install requests beautifulsoup4 lxml

    # 1달치 수집 + PDF 다운로드
    python naver_industry_crawler.py

    # 날짜 범위 지정
    python naver_industry_crawler.py --from-date 2026-01-23 --to-date 2026-04-23

    # 메타데이터만 수집 (PDF 다운로드 X)
    python naver_industry_crawler.py --no-download

    # 특정 업종만 수집
    python naver_industry_crawler.py --sector IT
"""

import re
import time
import json
import argparse
import requests
from bs4 import BeautifulSoup
from pathlib import Path


# ── 설정 ──────────────────────────────────────────────────────────────────────

BASE_URL      = "https://finance.naver.com"
LIST_URL      = BASE_URL + "/research/industry_list.naver"

DEFAULT_FROM  = "2026-03-23"
DEFAULT_TO    = "2026-04-23"

SAVE_DIR      = Path(__file__).parent.parent.parent / "data/reports/reports_naver_industry"
METADATA_FILE = str(Path(__file__).parent.parent.parent / "data/reports/metadata/naver_industry_metadata.json")

DELAY_PAGE    = 2.0
DELAY_PDF     = 1.5

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Referer": "https://finance.naver.com/research/",
    "Accept-Language": "ko-KR,ko;q=0.9",
}


# ── 파싱 ──────────────────────────────────────────────────────────────────────

def parse_report_list(html: str) -> list:
    """산업분석 리포트 목록 파싱"""
    soup  = BeautifulSoup(html, "lxml")
    table = soup.find("table", class_="type_1")
    if not table:
        return []

    reports = []
    for row in table.find_all("tr"):
        cols = row.find_all("td")
        if len(cols) < 5:
            continue

        # 구분선/빈 행 스킵
        if cols[0].get("class") and any(
            c in ["blank_07", "blank_08", "blank_09", "division_line", "division_line_1"]
            for c in cols[0].get("class", [])
        ):
            continue

        try:
            # 업종
            sector = cols[0].get_text(strip=True)
            if not sector:
                continue

            # 제목 + nid
            title_a = cols[1].find("a")
            if not title_a:
                continue
            title    = title_a.get_text(strip=True)
            href     = title_a.get("href", "")
            nid_match = re.search(r"nid=(\d+)", href)
            nid      = nid_match.group(1) if nid_match else None

            # 증권사
            broker = cols[2].get_text(strip=True)

            # PDF 링크 (직접 URL)
            pdf_a   = cols[3].find("a")
            pdf_url = pdf_a.get("href") if pdf_a else None

            # 날짜
            date_str = cols[4].get_text(strip=True)

            # 조회수
            views = cols[5].get_text(strip=True) if len(cols) > 5 else ""

            reports.append({
                "nid":         nid,
                "sector":      sector,
                "title":       title,
                "broker":      broker,
                "date_str":    date_str,
                "views":       views,
                "pdf_url":     pdf_url,
                "source":      "naver_industry",
                "report_type": "industry",
                "local_path":  None,
            })

        except Exception:
            continue

    return reports


def get_last_page(html: str) -> int:
    """마지막 페이지 번호 파싱 (pgRR 클래스)"""
    soup  = BeautifulSoup(html, "lxml")
    pg_rr = soup.find("td", class_="pgRR")
    if pg_rr:
        a = pg_rr.find("a")
        if a:
            m = re.search(r"page=(\d+)", a.get("href", ""))
            if m:
                return int(m.group(1))
    return 1


# ── 크롤러 ────────────────────────────────────────────────────────────────────

def crawl_all_pages(
    from_date: str = DEFAULT_FROM,
    to_date:   str = DEFAULT_TO,
    sector:    str = "",
) -> list:
    """전체 페이지 크롤링"""
    session = requests.Session()
    session.headers.update(HEADERS)

    params = {
        "keyword":       "",
        "brokerCode":    "",
        "searchType":    "writeDate",
        "writeFromDate": from_date,
        "writeToDate":   to_date,
        "upjong":        sector,
        "x":             0,
        "y":             0,
        "page":          1,
    }

    # 1페이지 먼저 가져와서 총 페이지 수 파악
    res = session.get(LIST_URL, params=params, timeout=10)
    res.raise_for_status()
    res.encoding = "euc-kr"

    last_page = get_last_page(res.text)
    print(f"총 페이지 수: {last_page}페이지")

    all_reports = []

    for page in range(1, last_page + 1):
        print(f"  페이지 {page}/{last_page} 수집 중...", end=" ")

        if page > 1:
            params["page"] = page
            res = session.get(LIST_URL, params=params, timeout=10)
            res.raise_for_status()
            res.encoding = "euc-kr"

        reports = parse_report_list(res.text)
        all_reports.extend(reports)
        print(f"{len(reports)}건 (누계: {len(all_reports)}건)")

        time.sleep(DELAY_PAGE)

    return all_reports


# ── PDF 다운로드 ───────────────────────────────────────────────────────────────

def sanitize_filename(name: str) -> str:
    for ch in r'\/:*?"<>|':
        name = name.replace(ch, "_")
    return name.strip()[:80]


def download_pdf(session: requests.Session, item: dict, save_dir: Path):
    """PDF 1개 다운로드"""
    if not item.get("pdf_url"):
        return None

    broker   = sanitize_filename(item.get("broker", "unknown"))
    date_str = item.get("date_str", "").replace(".", "")
    title    = sanitize_filename(item.get("title", "report"))
    sector   = sanitize_filename(item.get("sector", ""))

    filename   = f"{date_str}_{broker}_{sector}_{title}.pdf"
    broker_dir = save_dir / broker
    broker_dir.mkdir(parents=True, exist_ok=True)
    save_path  = broker_dir / filename

    if save_path.exists():
        return str(save_path)

    try:
        res = session.get(item["pdf_url"], timeout=15, stream=True)
        res.raise_for_status()

        content_type = res.headers.get("Content-Type", "")
        if "html" in content_type.lower():
            print(f"    접근 불가 (로그인 필요?), 스킵")
            return None

        with open(save_path, "wb") as f:
            for chunk in res.iter_content(chunk_size=8192):
                f.write(chunk)

        time.sleep(DELAY_PDF)
        return str(save_path)

    except Exception as e:
        print(f"    다운로드 실패: {e}")
        return None


def download_all_pdfs(reports: list, save_dir: str = SAVE_DIR) -> list:
    """전체 PDF 일괄 다운로드"""
    session  = requests.Session()
    session.headers.update(HEADERS)
    base_dir = Path(save_dir)
    base_dir.mkdir(exist_ok=True)

    has_pdf = [r for r in reports if r.get("pdf_url")]
    print(f"\nPDF 다운로드 시작 — {len(has_pdf)}건 (전체 {len(reports)}건 중 PDF 있는 것)")

    success, fail, skip = 0, 0, 0

    for i, item in enumerate(reports, 1):
        if not item.get("pdf_url"):
            skip += 1
            continue

        print(f"  [{i}/{len(reports)}] {item['broker']:15s} [{item['sector']}] {item['title'][:30]}")
        path = download_pdf(session, item, base_dir)

        if path:
            item["local_path"] = path
            success += 1
        else:
            fail += 1

    print(f"\n다운로드 완료: {success}건 성공 / {fail}건 실패 / {skip}건 PDF없음")
    return reports


# ── 저장 + 요약 ───────────────────────────────────────────────────────────────

def save_metadata(reports: list, output_path: str = METADATA_FILE):
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(reports, f, ensure_ascii=False, indent=2)
    print(f"메타데이터 저장: {output_path} ({len(reports)}건)")


def print_summary(reports: list):
    broker_counts = {}
    sector_counts = {}

    for r in reports:
        b = r.get("broker", "unknown")
        s = r.get("sector", "unknown")
        broker_counts[b] = broker_counts.get(b, 0) + 1
        sector_counts[s] = sector_counts.get(s, 0) + 1

    print("\n" + "=" * 50)
    print(f"총 수집 리포트: {len(reports)}건")
    pdf_count = sum(1 for r in reports if r.get("pdf_url"))
    print(f"PDF 있는 리포트: {pdf_count}건")

    print("\n증권사별 리포트 수 (Top 10):")
    for broker, count in sorted(broker_counts.items(), key=lambda x: -x[1])[:10]:
        print(f"  {broker:20s} {count:4d}건")

    print("\n업종별 리포트 수 (Top 10):")
    for sector, count in sorted(sector_counts.items(), key=lambda x: -x[1])[:10]:
        print(f"  {sector:15s} {count:4d}건")
    print("=" * 50)


# ── 메인 ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="네이버 증권 산업분석 리포트 크롤러")
    parser.add_argument("--from-date",   default=DEFAULT_FROM,  help="시작 날짜 (YYYY-MM-DD)")
    parser.add_argument("--to-date",     default=DEFAULT_TO,    help="종료 날짜 (YYYY-MM-DD)")
    parser.add_argument("--sector",      default="",            help="업종 필터 (예: IT, 반도체)")
    parser.add_argument("--no-download", action="store_true",   help="PDF 다운로드 안 함")
    parser.add_argument("--save-dir",    default=SAVE_DIR,      help="PDF 저장 폴더")
    parser.add_argument("--output",      default=METADATA_FILE, help="메타데이터 저장 경로")
    args = parser.parse_args()

    print(f"네이버 증권 산업분석 크롤링 시작")
    print(f"기간: {args.from_date} ~ {args.to_date}\n")

    # 1. 목록 수집
    reports = crawl_all_pages(
        from_date = args.from_date,
        to_date   = args.to_date,
        sector    = args.sector,
    )

    # 2. PDF 다운로드
    if not args.no_download:
        reports = download_all_pdfs(reports, save_dir=args.save_dir)

    # 3. 저장
    save_metadata(reports, args.output)

    # 4. 요약
    print_summary(reports)


if __name__ == "__main__":
    main()
