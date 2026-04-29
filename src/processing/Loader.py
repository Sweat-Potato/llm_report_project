"""
Loader.py
증권사 리포트 PDF 로더 (pymupdf4llm 기반)

[출력 형식]
  list[dict] (report 단위)
"""

import re
import json
from pathlib import Path
from openai import OpenAI
import pymupdf4llm
import os
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

BASE_DIR    = Path(__file__).parent.parent.parent
REPORTS_DIR = BASE_DIR / "data" / "reports" / "reports_naver_industry"
CACHE_PATH  = BASE_DIR / "data" / "loader_metadata" / "reports_cache.json"


# -----------------------------------------
# 1. 파일명 파싱
#    형식: YYMMDD_증권사명_섹터_제목.pdf
# -----------------------------------------
def parse_filename(filename: str, folder_name: str) -> dict:
    stem  = Path(filename).stem
    parts = stem.split("_")

    source_firm = folder_name
    date = sector = title = None

    date_raw = parts[0] if parts else ""
    if re.match(r'^\d{6}$', date_raw):
        date = f"20{date_raw[:2]}-{date_raw[2:4]}-{date_raw[4:]}"

    if len(parts) >= 2 and parts[1] != folder_name:
        print(f"    WARNING: 증권사명 불일치 - 폴더: {folder_name} / 파일명: {parts[1]}")

    if len(parts) >= 3:
        sector = parts[2]
    if len(parts) >= 4:
        title = " ".join(parts[3:]).strip()

    return {"source_firm": source_firm, "report_date": date, "sector": sector, "title": title}


# -----------------------------------------
# 2. LLM 메타데이터 추출
# -----------------------------------------
def extract_metadata_with_llm(first_page_text: str, source_firm: str) -> dict:
    prompt = f"""아래는 '{source_firm}' 증권사 리포트 텍스트입니다.
다음 정보를 JSON으로 추출하세요. 없으면 null.

{{
  "analyst": "애널리스트 이름 (여러 명이면 쉼표 구분, 성명만 추출)",
  "rating": "섹터 전체 투자의견 (비중확대/비중축소/중립/매수/매도 중 하나)",
  "target_price": 목표주가 숫자 (단일 종목 리포트만, 섹터 리포트면 null),
  "report_type": "섹터분석 or 종목분석 or 매크로 or 전략"
}}

주의:
- analyst: 이메일(@) 또는 전화번호(02-, 031- 등) 바로 앞에 있는 한글 이름
- rating: 리포트 전체 섹터 투자의견만 (개별 종목 투자의견 제외)
- target_price: 섹터 리포트는 null
- JSON만 반환 (설명 없이)

텍스트:
{first_page_text[:3000]}"""

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )
        raw = response.choices[0].message.content.strip()
        raw = re.sub(r'^```json\s*', '', raw)
        raw = re.sub(r'\s*```$', '', raw)
        result = json.loads(raw)
        return {
            "analyst":      result.get("analyst"),
            "rating":       result.get("rating"),
            "target_price": int(result["target_price"]) if result.get("target_price") else None,
            "report_type":  result.get("report_type", "섹터분석"),
        }
    except Exception as e:
        print(f"    WARNING: LLM 추출 실패: {e}")
        return {"analyst": None, "rating": None, "target_price": None, "report_type": "섹터분석"}


# -----------------------------------------
# 3. 단일 PDF 로드 (pymupdf4llm)
# -----------------------------------------
def load_single_pdf(pdf_path: str) -> dict:
    pdf_path    = Path(pdf_path)
    filename    = pdf_path.name
    folder_name = pdf_path.parent.name

    file_meta = parse_filename(filename, folder_name)

    try:
        pages_data = pymupdf4llm.to_markdown(str(pdf_path), page_chunks=True, show_progress=False)
    except Exception as e:
        print(f"    ERROR: pymupdf4llm 실패 ({filename}): {e}")
        return {}

    total_pages = len(pages_data)
    pages_text  = []

    for page_data in pages_data:
        text = page_data.get("text", "").strip()
        if not text:
            continue
        page_meta = page_data.get("metadata", {})
        pages_text.append({"page_num": page_meta.get("page", 0) + 1, "text": text})

    full_text = "\n\n".join(p["text"] for p in pages_text)

    first_pages_text = "\n\n".join(
        p["text"] for p in pages_text[:3] if len(p["text"]) > 50
    )
    llm_meta = extract_metadata_with_llm(first_pages_text, file_meta["source_firm"])

    return {
        "filename":    filename,
        "pdf_path":    str(pdf_path),
        "total_pages": total_pages,
        **file_meta,
        **llm_meta,
        "full_text":   full_text,
    }


# -----------------------------------------
# 4. 전체 폴더 로드 (증분 캐시)
# -----------------------------------------
def load_all_reports(
    reports_dir: str = str(REPORTS_DIR),
    cache_path:  str = str(CACHE_PATH),
) -> list[dict]:
    reports_dir = Path(reports_dir)
    pdf_files   = sorted(reports_dir.glob("**/*.pdf"))

    if not pdf_files:
        print(f"ERROR: {reports_dir} 에 PDF 없음")
        return []

    print(f"총 {len(pdf_files)}개 PDF 발견")

    cached_reports: list[dict] = []
    cached_filenames: set[str] = set()

    if Path(cache_path).exists():
        cached_reports   = load_cache(cache_path)
        cached_filenames = {r["filename"] for r in cached_reports}
        print(f"캐시 로드: {len(cached_reports)}개 기존 리포트")

    new_pdf_files = [f for f in pdf_files if f.name not in cached_filenames]

    if not new_pdf_files:
        print("새로 추가된 PDF 없음 -> 캐시 그대로 사용\n")
        return cached_reports

    print(f"신규 PDF {len(new_pdf_files)}개 파싱 시작\n")

    new_reports: list[dict] = []
    for pdf_path in new_pdf_files:
        print(f"  로딩 중: [{pdf_path.parent.name}] {pdf_path.name}")
        try:
            report = load_single_pdf(str(pdf_path))
            if not report:
                continue
            new_reports.append(report)
            print(f"  OK {report['source_firm']} | {report['report_type']} | "
                  f"{report['total_pages']}p | 투자의견: {report['rating'] or '-'} | "
                  f"목표주가: {report['target_price'] or '-'} | 애널리스트: {report['analyst'] or '-'}")
        except Exception as e:
            print(f"  ERROR: {pdf_path.name} 로드 실패: {e}")

    all_reports = cached_reports + new_reports
    save_cache(all_reports, cache_path)
    print(f"\n총 {len(all_reports)}개 리포트 (기존 {len(cached_reports)} + 신규 {len(new_reports)})")
    return all_reports


# -----------------------------------------
# 5. 캐시 저장 / 불러오기
# -----------------------------------------
def save_cache(reports: list[dict], cache_path: str = str(CACHE_PATH)) -> None:
    Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(reports, f, ensure_ascii=False, indent=2)
    print(f"캐시 저장: {cache_path} ({len(reports)}개)")


def load_cache(cache_path: str = str(CACHE_PATH)) -> list[dict]:
    with open(cache_path, encoding="utf-8") as f:
        return json.load(f)


# -----------------------------------------
# 단독 실행
# -----------------------------------------
if __name__ == "__main__":
    reports = load_all_reports()
    if reports:
        print(f"\n총 {len(reports)}개 리포트 로드 완료")
        for r in reports[:3]:
            print(f"  {r['source_firm']} | {r['sector']} | {r['report_date']} | {len(r['full_text'])}자")
