"""
loader.py
증권사 리포트 PDF 로더 (LLM 기반 메타데이터 추출)

파일명 형식: 날짜_증권사_섹터_제목.pdf
  예) 260330_DS투자증권_자동차_지역별 정책 점검.pdf
      20260420_industry_649803000.pdf

역할 분담:
  폴더명  → 증권사명  (규칙 기반, 100% 정확)
  파일명  → 날짜, 섹터, 제목 (규칙 기반)
  LLM     → 애널리스트, 투자의견, 목표주가, 리포트유형 (형식 제각각)
"""

import re
import json
from pathlib import Path
from openai import OpenAI
from langchain_community.document_loaders import PyPDFLoader
import os
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


# ─────────────────────────────────────
# 1. 파일명 파싱
#    고정 형식: YYMMDD_증권사명_섹터_제목.pdf
#    예) 260330_DS투자증권_자동차_지역별 정책 점검.pdf
#
#    parts[0] = 날짜   (YYMMDD)
#    parts[1] = 증권사 (폴더명과 동일, 교차 검증용)
#    parts[2] = 섹터
#    parts[3:] = 제목
# ─────────────────────────────────────
def parse_filename(filename: str, folder_name: str) -> dict:
    stem  = Path(filename).stem
    parts = stem.split("_")

    date        = None
    firm_in_filename = None  # 파일명 속 증권사명 (교차 검증용)
    sector      = None
    title       = None

    # 폴더명 = 증권사명 (1순위, 가장 정확)
    source_firm = folder_name

    # parts[0]: 날짜 변환 (YYMMDD → 20YY-MM-DD)
    date_raw = parts[0] if parts else ""
    if re.match(r'^\d{6}$', date_raw):
        date = f"20{date_raw[:2]}-{date_raw[2:4]}-{date_raw[4:]}"

    # parts[1]: 파일명 속 증권사명
    # 폴더명과 다를 경우 경고 출력 (데이터 품질 체크)
    if len(parts) >= 2:
        firm_in_filename = parts[1]
        if firm_in_filename != folder_name:
            print(f"    ⚠️  증권사명 불일치 - 폴더: {folder_name} / 파일명: {firm_in_filename}")

    # parts[2]: 섹터
    if len(parts) >= 3:
        sector = parts[2]

    # parts[3:]: 제목 (언더바로 나뉜 단어들 공백으로 합치기)
    if len(parts) >= 4:
        title = " ".join(parts[3:]).strip()

    return {
        "source_firm": source_firm,
        "report_date": date,
        "sector":      sector,
        "title":       title,
    }


# ─────────────────────────────────────
# 2. LLM으로 메타데이터 추출
# ─────────────────────────────────────
def extract_metadata_with_llm(first_page_text: str, source_firm: str) -> dict:
    """
    첫 페이지 텍스트 → LLM → 애널리스트, 투자의견, 목표주가, 리포트유형
    어떤 형식의 리포트든 동일하게 처리
    """

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
           예) "최태용  자동차·2차전지  02-709-2657  tyc@ds-sec.co.kr" → "최태용"
           여러 명이면 쉼표 구분, 성명만 추출 (직책/섹터명/전화번호 제외)
- rating: 리포트 전체 섹터 투자의견만 (개별 종목 투자의견 제외)
- target_price: 섹터 리포트는 null, 단일 종목 리포트만 추출
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

        # 마크다운 코드블록 제거
        raw = re.sub(r'^```json\s*', '', raw)
        raw = re.sub(r'\s*```$',     '', raw)

        result = json.loads(raw)

        return {
            "analyst":      result.get("analyst"),
            "rating":       result.get("rating"),
            "target_price": int(result["target_price"])
                            if result.get("target_price") else None,
            "report_type":  result.get("report_type", "섹터분석"),
        }

    except Exception as e:
        print(f"    ⚠️  LLM 추출 실패: {e}")
        return {
            "analyst":      None,
            "rating":       None,
            "target_price": None,
            "report_type":  "섹터분석",
        }


# ─────────────────────────────────────
# 3. 단일 PDF 로드
# ─────────────────────────────────────
def load_single_pdf(pdf_path: str) -> dict:
    pdf_path    = Path(pdf_path)
    filename    = pdf_path.name
    folder_name = pdf_path.parent.name   # 폴더명 = 증권사명

    # 파일명 파싱
    file_meta = parse_filename(filename, folder_name)

    # ── 텍스트 추출: PyPDFLoader (레이아웃 분리) ──────────────
    pypdf_loader = PyPDFLoader(str(pdf_path))
    pypdf_docs   = pypdf_loader.load()
    total_pages  = len(pypdf_docs)

    pages_text = []
    for doc in pypdf_docs:
        page_num = doc.metadata.get("page", 0) + 1  # 0-index → 1-index
        text     = doc.page_content.strip()
        if text:
            pages_text.append({
                "page_num": page_num,
                "text":     text
            })

    full_text = "\n\n".join([p["text"] for p in pages_text])

    # ── LLM 메타데이터 추출: 앞 3페이지 합쳐서 전달 ──────────
    # 1페이지가 이미지거나 목차인 경우 대비
    first_pages_text = "\n\n".join([
        p["text"] for p in pages_text[:3]
        if p["text"] and len(p["text"]) > 50
    ])

    llm_meta = extract_metadata_with_llm(
        first_pages_text,
        source_firm=file_meta["source_firm"]
    )

    return {
        "filename":    filename,
        "pdf_path":    str(pdf_path),
        "total_pages": total_pages,
        **file_meta,   # source_firm, report_date, sector, title
        **llm_meta,    # analyst, rating, target_price, report_type
        "full_text":   full_text,
    }


# ─────────────────────────────────────
# 4. 전체 폴더 로드 (하위 폴더 포함)
# ─────────────────────────────────────
def load_all_reports(reports_dir: str = "./reports") -> list[dict]:
    reports_dir = Path(reports_dir)
    pdf_files   = sorted(reports_dir.glob("**/*.pdf"))

    if not pdf_files:
        print(f"❌ {reports_dir} 에 PDF 없음")
        return []

    print(f"📂 총 {len(pdf_files)}개 PDF 발견\n")
    all_reports = []

    for pdf_path in pdf_files:
        print(f"  로딩 중: [{pdf_path.parent.name}] {pdf_path.name}")
        try:
            report = load_single_pdf(str(pdf_path))
            all_reports.append(report)
            print(f"  ✅ {report['source_firm']} | "
                  f"{report['report_type']} | "
                  f"{report['total_pages']}p | "
                  f"투자의견: {report['rating'] or '-'} | "
                  f"목표주가: {report['target_price'] or '-'} | "
                  f"애널리스트: {report['analyst'] or '-'}")

        except Exception as e:
            print(f"  ❌ {pdf_path.name} 로드 실패: {e}")

    print(f"\n✅ 총 {len(all_reports)}개 로드 완료")
    return all_reports


# ─────────────────────────────────────
# 5. JSON 캐시 저장 / 불러오기
# ─────────────────────────────────────
def save_cache(reports: list[dict],
               cache_path: str = "./data/reports_cache.json"):
    Path(cache_path).parent.mkdir(parents=True, exist_ok=True)

    # 그대로 저장 (load_single_pdf에서 이미 pages/tables 미포함)
    slim = reports

    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(slim, f, ensure_ascii=False, indent=2)

    print(f"✅ 캐시 저장 → {cache_path}")


def load_cache(cache_path: str = "./data/reports_cache.json") -> list[dict]:
    with open(cache_path, "r", encoding="utf-8") as f:
        return json.load(f)


# ─────────────────────────────────────
# 실행
# ─────────────────────────────────────
if __name__ == "__main__":
    reports = load_all_reports("./reports_naver_industry/교보증권")
    save_cache(reports, "./data/reports_cache.json")

    print("\n" + "="*60)
    print("📊 로드 결과 요약")
    print("="*60)

    for r in reports:
        print(f"""
파일명    : {r['filename']}
증권사    : {r['source_firm']}
날짜      : {r['report_date']}
섹터      : {r['sector']}
제목      : {r['title'] or '-'}
리포트유형: {r['report_type']}
투자의견  : {r['rating'] or '-'}
목표주가  : {r['target_price'] or '-'}
애널리스트: {r['analyst'] or '-'}
페이지수  : {r['total_pages']}
텍스트길이: {len(r['full_text'])}자
{'-'*40}
[첫 200자 미리보기]
{r['full_text'][:200]}
""")