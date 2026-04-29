"""
diagnose_chunks.py
청킹 결과가 원본 텍스트를 얼마나 커버하는지 진단

실행: python src/processing/chunking/diagnose_chunks.py
"""

import json
from pathlib import Path
from collections import defaultdict

BASE_DIR   = Path(__file__).parent.parent.parent.parent
CACHE_PATH = BASE_DIR / "data" / "loader_metadata" / "reports_cache.json"
CHUNKS_PATH= BASE_DIR / "data" / "chunks" / "chunking_01_recursive.json"

# ── 원본 캐시 로드 ─────────────────────────────────
print("=" * 60)
print("원본 reports_cache.json 분석")
print("=" * 60)

with open(CACHE_PATH, encoding="utf-8") as f:
    reports = json.load(f)

print(f"총 리포트 수: {len(reports)}개\n")

report_map = {}
for r in reports:
    fn = r["filename"]
    ft = r.get("full_text", "")
    report_map[fn] = ft
    print(f"  [{r['source_firm']:12s}] {r['sector']:8s} | "
          f"full_text={len(ft):6d}자 | {r['title'] or '-'}")

total_orig = sum(len(v) for v in report_map.values())
print(f"\n  원본 전체 텍스트 합계: {total_orig:,}자")

# ── 청킹 결과 로드 ────────────────────────────────
print("\n" + "=" * 60)
print("chunking_01_recursive.json 분석")
print("=" * 60)

with open(CHUNKS_PATH, encoding="utf-8") as f:
    data = json.load(f)

chunks = data["chunks"]
print(f"총 청크 수 : {len(chunks)}개")
print(f"평균 크기  : {data['avg_chunk_size']}자")

# 리포트별 청크 집계
by_report = defaultdict(list)
for c in chunks:
    by_report[c["filename"]].append(c["text"])

print("\n리포트별 청크 커버리지:")
print(f"{'파일명':45s} {'원본':>7s} {'청크합':>7s} {'커버율':>6s} {'청크수':>5s}")
print("-" * 75)

for fn, texts in by_report.items():
    chunk_total = sum(len(t) for t in texts)
    orig_len    = len(report_map.get(fn, ""))
    ratio       = chunk_total / orig_len * 100 if orig_len else 0
    chunk_cnt   = len(texts)
    short_fn    = fn[:43] + ".." if len(fn) > 45 else fn
    print(f"  {short_fn:45s} {orig_len:7,d} {chunk_total:7,d} {ratio:5.0f}%  {chunk_cnt:4d}개")

# ── 문제 청크 감지 ────────────────────────────────
print("\n" + "=" * 60)
print("이상 청크 감지 (너무 작은 청크)")
print("=" * 60)

tiny_chunks = [c for c in chunks if c["char_count"] < 30]
print(f"30자 미만 청크: {len(tiny_chunks)}개")
for c in tiny_chunks[:10]:
    print(f"  [{c['chunk_id']}] {repr(c['text'])}")

print("\n" + "=" * 60)
print("세 번째 리포트 첫 5청크 미리보기")
print("=" * 60)
first_fn = list(by_report.keys())[2]
first_chunks = [c for c in chunks if c["filename"] == first_fn]
for c in first_chunks:
    print(f"\n[chunk_{c['chunk_index']:03d} | {c['char