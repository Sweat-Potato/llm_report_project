"""
base.py
청킹 공통 유틸리티 및 데이터 타입 정의

모든 전략이 동일한 입출력 형식을 사용하도록 표준화.
"""

from __future__ import annotations
import json
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional


# ─────────────────────────────────────────────────
# 데이터 타입
# ─────────────────────────────────────────────────

@dataclass
class Chunk:
    """청크 단위 표준 구조체"""
    chunk_id:    str              # 고유 ID (예: "DS투자증권_20260330_0")
    text:        str              # 청크 텍스트
    char_count:  int              # 문자 수

    # 리포트 메타데이터
    source_firm:  str
    report_date:  Optional[str]
    sector:       Optional[str]
    title:        Optional[str]
    report_type:  Optional[str]
    analyst:      Optional[str]
    rating:       Optional[str]
    target_price: Optional[int]
    filename:     str

    # 청킹 메타데이터
    chunk_index:   int            # 이 리포트 안에서 몇 번째 청크
    total_chunks:  int            # 이 리포트의 총 청크 수
    strategy:      str            # 사용한 전략명

    # Parent-Child 전략 전용 (다른 전략에서는 None)
    parent_id:    Optional[str] = None
    chunk_level:  Optional[str] = None  # "parent" | "child"

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ChunkingResult:
    """전략 1회 실행 결과"""
    strategy:    str
    chunks:      list[Chunk]
    report_count: int
    total_chars: int             # 원본 총 문자 수

    @property
    def chunk_count(self) -> int:
        return len(self.chunks)

    @property
    def avg_chunk_size(self) -> float:
        if not self.chunks:
            return 0.0
        return sum(c.char_count for c in self.chunks) / len(self.chunks)

    def to_dict(self) -> dict:
        return {
            "strategy":     self.strategy,
            "report_count": self.report_count,
            "chunk_count":  self.chunk_count,
            "total_chars":  self.total_chars,
            "avg_chunk_size": round(self.avg_chunk_size, 1),
            "chunks":       [c.to_dict() for c in self.chunks],
        }

    def save(self, path: str) -> None:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)
        print(f"[저장] {out}  ({self.chunk_count}개 청크)")


# ─────────────────────────────────────────────────
# 공통 유틸
# ─────────────────────────────────────────────────

def load_reports_cache(cache_path: str) -> list[dict]:
    """reports_cache.json 로드"""
    with open(cache_path, "r", encoding="utf-8") as f:
        return json.load(f)


def make_chunk_id(source_firm: str, report_date: Optional[str],
                  index: int, prefix: str = "") -> str:
    """청크 고유 ID 생성"""
    date  = (report_date or "nodate").replace("-", "")
    firm  = source_firm.replace(" ", "")
    pre   = f"{prefix}_" if prefix else ""
    return f"{firm}_{date}_{pre}{index}"


def extract_meta(report: dict) -> dict:
    """리포트 딕셔너리에서 메타데이터만 추출"""
    return {
        "source_firm":  report.get("source_firm", ""),
        "report_date":  report.get("report_date"),
        "sector":       report.get("sector"),
        "title":        report.get("title"),
        "report_type":  report.get("report_type"),
        "analyst":      report.get("analyst"),
        "rating":       report.get("rating"),
        "target_price": report.get("target_price"),
        "filename":     report.get("filename", ""),
    }
