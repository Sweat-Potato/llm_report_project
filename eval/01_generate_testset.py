"""
eval/01_generate_testset.py
RAGAS TestsetGenerator 로 Q&A 테스트 데이터셋 생성

흐름:
  1. 청크 파일 로드 → 품질 필터 → 증권사별 균등 샘플링
  2. RAGAS TestsetGenerator 로 Q&A 쌍 자동 생성
  3. 한국어 질문만 필터링 (영어 제거) → 정확히 TESTSET_SIZE개 확보
  4. data/eval/testset.csv 저장

★ 설정 블록 — 이곳만 수정 ★
  CHUNKING    : 청킹 전략 선택 (03_eval_rag.py 와 반드시 일치)
  TESTSET_SIZE: 생성할 Q&A 쌍 수

실행:
  uv run python eval/01_generate_testset.py
"""

import re
import sys
import random
import time
from collections import defaultdict
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(BASE_DIR))

from dotenv import load_dotenv
load_dotenv()

# ══════════════════════════════════════════════════════════════════════════════
# ★ 설정 블록 — 전략 변경 시 이곳만 수정 ★
# ══════════════════════════════════════════════════════════════════════════════

# ── 청킹 전략 (하나만 주석 해제) ──────────────────────────────────────────────
# 주의: 03_eval_rag.py 의 청킹 전략과 반드시 일치해야 함 (DB 경로 동일해야 함)
# from src.processing.chunking import chunking_01_recursive as CHUNKING
# from src.processing.chunking import chunking_02_semantic as CHUNKING
from src.processing.chunking import chunking_03_hybrid   as CHUNKING
# from src.processing.chunking import chunking_04_sentence as CHUNKING

# ── 생성할 Q&A 쌍 수 (정확히 이 수만큼 한국어 Q&A 확보) ──────────────────────
TESTSET_SIZE = 20

# ── 테스트셋 생성에 사용할 최대 청크 수 ──────────────────────────────────────
# 300개 → NER 추출 8분 이상 → Connection timeout 발생
# 80개  → 약 2분, 안정적
MAX_CHUNKS_FOR_TESTSET = 80

# ── 청크 품질 필터 기준 ────────────────────────────────────────────────────────
MIN_CHUNK_LEN        = 200
MIN_KOREAN_RATIO     = 0.45
MIN_FINANCE_TERMS    = 2
MAX_SPECIAL_RATIO    = 0.25

# ══════════════════════════════════════════════════════════════════════════════

_FINANCE_TERMS = [
    "투자의견", "목표주가", "매수", "중립", "매도", "비중확대", "비중축소",
    "실적", "영업이익", "매출", "순이익", "밸류에이션", "PER", "PBR",
    "반도체", "2차전지", "배터리", "HBM", "AI", "전망", "리스크",
    "증권사", "리포트", "분석", "업황", "수요", "공급", "가격",
    "성장", "하락", "상승", "개선", "악화", "전년", "분기", "연간",
]


def _korean_ratio(text: str) -> float:
    if not text:
        return 0.0
    return len(re.findall(r'[가-힣]', text)) / len(text)


def _special_ratio(text: str) -> float:
    if not text:
        return 0.0
    return len(re.findall(r'[^\w가-힣\s]', text)) / len(text)


def _finance_term_count(text: str) -> int:
    return sum(1 for term in _FINANCE_TERMS if term in text)


def _is_quality_chunk(text: str) -> bool:
    s = text.strip()
    return (
        len(s) >= MIN_CHUNK_LEN
        and _korean_ratio(s) >= MIN_KOREAN_RATIO
        and _finance_term_count(s) >= MIN_FINANCE_TERMS
        and _special_ratio(s) <= MAX_SPECIAL_RATIO
    )


def _is_korean_question(text: str) -> bool:
    return isinstance(text, str) and _korean_ratio(text) >= 0.25


def _stratified_sample(chunks_by_firm: dict, total: int) -> list:
    firms    = list(chunks_by_firm.keys())
    per_firm = max(1, total // len(firms))
    sampled  = []
    for chunks in chunks_by_firm.values():
        sampled.extend(random.sample(chunks, min(per_firm, len(chunks))))
    if len(sampled) < total:
        remaining = [c for cs in chunks_by_firm.values() for c in cs if c not in sampled]
        random.shuffle(remaining)
        sampled.extend(remaining[:total - len(sampled)])
    random.shuffle(sampled)
    return sampled[:total]


def _load_quality_chunks() -> list[str]:
    from src.processing.chunking.base import ChunkingResult

    chunk_path = BASE_DIR / "data" / "chunks" / f"{CHUNKING.STRATEGY_NAME}.json"
    if not chunk_path.exists():
        print(f"  ERROR: 청크 파일 없음 ({chunk_path})")
        print(f"  먼저 pipeline/ingest.py 를 실행하세요.")
        sys.exit(1)

    print(f"  청크 파일 로드: {chunk_path}")
    result     = ChunkingResult.load(str(chunk_path))
    all_chunks = result.chunks
    print(f"  전체 청크    : {len(all_chunks)}개")

    quality = [c for c in all_chunks if _is_quality_chunk(c.text)]
    print(f"  품질 필터 후 : {len(quality)}개")

    seen, deduped = set(), []
    for c in quality:
        if c.text not in seen:
            seen.add(c.text)
            deduped.append(c)
    print(f"  중복 제거 후 : {len(deduped)}개")

    by_firm = defaultdict(list)
    for c in deduped:
        by_firm[c.source_firm or "미분류"].append(c.text)

    print(f"  증권사 수    : {len(by_firm)}개사")
    for firm, texts in sorted(by_firm.items()):
        print(f"    {firm}: {len(texts)}개 청크")

    random.seed(42)
    sampled = _stratified_sample(by_firm, MAX_CHUNKS_FOR_TESTSET)
    print(f"\n  균등 샘플링  : {len(sampled)}개 → TestsetGenerator 입력")
    return sampled


def _build_generator():
    from ragas.testset import TestsetGenerator
    from eval.base import get_generator_llm, get_generator_embeddings

    generator = TestsetGenerator(
        llm             = get_generator_llm(),
        embedding_model = get_generator_embeddings(),
    )
    print("  TestsetGenerator 초기화 완료")
    return generator


def main():
    print("=" * 65)
    print("RAGAS 테스트 데이터셋 생성 (품질 필터 + 균등 샘플링)")
    print("=" * 65)
    print(f"  청킹 전략: {CHUNKING.STRATEGY_NAME}")
    print(f"  목표 Q&A : 한국어 {TESTSET_SIZE}개")

    # STEP 1: 청크 로드 + 필터 + 샘플링
    print("\n[STEP 1] 청크 로드 및 품질 필터링")
    text_chunks = _load_quality_chunks()

    # STEP 2: Generator 초기화
    print("\n[STEP 2] TestsetGenerator 초기화")
    generator = _build_generator()

    # STEP 3: 한국어 Q&A 정확히 TESTSET_SIZE개 확보
    print(f"\n[STEP 3] Q&A 생성 (목표: 한국어 {TESTSET_SIZE}개)")
    MAX_ROUNDS  = 5
    collected   = []
    seen_inputs = set()

    for round_num in range(1, MAX_ROUNDS + 1):
        needed = TESTSET_SIZE - len(collected)
        if needed <= 0:
            break

        batch_size = max(needed, int(needed * 1.5))
        print(f"  [라운드 {round_num}] {batch_size}개 생성 중... (확보: {len(collected)}/{TESTSET_SIZE})")

        batch_df = None
        for attempt in range(3):
            try:
                batch    = generator.generate_with_chunks(
                    chunks       = text_chunks,
                    testset_size = batch_size,
                )
                batch_df = batch.to_pandas()
                break
            except Exception as e:
                if attempt < 2:
                    wait = (attempt + 1) * 10
                    print(f"    ⚠️  실패 ({str(e)[:50]}) → {wait}초 후 재시도")
                    time.sleep(wait)
                else:
                    print(f"    ❌  3회 모두 실패: {str(e)[:80]}")

        if batch_df is None:
            break

        for _, row in batch_df.iterrows():
            q = str(row.get("user_input", ""))
            if _is_korean_question(q) and q not in seen_inputs:
                seen_inputs.add(q)
                collected.append(row)

        korean_cnt = sum(1 for _, r in batch_df.iterrows() if _is_korean_question(str(r.get("user_input", ""))))
        print(f"    → 배치 한국어: {korean_cnt}/{len(batch_df)}개  누적: {len(collected)}개")

        if len(collected) >= TESTSET_SIZE:
            break

    if not collected:
        print("  ⚠️  한국어 Q&A 생성 실패.")
        sys.exit(1)

    import pandas as pd
    df = pd.DataFrame(collected[:TESTSET_SIZE]).reset_index(drop=True)
    print(f"\n  ✅ 최종 한국어 Q&A: {len(df)}개 확보")

    # STEP 4: 저장
    print("\n[STEP 4] 결과 저장")
    from eval.base import TESTSET_PATH

    TESTSET_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(TESTSET_PATH, index=False, encoding="utf-8-sig")
    print(f"  저장 완료: {TESTSET_PATH}")
    print(f"  컬럼: {list(df.columns)}")

    if "synthesizer_name" in df.columns:
        print(f"\n  질문 유형 분포:")
        for name, cnt in df["synthesizer_name"].value_counts().items():
            print(f"    {name}: {cnt}개")

    print(f"\n  예시 Q&A (상위 3개):")
    for i, row in df.head(3).iterrows():
        print(f"    Q{i+1}: {str(row['user_input'])[:80]}")
        print(f"    A{i+1}: {str(row.get('reference', ''))[:60]}...")
        print()

    print("\n" + "=" * 65)
    print(f"완료! 한국어 Q&A {len(df)}개 생성")
    print("다음 단계: eval/03_eval_rag.py 실행")
    print("=" * 65)


if __name__ == "__main__":
    main()
