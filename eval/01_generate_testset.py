"""
eval/01_generate_testset.py
RAGAS TestsetGenerator 로 Q&A 테스트 데이터셋 생성

흐름:
  1. reports_cache.json 로드
  2. 선택한 청킹 전략으로 청크 생성
  3. TestsetGenerator 로 Q&A 쌍 자동 생성
  4. data/eval/testset.csv 저장

★ 설정 블록 — 이곳만 수정 ★
  CHUNKING    : 청킹 전략 선택
  TESTSET_SIZE: 생성할 Q&A 쌍 수 (처음엔 10~20 권장)

실행:
  uv run python eval/01_generate_testset.py
"""

import sys
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(BASE_DIR))

from dotenv import load_dotenv
load_dotenv()

# ══════════════════════════════════════════════════════════════════════════════
# ★ 설정 블록 — 전략 변경 시 이곳만 수정 ★
# ══════════════════════════════════════════════════════════════════════════════

# ── 청킹 전략 (하나만 주석 해제) ──────────────────────────────────────────────
from src.processing.chunking import chunking_01_recursive as CHUNKING
# from src.processing.chunking import chunking_03_hybrid   as CHUNKING
# from src.processing.chunking import chunking_04_sentence as CHUNKING

# ── 생성할 Q&A 쌍 수 ─────────────────────────────────────────────────────────
TESTSET_SIZE = 20   # 처음엔 10~20 으로 시작, 충분한 테스트 원하면 50 이상

# ══════════════════════════════════════════════════════════════════════════════


def _load_text_chunks() -> list[str]:
    """
    reports_cache.json → 청크 → 텍스트 리스트 변환.

    TestsetGenerator.generate_with_chunks() 는 순수 텍스트 리스트를 받으므로
    Chunk 객체에서 page_content(text) 만 추출.
    """
    from src.processing.Loader import load_all_reports

    cache_path = BASE_DIR / "data" / "loader_metadata" / "reports_cache.json"
    if not cache_path.exists():
        print(f"  ERROR: 캐시 없음 ({cache_path})")
        print(f"  먼저 src/processing/Loader.py 를 실행해 reports_cache.json 을 생성하세요.")
        sys.exit(1)

    print(f"  캐시 로드 중: {cache_path}")
    reports = load_all_reports(cache_path=str(cache_path))
    print(f"  리포트 {len(reports)}개 로드 완료")

    result = CHUNKING.chunk_reports(reports)
    text_chunks = [chunk.text for chunk in result.chunks]

    print(f"  청킹 전략  : {CHUNKING.STRATEGY_NAME}")
    print(f"  생성된 청크: {len(text_chunks)}개")
    print(f"  평균 크기  : {result.avg_chunk_size:.0f}자")
    return text_chunks


def _build_generator():
    """
    TestsetGenerator 초기화.

    - LLM: 질문·정답 생성 + 품질 검증
    - Embeddings: 청크 벡터화 및 관련 컨텍스트 선택
      → RAGAS 네이티브 OpenAIEmbeddings (LangChain 의 것과 다름!)
    """
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
    print("RAGAS 테스트 데이터셋 생성")
    print("=" * 65)
    print(f"  청킹 전략: {CHUNKING.STRATEGY_NAME}")
    print(f"  목표 Q&A : {TESTSET_SIZE}개")

    # STEP 1: 청크 준비
    print("\n[STEP 1] 청크 로드")
    text_chunks = _load_text_chunks()
    
    # 1. 빈 chunk 방어
    if not text_chunks:
        print("  ERROR: 생성된 청크가 없습니다.")
        sys.exit(1)

    # 2. 짧은 chunk 제거
    text_chunks = [c for c in text_chunks if len(c.strip()) > 200]

    # 3. 중복 제거
    text_chunks = list(set(text_chunks))

    # 4. 샘플링
    import random

    MAX_CHUNKS_FOR_TESTSET = 300

    if len(text_chunks) > MAX_CHUNKS_FOR_TESTSET:
        print(f"\n  ⚠️ 청크 샘플링 적용 ({len(text_chunks)} → {MAX_CHUNKS_FOR_TESTSET})")
        random.seed(42)
        text_chunks = random.sample(text_chunks, MAX_CHUNKS_FOR_TESTSET)

    print(f"  최종 사용 청크: {len(text_chunks)}개")

    # STEP 2: Generator 초기화
    print("\n[STEP 2] TestsetGenerator 초기화")
    generator = _build_generator()

    # STEP 3: 테스트셋 생성
    print(f"\n[STEP 3] 테스트셋 생성 (size={TESTSET_SIZE})...")
    print("  (LLM 호출로 시간이 걸립니다 — 대략 1~5분)")
    testset = generator.generate_with_chunks(
        chunks       = text_chunks,
        testset_size = TESTSET_SIZE,
    )

    # STEP 4: 저장
    print("\n[STEP 4] 결과 저장")
    from eval.base import TESTSET_PATH

    df = testset.to_pandas()
    actual_size = len(df)
    print(f"  생성된 Q&A 쌍: {actual_size}개")
    print(f"  컬럼: {list(df.columns)}")

    TESTSET_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(TESTSET_PATH, index=False, encoding="utf-8-sig")
    print(f"  저장 완료: {TESTSET_PATH}")

    # 질문 유형 분포 출력
    if "synthesizer_name" in df.columns:
        print(f"\n  질문 유형 분포:")
        for synth_name, count in df["synthesizer_name"].value_counts().items():
            print(f"    {synth_name}: {count}개")

    # 예시 출력
    if actual_size > 0:
        sample = df.iloc[0]
        print(f"\n  예시 Q&A:")
        print(f"    Q: {sample['user_input'][:80]}...")
        print(f"    A: {str(sample.get('reference', ''))[:80]}...")

    print("\n" + "=" * 65)
    print("완료! 다음 단계: eval/03_eval_rag.py 실행")
    print("=" * 65)


if __name__ == "__main__":
    main()
