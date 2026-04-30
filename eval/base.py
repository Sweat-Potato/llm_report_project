"""
eval/base.py
평가 공통 유틸리티

역할:
  - RAGAS LLM / Embedding 래퍼 생성 함수
  - 경로 상수 (EVAL_DIR, TESTSET_PATH 등)
  - CSV 로드 (reference_contexts 리스트 자동 복원)
  - 결과 저장 / 점수 출력 유틸

주의:
  TestsetGenerator 와 evaluate() 는 서로 다른 Embeddings 클래스를 사용함.
  - 생성(Generator) → ragas.embeddings.OpenAIEmbeddings   (RAGAS 네이티브)
  - 평가(evaluate)  → LangchainEmbeddingsWrapper            (LangChain 래퍼)
"""

from __future__ import annotations

import ast
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).parent.parent

# ── 경로 상수 ─────────────────────────────────────────────────────────────────
EVAL_DIR          = BASE_DIR / "data" / "eval"
TESTSET_PATH      = EVAL_DIR / "testset.csv"
RAG_EVAL_PATH     = EVAL_DIR / "rag_eval_result.csv"
CHUNK_EVAL_PATH   = EVAL_DIR / "chunk_eval_result.csv"
REPORT_EVAL_PATH  = EVAL_DIR / "report_eval_result.csv"


# ── LLM / Embedding 팩토리 ────────────────────────────────────────────────────

def get_generator_llm():
    """
    TestsetGenerator 용 LLM 래퍼.
    RAGAS 내부에서 질문·정답 생성 및 품질 검증에 사용.
    """
    from langchain_openai import ChatOpenAI
    from ragas.llms import LangchainLLMWrapper
    return LangchainLLMWrapper(ChatOpenAI(model="gpt-4o-mini"))


def get_generator_embeddings():
    """
    TestsetGenerator 용 RAGAS 네이티브 Embeddings.
    청크 간 유사도 계산 및 관련 컨텍스트 선택에 사용.
    (LangChain OpenAIEmbeddings 가 아닌 RAGAS 자체 클래스 사용)
    """
    import openai
    from ragas.embeddings import OpenAIEmbeddings as RagasOpenAIEmbeddings
    return RagasOpenAIEmbeddings(
        client=openai.OpenAI(),
        model="text-embedding-3-small",
    )


def get_evaluator_llm():
    """
    RAGAS evaluate() 용 LLM 래퍼.
    context_precision, context_recall, faithfulness 계산에 사용.
    temperature=0 으로 일관된 점수 확보.
    """
    from langchain_openai import ChatOpenAI
    from ragas.llms import LangchainLLMWrapper
    return LangchainLLMWrapper(ChatOpenAI(model="gpt-4o-mini", temperature=0))


def get_evaluator_embeddings():
    """
    RAGAS evaluate() 용 Embeddings 래퍼.
    answer_relevancy 계산에 사용.
    (TestsetGenerator 의 RagasOpenAIEmbeddings 와 다름 — LangchainEmbeddingsWrapper 사용)
    """
    from langchain_openai import OpenAIEmbeddings
    from ragas.embeddings import LangchainEmbeddingsWrapper
    return LangchainEmbeddingsWrapper(OpenAIEmbeddings(model="text-embedding-3-small"))


# ── 데이터셋 I/O ──────────────────────────────────────────────────────────────

def load_testset(path: Path = TESTSET_PATH):
    """
    CSV 에서 테스트셋 로드.

    핵심: CSV 저장 시 reference_contexts 리스트가 문자열로 변환됨.
    ast.literal_eval() 로 복원하지 않으면 RAGAS 가 문자를 한 글자씩
    순회해 완전히 잘못된 점수가 나옴.
    """
    from datasets import Dataset

    if not Path(path).exists():
        raise FileNotFoundError(
            f"테스트셋 없음: {path}\n"
            f"먼저 eval/01_generate_testset.py 를 실행하세요."
        )

    df = pd.read_csv(path)
    dataset = Dataset.from_pandas(df)

    def _restore_list(example):
        val = example["reference_contexts"]
        try:
            restored = ast.literal_eval(val) if isinstance(val, str) else []
        except (ValueError, SyntaxError):
            restored = []
        return {"reference_contexts": restored}

    before = len(dataset)
    dataset = dataset.map(_restore_list)
    empty = sum(1 for row in dataset if not row["reference_contexts"])
    if empty:
        print(f"  [경고] reference_contexts 복원 실패 {empty}/{before}행 → 빈 리스트로 대체")
        print(f"         해당 행은 context_recall 점수가 0으로 계산됩니다")

    return dataset


def save_results(df: pd.DataFrame, path: Path, label: str = "결과") -> None:
    """결과 DataFrame 을 CSV 로 저장 (디렉터리 자동 생성)."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"  {label} 저장 완료: {path}")


# ── 점수 출력 ─────────────────────────────────────────────────────────────────

def print_score_summary(scores: dict[str, float], label: str = "평가 점수") -> None:
    """
    점수 딕셔너리를 이쁘게 출력.
    0.8 이상 ✅ / 0.6~0.8 ⚠️ / 0.6 미만 ❌
    """
    print(f"\n{'='*55}")
    print(f"📊 {label}")
    print(f"{'='*55}")
    for key, val in scores.items():
        status = "✅" if val > 0.8 else "⚠️" if val > 0.6 else "❌"
        print(f"  {status} {key:30s}: {val:.3f}")
    overall = sum(scores.values()) / len(scores) if scores else 0
    print(f"{'─'*55}")
    print(f"  🎯 {'전체 평균':30s}: {overall:.3f}")
    print(f"{'='*55}")
