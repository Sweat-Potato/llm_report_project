"""
app2.py — 리서치 리포트 RAG 시스템 웹 UI
Mastercard Design System 참고 (크림 캔버스 · 잉크 블랙 · 시그널 오렌지)
"""
import os
import re
import sys
import time
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ["ANONYMIZED_TELEMETRY"] = "False"
os.environ["CHROMA_TELEMETRY"] = "False"

try:
    import chromadb.telemetry.product.posthog as _ph
    _ph.Posthog.capture = lambda *a, **kw: None
except Exception:
    pass

import streamlit as st

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv()

_KO_SENTENCE_ENDS = ('다.', '다,', '며,', '며.', '고,', '고.', '한다.', '된다.', '한다,', '이다.', '이다,')

def _fix_md_paragraphs(text: str) -> str:
    """LLM 마크다운 출력 정규화:
    1) 한국어 문장으로 끝나는 ## / ### 헤딩 → 일반 텍스트로 변환 (가짜 헤딩 제거)
    2) 서술형 문단의 단일 \\n → \\n\\n (Markdown 단락 분리 보장)
    표(|), 인용(>), 리스트(-/*), 코드블록(```) 은 그대로 유지."""
    lines = text.split('\n')
    out = []
    in_code = False
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith('```'):
            in_code = not in_code

        # 한국어 문장 어미로 끝나는 ## / ### → 일반 텍스트
        if not in_code and re.match(r'^#{2,3}\s+', stripped):
            content = re.sub(r'^#{2,3}\s+', '', stripped)
            if any(content.endswith(e) for e in _KO_SENTENCE_ENDS):
                line = content

        out.append(line)
        if in_code or i >= len(lines) - 1:
            continue
        curr, nxt = line.strip(), lines[i + 1].strip()
        special = lambda s: not s or s.startswith(('#', '|', '>', '-', '*', '`', '!'))
        if curr and nxt and not special(curr) and not special(nxt):
            out.append('')
    return '\n'.join(out)


# ── 페이지 설정 ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ResearchRAG",
    page_icon="🟠",
    layout="wide",
    initial_sidebar_state="auto",
)

# ── Mastercard 디자인 CSS ──────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&display=swap');

  :root {
    --canvas:  #F3F0EE;
    --ink:     #141413;
    --orange:  #C87B52;
    --orange-light: #D8956E;
    --white:   #FFFFFF;
    --gray:    #7A7570;
    --gray-light: #B5B0AB;
    --border:  #DDD9D5;
    --card-bg: #FFFFFF;
    --ink-soft: #2A2926;
  }

  /* ── 전체 폰트·배경 ── */
  html, body, [class*="css"] {
    font-family: 'DM Sans', 'Sofia Sans', -apple-system, sans-serif !important;
  }
  .stApp { background: var(--canvas); }

  /* ── 사이드바 ── */
  [data-testid="stSidebar"] {
    background: var(--ink) !important;
    border-right: none;
  }
  [data-testid="stSidebar"] * { color: #E8E4E0 !important; }
  [data-testid="stSidebar"] .stRadio label { color: #B5B0AB !important; }
  [data-testid="stSidebar"] hr { border-color: #2A2926 !important; }

  /* ── 메인 여백 ── */
  .main .block-container {
    padding: 2rem 2.5rem 3rem;
    max-width: 1300px;
  }

  /* ── 카드 ── */
  .mc-card {
    background: var(--white);
    border-radius: 32px;
    padding: 2rem 2.2rem;
    border: 1px solid var(--border);
    box-shadow: 0 2px 24px rgba(20,20,19,0.06);
    margin-bottom: 1.2rem;
  }
  .mc-card-sm {
    background: var(--white);
    border-radius: 24px;
    padding: 1.2rem 1.5rem;
    border: 1px solid var(--border);
    box-shadow: 0 1px 12px rgba(20,20,19,0.04);
    margin-bottom: 0.75rem;
  }

  /* ── 히어로 배너 ── */
  .mc-hero {
    background: var(--ink);
    border-radius: 40px;
    padding: 2.8rem 3rem;
    margin-bottom: 1.5rem;
    position: relative;
    overflow: hidden;
  }
  .mc-hero::after {
    content: '';
    position: absolute;
    right: -60px; top: -60px;
    width: 280px; height: 280px;
    border-radius: 50%;
    background: var(--orange);
    opacity: 0.15;
  }
  .mc-hero-title {
    font-size: 2.2rem;
    font-weight: 700;
    color: var(--white);
    letter-spacing: -0.02em;
    line-height: 1.2;
    margin: 0 0 0.5rem;
  }
  .mc-hero-sub {
    font-size: 1rem;
    color: #B5B0AB;
    font-weight: 400;
    margin: 0;
  }

  /* ── 스탯 카드 ── */
  .mc-stat {
    background: var(--white);
    border-radius: 28px;
    padding: 1.6rem 1.8rem;
    border: 1px solid var(--border);
    box-shadow: 0 2px 16px rgba(20,20,19,0.05);
  }
  .mc-stat-label {
    font-size: 0.72rem;
    font-weight: 600;
    color: var(--gray);
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: 0.5rem;
  }
  .mc-stat-value {
    font-size: 2rem;
    font-weight: 700;
    color: var(--ink);
    letter-spacing: -0.02em;
    line-height: 1;
  }
  .mc-stat-icon {
    font-size: 1.4rem;
    margin-bottom: 0.6rem;
  }

  /* ── 섹션 헤더 ── */
  .mc-section {
    font-size: 1.1rem;
    font-weight: 700;
    color: var(--ink);
    letter-spacing: -0.01em;
    margin-bottom: 0.9rem;
  }

  /* ── 배지·필 ── */
  .mc-pill {
    display: inline-block;
    padding: 0.25rem 0.9rem;
    border-radius: 999px;
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.03em;
  }
  .pill-ink    { background: var(--ink); color: var(--white); }
  .pill-orange { background: var(--orange); color: var(--white); }
  .pill-canvas { background: var(--canvas); color: var(--ink); border: 1px solid var(--border); }
  .pill-outline{ background: transparent; color: var(--orange); border: 1.5px solid var(--orange); }

  /* ── 질문 유형별 색 ── */
  .qt-fact      { background: #1A1918; color: #F3F0EE; }
  .qt-coverage  { background: #E8F5E9; color: #2E7D32; border: 1px solid #C8E6C9; }
  .qt-timeline  { background: #FFF3E0; color: #E65100; border: 1px solid #FFE0B2; }
  .qt-broker    { background: #141413; color: var(--white); }
  .qt-risk      { background: #FFEBEE; color: #B71C1C; border: 1px solid #FFCDD2; }
  .qt-consensus { background: #E8F5E9; color: #1B5E20; border: 1px solid #C8E6C9; }
  .qt-other     { background: var(--canvas); color: var(--gray); border: 1px solid var(--border); }

  /* ── 검색 결과 카드 ── */
  .mc-result {
    background: var(--white);
    border-radius: 20px;
    padding: 1.3rem 1.6rem;
    border: 1px solid var(--border);
    border-left: 5px solid var(--orange);
    margin-bottom: 0.9rem;
    box-shadow: 0 1px 8px rgba(20,20,19,0.04);
  }
  .mc-result-meta {
    font-size: 0.74rem;
    color: var(--gray);
    margin-bottom: 0.35rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
    flex-wrap: wrap;
  }
  .mc-result-title {
    font-size: 0.92rem;
    font-weight: 600;
    color: var(--ink);
    margin-bottom: 0.35rem;
  }
  .mc-result-body {
    font-size: 0.84rem;
    color: var(--gray);
    line-height: 1.7;
  }

  /* ── 리포트 본문 ── */
  .mc-report {
    background: var(--white);
    border-radius: 32px;
    padding: 2.5rem 3rem;
    border: 1px solid var(--border);
    box-shadow: 0 4px 32px rgba(20,20,19,0.07);
    line-height: 1.85;
    color: var(--ink-soft);
    font-size: 0.95rem;
  }

  /* ── 입력 필드 ── */
  .stTextInput > div > div > input,
  .stTextArea > div > div > textarea {
    border-radius: 16px !important;
    border: 1.5px solid var(--border) !important;
    background: var(--white) !important;
    color: var(--ink) !important;
    font-size: 0.93rem !important;
    padding: 0.65rem 1rem !important;
    font-family: 'DM Sans', sans-serif !important;
  }
  .stTextInput > div > div > input:focus,
  .stTextArea > div > div > textarea:focus {
    border-color: var(--orange) !important;
    box-shadow: 0 0 0 3px rgba(207,69,0,0.10) !important;
  }

  /* ── 버튼 ── */
  .stButton > button {
    background: var(--ink) !important;
    color: var(--white) !important;
    border: none !important;
    border-radius: 999px !important;
    padding: 0.6rem 2rem !important;
    font-weight: 600 !important;
    font-size: 0.88rem !important;
    letter-spacing: 0.01em !important;
    transition: background 0.15s !important;
    font-family: 'DM Sans', sans-serif !important;
  }
  .stButton > button:hover {
    background: var(--orange) !important;
  }

  /* ── 폼 제출 버튼 (오렌지) ── */
  [data-testid="stFormSubmitButton"] > button {
    background: var(--orange) !important;
    padding: 0.65rem 2.5rem !important;
  }
  [data-testid="stFormSubmitButton"] > button:hover {
    background: var(--orange-light) !important;
  }

  /* ── 슬라이더·토글 ── */
  .stSlider [data-baseweb="slider"] div[role="slider"] {
    background: var(--orange) !important;
  }

  /* ── 사이드바 라디오 ── */
  .stRadio > div { gap: 0.2rem !important; }
  .stRadio label {
    padding: 0.55rem 1rem !important;
    border-radius: 999px !important;
    font-size: 0.88rem !important;
    font-weight: 500 !important;
    cursor: pointer !important;
    transition: background 0.12s !important;
  }

  /* ── 로고 영역 ── */
  .mc-logo {
    display: flex;
    align-items: center;
    gap: 0.8rem;
    padding: 0.3rem 0 1.6rem;
    margin-bottom: 0.5rem;
    border-bottom: 1px solid #2A2926;
  }
  .mc-logo-circles {
    position: relative;
    width: 42px;
    height: 26px;
  }
  .mc-logo-c1, .mc-logo-c2 {
    position: absolute;
    width: 26px; height: 26px;
    border-radius: 50%;
  }
  .mc-logo-c1 { background: #EB001B; left: 0; }
  .mc-logo-c2 { background: #F79E1B; right: 0; opacity: 0.9; }
  .mc-logo-name {
    font-size: 1rem;
    font-weight: 700;
    color: #F3F0EE !important;
    letter-spacing: -0.01em;
  }
  .mc-logo-sub {
    font-size: 0.7rem;
    color: #7A7570 !important;
    margin-top: 0.1rem;
  }

  /* ── 구분선 ── */
  hr { border: none; border-top: 1px solid var(--border); margin: 1.4rem 0; }

  /* ── 시스템 정보 텍스트 ── */
  .sys-info {
    font-size: 0.76rem;
    line-height: 1.8;
    color: #7A7570 !important;
    padding: 0 0.3rem;
  }
  .sys-info b { color: #B5B0AB !important; }

  /* ── expander ── */
  details {
    background: var(--canvas) !important;
    border-radius: 16px !important;
    border: 1px solid var(--border) !important;
    margin-bottom: 0.5rem !important;
  }

  /* ── 프로그레스 ── */
  .stProgress > div > div { background: var(--orange) !important; }

  /* ── 숨김 ── */
  #MainMenu { visibility: hidden; }
  footer { visibility: hidden; }
  header { background: transparent !important; box-shadow: none !important; }
</style>
""", unsafe_allow_html=True)


# ── 백엔드 로드 (app.py와 동일) ───────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def load_backend():
    from src.processing.chunking import chunking_01_recursive as CHUNKING
    from src.embedding import embedding_01_openai as EMBEDDING
    from src.vectorstore import vectorstore_01_chroma as VECTORSTORE
    from src.retriever import router as ROUTER
    from src.reranker import reranker_01_crossencoder as RERANKER
    from langchain.schema import Document

    VS_BASE_DIR = PROJECT_ROOT / "data" / "vectorstore"
    DB_PATH = str(VS_BASE_DIR / VECTORSTORE.STRATEGY_NAME / EMBEDDING.STRATEGY_NAME / CHUNKING.STRATEGY_NAME)

    if not VECTORSTORE.exists(DB_PATH):
        return None, None, None, None, None, None, 0

    embeddings  = EMBEDDING.get_embeddings()
    vectorstore = VECTORSTORE.load(DB_PATH, embeddings)
    results     = vectorstore.get(include=["documents", "metadatas"])
    all_docs    = [
        Document(page_content=text, metadata=meta)
        for text, meta in zip(results["documents"], results["metadatas"])
    ]
    retriever_tuple = ROUTER.build_retriever(vectorstore, all_docs, k=40)
    return retriever_tuple, ROUTER, RERANKER, EMBEDDING, VECTORSTORE, CHUNKING, len(all_docs)


# ── 헬퍼 ───────────────────────────────────────────────────────────────────────

QUESTION_TYPE_META = {
    "fact_lookup":       ("🔍", "사실 확인",    "qt-fact"),
    "coverage_summary":  ("📋", "커버리지",     "qt-coverage"),
    "timeline":          ("📅", "타임라인",     "qt-timeline"),
    "broker_comparison": ("⚖️",  "증권사 비교", "qt-broker"),
    "risk":              ("⚠️",  "리스크",      "qt-risk"),
    "consensus":         ("🤝", "컨센서스",     "qt-consensus"),
    "other":             ("📝", "종합 리포트",  "qt-other"),
}

SECTOR_COLORS = {
    "반도체": "pill-ink",
    "조선":   "pill-canvas",
    "AI인프라":"pill-orange",
    "자동차": "pill-canvas",
    "바이오": "pill-canvas",
}

def sector_pill(sector: str) -> str:
    cls = SECTOR_COLORS.get(sector, "pill-canvas")
    return f'<span class="mc-pill {cls}">{sector}</span>'


# ── 사이드바 ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("""
    <div class="mc-logo">
      <div>
        <div class="mc-logo-name">ResearchRAG</div>
        <div class="mc-logo-sub">증권사 리포트 분석</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    page = st.radio(
        "페이지 선택",
        options=[" 대시보드", " 질문 · 분석", " 최근 리포트"],
        label_visibility="collapsed",
    )
    page = page.split("  ", 1)[-1].strip()

    top_n = 15

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown('<div style="font-size:0.68rem;font-weight:700;color:#4A4744;letter-spacing:0.1em;text-transform:uppercase;padding:0 0.3rem 0.5rem;">시스템</div>', unsafe_allow_html=True)

    with st.spinner("로드 중…"):
        retriever, ROUTER, RERANKER, EMBEDDING, VECTORSTORE, CHUNKING, total_chunks = load_backend()

    db_ok = retriever is not None
    if db_ok:
        st.markdown(f"""
        <div class="sys-info">
          <div style="color:#C87B52;font-weight:700;margin-bottom:0.3rem;">● 연결됨</div>
          <div>청크 <b>{total_chunks:,}개</b></div>
          <div>청킹 <b>{CHUNKING.STRATEGY_NAME}</b></div>
          <div>리트리버 <b>router</b></div>
          <div>리랭커 <b>{RERANKER.STRATEGY_NAME}</b></div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.error("DB 없음 — ingest.py 실행 필요")


# ── 페이지: 대시보드 ───────────────────────────────────────────────────────────

if page == "대시보드":

    # 히어로
    st.markdown("""
    <div class="mc-hero">
      <p class="mc-hero-title">증권사 리포트<br>RAG 분석 시스템</p>
      <p class="mc-hero-sub">자유형 질문 → 유형 자동 감지 → 분석 리포트 생성</p>
    </div>
    """, unsafe_allow_html=True)

    # 스탯 카드
    c1, c2, c3, c4 = st.columns(4)
    def stat(col, icon, label, val):
        col.markdown(f"""
        <div class="mc-stat">
          <div class="mc-stat-icon">{icon}</div>
          <div class="mc-stat-label">{label}</div>
          <div class="mc-stat-value">{val}</div>
        </div>
        """, unsafe_allow_html=True)

    stat(c1, "📦", "총 청크 수", f"{total_chunks:,}" if db_ok else "—")
    stat(c2, "🏢", "커버 증권사", "12")
    stat(c3, "🧩", "질문 유형", "7")
    stat(c4, "⚡", "임베딩", "3-small")

    st.markdown("<br>", unsafe_allow_html=True)
    left, right = st.columns([3, 2], gap="large")

    with left:
        st.markdown('<div class="mc-section">지원 질문 유형</div>', unsafe_allow_html=True)
        for qt, (icon, label, cls) in QUESTION_TYPE_META.items():
            st.markdown(f"""
            <div class="mc-card-sm" style="display:flex;align-items:center;gap:1rem;">
              <span style="font-size:1.15rem;">{icon}</span>
              <div style="flex:1;">
                <span class="mc-pill {cls}">{label}</span>
                <span style="font-size:0.76rem;color:var(--gray);margin-left:0.5rem;">{qt}</span>
              </div>
            </div>
            """, unsafe_allow_html=True)

    with right:
        st.markdown('<div class="mc-section">커버 증권사</div>', unsafe_allow_html=True)
        brokers = [
            "하나증권","키움증권","DS투자증권","IBK투자증권","SK증권",
            "교보증권","대신증권","유안타증권","유진투자증권",
            "한화투자증권","iM증권","한국IR협의회",
        ]
        for b in brokers:
            st.markdown(f"""
            <div style="display:flex;align-items:center;gap:0.6rem;
                        padding:0.42rem 0.9rem;background:var(--white);
                        border-radius:999px;border:1px solid var(--border);
                        margin-bottom:0.35rem;font-size:0.82rem;color:var(--ink);">
              <span style="width:6px;height:6px;background:var(--orange);
                           border-radius:50%;display:inline-block;"></span>
              {b}
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="mc-card">', unsafe_allow_html=True)
    st.markdown('<div class="mc-section">예시 질문</div>', unsafe_allow_html=True)
    examples = [
        ("⚖️", "broker_comparison", "하나증권과 키움증권의 3월 반도체 의견 차이"),
        ("⚠️", "risk",              "조선업에서 언급된 리스크 요인 정리해줘"),
        ("🤝", "consensus",         "AI 인프라에 대해 증권사들이 공통으로 강조하는 게 뭐야"),
        ("📅", "timeline",          "이번 달 반도체 섹터 투자의견 변화"),
        ("📋", "coverage_summary",  "최근 AI 인프라 리포트 현황 정리해줘"),
    ]
    for icon, qt, ex in examples:
        _, cls = QUESTION_TYPE_META[qt][1], QUESTION_TYPE_META[qt][2]
        st.markdown(f"""
        <div style="display:flex;align-items:center;gap:0.8rem;
                    padding:0.6rem 1rem;background:var(--canvas);
                    border-radius:14px;margin-bottom:0.4rem;">
          <span>{icon}</span>
          <span style="font-size:0.85rem;color:var(--ink-soft);">{ex}</span>
        </div>
        """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)


# ── 페이지: 질문 · 분석 ────────────────────────────────────────────────────────

elif page == "질문 · 분석":

    st.markdown("""
    <div class="mc-hero" style="padding:2rem 2.5rem;">
      <p class="mc-hero-title" style="font-size:1.6rem;">질문 · 분석 리포트</p>
      <p class="mc-hero-sub">자유형 질문 → 유형 자동 감지 → 맞춤 리포트 생성</p>
    </div>
    """, unsafe_allow_html=True)

    if not db_ok:
        st.error("ChromaDB가 없습니다. `pipeline/ingest.py`를 실행해 주세요.")
        st.stop()

    from src.reportcreator.freeform_chain import answer_question

    # 예시 버튼
    if "preset_q2" not in st.session_state:
        st.session_state.preset_q2 = ""

    examples = [
        "하나증권과 키움증권의 3월 반도체 의견 차이",
        "조선업에서 언급된 리스크 요인 정리해줘",
        "AI 인프라에 대해 공통으로 강조하는 게 뭐야",
        "이번 달 반도체 투자의견 변화",
    ]
    ex_cols = st.columns(len(examples))
    for col, ex in zip(ex_cols, examples):
        if col.button(ex[:14] + "…", key=f"ex2_{ex}"):
            st.session_state.preset_q2 = ex

    st.markdown("<br>", unsafe_allow_html=True)

    with st.form("ask_form2"):
        question = st.text_area(
            "질문",
            value=st.session_state.preset_q2,
            placeholder="예: 하나증권과 키움증권의 3월 반도체 의견 차이를 설명해줘",
            height=110,
            label_visibility="collapsed",
        )
        submitted = st.form_submit_button("분석 리포트 생성 →")

    if submitted and question.strip():
        st.session_state.preset_q2 = ""
        progress_bar = st.progress(0, text="분석 준비 중…")

        with st.spinner(""):
            t0 = time.time()
            progress_bar.progress(15, text="질문 유형 분류 중…")
            result = answer_question(
                retriever, question,
                retrieve_fn=lambda r, q, k=40: ROUTER.retrieve(r, q, k=k),
                #rerank_fn=lambda q, docs, top_n: RERANKER.rerank(q, docs, top_n=top_n),
            )
            elapsed = time.time() - t0
            progress_bar.progress(100, text="완료!")

        qt   = result.get("question_type", "other")
        icon, label, cls = QUESTION_TYPE_META.get(qt, ("📝", "분석", "qt-other"))
        sources = result.get("sources", [])

        # 메타 요약
        st.markdown(f"""
        <div class="mc-card" style="display:flex;flex-wrap:wrap;gap:1.5rem;
                                     align-items:center;padding:1.2rem 1.8rem;">
          <div>
            <div style="font-size:0.68rem;font-weight:700;color:var(--gray);
                        text-transform:uppercase;letter-spacing:0.08em;margin-bottom:0.4rem;">질문 유형</div>
            <span class="mc-pill {cls}" style="font-size:0.82rem;padding:0.3rem 1rem;">
              {icon} {label}
            </span>
          </div>
          <div>
            <div style="font-size:0.68rem;font-weight:700;color:var(--gray);
                        text-transform:uppercase;letter-spacing:0.08em;margin-bottom:0.4rem;">참고 증권사</div>
            <div style="font-size:0.9rem;font-weight:600;color:var(--ink);">
              {', '.join(sources) if sources else '—'}
            </div>
          </div>
          <div style="margin-left:auto;text-align:right;">
            <div style="font-size:0.68rem;font-weight:700;color:var(--gray);
                        text-transform:uppercase;letter-spacing:0.08em;margin-bottom:0.4rem;">생성 시간</div>
            <div style="font-size:1.2rem;font-weight:700;color:var(--orange);">{elapsed:.1f}s</div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        # 리포트 본문
        st.markdown('<div class="mc-report">', unsafe_allow_html=True)
        st.markdown(_fix_md_paragraphs(result["answer"]))
        st.markdown('</div>', unsafe_allow_html=True)

        # 다운로드
        st.download_button(
            "리포트 저장 (.md)",
            data=result["answer"],
            file_name=f"report_{qt}_{int(t0)}.md",
            mime="text/markdown",
        )

        # 참조 청크
        retrieved_docs = result.get("docs", [])
        if retrieved_docs:
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown(f'<div class="mc-section">📎 참조 청크 — {len(retrieved_docs)}개</div>', unsafe_allow_html=True)

            for i, doc in enumerate(retrieved_docs, 1):
                broker  = doc.metadata.get("source_firm", doc.metadata.get("broker", "-"))
                date    = doc.metadata.get("report_date", "-")
                sector  = doc.metadata.get("sector", "-")
                title   = doc.metadata.get("title", "")
                score   = doc.metadata.get("rerank_score", None)
                content = doc.page_content

                score_html = ""
                if score is not None:
                    try:
                        s = float(score)
                        score_html = f'<span class="mc-pill pill-outline" style="font-size:0.68rem;">score {s:.3f}</span>'
                    except Exception:
                        pass

                sector_html = sector_pill(sector) if sector and sector != "-" else ""
                preview = content[:200] + ("…" if len(content) > 200 else "")

                with st.expander(f"#{i}  {broker}  ·  {date}  {'· ' + title[:28] if title else ''}"):
                    st.markdown(f"""
                    <div style="display:flex;flex-wrap:wrap;gap:0.4rem;margin-bottom:0.7rem;align-items:center;">
                      {sector_html} {score_html}
                    </div>
                    <div style="font-size:0.84rem;color:var(--gray);line-height:1.75;">{preview}</div>
                    """, unsafe_allow_html=True)
                    if len(content) > 200:
                        if st.toggle("전문 보기", key=f"full2_{i}"):
                            st.text(content)

    elif submitted:
        st.warning("질문을 입력해 주세요.")


# ── 페이지: 최근 리포트 ─────────────────────────────────────────────────────────

elif page == "최근 리포트":

    st.markdown("""
    <div class="mc-hero" style="padding:2rem 2.5rem;">
      <p class="mc-hero-title" style="font-size:1.6rem;">생성된 리포트</p>
      <p class="mc-hero-sub">data/reports_output/ 저장 파일</p>
    </div>
    """, unsafe_allow_html=True)

    reports_dir = PROJECT_ROOT / "data" / "reports_output"
    md_files = (
        sorted(reports_dir.glob("*.md"), key=lambda f: f.stat().st_mtime, reverse=True)
        if reports_dir.exists() else []
    )

    if not md_files:
        st.markdown("""
        <div class="mc-card" style="text-align:center;padding:4rem 2rem;">
          <div style="font-size:3rem;margin-bottom:1rem;">📂</div>
          <div style="font-size:1.1rem;font-weight:700;color:var(--ink);letter-spacing:-0.01em;">
            아직 생성된 리포트가 없습니다
          </div>
          <div style="font-size:0.88rem;color:var(--gray);margin-top:0.5rem;">
            질문·분석 탭에서 리포트를 생성해보세요.
          </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f'<div style="font-size:0.82rem;color:var(--gray);margin-bottom:1rem;">총 {len(md_files)}개</div>', unsafe_allow_html=True)
        for f in md_files:
            mtime    = time.strftime("%Y-%m-%d %H:%M", time.localtime(f.stat().st_mtime))
            size_kb  = f.stat().st_size / 1024
            with st.expander(f"📄  {f.stem[:50]}  —  {mtime}  ({size_kb:.1f} KB)"):
                content = f.read_text(encoding="utf-8", errors="replace")
                st.markdown(content)
                st.download_button(
                    "다운로드",
                    data=content,
                    file_name=f.name,
                    mime="text/markdown",
                    key=f"dl2_{f.name}",
                )
