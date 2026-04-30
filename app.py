"""
app.py
금융 리서치 리포트 RAG 시스템 — Streamlit UI

실행:
    uv run streamlit run app.py
"""
import sys
import re
from pathlib import Path
from datetime import datetime

import streamlit as st

# ── 프로젝트 루트 경로 ────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv()

# ── 페이지 설정 (반드시 첫 번째 st 호출) ─────────────────────────────────────────
st.set_page_config(
    page_title="리서치 리포트 분석 시스템",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ── 전략 모듈 임포트 ─────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def _import_modules():
    from src.processing.chunking import chunking_01_recursive as c1
    from src.processing.chunking import chunking_02_semantic  as c2
    from src.processing.chunking import chunking_03_hybrid    as c3
    from src.processing.chunking import chunking_04_sentence  as c4
    from src.embedding   import embedding_01_openai      as emb1
    from src.vectorstore import vectorstore_01_chroma    as vs1
    from src.retriever   import retriever_01_ensemble    as ret1
    from src.retriever   import retriever_02_balanced    as ret2
    from src.reranker    import reranker_01_crossencoder as rer1
    from src.reportcreator.report_chain   import (
        generate_report,
        step_retrieve,
        step_summarize_by_broker,
        step_analyze_consensus,
        step_extract_insights,
        step_generate_final_report,
    )
    from src.reportcreator.freeform_chain import answer_question
    return (
        {"RecursiveCharacterTextSplitter (기본)": c1,
         "SemanticChunker": c2,
         "Hybrid (길이별 자동 분기)": c3,
         "문단 기준 (Sentence)": c4},
        {"OpenAI text-embedding-3-small": emb1},
        {"ChromaDB": vs1},
        {"BM25 + Vector Ensemble": ret1,
         "BM25 + Vector Balanced": ret2},
        {"BGE Cross-Encoder": rer1},
        generate_report,
        (step_retrieve, step_summarize_by_broker,
         step_analyze_consensus, step_extract_insights, step_generate_final_report),
        answer_question,
    )

(
    CHUNKING_OPTIONS, EMBEDDING_OPTIONS, VECTORSTORE_OPTIONS,
    RETRIEVER_OPTIONS, RERANKER_OPTIONS,
    generate_report, REPORT_STEPS, answer_question,
) = _import_modules()

VS_BASE_DIR = PROJECT_ROOT / "data" / "vectorstore"


# ══════════════════════════════════════════════════════════════════════════════
# CSS
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:ital,opsz,wght@0,9..40,300;0,9..40,400;0,9..40,500;0,9..40,600;1,9..40,400&family=DM+Serif+Display&family=DM+Mono&family=Noto+Sans+KR:wght@300;400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'Noto Sans KR', 'DM Sans', sans-serif;
}

/* ─ 사이드바 ─ */
[data-testid="stSidebar"] {
    background: #1A1A1C !important;
    border-right: 1px solid #2A2A2C;
}
[data-testid="stSidebar"] * { color: #D8D8D8 !important; }
[data-testid="stSidebar"] .stSelectbox > label,
[data-testid="stSidebar"] .stNumberInput > label,
[data-testid="stSidebar"] .stTextInput > label { color: #A8A8A8 !important; font-size: 0.8rem !important; }
[data-testid="stSidebar"] .stSelectbox [data-baseweb="select"] > div {
    background: #28282A !important;
    border-color: #3A3A3C !important;
}
[data-testid="stSidebar"] .stNumberInput input {
    background: #28282A !important;
    border-color: #3A3A3C !important;
    color: #E8E8E8 !important;
}

/* ─ 버튼 ─ */
.stButton > button {
    background: linear-gradient(135deg, #C9A84C 0%, #E8C96A 100%) !important;
    color: #1A1A1C !important;
    font-weight: 600 !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 0.55rem 1.4rem !important;
    font-size: 0.88rem !important;
    letter-spacing: 0.025em !important;
    transition: all 0.18s ease !important;
    box-shadow: 0 2px 8px rgba(201,168,76,0.25) !important;
}
.stButton > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 18px rgba(201,168,76,0.45) !important;
}
.stButton > button:active { transform: translateY(0) !important; }

/* ─ 탭 ─ */
.stTabs [data-baseweb="tab-list"] {
    gap: 0.4rem;
    border-bottom: 2px solid #EDE9E0;
    padding-bottom: 0;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 8px 8px 0 0;
    padding: 0.55rem 1.3rem;
    font-size: 0.9rem;
    font-weight: 500;
    border: 1px solid transparent;
    border-bottom: none;
    color: #666;
    background: transparent;
}
.stTabs [aria-selected="true"] {
    background: #1A1A1C !important;
    color: #C9A84C !important;
    border-color: #EDE9E0 !important;
    border-bottom-color: #1A1A1C !important;
}

/* ─ 공통 카드 ─ */
.card {
    background: #FFFFFF;
    border: 1px solid #EDE9E0;
    border-radius: 12px;
    padding: 1.3rem 1.6rem;
    margin-bottom: 1rem;
}

/* ─ 검색 결과 카드 ─ */
.result-card {
    background: #FFFFFF;
    border: 1px solid #EDE9E0;
    border-left: 4px solid #C9A84C;
    border-radius: 0 10px 10px 0;
    padding: 1.1rem 1.5rem;
    margin-bottom: 0.9rem;
    transition: box-shadow 0.18s ease;
}
.result-card:hover { box-shadow: 0 4px 18px rgba(0,0,0,0.08); }
.result-card .rc-top {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 0.6rem;
}
.badge-broker {
    background: #1A1A1C;
    color: #C9A84C !important;
    font-size: 0.72rem;
    font-weight: 600;
    padding: 0.18rem 0.7rem;
    border-radius: 20px;
    letter-spacing: 0.06em;
    text-transform: uppercase;
}
.badge-score {
    background: #F4EFE4;
    color: #8A6A20 !important;
    font-size: 0.72rem;
    font-family: 'DM Mono', monospace;
    padding: 0.18rem 0.6rem;
    border-radius: 20px;
}
.rc-title {
    font-weight: 600;
    font-size: 0.93rem;
    color: #1A1A1C;
    margin-bottom: 0.35rem;
    line-height: 1.4;
}
.rc-meta {
    font-size: 0.76rem;
    color: #999;
    margin-bottom: 0.6rem;
}
.rc-body {
    font-size: 0.84rem;
    color: #555;
    line-height: 1.75;
    border-top: 1px solid #F4EFE4;
    padding-top: 0.7rem;
}

/* ─ 리포트 헤더 배너 ─ */
.report-banner {
    background: linear-gradient(135deg, #1A1A1C 0%, #2A1E14 60%, #1A1A1C 100%);
    border-radius: 14px;
    padding: 2rem 2.5rem;
    margin-bottom: 1.8rem;
    position: relative;
    overflow: hidden;
}
.report-banner::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0; bottom: 0;
    background: repeating-linear-gradient(
        -45deg, transparent, transparent 40px,
        rgba(201,168,76,0.04) 40px, rgba(201,168,76,0.04) 41px
    );
}
.report-banner h2 {
    font-family: 'DM Serif Display', serif !important;
    font-size: 1.7rem !important;
    color: #C9A84C !important;
    margin-bottom: 0.5rem !important;
    position: relative;
}
.report-banner .rb-meta {
    font-size: 0.82rem;
    color: #888 !important;
    position: relative;
}

/* ─ 스텝 카드 ─ */
.step-row {
    display: flex;
    align-items: flex-start;
    gap: 1rem;
    padding: 0.9rem 1.2rem;
    border-radius: 10px;
    margin-bottom: 0.5rem;
    border: 1px solid #EDE9E0;
    background: #FAFAFA;
    transition: all 0.2s ease;
}
.step-row.s-done  { border-color: #81C784; background: #F1FBF2; }
.step-row.s-active { border-color: #C9A84C; background: #FDF9F0; }
.step-circle {
    width: 30px; height: 30px;
    border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    font-size: 0.8rem; font-weight: 700;
    flex-shrink: 0;
    background: #EDE9E0; color: #AAA;
}
.step-circle.s-done   { background: #4CAF50; color: white; }
.step-circle.s-active { background: #C9A84C; color: white; }
.step-info .step-name { font-size: 0.9rem; font-weight: 600; color: #1A1A1C; }
.step-info .step-desc { font-size: 0.78rem; color: #999; margin-top: 0.1rem; }

/* ─ 메트릭 카드 ─ */
.m-card {
    background: #FAFAFA;
    border: 1px solid #EDE9E0;
    border-radius: 10px;
    padding: 1rem 1.2rem;
    text-align: center;
}
.m-card .m-val {
    font-family: 'DM Serif Display', serif;
    font-size: 1.8rem;
    color: #C9A84C;
    line-height: 1;
}
.m-card .m-lbl {
    font-size: 0.7rem;
    color: #AAA;
    margin-top: 0.25rem;
    letter-spacing: 0.08em;
    text-transform: uppercase;
}

/* ─ 칩 버튼 ─ */
.chip-wrap { display: flex; flex-wrap: wrap; gap: 0.4rem; margin-bottom: 1rem; }
div.stButton.chip > button {
    background: #F4EFE4 !important;
    color: #7A5A1A !important;
    font-size: 0.78rem !important;
    font-weight: 500 !important;
    padding: 0.3rem 0.9rem !important;
    border-radius: 20px !important;
    border: 1px solid #DDD5C0 !important;
    box-shadow: none !important;
}
div.stButton.chip > button:hover {
    background: #EBE2CC !important;
    transform: none !important;
    box-shadow: none !important;
}

/* ─ 질문 박스 ─ */
.q-box {
    background: #F8F7F4;
    border: 1px solid #EDE9E0;
    border-left: 4px solid #C9A84C;
    border-radius: 0 8px 8px 0;
    padding: 0.9rem 1.3rem;
    margin-bottom: 1rem;
    font-size: 0.92rem;
    color: #333;
    line-height: 1.6;
}

/* ─ 연결 상태 배지 ─ */
.badge-on  { background:#E8F5E9; color:#2E7D32!important; border:1px solid #A5D6A7;
             border-radius:20px; padding:0.2rem 0.8rem; font-size:0.75rem; font-weight:600; display:inline-block; }
.badge-off { background:#FFF8E1; color:#E65100!important; border:1px solid #FFE082;
             border-radius:20px; padding:0.2rem 0.8rem; font-size:0.75rem; font-weight:600; display:inline-block; }

/* ─ 구분선 ─ */
.div-line { height:1px; background:#EDE9E0; margin:1.2rem 0; }

/* ─ 섹션 제목 ─ */
.sec-title {
    font-family: 'DM Serif Display', serif;
    font-size: 1.3rem;
    color: #1A1A1C;
    margin-bottom: 0.3rem;
}
.sec-sub { font-size: 0.82rem; color: #AAA; margin-bottom: 1.2rem; }

/* ─ 입력창 ─ */
.stTextInput input, .stTextArea textarea {
    border-radius: 8px !important;
    border-color: #DDD5C0 !important;
    font-family: 'Noto Sans KR', sans-serif !important;
}
.stTextInput input:focus, .stTextArea textarea:focus {
    border-color: #C9A84C !important;
    box-shadow: 0 0 0 2px rgba(201,168,76,0.18) !important;
}

/* ─ 다운로드 버튼 ─ */
.stDownloadButton > button {
    background: #1A1A1C !important;
    color: #C9A84C !important;
    border: 1px solid #3A3A3C !important;
    border-radius: 8px !important;
    font-size: 0.85rem !important;
    box-shadow: none !important;
}
.stDownloadButton > button:hover {
    background: #2A2A2C !important;
    transform: none !important;
    box-shadow: none !important;
}
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# 세션 상태 초기화
# ══════════════════════════════════════════════════════════════════════════════
def _init_state():
    defaults = {
        "retriever":       None,
        "strategies":      {},
        "db_stats":        {},
        "search_results":  [],
        "search_query":    "",
        "report_result":   None,
        "freeform_result": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init_state()


# ══════════════════════════════════════════════════════════════════════════════
# 사이드바
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    # 로고
    st.markdown("""
    <div style="text-align:center; padding:1.8rem 0 1rem;">
        <div style="font-size:2rem;">📊</div>
        <div style="font-family:'DM Serif Display',serif; font-size:1.25rem;
                    color:#C9A84C; letter-spacing:0.03em; margin-top:0.3rem;">
            리서치 RAG
        </div>
        <div style="font-size:0.68rem; color:#555; margin-top:0.25rem;
                    letter-spacing:0.12em; text-transform:uppercase;">
            Financial Report Intelligence
        </div>
    </div>
    """, unsafe_allow_html=True)

    # 연결 상태
    if st.session_state.retriever:
        st.markdown('<div class="badge-on">● DB 연결됨</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="badge-off">● DB 미연결</div>', unsafe_allow_html=True)

    st.markdown('<div class="div-line"></div>', unsafe_allow_html=True)

    # 전략 선택
    st.markdown("**⚙️ 전략 설정**")
    st.caption("ingest.py 와 동일한 전략을 선택하세요")

    chunking_key    = st.selectbox("청킹 전략",   list(CHUNKING_OPTIONS.keys()),    index=0)
    embedding_key   = st.selectbox("임베딩",       list(EMBEDDING_OPTIONS.keys()),   index=0)
    vectorstore_key = st.selectbox("벡터스토어",   list(VECTORSTORE_OPTIONS.keys()), index=0)
    retriever_key   = st.selectbox("리트리버",     list(RETRIEVER_OPTIONS.keys()),   index=0)
    reranker_key    = st.selectbox("리랭커",       list(RERANKER_OPTIONS.keys()),    index=0)

    CHUNKING    = CHUNKING_OPTIONS[chunking_key]
    EMBEDDING   = EMBEDDING_OPTIONS[embedding_key]
    VECTORSTORE = VECTORSTORE_OPTIONS[vectorstore_key]
    RETRIEVER   = RETRIEVER_OPTIONS[retriever_key]
    RERANKER    = RERANKER_OPTIONS[reranker_key]

    db_path = str(
        VS_BASE_DIR
        / VECTORSTORE.STRATEGY_NAME
        / EMBEDDING.STRATEGY_NAME
        / CHUNKING.STRATEGY_NAME
    )
    st.caption(f"🗂 `…/{VECTORSTORE.STRATEGY_NAME}/{EMBEDDING.STRATEGY_NAME}/{CHUNKING.STRATEGY_NAME}`")

    col_k, col_n = st.columns(2)
    with col_k:
        k_val     = st.number_input("후보 k",    value=20, min_value=5,  max_value=50, step=5)
    with col_n:
        top_n_val = st.number_input("Rerank N", value=8,  min_value=3,  max_value=20, step=1)

    st.markdown("")

    if st.button("🔗  DB 연결하기", use_container_width=True):
        if not VECTORSTORE.exists(db_path):
            st.error("DB 없음. 먼저 `uv run python pipeline/ingest.py` 실행")
        else:
            with st.spinner("벡터스토어 로드 중..."):
                from langchain.schema import Document
                embeddings  = EMBEDDING.get_embeddings()
                vectorstore = VECTORSTORE.load(db_path, embeddings)
                results     = vectorstore.get(include=["documents", "metadatas"])
                all_docs    = [
                    Document(page_content=t, metadata=m)
                    for t, m in zip(results["documents"], results["metadatas"])
                ]
                retriever = RETRIEVER.build_retriever(vectorstore, all_docs, k=k_val)

                brokers, sectors = set(), set()
                for m in results["metadatas"]:
                    if m.get("source_firm"): brokers.add(m["source_firm"])
                    if m.get("sector"):      sectors.add(m["sector"])

                st.session_state.retriever  = retriever
                st.session_state.strategies = {
                    "RETRIEVER": RETRIEVER,
                    "RERANKER":  RERANKER,
                    "k":         k_val,
                    "top_n":     top_n_val,
                }
                st.session_state.db_stats = {
                    "chunks":  len(all_docs),
                    "brokers": sorted(brokers),
                    "sectors": sorted(sectors),
                }
            st.success("연결 완료!")
            st.rerun()

    # DB 통계
    if st.session_state.db_stats:
        stats = st.session_state.db_stats
        st.markdown('<div class="div-line"></div>', unsafe_allow_html=True)
        st.markdown(f"""
        <div style="display:flex; gap:0.5rem; margin-bottom:0.7rem;">
            <div class="m-card" style="flex:1;">
                <div class="m-val">{stats['chunks']:,}</div>
                <div class="m-lbl">청크</div>
            </div>
            <div class="m-card" style="flex:1;">
                <div class="m-val">{len(stats['brokers'])}</div>
                <div class="m-lbl">증권사</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        if stats["brokers"]:
            with st.expander("🏦 증권사 목록"):
                for b in stats["brokers"]:
                    st.caption(f"• {b}")
        if stats["sectors"]:
            with st.expander("📂 섹터 목록"):
                for s in stats["sectors"]:
                    st.caption(f"• {s}")


# ══════════════════════════════════════════════════════════════════════════════
# 메인 헤더
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div style="margin-bottom:1.6rem;">
    <div style="font-family:'DM Serif Display',serif; font-size:2rem;
                color:#1A1A1C; line-height:1.2; margin-bottom:0.4rem;">
        금융 리서치 분석 시스템
    </div>
    <div style="font-size:0.88rem; color:#AAA;">
        증권사 리포트를 AI로 분석 · 종합하여 인사이트를 제공합니다
    </div>
</div>
""", unsafe_allow_html=True)

if not st.session_state.retriever:
    st.info("👈 사이드바에서 전략을 선택하고 **DB 연결하기** 버튼을 눌러주세요.")
    st.stop()


# ══════════════════════════════════════════════════════════════════════════════
# 탭 레이아웃
# ══════════════════════════════════════════════════════════════════════════════
tab1, tab2, tab3 = st.tabs(["🔍  키워드 검색", "📊  종합 리포트 생성", "💬  자유형 질문"])


# ──────────────────────────────────────────────────────────────────────────────
# TAB 1 — 키워드 검색
# ──────────────────────────────────────────────────────────────────────────────
with tab1:
    st.markdown('<div class="sec-title">키워드 검색</div>', unsafe_allow_html=True)
    st.markdown('<div class="sec-sub">Hybrid Search (BM25 + Vector) → Cross-Encoder Rerank</div>',
                unsafe_allow_html=True)

    QUICK = ["반도체 HBM", "AI 인프라", "2차전지", "조선업 수주",
             "바이오 신약", "금리 인하", "K-방산", "자동차 EV"]

    st.markdown("**빠른 검색**")
    q_cols = st.columns(len(QUICK))
    for i, q in enumerate(QUICK):
        with q_cols[i]:
            if st.button(q, key=f"qchip_{i}"):
                st.session_state["_search_prefill"] = q

    search_val = st.session_state.pop("_search_prefill", st.session_state.search_query)

    search_input = st.text_input(
        "검색어",
        value=search_val,
        placeholder="예: 반도체 HBM 공급 업황 전망 2026",
        label_visibility="collapsed",
    )
    st.session_state.search_query = search_input

    col_btn, _, col_n = st.columns([2, 3, 1])
    with col_n:
        top_n_s = st.number_input("결과 수", value=5, min_value=1, max_value=20,
                                  key="search_topn", label_visibility="visible")
    with col_btn:
        search_go = st.button("🔍  검색", key="search_go")

    if search_go and search_input.strip():
        strat = st.session_state.strategies
        ret   = st.session_state.retriever
        with st.spinner("검색 중..."):
            candidates = strat["RETRIEVER"].retrieve(ret, search_input, k=strat["k"])
            docs       = strat["RERANKER"].rerank(search_input, candidates, top_n=top_n_s)
        st.session_state.search_results = docs
        st.session_state.search_query   = search_input

    results = st.session_state.search_results
    if results:
        q_used = st.session_state.search_query
        st.markdown('<div class="div-line"></div>', unsafe_allow_html=True)
        st.markdown(f"**'{q_used}'** 검색 결과 — **{len(results)}**개")
        st.markdown("")

        for i, doc in enumerate(results, 1):
            score   = doc.metadata.get("rerank_score")
            broker  = doc.metadata.get("source_firm", "-")
            date    = doc.metadata.get("report_date",  "-")
            sector  = doc.metadata.get("sector",       "-")
            title   = (doc.metadata.get("title") or "")[:90]
            score_s = f"{score:.4f}" if isinstance(score, float) else "—"

            st.markdown(f"""
            <div class="result-card">
                <div class="rc-top">
                    <span class="badge-broker">{broker}</span>
                    <span class="badge-score">rerank {score_s}</span>
                </div>
                <div class="rc-title">{i}. {title or '(제목 없음)'}</div>
                <div class="rc-meta">📅 {date}&nbsp;&nbsp;|&nbsp;&nbsp;📂 {sector}</div>
                <div class="rc-body">{doc.page_content[:450]}…</div>
            </div>
            """, unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────────────────
# TAB 2 — 종합 리포트 생성 (5-step)
# ──────────────────────────────────────────────────────────────────────────────
with tab2:
    st.markdown('<div class="sec-title">종합 리서치 리포트 생성</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sec-sub">5단계 멀티스텝 체인 → Executive Summary · 시장 현황 · 증권사 비교 · 리스크 분석 · 투자 전략</div>',
        unsafe_allow_html=True,
    )

    TOPIC_EXAMPLES = ["AI 반도체 업황 및 투자 전략", "2차전지 소재 공급망 분석",
                      "조선업 수주 회복과 밸류에이션", "바이오 신약 파이프라인 전망"]

    st.markdown("**주제 예시**")
    te_cols = st.columns(len(TOPIC_EXAMPLES))
    for i, ex in enumerate(TOPIC_EXAMPLES):
        with te_cols[i]:
            if st.button(ex, key=f"tex_{i}"):
                st.session_state["_topic_prefill"] = ex

    topic_val = st.session_state.pop("_topic_prefill", "")

    report_topic = st.text_input(
        "분석 주제",
        value=topic_val,
        placeholder="예: AI 반도체 HBM 공급 업황 및 투자 전략",
        label_visibility="collapsed",
    )

    report_go = st.button("📊  리포트 생성", key="report_go")

    if report_go and report_topic.strip():
        strat = st.session_state.strategies
        ret   = st.session_state.retriever

        STEP_META = [
            ("검색",           f"'{report_topic}' 관련 청크 수집 (Hybrid + Rerank)"),
            ("증권사별 요약",   "각 증권사 핵심 논거 요약 — GPT-4o-mini"),
            ("컨센서스 분석",   "공통 의견 & 이견 도출 — GPT-4o-mini"),
            ("인사이트 도출",   "핵심 인사이트 & 포트폴리오 시사점 — GPT-4o"),
            ("리포트 작성",     "최종 종합 리포트 생성 — GPT-4o"),
        ]

        def render_steps(active: int, done: set[int], placeholder):
            html = ""
            for idx, (name, desc) in enumerate(STEP_META):
                if idx in done:
                    cls, c_cls, icon = "s-done", "s-done", "✓"
                elif idx == active:
                    cls, c_cls, icon = "s-active", "s-active", "▶"
                else:
                    cls, c_cls, icon = "", "", str(idx + 1)
                html += f"""
                <div class="step-row {cls}">
                    <div class="step-circle {c_cls}">{icon}</div>
                    <div class="step-info">
                        <div class="step-name">Step {idx+1}. {name}</div>
                        <div class="step-desc">{desc}</div>
                    </div>
                </div>"""
            placeholder.markdown(html, unsafe_allow_html=True)

        ph = st.empty()
        done: set[int] = set()
        (step_retrieve, step_summarize, step_consensus,
         step_insights, step_final) = REPORT_STEPS

        with st.spinner("리포트 생성 중 (약 2~5분 소요)..."):
            render_steps(0, done, ph)
            docs = step_retrieve(
                ret, report_topic,
                retrieve_fn = strat["RETRIEVER"].retrieve,
                rerank_fn   = strat["RERANKER"].rerank,
                k=strat["k"], top_n=strat["top_n"],
            )
            done.add(0)

            render_steps(1, done, ph)
            summaries = step_summarize(docs, report_topic)
            done.add(1)

            render_steps(2, done, ph)
            consensus, differences = step_consensus(summaries, report_topic)
            done.add(2)

            render_steps(3, done, ph)
            insights = step_insights(summaries, consensus, differences, report_topic)
            done.add(3)

            render_steps(4, done, ph)
            final_md = step_final(report_topic, summaries, consensus, differences, insights)
            done.add(4)
            render_steps(-1, done, ph)

        st.session_state.report_result = {
            "topic":     report_topic,
            "report":    final_md,
            "brokers":   list(summaries.keys()),
            "timestamp": datetime.now().strftime("%Y년 %m월 %d일 %H:%M"),
        }
        st.rerun()

    if st.session_state.report_result:
        res = st.session_state.report_result

        st.markdown(f"""
        <div class="report-banner">
            <h2>📊 {res['topic']}</h2>
            <div class="rb-meta">
                생성일시: {res['timestamp']} &nbsp;|&nbsp;
                참고 증권사: {', '.join(res['brokers']) if res.get('brokers') else '—'}
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(res["report"])

        st.markdown("")
        col_dl1, col_dl2 = st.columns([2, 5])
        with col_dl1:
            safe_name = re.sub(r'[^\w가-힣]', '_', res['topic'])[:40]
            st.download_button(
                "⬇️  마크다운 다운로드",
                data      = res["report"].encode("utf-8"),
                file_name = f"report_{safe_name}_{datetime.now().strftime('%Y%m%d')}.md",
                mime      = "text/markdown",
            )


# ──────────────────────────────────────────────────────────────────────────────
# TAB 3 — 자유형 질문
# ──────────────────────────────────────────────────────────────────────────────
with tab3:
    st.markdown('<div class="sec-title">자유형 질문</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sec-sub">질문 유형 자동 분류 → 다중 쿼리 검색 → 시장 분석 · 증권사별 관점 · 주요 논거 포함 답변</div>',
        unsafe_allow_html=True,
    )

    FREEFORM_EX = [
        ("🏦", "하나증권과 키움증권의 반도체 의견 차이를 설명해줘"),
        ("📅", "이번 달 AI 섹터 투자의견 변화 알려줘"),
        ("⚠️", "조선업에서 언급된 리스크 요인 정리해줘"),
        ("💰", "각 증권사 목표주가 상향 근거가 뭐야"),
        ("🤝", "AI 인프라에 대해 증권사들이 공통으로 강조하는 게 뭐야"),
    ]

    TYPE_LABEL = {
        "broker_comparison": "🏦 증권사 비교",
        "timeline":          "📅 타임라인",
        "valuation":         "💰 밸류에이션",
        "risk":              "⚠️ 리스크",
        "consensus":         "🤝 컨센서스",
        "other":             "💬 기타",
    }

    st.markdown("**질문 예시**")
    ex_cols = st.columns(len(FREEFORM_EX))
    for i, (icon, ex) in enumerate(FREEFORM_EX):
        with ex_cols[i]:
            if st.button(f"{icon} {ex[:12]}…", key=f"fex_{i}"):
                st.session_state["_freeform_prefill"] = ex

    fq_val = st.session_state.pop("_freeform_prefill", "")

    freeform_q = st.text_area(
        "질문 입력",
        value=fq_val,
        height=110,
        placeholder=(
            "자유롭게 질문하세요.\n"
            "예: 하나증권과 키움증권의 3월 반도체 의견 차이를 설명해줘"
        ),
        label_visibility="collapsed",
    )

    freeform_go = st.button("💬  질문하기", key="freeform_go")

    if freeform_go and freeform_q.strip():
        strat = st.session_state.strategies
        ret   = st.session_state.retriever

        with st.status("질문 분석 중...", expanded=True) as status:
            st.write("📌 Step 1 — 질문 유형 분류 중...")
            # answer_question 내부에서 Step 1~4 수행
            result = answer_question(
                ret,
                freeform_q,
                retrieve_fn = strat["RETRIEVER"].retrieve,
                rerank_fn   = strat["RERANKER"].rerank,
            )
            status.update(label="분석 완료!", state="complete", expanded=False)

        st.session_state.freeform_result = result
        st.rerun()

    if st.session_state.freeform_result:
        res = st.session_state.freeform_result

        # 메트릭 행
        mc1, mc2, mc3 = st.columns(3)
        mc1.markdown(f"""
        <div class="m-card">
            <div class="m-val" style="font-size:1.2rem;">{TYPE_LABEL.get(res['question_type'], res['question_type'])}</div>
            <div class="m-lbl">질문 유형</div>
        </div>""", unsafe_allow_html=True)
        mc2.markdown(f"""
        <div class="m-card">
            <div class="m-val">{res['chunk_count']}</div>
            <div class="m-lbl">참고 청크</div>
        </div>""", unsafe_allow_html=True)
        mc3.markdown(f"""
        <div class="m-card">
            <div class="m-val">{len(res['sources'])}</div>
            <div class="m-lbl">참고 증권사</div>
        </div>""", unsafe_allow_html=True)

        st.markdown("")
        st.markdown(f'<div class="q-box"><b>Q.</b> {res["question"]}</div>',
                    unsafe_allow_html=True)
        st.markdown("---")
        st.markdown(res["answer"])

        if res["sources"]:
            with st.expander("📋 참고 증권사"):
                for s in res["sources"]:
                    st.caption(f"• {s}")

        st.markdown("")
        col_fdl, _ = st.columns([2, 5])
        with col_fdl:
            safe_q = re.sub(r'[^\w가-힣]', '_', res['question'])[:30]
            st.download_button(
                "⬇️  답변 다운로드",
                data      = res["answer"].encode("utf-8"),
                file_name = f"freeform_{safe_q}_{datetime.now().strftime('%Y%m%d_%H%M')}.md",
                mime      = "text/markdown",
            )
