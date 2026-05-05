"""
app.py — 리서치 리포트 RAG 시스템 웹 UI
Horizon UI Chakra 디자인 참고
"""
import os
import sys
import time
from pathlib import Path

# PyTorch/OpenMP 충돌 방지 — 반드시 torch import 전에 설정
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
# ChromaDB 텔레메트리 완전 비활성화
os.environ["ANONYMIZED_TELEMETRY"] = "False"
os.environ["CHROMA_TELEMETRY"] = "False"

# ChromaDB telemetry 패치 (0.6.x capture() 버그 무음 처리)
try:
    import chromadb.telemetry.product.posthog as _ph
    _ph.Posthog.capture = lambda *a, **kw: None  # noqa: E731
except Exception:
    pass

import streamlit as st

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv()

# ── 페이지 설정 ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="리서치 리포트 RAG",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="auto",
)

# ── 전역 CSS (Horizon UI Chakra 스타일) ───────────────────────────────────────
st.markdown("""
<style>
  /* ── 기본 색상 토큰 ── */
  :root {
    --brand:      #4318FF;
    --brand-light:#6B48FF;
    --bg-page:    #F4F7FE;
    --bg-card:    #FFFFFF;
    --text-main:  #2B3674;
    --text-sub:   #A3AED0;
    --text-body:  #707EAE;
    --border:     #E9EDF7;
    --success:    #01B574;
    --warning:    #FFB547;
    --danger:     #EE5D50;
    --info:       #39B8FF;
    --gradient:   linear-gradient(135deg, #4318FF 0%, #868CFF 100%);
  }

  /* ── 전체 배경 ── */
  .stApp { background: var(--bg-page); }

  /* ── 사이드바 ── */
  [data-testid="stSidebar"] {
    background: var(--bg-card);
    border-right: 1px solid var(--border);
  }
  [data-testid="stSidebar"] .stMarkdown h1,
  [data-testid="stSidebar"] .stMarkdown h2,
  [data-testid="stSidebar"] .stMarkdown h3 {
    color: var(--text-main);
  }

  /* ── 메인 컨텐츠 여백 ── */
  .main .block-container {
    padding: 1.5rem 2rem 2rem;
    max-width: 1400px;
  }

  /* ── 카드 컴포넌트 ── */
  .hz-card {
    background: var(--bg-card);
    border-radius: 20px;
    padding: 1.5rem 1.8rem;
    border: 1px solid var(--border);
    box-shadow: 14px 17px 40px 4px rgba(112,144,176,0.08);
    margin-bottom: 1rem;
  }
  .hz-card-sm {
    background: var(--bg-card);
    border-radius: 16px;
    padding: 1.2rem 1.4rem;
    border: 1px solid var(--border);
    box-shadow: 14px 17px 40px 4px rgba(112,144,176,0.06);
    margin-bottom: 0.75rem;
  }

  /* ── 스탯 카드 ── */
  .hz-stat {
    background: var(--bg-card);
    border-radius: 20px;
    padding: 1.4rem 1.6rem;
    border: 1px solid var(--border);
    box-shadow: 14px 17px 40px 4px rgba(112,144,176,0.08);
    display: flex;
    align-items: center;
    gap: 1rem;
  }
  .hz-stat-icon {
    width: 56px; height: 56px;
    border-radius: 50%;
    background: var(--gradient);
    display: flex; align-items: center; justify-content: center;
    font-size: 1.4rem;
    flex-shrink: 0;
  }
  .hz-stat-label {
    font-size: 0.75rem;
    font-weight: 500;
    color: var(--text-sub);
    text-transform: uppercase;
    letter-spacing: 0.04em;
    margin-bottom: 0.25rem;
  }
  .hz-stat-value {
    font-size: 1.5rem;
    font-weight: 700;
    color: var(--text-main);
    line-height: 1.2;
  }

  /* ── 페이지 타이틀 ── */
  .hz-page-title {
    font-size: 1.6rem;
    font-weight: 700;
    color: var(--text-main);
    margin: 0 0 0.2rem;
  }
  .hz-page-sub {
    font-size: 0.88rem;
    color: var(--text-sub);
    margin: 0 0 1.5rem;
  }

  /* ── 섹션 헤더 ── */
  .hz-section-header {
    font-size: 1rem;
    font-weight: 700;
    color: var(--text-main);
    margin-bottom: 0.75rem;
  }

  /* ── 배지 ── */
  .hz-badge {
    display: inline-block;
    padding: 0.18rem 0.65rem;
    border-radius: 999px;
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.02em;
  }
  .badge-brand  { background: #EBE9FF; color: var(--brand); }
  .badge-green  { background: #E6FAF5; color: var(--success); }
  .badge-orange { background: #FFF3DF; color: var(--warning); }
  .badge-red    { background: #FFEAEA; color: var(--danger); }
  .badge-blue   { background: #E3F8FF; color: var(--info); }
  .badge-gray   { background: var(--border); color: var(--text-body); }

  /* ── 질문 유형 레이블 ── */
  .qt-pill {
    display: inline-flex; align-items: center; gap: 0.3rem;
    padding: 0.3rem 0.9rem;
    border-radius: 999px;
    font-size: 0.78rem;
    font-weight: 600;
  }

  /* ── 검색 결과 카드 ── */
  .result-card {
    background: var(--bg-card);
    border-radius: 16px;
    padding: 1.2rem 1.5rem;
    border-left: 4px solid var(--brand);
    box-shadow: 0 2px 12px rgba(67,24,255,0.06);
    margin-bottom: 0.85rem;
  }
  .result-meta {
    font-size: 0.75rem;
    color: var(--text-sub);
    margin-bottom: 0.4rem;
  }
  .result-title {
    font-size: 0.92rem;
    font-weight: 600;
    color: var(--text-main);
    margin-bottom: 0.4rem;
  }
  .result-body {
    font-size: 0.85rem;
    color: var(--text-body);
    line-height: 1.65;
  }
  .result-score {
    font-size: 0.72rem;
    font-weight: 700;
    color: var(--brand);
    background: #EBE9FF;
    padding: 0.15rem 0.5rem;
    border-radius: 999px;
  }

  /* ── 리포트 출력 영역 ── */
  .report-area {
    background: var(--bg-card);
    border-radius: 20px;
    padding: 2rem 2.4rem;
    border: 1px solid var(--border);
    box-shadow: 14px 17px 40px 4px rgba(112,144,176,0.08);
    line-height: 1.8;
    color: var(--text-main);
  }
  .report-area h1,h2,h3 { color: var(--text-main); }

  /* ── 입력 필드 커스터마이징 ── */
  .stTextInput > div > div > input,
  .stTextArea > div > div > textarea {
    border-radius: 12px !important;
    border: 1px solid var(--border) !important;
    background: var(--bg-page) !important;
    color: var(--text-main) !important;
    font-size: 0.92rem !important;
    padding: 0.6rem 1rem !important;
  }
  .stTextInput > div > div > input:focus,
  .stTextArea > div > div > textarea:focus {
    border-color: var(--brand) !important;
    box-shadow: 0 0 0 3px rgba(67,24,255,0.12) !important;
  }

  /* ── 버튼 ── */
  .stButton > button {
    background: var(--gradient) !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 0.55rem 1.6rem !important;
    font-weight: 600 !important;
    font-size: 0.88rem !important;
    transition: opacity 0.15s ease !important;
    box-shadow: 0 4px 16px rgba(67,24,255,0.3) !important;
  }
  .stButton > button:hover { opacity: 0.88 !important; }

  /* ── 라디오/셀렉트 ── */
  .stRadio label { color: var(--text-body) !important; font-size: 0.88rem !important; }
  .stSelectbox > div > div { border-radius: 12px !important; }

  /* ── 구분선 ── */
  hr { border: none; border-top: 1px solid var(--border); margin: 1.2rem 0; }

  /* ── 사이드바 메뉴 아이템 ── */
  .nav-item {
    display: flex;
    align-items: center;
    gap: 0.7rem;
    padding: 0.65rem 1rem;
    border-radius: 12px;
    cursor: pointer;
    margin-bottom: 0.2rem;
    font-size: 0.88rem;
    font-weight: 500;
    color: var(--text-body);
    transition: all 0.15s;
  }
  .nav-item.active {
    background: var(--gradient);
    color: white;
    font-weight: 600;
  }
  .nav-item:hover:not(.active) {
    background: var(--bg-page);
    color: var(--brand);
  }

  /* ── 로고 ── */
  .sidebar-logo {
    display: flex;
    align-items: center;
    gap: 0.75rem;
    padding: 0.5rem 0 1.5rem;
    margin-bottom: 0.5rem;
    border-bottom: 1px solid var(--border);
  }
  .sidebar-logo-icon {
    width: 42px; height: 42px;
    background: var(--gradient);
    border-radius: 12px;
    display: flex; align-items: center; justify-content: center;
    font-size: 1.3rem;
  }
  .sidebar-logo-text {
    font-size: 1rem;
    font-weight: 700;
    color: var(--text-main);
    line-height: 1.2;
  }
  .sidebar-logo-sub {
    font-size: 0.72rem;
    color: var(--text-sub);
  }

  /* ── 스피너 오버라이드 ── */
  .stSpinner > div { color: var(--brand) !important; }

  /* ── expander ── */
  details { background: var(--bg-page) !important; border-radius: 12px !important; }

  /* ── hide default streamlit elements ── */
  #MainMenu { visibility: hidden; }
  footer { visibility: hidden; }
  /* 헤더 자체는 숨기지 않음 — 사이드바 토글 버튼이 여기 있음 */
  /* 헤더를 투명하게만 처리 */
  header { background: transparent !important; box-shadow: none !important; }
</style>
""", unsafe_allow_html=True)


# ── 전략 모듈 import (캐싱) ────────────────────────────────────────────────────

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
    # router.build_retriever → (ret1_instance, ret2_instance, all_docs) 튜플 반환
    retriever_tuple = ROUTER.build_retriever(vectorstore, all_docs, k=40)
    return retriever_tuple, ROUTER, RERANKER, EMBEDDING, VECTORSTORE, CHUNKING, len(all_docs)


# ── 헬퍼 ───────────────────────────────────────────────────────────────────────

QUESTION_TYPE_META = {
    "fact_lookup":       ("🔍", "사실 확인",     "badge-blue"),
    "coverage_summary":  ("📋", "커버리지 요약",  "badge-green"),
    "timeline":          ("📅", "타임라인",       "badge-orange"),
    "broker_comparison": ("⚖️",  "증권사 비교",   "badge-brand"),
    "risk":              ("⚠️",  "리스크 분석",   "badge-red"),
    "consensus":         ("🤝", "컨센서스",       "badge-green"),
    "other":             ("📝", "종합 리포트",    "badge-gray"),
}

SECTOR_COLORS = {
    "반도체": "badge-blue",
    "조선":   "badge-green",
    "AI인프라":"badge-brand",
    "자동차": "badge-orange",
    "바이오": "badge-red",
}

def score_bar(score: float | str) -> str:
    try:
        s = float(score)
        pct = max(0, min(100, int(s * 100)))
        color = "#01B574" if pct >= 70 else "#FFB547" if pct >= 40 else "#EE5D50"
        return f"""
        <div style="display:flex;align-items:center;gap:0.5rem;margin-top:0.4rem;">
          <div style="flex:1;height:6px;background:#E9EDF7;border-radius:999px;overflow:hidden;">
            <div style="width:{pct}%;height:100%;background:{color};border-radius:999px;"></div>
          </div>
          <span style="font-size:0.72rem;font-weight:700;color:{color};">{s:.3f}</span>
        </div>"""
    except Exception:
        return ""


def sector_badge(sector: str) -> str:
    cls = SECTOR_COLORS.get(sector, "badge-gray")
    return f'<span class="hz-badge {cls}">{sector}</span>'


# ── 사이드바 ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("""
    <div class="sidebar-logo">
      <div class="sidebar-logo-icon">📊</div>
      <div>
        <div class="sidebar-logo-text">ResearchRAG</div>
        <div class="sidebar-logo-sub">증권사 리포트 분석 시스템</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    page = st.radio(
        "페이지 선택",
        options=["🏠  대시보드", "💬  질문 · 분석", "📂  최근 리포트"],
        label_visibility="collapsed",
    )
    page = page.split("  ", 1)[-1].strip()

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown('<div style="font-size:0.72rem;font-weight:700;color:var(--text-sub);letter-spacing:0.08em;padding:0 0.3rem 0.4rem;">검색 설정</div>', unsafe_allow_html=True)

    top_n = st.slider("최종 결과 수 (Top-N)", min_value=3, max_value=20, value=10)
    show_content = st.toggle("청크 전문 표시", value=False)

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown('<div style="font-size:0.72rem;font-weight:700;color:var(--text-sub);letter-spacing:0.08em;padding:0 0.3rem 0.4rem;">시스템 정보</div>', unsafe_allow_html=True)

    with st.spinner("벡터스토어 로드 중…"):
        retriever, ROUTER, RERANKER, EMBEDDING, VECTORSTORE, CHUNKING, total_chunks = load_backend()

    db_ok = retriever is not None
    if db_ok:
        st.markdown(f"""
        <div style="font-size:0.78rem;color:var(--text-body);padding:0.2rem 0.3rem;">
          <div>✅ ChromaDB 연결됨</div>
          <div style="color:var(--text-sub);margin-top:0.3rem;">청크 수: <b style="color:var(--text-main);">{total_chunks:,}</b></div>
          <div style="color:var(--text-sub);">청킹: <b style="color:var(--text-main);">{CHUNKING.STRATEGY_NAME}</b></div>
          <div style="color:var(--text-sub);">리트리버: <b style="color:var(--text-main);">router (ensemble ↔ balanced)</b></div>
          <div style="color:var(--text-sub);">리랭커: <b style="color:var(--text-main);">{RERANKER.STRATEGY_NAME}</b></div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.error("ChromaDB 없음 — `pipeline/ingest.py` 실행 필요")


# ── 페이지: 대시보드 ───────────────────────────────────────────────────────────

if page == "대시보드":
    st.markdown('<p class="hz-page-title">대시보드</p>', unsafe_allow_html=True)
    st.markdown('<p class="hz-page-sub">리서치 리포트 RAG 시스템 현황</p>', unsafe_allow_html=True)

    # 스탯 카드 행
    c1, c2, c3, c4 = st.columns(4)

    def stat_card(col, icon, label, value):
        col.markdown(f"""
        <div class="hz-stat">
          <div class="hz-stat-icon">{icon}</div>
          <div>
            <div class="hz-stat-label">{label}</div>
            <div class="hz-stat-value">{value}</div>
          </div>
        </div>
        """, unsafe_allow_html=True)

    stat_card(c1, "📦", "총 청크 수", f"{total_chunks:,}" if db_ok else "—")
    stat_card(c2, "🏢", "커버 증권사", "12개")
    stat_card(c3, "📄", "질문 유형", "7가지")
    stat_card(c4, "🧠", "임베딩 모델", "text-embedding-3-small")

    st.markdown("<br>", unsafe_allow_html=True)

    left, right = st.columns([3, 2])

    with left:
        st.markdown('<div class="hz-section-header">지원 질문 유형</div>', unsafe_allow_html=True)
        for qt, (icon, label, badge) in QUESTION_TYPE_META.items():
            st.markdown(f"""
            <div class="hz-card-sm" style="display:flex;align-items:center;gap:0.8rem;">
              <span style="font-size:1.2rem;">{icon}</span>
              <div style="flex:1;">
                <span class="hz-badge {badge}">{label}</span>
                <span style="font-size:0.78rem;color:var(--text-sub);margin-left:0.5rem;">{qt}</span>
              </div>
            </div>
            """, unsafe_allow_html=True)

    with right:
        st.markdown('<div class="hz-section-header">커버 증권사</div>', unsafe_allow_html=True)
        brokers = ["하나증권", "키움증권", "DS투자증권", "IBK투자증권", "SK증권",
                   "교보증권", "대신증권", "유안타증권", "유진투자증권",
                   "한화투자증권", "iM증권", "한국IR협의회"]
        for b in brokers:
            st.markdown(f"""
            <div style="display:flex;align-items:center;gap:0.6rem;padding:0.45rem 0.8rem;
                        background:var(--bg-card);border-radius:10px;margin-bottom:0.4rem;
                        border:1px solid var(--border);font-size:0.82rem;color:var(--text-body);">
              <span style="width:8px;height:8px;background:var(--brand);border-radius:50%;display:inline-block;"></span>
              {b}
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="hz-section-header">커버 섹터</div>', unsafe_allow_html=True)
        sectors = ["반도체", "조선", "AI인프라", "자동차", "바이오"]
        cols = st.columns(3)
        for i, s in enumerate(sectors):
            cols[i % 3].markdown(f'<span class="hz-badge {SECTOR_COLORS.get(s,"badge-gray")}" style="margin:0.2rem;">{s}</span>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="hz-card">', unsafe_allow_html=True)
    st.markdown('<div class="hz-section-header">빠른 시작 — 질문 예시</div>', unsafe_allow_html=True)
    examples = [
        ("⚖️", "하나증권과 키움증권의 3월 반도체 의견 차이"),
        ("⚠️", "조선업에서 언급된 리스크 요인 정리해줘"),
        ("🤝", "AI 인프라에 대해 증권사들이 공통으로 강조하는 게 뭐야"),
        ("📅", "이번 달 반도체 섹터 투자의견 변화"),
        ("📋", "최근 AI 인프라 리포트 현황 정리해줘"),
    ]
    for icon, ex in examples:
        st.markdown(f"""
        <div style="display:flex;align-items:center;gap:0.7rem;padding:0.5rem 0.8rem;
                    background:var(--bg-page);border-radius:10px;margin-bottom:0.35rem;
                    font-size:0.84rem;color:var(--text-body);">
          <span>{icon}</span> {ex}
        </div>
        """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)


# ── 페이지: 검색 ───────────────────────────────────────────────────────────────

elif page == "__검색_삭제됨__":
    st.markdown('<p class="hz-page-title">청크 검색</p>', unsafe_allow_html=True)
    st.markdown('<p class="hz-page-sub">Hybrid Search (BM25 + Vector) + BGE Cross-Encoder 리랭킹</p>', unsafe_allow_html=True)

    if not db_ok:
        st.error("ChromaDB가 없습니다. 먼저 `pipeline/ingest.py`를 실행해 주세요.")
        st.stop()

    with st.form("search_form"):
        query = st.text_input("검색 키워드", placeholder="예: 반도체 업황, 조선 수주, AI 인프라 투자…")
        submitted = st.form_submit_button("🔎  검색")

    if submitted and query.strip():
        with st.spinner("검색 중…"):
            t0 = time.time()
            candidates = RETRIEVER.retrieve(retriever, query, k=40)
            docs       = RERANKER.rerank(query, candidates, top_n=top_n)
            elapsed    = time.time() - t0

        st.markdown(f"""
        <div class="hz-card" style="display:flex;align-items:center;gap:1rem;padding:1rem 1.5rem;">
          <div style="flex:1;">
            <span style="font-weight:700;color:var(--text-main);">"{query}"</span>
            <span style="color:var(--text-sub);font-size:0.82rem;margin-left:0.5rem;">검색 완료</span>
          </div>
          <div style="text-align:right;">
            <div style="font-size:0.75rem;color:var(--text-sub);">후보 → 최종</div>
            <div style="font-weight:700;color:var(--brand);">{len(candidates)} → {len(docs)}개</div>
          </div>
          <div style="text-align:right;">
            <div style="font-size:0.75rem;color:var(--text-sub);">소요 시간</div>
            <div style="font-weight:700;color:var(--text-main);">{elapsed:.2f}s</div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        for i, doc in enumerate(docs, 1):
            score  = doc.metadata.get("rerank_score", "-")
            broker = doc.metadata.get("source_firm",  doc.metadata.get("broker", "-"))
            date   = doc.metadata.get("report_date",  "-")
            sector = doc.metadata.get("sector",        "-")
            title  = doc.metadata.get("title",         "")
            content = doc.page_content

            sector_html = sector_badge(sector) if sector and sector != "-" else ""

            try:
                score_val = float(score)
                pct = max(0, min(100, int(score_val * 100)))
                bar_color = "#01B574" if pct >= 70 else "#FFB547" if pct >= 40 else "#EE5D50"
                score_html = f"""
                <div style="display:flex;align-items:center;gap:0.5rem;margin-top:0.5rem;">
                  <div style="width:120px;height:5px;background:#E9EDF7;border-radius:999px;overflow:hidden;">
                    <div style="width:{pct}%;height:100%;background:{bar_color};border-radius:999px;"></div>
                  </div>
                  <span style="font-size:0.72rem;font-weight:700;color:{bar_color};">Rerank {score_val:.3f}</span>
                </div>"""
            except Exception:
                score_html = ""

            preview = content[:350] + ("…" if len(content) > 350 else "")

            st.markdown(f"""
            <div class="result-card">
              <div class="result-meta">
                <b style="color:var(--text-main);">#{i}</b>
                &nbsp;·&nbsp; 🏢 {broker}
                &nbsp;·&nbsp; 📅 {date}
                &nbsp;&nbsp; {sector_html}
              </div>
              {'<div class="result-title">' + title[:60] + '</div>' if title else ''}
              {score_html}
              {'<div class="result-body" style="margin-top:0.6rem;">' + preview + '</div>' if show_content else ''}
            </div>
            """, unsafe_allow_html=True)

            if show_content and len(content) > 350:
                with st.expander("전문 보기"):
                    st.text(content)

    elif submitted:
        st.warning("검색 키워드를 입력해 주세요.")


# ── 페이지: 질문 · 분석 ────────────────────────────────────────────────────────

elif page == "질문 · 분석":
    st.markdown('<p class="hz-page-title">질문 · 분석 리포트</p>', unsafe_allow_html=True)
    st.markdown('<p class="hz-page-sub">자유형 질문을 입력하면 유형을 자동 감지해 분석 리포트를 생성합니다.</p>', unsafe_allow_html=True)

    if not db_ok:
        st.error("ChromaDB가 없습니다. 먼저 `pipeline/ingest.py`를 실행해 주세요.")
        st.stop()

    from src.reportcreator.freeform_chain import answer_question

    # 예시 질문 버튼
    st.markdown('<div class="hz-section-header">빠른 질문 선택</div>', unsafe_allow_html=True)
    examples = [
        "하나증권과 키움증권의 3월 반도체 의견 차이",
        "조선업에서 언급된 리스크 요인 정리해줘",
        "AI 인프라에 대해 증권사들이 공통으로 강조하는 게 뭐야",
        "이번 달 반도체 섹터 투자의견 변화",
    ]
    ex_cols = st.columns(len(examples))
    preset_q = ""
    for col, ex in zip(ex_cols, examples):
        if col.button(ex[:18] + "…", key=f"ex_{ex}"):
            preset_q = ex

    st.markdown("<br>", unsafe_allow_html=True)

    with st.form("ask_form"):
        question = st.text_area(
            "질문 입력",
            value=preset_q,
            placeholder="예: 하나증권과 키움증권의 3월 반도체 의견 차이를 설명해줘",
            height=100,
        )
        submitted = st.form_submit_button("💬  분석 리포트 생성")

    if submitted and question.strip():
        progress_bar = st.progress(0, text="질문 유형 분류 중…")

        with st.spinner("분석 리포트 생성 중 (수십 초 소요)…"):
            t0 = time.time()
            progress_bar.progress(20, text="문서 검색 중…")
            result = answer_question(
                retriever, question,
                retrieve_fn=lambda r, q, k: ROUTER.retrieve(r, q, k=k),
                rerank_fn=lambda q, docs, top_n: RERANKER.rerank(q, docs, top_n=top_n),
            )
            elapsed = time.time() - t0
            progress_bar.progress(100, text="완료!")

        qt   = result.get("question_type", "other")
        icon, label, badge = QUESTION_TYPE_META.get(qt, ("📝", "분석", "badge-gray"))
        sources = result.get("sources", [])

        # 메타 요약 카드
        st.markdown(f"""
        <div class="hz-card" style="display:flex;flex-wrap:wrap;gap:1.2rem;align-items:center;padding:1rem 1.5rem;">
          <div>
            <div style="font-size:0.72rem;color:var(--text-sub);margin-bottom:0.25rem;">질문 유형</div>
            <span class="hz-badge {badge}" style="font-size:0.82rem;padding:0.3rem 0.8rem;">
              {icon} {label}
            </span>
          </div>
          <div>
            <div style="font-size:0.72rem;color:var(--text-sub);margin-bottom:0.25rem;">참고 증권사</div>
            <div style="font-size:0.85rem;font-weight:600;color:var(--text-main);">
              {', '.join(sources) if sources else '—'}
            </div>
          </div>
          <div style="margin-left:auto;text-align:right;">
            <div style="font-size:0.72rem;color:var(--text-sub);">생성 시간</div>
            <div style="font-weight:700;color:var(--brand);">{elapsed:.1f}s</div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        # 리포트 본문
        st.markdown('<div class="report-area">', unsafe_allow_html=True)
        st.markdown(result["answer"])
        st.markdown('</div>', unsafe_allow_html=True)

        # 다운로드
        st.download_button(
            "📥  리포트 저장 (.md)",
            data=result["answer"],
            file_name=f"report_{qt}_{int(t0)}.md",
            mime="text/markdown",
        )

        # 참조 청크 출처
        retrieved_docs = result.get("docs", [])
        if retrieved_docs:
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown('<div class="hz-section-header">📎 참조 청크 출처</div>', unsafe_allow_html=True)
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
                        pct = max(0, min(100, int(s * 100)))
                        bar_color = "#01B574" if pct >= 70 else "#FFB547" if pct >= 40 else "#EE5D50"
                        score_html = f'<span style="font-size:0.7rem;font-weight:700;color:{bar_color};background:{bar_color}22;padding:0.1rem 0.45rem;border-radius:999px;">score {s:.3f}</span>'
                    except Exception:
                        pass

                sector_html = sector_badge(sector) if sector and sector != "-" else ""
                preview = content[:200] + ("…" if len(content) > 200 else "")

                with st.expander(f"#{i}  {broker}  ·  {date}  {'· ' + title[:30] if title else ''}"):
                    st.markdown(f"""
                    <div style="display:flex;flex-wrap:wrap;gap:0.5rem;margin-bottom:0.6rem;align-items:center;">
                      {sector_html} {score_html}
                    </div>
                    <div style="font-size:0.83rem;color:var(--text-body);line-height:1.7;">{preview}</div>
                    """, unsafe_allow_html=True)
                    if len(content) > 200:
                        if st.toggle("전문 보기", key=f"chunk_full_{i}"):
                            st.text(content)

    elif submitted:
        st.warning("질문을 입력해 주세요.")


# ── 페이지: 최근 리포트 ─────────────────────────────────────────────────────────

elif page == "최근 리포트":
    st.markdown('<p class="hz-page-title">생성된 리포트</p>', unsafe_allow_html=True)
    st.markdown('<p class="hz-page-sub">data/reports_output/ 폴더에 저장된 리포트 목록</p>', unsafe_allow_html=True)

    reports_dir = PROJECT_ROOT / "data" / "reports_output"
    md_files = sorted(reports_dir.glob("*.md"), key=lambda f: f.stat().st_mtime, reverse=True) if reports_dir.exists() else []

    if not md_files:
        st.markdown("""
        <div class="hz-card" style="text-align:center;padding:3rem;">
          <div style="font-size:2.5rem;margin-bottom:0.8rem;">📂</div>
          <div style="font-size:1rem;font-weight:600;color:var(--text-main);">아직 생성된 리포트가 없습니다</div>
          <div style="font-size:0.85rem;color:var(--text-sub);margin-top:0.4rem;">
            질문·분석 탭에서 리포트를 생성해보세요.
          </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f'<div style="font-size:0.82rem;color:var(--text-sub);margin-bottom:1rem;">총 {len(md_files)}개의 리포트</div>', unsafe_allow_html=True)
        for f in md_files:
            mtime = time.strftime("%Y-%m-%d %H:%M", time.localtime(f.stat().st_mtime))
            size_kb = f.stat().st_size / 1024
            with st.expander(f"📄 {f.name}  —  {mtime}  ({size_kb:.1f} KB)"):
                content = f.read_text(encoding="utf-8", errors="replace")
                st.markdown(content)
                st.download_button(
                    "📥 다운로드",
                    data=content,
                    file_name=f.name,
                    mime="text/markdown",
                    key=f"dl_{f.name}",
                )
