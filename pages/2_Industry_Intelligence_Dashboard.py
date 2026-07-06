"""
Industry Collaboration Intelligence Dashboard
NUS Research Strategy & Industry Collaboration Office
Singapore Landscape · 2020–2025
"""

import re
import warnings
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

try:
    import anthropic as _ant
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False

try:
    import snowflake.connector
    SNOWFLAKE_AVAILABLE = True
except ImportError:
    SNOWFLAKE_AVAILABLE = False

warnings.filterwarnings("ignore")

# ── NUS COLOUR PALETTE ────────────────────────────────────────────────────────
NUS_BLUE   = "#003D7C"
NUS_ORANGE = "#EF7C00"
NUS_LBLUE  = "#006CB7"
NUS_GOLD   = "#C0923F"
WHITE      = "#FFFFFF"
OFFWHITE   = "#F5F7FA"
SLATE      = "#3D4F61"
MUTED      = "#7A8FA6"
LIGHT      = "#D6E4F0"
RED        = "#B03030"
GREEN      = "#1A7A4A"
AMBER      = "#C07000"
CHART_COLS = [
    "#1f77b4",  # steel blue
    "#ff7f0e",  # orange
    "#2ca02c",  # green
    "#d62728",  # red
    "#9467bd",  # purple
    "#17becf",  # cyan
    "#e377c2",  # pink
    "#bcbd22",  # olive
    "#8c564b",  # brown
    "#003D7C",  # NUS navy
]

# ── GLOBAL STYLES ─────────────────────────────────────────────────────────────
st.markdown(f"""
<style>
/* ----- hide sidebar page nav (top buttons handle navigation) ----- */
[data-testid="stSidebarNav"]     {{ display:none !important; }}

/* ----- base ----- */
html, body, [class*="css"]          {{ font-size:15px; }}

/* ----- sidebar ----- */
[data-testid="stSidebar"]           {{ background:{NUS_BLUE} !important; }}
[data-testid="stSidebar"] *         {{ color:{LIGHT} !important; }}
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3        {{ color:{WHITE} !important; }}
[data-testid="stSidebar"] .stTextArea textarea {{
    background:#0A3060; color:{WHITE} !important;
    border:1px solid #1A5090; border-radius:6px; font-size:14px;
}}
[data-testid="stSidebar"] button,
[data-testid="stSidebar"] .stButton > button,
section[data-testid="stSidebar"] button {{
    background:#2E6DA4 !important; color:{WHITE} !important; border:none !important;
    border-radius:6px !important; font-weight:600 !important; font-size:14px !important;
}}
[data-testid="stSidebar"] button:hover,
[data-testid="stSidebar"] .stButton > button:hover,
section[data-testid="stSidebar"] button:hover {{
    background:#245888 !important; color:{WHITE} !important;
}}

/* ----- cards ----- */
.kpi-card {{
    background:{WHITE}; border-radius:10px;
    padding:1rem 1.2rem; border-left:5px solid {NUS_ORANGE};
    box-shadow:0 2px 8px rgba(0,0,0,0.07);
}}
.kpi-val {{ font-size:2rem; font-weight:800; color:{NUS_BLUE}; line-height:1.1; }}
.kpi-lbl {{ font-size:14px; color:{SLATE}; margin-top:0.2rem; }}
.kpi-sub {{ font-size:13px; color:{MUTED}; }}

.spotlight-card {{
    background:linear-gradient(135deg,{NUS_BLUE},{NUS_LBLUE});
    color:{WHITE}; border-radius:10px; padding:1rem 1.2rem;
    box-shadow:0 4px 12px rgba(0,61,124,0.22);
    min-height:160px; display:flex; flex-direction:column;
}}
.spotlight-num  {{ font-size:2.2rem; font-weight:800; color:{NUS_ORANGE}; }}
.spotlight-lbl  {{ font-size:14px; color:#B0CCE8; margin-top:0.1rem; font-weight:600; }}
.spotlight-body {{ font-size:13px; color:#D6E4F0; margin-top:0.4rem; line-height:1.55; flex:1; }}

.exec-box {{
    background:{WHITE}; border-radius:10px; padding:1.2rem 1.4rem;
    border-top:4px solid {NUS_ORANGE};
    box-shadow:0 2px 8px rgba(0,0,0,0.06); margin-bottom:0.8rem;
}}
.exec-title {{ font-size:16px; font-weight:700; color:{NUS_BLUE}; margin-bottom:0.5rem; }}
.exec-body  {{ font-size:14px; color:{SLATE}; line-height:1.7; }}

/* ----- callout boxes ----- */
.insight {{
    background:#EAF2FB; border-left:5px solid {NUS_BLUE};
    padding:0.7rem 1rem; border-radius:0 8px 8px 0;
    margin:0.5rem 0 0.8rem; font-size:14px; color:{NUS_BLUE};
    line-height:1.6;
}}
.warn-box {{
    background:#FEF3E2; border-left:5px solid {NUS_ORANGE};
    padding:0.7rem 1rem; border-radius:0 8px 8px 0;
    margin:0.5rem 0 0.8rem; font-size:14px; color:#7A3800;
    line-height:1.6;
}}
.alert-box {{
    background:#FDEDEC; border-left:5px solid {RED};
    padding:0.7rem 1rem; border-radius:0 8px 8px 0;
    margin:0.5rem 0 0.8rem; font-size:14px; color:#78281F;
    line-height:1.6;
}}

/* ----- section label ----- */
/* min-height + bottom-aligned text so 1-line and 2-line headers reserve the same
   vertical space — keeps side-by-side charts aligned across columns */
.section-label {{
    font-size:13px; font-weight:700; letter-spacing:0.07em;
    color:{NUS_ORANGE}; text-transform:uppercase;
    margin:1.2rem 0 0.4rem; border-bottom:1px solid {LIGHT};
    padding-bottom:0.2rem;
    min-height:2.5em; display:flex; align-items:flex-end;
}}

/* ----- sg banner ----- */
.sg-banner {{
    background:linear-gradient(90deg,{NUS_BLUE},{NUS_LBLUE});
    color:{WHITE}; padding:0.5rem 1rem; border-radius:8px;
    font-size:14px; margin-bottom:0.8rem;
}}

/* ----- llm result panel ----- */
.llm-panel {{
    background:{WHITE}; border:1px solid {LIGHT};
    border-radius:10px; padding:1rem 1.2rem;
    box-shadow:0 2px 10px rgba(0,0,0,0.07);
    margin-bottom:0.8rem;
}}
.llm-q   {{ font-size:14px; font-weight:700; color:{NUS_BLUE}; margin-bottom:0.4rem; }}
.llm-err {{ font-size:14px; color:{RED}; }}

/* ----- page title / subtitle ----- */
.page-title    {{ color:{NUS_BLUE}; margin:0 0 0.1rem; font-size:1.7rem; font-weight:700; line-height:1.2; }}
.page-subtitle {{ color:{SLATE}; font-size:14px; margin:0; }}

/* ----- dark mode overrides ----- */
@media (prefers-color-scheme: dark) {{
    .page-title    {{ color:#7BB3E0; }}
    .page-subtitle {{ color:#8AAAC4; }}
    .kpi-card {{
        background:#1C2D3E;
        box-shadow:0 2px 8px rgba(0,0,0,0.4);
    }}
    .kpi-val  {{ color:#B0CCE8; }}
    .kpi-lbl  {{ color:#8AAAC4; }}
    .kpi-sub  {{ color:#6A8A9E; }}

    .exec-box {{
        background:#1C2D3E;
        box-shadow:0 2px 8px rgba(0,0,0,0.4);
    }}
    .exec-title {{ color:#B0CCE8; }}
    .exec-body  {{ color:#A0B8CC; }}

    .insight {{
        background:#0A2A4A; border-left-color:#4A8FCC;
        color:#B0CCE8;
    }}
    .warn-box {{
        background:#2A1800; border-left-color:{NUS_ORANGE};
        color:#F4B860;
    }}
    .alert-box {{
        background:#2A0A0A; border-left-color:#CC4444;
        color:#F4A0A0;
    }}

    .section-label {{ border-bottom-color:#2A4A6A; }}

    .llm-panel {{
        background:#1C2D3E; border-color:#2A4A6A;
        box-shadow:0 2px 10px rgba(0,0,0,0.4);
    }}
    .llm-q {{ color:#B0CCE8; }}
}}
</style>
""", unsafe_allow_html=True)


# ── DATA LAYER ────────────────────────────────────────────────────────────────
def _load_private_key(pem_str: str) -> bytes:
    """Load a PEM private key string and return DER bytes for Snowflake auth."""
    from cryptography.hazmat.primitives import serialization
    p_key = serialization.load_pem_private_key(pem_str.encode(), password=None)
    return p_key.private_bytes(
        encoding=serialization.Encoding.DER,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    )


@st.cache_resource(show_spinner=False)
def get_conn():
    if not SNOWFLAKE_AVAILABLE:
        return None
    cfg = st.secrets["snowflake_intel"]
    return snowflake.connector.connect(
        account=cfg["account"], user=cfg["user"],
        private_key=_load_private_key(cfg["private_key"]),
        database=cfg["database"], schema=cfg["schema"],
        warehouse=cfg["warehouse"], role=cfg["role"],
        client_session_keep_alive=True,
    )

@st.cache_data(ttl=3600, show_spinner="Loading data…")
def sql(query: str) -> pd.DataFrame:
    conn = get_conn()
    cur  = conn.cursor()
    cur.execute(query)
    return cur.fetch_pandas_all()

TBL = (f"{st.secrets['snowflake_intel']['database']}."
       f"{st.secrets['snowflake_intel']['schema']}."
       f"{st.secrets['snowflake_intel']['table']}")


# ── CHART HELPERS ─────────────────────────────────────────────────────────────
def clean_fig(fig, h=380):
    fig.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(size=13),
        margin=dict(t=25, b=50, l=10, r=30),
        height=h,
        legend=dict(font=dict(size=12), bgcolor="rgba(0,0,0,0)"),
    )
    fig.update_xaxes(showgrid=False, linecolor="rgba(128,128,128,0.4)", tickcolor="rgba(128,128,128,0.4)",
                     tickfont=dict(size=12))
    fig.update_yaxes(gridcolor="rgba(128,128,128,0.2)", linecolor="rgba(0,0,0,0)",
                     tickfont=dict(size=12))
    return fig

def hbar(df, x, y, h=380, color=NUS_BLUE):
    # Truncate long labels so they don't get clipped
    df = df.copy()
    df[y] = df[y].astype(str).str[:40]
    fig = px.bar(df, x=x, y=y, orientation="h",
                 color_discrete_sequence=[color],
                 labels={x: "", y: ""})
    fig.update_layout(
        yaxis=dict(autorange="reversed", automargin=True),
        margin=dict(t=25, b=50, l=220, r=30),  # generous left margin for labels
    )
    return clean_fig(fig, h)

def insight(md):
    st.markdown(f"<div class='insight'>▶ {md}</div>", unsafe_allow_html=True)

def warn(md):
    st.markdown(f"<div class='warn-box'>⚠ {md}</div>", unsafe_allow_html=True)

def alert(md):
    st.markdown(f"<div class='alert-box'>⚠ {md}</div>", unsafe_allow_html=True)

def section(label):
    st.markdown(f"<div class='section-label'>{label}</div>", unsafe_allow_html=True)


# ── LLM HELPERS ───────────────────────────────────────────────────────────────
SCHEMA_CONTEXT = f"""
You are a senior data analyst writing SQL for Snowflake.

TABLE: {TBL}  (Singapore research ecosystem — patents and publications, 2020–2025)

COLUMNS:
- UID                          VARCHAR  unique record ID
- TITLE                        VARCHAR  title of work
- IP_TYPE                      VARCHAR  'Publications' or 'Patents'
- APPLICATION_PUBLICATION_YEAR NUMBER   year 2020–2025
- NUS_IP                       BOOLEAN  TRUE = NUS-owned/affiliated
- QS_SUBJECT_AREA              VARCHAR  broad QS faculty area — PREFERRED subject dimension. 5 values: ENGINEERING & TECHNOLOGY, NATURAL SCIENCES, LIFE SCIENCES & MEDICINE, SOCIAL SCIENCES & MANAGEMENT, ARTS & HUMANITIES. Multiple values '|' separated (both IP types). Placeholder '-' means unmapped — exclude it.
- QS_SUBJECT                   VARCHAR  granular subject (e.g. 'DATA SCIENCE'); pipe '|' separated (both IP types). Use only when a specific granular subject is requested.
- NORMALIZED_NAMES_CONCAT      VARCHAR  institution names separated by '|'
- N_CORPORATE                  NUMBER   count of corporate co-authors/co-inventors
- N_INSTITUTE                  NUMBER   count of institute collaborators
- N_HOSPITAL                   NUMBER   count of hospital collaborators
- N_GOV_NONPROFIT              NUMBER   count of gov/nonprofit collaborators
- UNITS                        VARCHAR  NUS unit/dept, '|' separated for multiple
- AUTHORS                      VARCHAR  pipe '|' separated author names (publications only)
- AUTHOR_COUNT                 NUMBER   number of authors

KEY PATTERNS:
- NUS records:             WHERE NUS_IP = TRUE
- Industry collaborations: WHERE N_CORPORATE > 0
- Split subject area (default, both IP types):
                           LATERAL FLATTEN(INPUT=>SPLIT(QS_SUBJECT_AREA,'|')) f
                           then TRIM(f.VALUE::STRING); exclude '' and '-'
- Split granular subject (both IP types, '|'):  only when a specific subject is asked for
- Split institutions:      LATERAL FLATTEN(INPUT=>SPLIT(NORMALIZED_NAMES_CONCAT,'|')) f
- Split units:             LATERAL FLATTEN(INPUT=>SPLIT(UNITS,'|')) f
- Exclude NUS itself:      NOT CONTAINS(UPPER(TRIM(f.VALUE::STRING)),'NATIONAL UNIVERSITY')
- Filter by subject value: AND CONTAINS(UPPER(TRIM(f.VALUE::STRING)),'DATA SCIENCE')
  (NEVER use the SELECT alias in WHERE — always repeat the full TRIM(f.VALUE::STRING) expression)

RULES:
1. Write only SELECT statements — no DDL, DML, or destructive operations.
2. Always add LIMIT 200 unless the question needs aggregated totals only.
3. Use TRIM() when parsing FLATTEN values.
4. Return ONLY the SQL — no markdown, no explanation.
5. NEVER reference a SELECT alias in WHERE or HAVING — repeat the full expression.
6. Use positional GROUP BY (GROUP BY 1, 2, 3) rather than alias names.
7. LATERAL FLATTEN must appear directly in the top-level FROM clause — NEVER inside a subquery, CTE, or derived table. Write:
   SELECT ... FROM {TBL}, LATERAL FLATTEN(INPUT=>SPLIT(col, 'sep')) f WHERE ...
   NOT: SELECT ... FROM (SELECT ... FROM {TBL}, LATERAL FLATTEN(...) f) sub
"""

def run_llm_query(question: str, api_key: str):
    """Returns (sql_str, dataframe, error_str)"""
    try:
        client  = _ant.Anthropic(api_key=api_key)
        msg     = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=1024,
            messages=[{"role":"user",
                        "content": f"{SCHEMA_CONTEXT}\n\nQuestion: {question}"}],
        )
        gen_sql = re.sub(r"```sql|```", "", msg.content[0].text).strip()
    except Exception as e:
        return None, None, f"AI error: {e}"
    try:
        df = sql(gen_sql)
        return gen_sql, df, None
    except Exception as e:
        return gen_sql, None, f"Query error: {e}"


# ════════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ════════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown(
        f"<h2 style='color:{WHITE};margin:0;font-size:1.1rem'>🎓 IC Intelligence</h2>"
        f"<p style='color:{MUTED};font-size:13px;margin:0 0 0.6rem'>NUS Industry Collaboration</p>",
        unsafe_allow_html=True,
    )

    if st.button("↺  Refresh Data", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

    st.markdown(f"<div style='height:1px;background:#1A4A6E;margin:0.6rem 0'></div>",
                unsafe_allow_html=True)
    st.markdown(f"<div style='font-size:13px;color:{MUTED};font-weight:700;letter-spacing:.05em'>FILTERS</div>",
                unsafe_allow_html=True)

    year_range = st.slider("Publication year", 2020, 2025, (2020, 2025))
    ip_types   = st.multiselect("IP Type",
                                ["Publications","Patents"],
                                default=["Publications","Patents"])

    # ── ASK THE DATA (sidebar) ──────────────────────────────────────────────
    st.markdown(f"<div style='height:1px;background:#1A4A6E;margin:0.8rem 0'></div>",
                unsafe_allow_html=True)
    st.markdown(
        f"<div style='font-size:13px;color:{MUTED};font-weight:700;letter-spacing:.05em'>ASK THE DATA</div>"
        f"<div style='font-size:13px;color:{LIGHT};margin:0.3rem 0 0.5rem'>Ask a question in plain English</div>",
        unsafe_allow_html=True,
    )

    api_key = st.secrets.get("anthropic", {}).get("api_key", "")
    if not api_key or api_key.startswith("YOUR_"):
        st.markdown(
            f"<div style='font-size:13px;color:#7EB3D8'>⚠ Add Anthropic API key to secrets.toml</div>",
            unsafe_allow_html=True,
        )
        api_key = st.text_input("API key:", type="password", key="tmp_key",
                                 label_visibility="collapsed")

    EXAMPLES = [
        "NUS vs industry patent count by year",
        "Total NUS publications vs patents 2020–2024",
        "Compare Data Science vs Medicine patent trends",
    ]
    st.markdown(f"<div style='font-size:13px;color:{MUTED};margin:0.4rem 0 0.2rem'>Quick examples:</div>",
                unsafe_allow_html=True)
    for ex in EXAMPLES:
        if st.button(ex, key=f"ex_{ex}", use_container_width=True):
            st.session_state["pending_q"] = ex

    sidebar_q = st.text_area(
        "Question:", height=90, key="sidebar_q",
        placeholder="e.g. Which partners collaborated most with NUS in biomedical patents?",
        label_visibility="collapsed",
    )

    ask_btn = st.button("▶ Search Data", use_container_width=True, key="ask_btn")

    if (ask_btn or "pending_q" in st.session_state) and (sidebar_q or st.session_state.get("pending_q")):
        q = st.session_state.pop("pending_q", None) or sidebar_q
        if q and api_key:
            if not ANTHROPIC_AVAILABLE:
                st.session_state["llm_result"] = {
                    "q": q, "sql": None, "df": None,
                    "error": "Run: python3 -m pip install anthropic"
                }
            else:
                with st.spinner("Analysing…"):
                    gen_sql, df, err = run_llm_query(q, api_key)
                    st.session_state["llm_result"] = {
                        "q": q, "sql": gen_sql, "df": df, "error": err
                    }

    st.markdown(f"<div style='height:1px;background:#1A4A6E;margin:0.8rem 0'></div>",
                unsafe_allow_html=True)
    st.markdown(
        f"<div style='font-size:13px;color:{MUTED}'>Live data · Cache 1 hr</div>",
        unsafe_allow_html=True,
    )

yr_min, yr_max = year_range
ip_filter = "(" + ", ".join(f"'{t}'" for t in ip_types) + ")" if ip_types else "('Publications','Patents')"


# ── PAGE NAVIGATION ──────────────────────────────────────────────────────────
nav_col1, nav_col2, nav_spacer = st.columns([2, 2, 6])
with nav_col1:
    if st.button("🧲 Potential Collaborator Finder", use_container_width=True):
        st.switch_page("pages/1_Research_Collaboration_Explorer.py")
with nav_col2:
    st.button("📊 Industry Collaboration Overview", disabled=True, use_container_width=True)

# ── SINGAPORE BANNER ─────────────────────────────────────────────────────────
st.markdown(
    "<div class='sg-banner'>"
    "🇸🇬 <strong>Singapore Landscape Only</strong> — "
    "All records represent patents filed and publications authored within the Singapore "
    "research ecosystem, 2020–2025. Includes NUS, partner universities, hospitals, "
    "government agencies, and industry."
    "</div>",
    unsafe_allow_html=True,
)

st.markdown(
    "<h1 class='page-title'>SG Industry Collaboration Overview</h1>"
    f"<p class='page-subtitle'>NUS Research Strategy &nbsp;·&nbsp; "
    f"{yr_min}–{yr_max} &nbsp;·&nbsp; {', '.join(ip_types) if ip_types else 'All IP types'}</p>",
    unsafe_allow_html=True,
)
st.markdown("<div style='height:0.4rem'></div>", unsafe_allow_html=True)


# ── LLM RESULT PANEL (main area) ─────────────────────────────────────────────
if "llm_result" in st.session_state:
    res = st.session_state["llm_result"]
    with st.container():
        st.markdown("<div class='llm-panel'>", unsafe_allow_html=True)
        st.markdown(f"<div class='llm-q'>🤖 {res['q']}</div>", unsafe_allow_html=True)

        if res.get("error"):
            st.markdown(f"<div class='llm-err'>❌ {res['error']}</div>",
                        unsafe_allow_html=True)
        elif res.get("df") is not None:
            df_r = res["df"]
            rows = len(df_r)
            num_cols = df_r.select_dtypes(include="number").columns.tolist()
            str_cols = df_r.select_dtypes(exclude="number").columns.tolist()

            if num_cols and str_cols and rows <= 50:
                try:
                    year_cols   = [c for c in num_cols
                                   if df_r[c].dropna().between(1990, 2030).all()
                                   and df_r[c].nunique() <= 15]
                    metric_cols = [c for c in num_cols if c not in year_cols]
                    if metric_cols:
                        mc = metric_cols[0]
                        if year_cols:
                            # time-series: grouped bar by year, colour per category
                            af = px.bar(df_r, x=year_cols[0], y=mc,
                                        color=str_cols[0],
                                        barmode="group",
                                        color_discrete_sequence=CHART_COLS,
                                        labels={mc:"", year_cols[0]:"Year", str_cols[0]:""})
                        else:
                            # aggregate totals per category → horizontal bar
                            agg = (df_r.groupby(str_cols[0])[mc].sum()
                                   .reset_index().sort_values(mc, ascending=False))
                            af = px.bar(agg.head(25), x=mc, y=str_cols[0],
                                        orientation="h",
                                        color_discrete_sequence=[NUS_BLUE],
                                        labels={mc:"", str_cols[0]:""})
                            af.update_layout(yaxis=dict(autorange="reversed"))
                        st.plotly_chart(clean_fig(af, min(380, 80 + rows*26)),
                                        use_container_width=True)
                except Exception:
                    pass

            st.dataframe(df_r, use_container_width=True,
                         height=min(380, 70 + rows * 35))
            st.markdown(f"<div style='font-size:13px;color:{MUTED}'>{rows:,} rows</div>",
                        unsafe_allow_html=True)
            with st.expander("View generated SQL"):
                st.code(res.get("sql",""), language="sql")

        st.markdown("</div>", unsafe_allow_html=True)
        if st.button("✕ Clear result", key="clr_llm"):
            del st.session_state["llm_result"]
            st.rerun()
    st.markdown("<div style='height:0.3rem'></div>", unsafe_allow_html=True)


# ── TABS ──────────────────────────────────────────────────────────────────────
t1, t2, t3, t4, t5 = st.tabs([
    "📊 Overview",
    "🤝 Collaboration Areas",
    "💰 Funding Alignment",
    "📈 Research Demands",
    "🏛 NUS Units",
])


# ════════════════════════════════════════════════════════════════════════════════
# TAB 1 — OVERVIEW
# ════════════════════════════════════════════════════════════════════════════════
with t1:
    kpi_df = sql(f"""
        SELECT
            COUNT(*) AS TOTAL,
            SUM(CASE WHEN NUS_IP=TRUE  THEN 1 ELSE 0 END) AS NUS_RECS,
            SUM(CASE WHEN NUS_IP=TRUE AND N_CORPORATE>0 THEN 1 ELSE 0 END) AS NUS_IND,
            SUM(CASE WHEN IP_TYPE='Patents'      THEN 1 ELSE 0 END) AS PATENTS,
            SUM(CASE WHEN IP_TYPE='Publications' THEN 1 ELSE 0 END) AS PUBS,
            SUM(CASE WHEN NUS_IP=TRUE AND IP_TYPE='Patents' THEN 1 ELSE 0 END) AS NUS_PAT
        FROM {TBL}
        WHERE APPLICATION_PUBLICATION_YEAR BETWEEN {yr_min} AND {yr_max}
          AND IP_TYPE IN {ip_filter}
    """)
    r = kpi_df.iloc[0]

    k1,k2,k3,k4,k5 = st.columns(5)
    def kpi(col, val, lbl, sub="", border=NUS_ORANGE):
        col.markdown(
            f"<div class='kpi-card' style='border-color:{border}'>"
            f"<div class='kpi-val'>{val:,}</div>"
            f"<div class='kpi-lbl'>{lbl}</div>"
            f"<div class='kpi-sub'>{sub}</div></div>",
            unsafe_allow_html=True,
        )
    kpi(k1, int(r["TOTAL"]),   "Total Records",    "Patents + Publications", NUS_BLUE)
    kpi(k2, int(r["NUS_RECS"]),"NUS IP Records",   f"{int(r['NUS_RECS']/r['TOTAL']*100)}% of corpus")
    kpi(k3, int(r["NUS_IND"]), "NUS × Industry",   "Co-patents & co-pubs", NUS_ORANGE)
    kpi(k4, int(r["PATENTS"]), "Patents",           f"NUS: {int(r['NUS_PAT'])}", NUS_LBLUE)
    kpi(k5, int(r["PUBS"]),    "Publications",      "Within selection",      NUS_GOLD)

    st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)

    trend_df = sql(f"""
        SELECT APPLICATION_PUBLICATION_YEAR AS YEAR, IP_TYPE,
               CASE WHEN NUS_IP=TRUE THEN 'NUS' ELSE 'External' END AS SRC,
               COUNT(*) AS CNT
        FROM {TBL}
        WHERE APPLICATION_PUBLICATION_YEAR BETWEEN {yr_min} AND {yr_max}
          AND IP_TYPE IN {ip_filter}
        GROUP BY 1,2,3 ORDER BY 1
    """)

    pat_yr  = trend_df[(trend_df["SRC"]=="NUS") & (trend_df["IP_TYPE"]=="Patents")]
    pat_chg = 0
    if not pat_yr.empty and len(pat_yr) > 1:
        p0 = pat_yr.sort_values("YEAR").iloc[0]["CNT"]
        p1 = pat_yr.sort_values("YEAR").iloc[-1]["CNT"]
        pat_chg = int((p1-p0)/p0*100) if p0 else 0

    st.markdown(
        f"<div class='exec-box'>"
        f"<div class='exec-title'>📋 Executive Summary</div>"
        f"<div class='exec-body'>"
        f"This dashboard covers <strong>{int(r['TOTAL']):,} records</strong> (patents and publications) "
        f"from the Singapore research ecosystem ({yr_min}–{yr_max}). "
        f"NUS owns or co-owns <strong>{int(r['NUS_RECS']):,} records</strong> "
        f"({int(r['NUS_RECS']/r['TOTAL']*100)}% of the corpus). "
        f"Only <strong>{int(r['NUS_IND']):,} NUS records involve a corporate collaborator</strong>, "
        f"revealing a major gap between research production and industry engagement.<br><br>"
        f"Three signals require immediate management attention: "
        f"<strong>(1)</strong> Applied Materials and Singapore Health Services are NUS's deepest patent "
        f"co-inventors — formalise joint IP roadmaps with them now. "
        f"<strong>(2)</strong> Data Science is the only external patent domain growing (+32%), "
        f"signalling where industry funding will flow in the next 12–24 months — "
        f"NUS publication growth in this domain (+53%) creates a rare alignment opportunity. "
        f"<strong>(3)</strong> NUS patent output has fallen {abs(pat_chg)}% over the period "
        f"while publications grew — research is not being converted into IP."
        f"</div></div>",
        unsafe_allow_html=True,
    )

    c1, c2 = st.columns(2)

    with c1:
        section("What this tells us")
        insight("NUS publication volume is growing steadily each year. However, patent "
                "filings have declined sharply — NUS is producing more research but "
                "protecting less of it commercially. Industry co-publication remains "
                "a small fraction of total output.")
        section("Annual output — NUS publications and patents")
        nus_t = trend_df[trend_df["SRC"]=="NUS"]
        if not nus_t.empty:
            fig = px.bar(nus_t, x="YEAR", y="CNT", color="IP_TYPE", barmode="group",
                         color_discrete_map={"Publications":NUS_BLUE,"Patents":NUS_ORANGE},
                         labels={"CNT":"Count","YEAR":"Year","IP_TYPE":""})
            fig.update_layout(xaxis=dict(tickmode="linear",dtick=1))
            st.plotly_chart(clean_fig(fig,380), use_container_width=True)

    with c2:
        section("What this tells us")
        insight("NUS holds a small and shrinking share of Singapore patents relative to "
                "external filers. The external ecosystem — particularly CS and Data Science "
                "companies — is filing far more. This gap is the primary opportunity for "
                "NUS to grow industry partnerships.")
        section("NUS vs external patent share by year")
        pat_t = trend_df[trend_df["IP_TYPE"]=="Patents"] if "Patents" in ip_types else pd.DataFrame()
        if not pat_t.empty:
            fig2 = px.bar(pat_t, x="YEAR", y="CNT", color="SRC", barmode="stack",
                          color_discrete_map={"NUS":NUS_BLUE,"External":"#6FA3C8"},
                          labels={"CNT":"Patents","YEAR":"Year","SRC":""})
            fig2.update_layout(xaxis=dict(tickmode="linear",dtick=1))
            st.plotly_chart(clean_fig(fig2,380), use_container_width=True)


# ════════════════════════════════════════════════════════════════════════════════
# TAB 2 — COLLABORATION AREAS
# ════════════════════════════════════════════════════════════════════════════════
with t2:
    # ── queries ──
    pp = sql(f"""
        SELECT TRIM(f.VALUE::STRING) AS PARTNER, QS_SUBJECT AS SUBJECT,
               APPLICATION_PUBLICATION_YEAR AS YEAR, COUNT(*) AS CNT
        FROM {TBL}, LATERAL FLATTEN(INPUT=>SPLIT(NORMALIZED_NAMES_CONCAT,'|')) f
        WHERE NUS_IP=TRUE AND IP_TYPE='Patents' AND N_CORPORATE>0
          AND NOT CONTAINS(UPPER(TRIM(f.VALUE::STRING)),'NATIONAL UNIVERSITY')
          AND TRIM(f.VALUE::STRING)<>''
          AND APPLICATION_PUBLICATION_YEAR BETWEEN {yr_min} AND {yr_max}
        GROUP BY 1,2,3 ORDER BY CNT DESC
    """)
    ud = sql(f"""
        SELECT TRIM(f.VALUE::STRING) AS UNIT, COUNT(*) AS CNT
        FROM {TBL}, LATERAL FLATTEN(INPUT=>SPLIT(UNITS,'|')) f
        WHERE NUS_IP=TRUE AND IP_TYPE='Publications' AND N_CORPORATE>0
          AND UNITS IS NOT NULL AND UNITS<>''
          AND APPLICATION_PUBLICATION_YEAR BETWEEN {yr_min} AND {yr_max}
        GROUP BY 1 ORDER BY 2 DESC LIMIT 12
    """)
    sd = sql(f"""
        SELECT TRIM(f.VALUE::STRING) AS SUBJECT, COUNT(*) AS CNT
        FROM {TBL}, LATERAL FLATTEN(INPUT=>SPLIT(QS_SUBJECT_AREA,'|')) f
        WHERE NUS_IP=TRUE AND IP_TYPE='Publications' AND N_CORPORATE>0
          AND APPLICATION_PUBLICATION_YEAR BETWEEN {yr_min} AND {yr_max}
          AND TRIM(f.VALUE::STRING) NOT IN ('','-')
        GROUP BY 1 ORDER BY 2 DESC
    """)

    # ── Row 1 — context ──
    ca, cb = st.columns([1.1, 0.9])
    with ca:
        section("What this tells us — patent partners")
        insight("Applied Materials Inc (semiconductor & materials) and Singapore Health "
                "Services (biomedical) are NUS's two most active patent co-inventors. "
                "Both have been consistent multi-year partners. Formalising joint IP "
                "roadmaps with them should be the immediate priority.")
    with cb:
        section("What this tells us — publication units")
        insight("CSI SPORE (Cancer Science) and LSI (Life Sciences) generate more than "
                "half of all NUS–industry co-publications. These units are the natural "
                "bridge to pharmaceutical companies and health-tech investors. Target "
                "them first for industry grant applications.")

    # ── Row 2 — top charts (matched height) ──
    ca, cb = st.columns([1.1, 0.9])
    with ca:
        section("Top industry patent partners — NUS co-owned patents")
        if not pp.empty:
            top_p = pp.groupby("PARTNER")["CNT"].sum().reset_index() \
                      .sort_values("CNT",ascending=False).head(12)
            st.plotly_chart(hbar(top_p,"CNT","PARTNER",h=420), use_container_width=True)
    with cb:
        section("NUS units with the most industry co-publications")
        if not ud.empty:
            st.plotly_chart(hbar(ud,"CNT","UNIT",h=420,color=NUS_ORANGE),
                            use_container_width=True)

    # ── Row 3 — bottom charts (matched height) ──
    ca, cb = st.columns([1.1, 0.9])
    with ca:
        section("Year trend — top 5 patent partners")
        if not pp.empty:
            top5 = top_p["PARTNER"].head(5).tolist()
            tr5  = pp[pp["PARTNER"].isin(top5)].groupby(["YEAR","PARTNER"])["CNT"].sum().reset_index()
            tr5["PARTNER"] = tr5["PARTNER"].str[:28]
            fig  = px.line(tr5, x="YEAR", y="CNT", color="PARTNER", markers=True,
                           color_discrete_sequence=CHART_COLS,
                           labels={"CNT":"Co-patents","YEAR":"Year","PARTNER":""})
            fig.update_layout(
                xaxis=dict(tickmode="linear", dtick=1),
                legend=dict(orientation="h", yanchor="top", y=-0.28,
                            xanchor="left", x=0, font=dict(size=11)),
                margin=dict(t=25, b=110, l=10, r=10),
            )
            st.plotly_chart(clean_fig(fig, 360), use_container_width=True)
    with cb:
        section("Subject mix — industry co-publications")
        if not sd.empty:
            _total = sd["CNT"].sum()
            sd = sd.copy()
            sd["Area"] = sd["SUBJECT"].str.title()
            sd["PCT"] = (sd["CNT"] / _total * 100).round(1)
            sd["LABEL"] = sd.apply(lambda r: f"{int(r['CNT'])}  ({r['PCT']}%)", axis=1)
            AREA_COLS = {
                "Life Sciences & Medicine":      "#2C6BAD",
                "Engineering & Technology":      "#EF7C00",
                "Natural Sciences":              "#2E9E6B",
                "Social Sciences & Management":  "#8E6BAD",
                "Arts & Humanities":             "#C0504D",
            }
            fig = px.bar(sd, x="CNT", y="Area", orientation="h", text="LABEL",
                         color="Area", color_discrete_map=AREA_COLS,
                         labels={"CNT": "", "Area": ""})
            fig.update_traces(textposition="outside", textfont_size=12, cliponaxis=False)
            fig.update_layout(
                showlegend=False,
                yaxis=dict(autorange="reversed", automargin=True),
                xaxis=dict(range=[0, sd["CNT"].max() * 1.18]),
                margin=dict(t=20, b=30, l=185, r=50),
            )
            st.plotly_chart(clean_fig(fig, 360), use_container_width=True)


# ════════════════════════════════════════════════════════════════════════════════
# TAB 3 — FUNDING ALIGNMENT
# ════════════════════════════════════════════════════════════════════════════════
with t3:
    sp1, sp2, sp3, sp4 = st.columns(4)
    for col, num, lbl, body in [
        (sp1, "+31.7%", "Data Science patents growing",
         "The only domain with rising external filings — everything else is declining"),
        (sp2, "949", "External DS patents filed",
         "vs just 5 NUS DS patents — a 99.5% gap in the market's most active growth area"),
        (sp3, "10", "Untapped CS/AI partners",
         "Top filers (Lenovo, MediaTek, Grab, Huawei…) have zero current NUS patent deal"),
        (sp4, "53%", "NUS DS publication growth",
         "NUS is building capacity fast — but not converting it to patents or partnerships"),
    ]:
        col.markdown(
            f"<div class='spotlight-card'>"
            f"<div class='spotlight-num'>{num}</div>"
            f"<div class='spotlight-lbl'>{lbl}</div>"
            f"<div class='spotlight-body'>{body}</div></div>",
            unsafe_allow_html=True,
        )

    st.markdown("<div style='height:0.8rem'></div>", unsafe_allow_html=True)

    ext_df = sql(f"""
        SELECT APPLICATION_PUBLICATION_YEAR AS YEAR,
               TRIM(f.VALUE::STRING) AS SUBJECT, COUNT(*) AS CNT
        FROM {TBL}, LATERAL FLATTEN(INPUT=>SPLIT(QS_SUBJECT,'|')) f
        WHERE NUS_IP=FALSE AND IP_TYPE='Patents'
          AND APPLICATION_PUBLICATION_YEAR BETWEEN {yr_min} AND {yr_max}
          AND TRIM(f.VALUE::STRING)<>''
        GROUP BY 1,2 ORDER BY 1,3 DESC
    """)

    filers = sql(f"""
        SELECT TRIM(f.VALUE::STRING) AS ENTITY, COUNT(*) AS CNT
        FROM {TBL}, LATERAL FLATTEN(INPUT=>SPLIT(NORMALIZED_NAMES_CONCAT,'|')) f
        WHERE NUS_IP=FALSE AND IP_TYPE='Patents'
          AND (CONTAINS(UPPER(QS_SUBJECT),'DATA SCIENCE')
               OR CONTAINS(UPPER(QS_SUBJECT),'COMPUTER SCIENCE'))
          AND APPLICATION_PUBLICATION_YEAR BETWEEN {yr_min} AND {yr_max}
          AND TRIM(f.VALUE::STRING)<>''
        GROUP BY 1 ORDER BY 2 DESC LIMIT 12
    """)

    # ── Row 1 — context ──
    c1, c2 = st.columns(2)
    with c1:
        section("What this tells us — patent domain trends")
        warn("Data Science is the single growing external patent domain. Every other "
             "sector is contracting, signalling that industry is consolidating spend. "
             "Align funding proposals to Data Science and CS now — before RFPs are issued.")
    with c2:
        section("What this tells us — untapped partners")
        insight("None of the top CS and Data Science patent filers currently hold a "
                "NUS patent collaboration. Each company on this list is a direct, "
                "data-backed outreach target for joint research and co-IP agreements. "
                "Prioritise outreach within the next 6 months.")

    # ── Row 2 — charts (matched height) ──
    c1, c2 = st.columns(2)
    with c1:
        section("External patent growth rate — early vs late period")
        if not ext_df.empty:
            mid = (yr_min+yr_max)//2
            ey  = max(1,mid-yr_min+1); ly = max(1,yr_max-mid)
            e   = ext_df[ext_df["YEAR"]<=mid].groupby("SUBJECT")["CNT"].sum()
            l   = ext_df[ext_df["YEAR"]> mid].groupby("SUBJECT")["CNT"].sum()
            gd  = pd.DataFrame({"E":e,"L":l}).dropna()
            gd  = gd[gd["E"]>50]
            gd["PCT"] = ((gd["L"]/ly-gd["E"]/ey)/(gd["E"]/ey)*100).round(1)
            gd  = gd.sort_values("PCT").reset_index()
            gd["COL"] = gd["PCT"].apply(lambda v: GREEN if v>0 else RED)

            fig = px.bar(gd, x="PCT", y="SUBJECT", orientation="h",
                         color="COL", color_discrete_map={GREEN:GREEN, RED:RED},
                         labels={"PCT":"Growth %","SUBJECT":""})
            fig.add_vline(x=0, line_color=SLATE, line_width=1)
            fig.update_layout(showlegend=False)
            st.plotly_chart(clean_fig(fig,440), use_container_width=True)
            _early = f"{yr_min}" if yr_min == mid else f"{yr_min}–{mid}"
            _late = f"{mid+1}" if mid + 1 == yr_max else f"{mid+1}–{yr_max}"
            st.caption(
                f"📊 Growth compares the **average annual** external patents in **{_late}** against **{_early}** "
                f"— the later vs earlier half of the selected {yr_min}–{yr_max} range. "
                f"Adjust the year filter to change the comparison."
            )
    with c2:
        section("Top CS & Data Science filers — not yet NUS partners")
        if not filers.empty:
            st.plotly_chart(hbar(filers,"CNT","ENTITY",h=440,color=NUS_BLUE),
                            use_container_width=True)


# ════════════════════════════════════════════════════════════════════════════════
# TAB 4 — RESEARCH DEMANDS
# ════════════════════════════════════════════════════════════════════════════════
with t4:
    # ── Row 1 — context ──
    r1, r2 = st.columns(2)
    with r1:
        section("What this tells us — capacity build")
        insight("Data Science (+53%) and CS & Info Systems (+43%) are NUS's "
                "fastest-growing publication domains — precisely where industry "
                "patent demand is also growing. This alignment makes a compelling "
                "case for priority investment and targeted IP conversion.")
    with r2:
        section("What this tells us — capacity gap")
        insight("CS and Data Science have the biggest gap between external demand and "
                "NUS IP presence (under 2% share). These are the domains where NUS "
                "research is growing fastest — closing this IP gap is the highest-ROI "
                "action available to the TTO right now.")

    # ── Row 2 — charts (matched height) ──
    r1, r2 = st.columns(2)
    with r1:
        section("NUS publication growth by domain")
        pub_s = sql(f"""
            SELECT APPLICATION_PUBLICATION_YEAR AS YEAR,
                   TRIM(f.VALUE::STRING) AS SUBJECT, COUNT(*) AS CNT
            FROM {TBL}, LATERAL FLATTEN(INPUT=>SPLIT(QS_SUBJECT,'|')) f
            WHERE NUS_IP=TRUE AND IP_TYPE='Publications'
              AND APPLICATION_PUBLICATION_YEAR BETWEEN {yr_min} AND {yr_max}
              AND TRIM(f.VALUE::STRING)<>''
            GROUP BY 1,2 ORDER BY 1
        """)
        if not pub_s.empty:
            mid  = (yr_min+yr_max)//2
            ey   = max(1,mid-yr_min+1); ly = max(1,yr_max-mid)
            e    = pub_s[pub_s["YEAR"]<=mid].groupby("SUBJECT")["CNT"].sum()
            l    = pub_s[pub_s["YEAR"]> mid].groupby("SUBJECT")["CNT"].sum()
            pg   = pd.DataFrame({"E":e,"L":l}).dropna()
            pg   = pg[pg["E"]>100]
            pg["PCT"] = ((pg["L"]/ly-pg["E"]/ey)/(pg["E"]/ey)*100).round(1)
            pg   = pg.sort_values("PCT",ascending=False).head(12).reset_index()
            fig  = px.bar(pg, x="PCT", y="SUBJECT", orientation="h",
                          color="PCT",
                          color_continuous_scale=[[0,NUS_BLUE],[0.5,NUS_LBLUE],[1,NUS_ORANGE]],
                          labels={"PCT":"Growth %","SUBJECT":""})
            fig.update_layout(coloraxis_showscale=False,
                               yaxis=dict(autorange="reversed"))
            st.plotly_chart(clean_fig(fig,440), use_container_width=True)
            _early = f"{yr_min}" if yr_min == mid else f"{yr_min}–{mid}"
            _late = f"{mid+1}" if mid + 1 == yr_max else f"{mid+1}–{yr_max}"
            st.caption(
                f"📊 Growth compares the **average annual** publications in **{_late}** against **{_early}** "
                f"— the later vs earlier half of the selected {yr_min}–{yr_max} range. "
                f"Adjust the year filter to change the comparison."
            )

    with r2:
        section("NUS share of all Singapore patents — by domain")
        gap = sql(f"""
            SELECT TRIM(f.VALUE::STRING) AS SUBJECT,
                   SUM(CASE WHEN NUS_IP=TRUE  THEN 1 ELSE 0 END) AS NUS_N,
                   SUM(CASE WHEN NUS_IP=FALSE THEN 1 ELSE 0 END) AS EXT_N
            FROM {TBL}, LATERAL FLATTEN(INPUT=>SPLIT(QS_SUBJECT,'|')) f
            WHERE IP_TYPE='Patents'
              AND APPLICATION_PUBLICATION_YEAR BETWEEN {yr_min} AND {yr_max}
              AND TRIM(f.VALUE::STRING)<>''
            GROUP BY 1 HAVING EXT_N>50
            ORDER BY NUS_N/(NUS_N+EXT_N)
        """)
        if not gap.empty:
            gap["SHARE"] = (gap["NUS_N"]/(gap["NUS_N"]+gap["EXT_N"])*100).round(1)
            gap["COL"]   = gap["SHARE"].apply(
                lambda v: RED if v<2 else (AMBER if v<7 else GREEN))
            fig2 = px.bar(gap, x="SHARE", y="SUBJECT", orientation="h",
                          color="COL",
                          color_discrete_map={RED:RED,AMBER:AMBER,GREEN:GREEN},
                          text="SHARE",
                          labels={"SHARE":"NUS Share %","SUBJECT":""})
            fig2.update_traces(texttemplate="%{text}%", textposition="outside",
                                textfont_size=13)
            fig2.update_layout(showlegend=False,
                                xaxis=dict(range=[0, gap["SHARE"].max()*1.3]))
            st.plotly_chart(clean_fig(fig2,440), use_container_width=True)
            st.markdown(
                f"<span style='color:{RED};font-weight:700;font-size:14px'>■ &lt;2% Critical Gap &nbsp;</span>"
                f"<span style='color:{AMBER};font-weight:700;font-size:14px'>■ 2–7% Moderate &nbsp;</span>"
                f"<span style='color:{GREEN};font-weight:700;font-size:14px'>■ &gt;7% Strength</span>",
                unsafe_allow_html=True,
            )

    section("What this tells us — patent decline")
    alert("NUS patent filings have declined sharply while publications grew — "
          "a critical commercialisation pipeline failure. CS & Information Systems "
          "is the only domain with growing patent output. Immediate TTO capacity "
          "investment is recommended, particularly in Data Science and CS.")

    pd_df = sql(f"""
        SELECT APPLICATION_PUBLICATION_YEAR AS YEAR,
               TRIM(f.VALUE::STRING) AS SUBJECT, COUNT(*) AS CNT
        FROM {TBL}, LATERAL FLATTEN(INPUT=>SPLIT(QS_SUBJECT,'|')) f
        WHERE NUS_IP=TRUE AND IP_TYPE='Patents'
          AND APPLICATION_PUBLICATION_YEAR BETWEEN {yr_min} AND {yr_max}
          AND TRIM(f.VALUE::STRING)<>''
        GROUP BY 1,2 ORDER BY 1
    """)

    pa, pb = st.columns([1,1.6])
    with pa:
        section("Total NUS patents filed per year")
        if not pd_df.empty:
            tot = pd_df.groupby("YEAR")["CNT"].sum().reset_index()
            fig3 = px.line(tot, x="YEAR", y="CNT", markers=True,
                           color_discrete_sequence=[RED],
                           labels={"CNT":"NUS Patents Filed","YEAR":"Year"})
            fig3.update_traces(line_width=3, marker_size=10)
            fig3.update_layout(xaxis=dict(tickmode="linear",dtick=1))
            st.plotly_chart(clean_fig(fig3,300), use_container_width=True)

    with pb:
        section("NUS patents by domain — stacked by year")
        if not pd_df.empty:
            top8 = pd_df.groupby("SUBJECT")["CNT"].sum().sort_values(ascending=False).head(8).index
            fig4 = px.bar(pd_df[pd_df["SUBJECT"].isin(top8)],
                          x="YEAR", y="CNT", color="SUBJECT", barmode="stack",
                          color_discrete_sequence=CHART_COLS,
                          labels={"CNT":"Patents","YEAR":"Year","SUBJECT":""})
            fig4.update_layout(xaxis=dict(tickmode="linear",dtick=1),
                                legend=dict(font=dict(size=12)))
            st.plotly_chart(clean_fig(fig4,300), use_container_width=True)


# ════════════════════════════════════════════════════════════════════════════════
# TAB 5 — NUS UNITS
# ════════════════════════════════════════════════════════════════════════════════
with t5:
    unit_list = sql(f"""
        SELECT DISTINCT TRIM(f.VALUE::STRING) AS U
        FROM {TBL}, LATERAL FLATTEN(INPUT=>SPLIT(UNITS,'|')) f
        WHERE NUS_IP=TRUE AND UNITS IS NOT NULL AND UNITS<>''
        ORDER BY U
    """)["U"].dropna().tolist()

    unit_sel = st.selectbox("Drill into a specific unit (or view All):",
                            ["All"] + unit_list, key="unit_sel")
    unit_where = f"AND CONTAINS(UNITS, '{unit_sel}')" if unit_sel != "All" else ""

    # ── Row 1: insight boxes (own row so wrapping doesn't offset charts below) ──
    ins1, ins2 = st.columns(2)
    with ins1:
        section("What this tells us — unit engagement")
        insight("Units where the orange industry bar is a large fraction of the blue "
                "total bar are already well-connected with industry. Units with high "
                "total output but low industry overlay are under-commercialised — "
                "prime targets for industry outreach programmes.")
    with ins2:
        section("What this tells us — unit trends")
        insight("Units growing year-on-year are building momentum that makes them "
                "more attractive to industry partners. A flat or declining trend "
                "may indicate capacity constraints or shifting research priorities.")

    # ── Row 2: compute shared top-10 unit list, then render all 3 charts ────────
    uo = sql(f"""
        SELECT TRIM(f.VALUE::STRING) AS UNIT, IP_TYPE,
               COUNT(*) AS TOTAL,
               SUM(CASE WHEN N_CORPORATE>0 THEN 1 ELSE 0 END) AS IND
        FROM {TBL}, LATERAL FLATTEN(INPUT=>SPLIT(UNITS,'|')) f
        WHERE NUS_IP=TRUE AND UNITS IS NOT NULL AND UNITS<>''
          AND APPLICATION_PUBLICATION_YEAR BETWEEN {yr_min} AND {yr_max}
          {unit_where}
        GROUP BY 1,2 ORDER BY TOTAL DESC LIMIT 40
    """)
    TOP_N = 10
    top_units = (
        uo.groupby("UNIT")[["TOTAL"]].sum()
        .sort_values("TOTAL", ascending=False)
        .head(TOP_N).index.tolist()
        if not uo.empty else []
    )

    u1, u2 = st.columns(2)

    with u1:
        section("Total output vs industry collaborations — by unit")
        if not uo.empty:
            agg = uo[uo["UNIT"].isin(top_units)].groupby("UNIT")[["TOTAL","IND"]].sum().reset_index() \
                    .sort_values("TOTAL", ascending=False)
            fig = go.Figure()
            fig.add_bar(name="Total Output", x=agg["TOTAL"], y=agg["UNIT"],
                        orientation="h", marker_color=NUS_BLUE)
            fig.add_bar(name="Industry Co-authored", x=agg["IND"], y=agg["UNIT"],
                        orientation="h", marker_color=NUS_ORANGE)
            fig.update_layout(barmode="overlay",
                               yaxis=dict(autorange="reversed"),
                               legend=dict(font=dict(size=12)))
            st.plotly_chart(clean_fig(fig, 450), use_container_width=True)

    with u2:
        section("Unit activity trend by year")
        ut = sql(f"""
            SELECT APPLICATION_PUBLICATION_YEAR AS YEAR,
                   TRIM(f.VALUE::STRING) AS UNIT, COUNT(*) AS CNT
            FROM {TBL}, LATERAL FLATTEN(INPUT=>SPLIT(UNITS,'|')) f
            WHERE NUS_IP=TRUE AND UNITS IS NOT NULL AND UNITS<>''
              AND APPLICATION_PUBLICATION_YEAR BETWEEN {yr_min} AND {yr_max}
              {unit_where}
            GROUP BY 1,2 ORDER BY 1
        """)
        if not ut.empty:
            fig2  = px.line(ut[ut["UNIT"].isin(top_units)],
                            x="YEAR", y="CNT", color="UNIT", markers=True,
                            color_discrete_sequence=CHART_COLS,
                            labels={"CNT":"Records","YEAR":"Year","UNIT":""})
            fig2.update_layout(xaxis=dict(tickmode="linear",dtick=1),
                                legend=dict(font=dict(size=12)))
            st.plotly_chart(clean_fig(fig2, 450), use_container_width=True)

    section("Subject specialisation heatmap — unit vs domain")
    us = sql(f"""
        SELECT TRIM(u.VALUE::STRING) AS UNIT,
               TRIM(s.VALUE::STRING) AS SUBJECT,
               COUNT(*) AS CNT
        FROM {TBL},
             LATERAL FLATTEN(INPUT=>SPLIT(UNITS,'|')) u,
             LATERAL FLATTEN(INPUT=>SPLIT(QS_SUBJECT_AREA,'|')) s
        WHERE NUS_IP=TRUE AND IP_TYPE='Publications'
          AND UNITS IS NOT NULL AND UNITS<>''
          AND APPLICATION_PUBLICATION_YEAR BETWEEN {yr_min} AND {yr_max}
          {unit_where}
          AND TRIM(u.VALUE::STRING)<>'' AND TRIM(s.VALUE::STRING) NOT IN ('','-')
        GROUP BY 1,2 ORDER BY CNT DESC LIMIT 100
    """)
    if not us.empty:
        ts = us.groupby("SUBJECT")["CNT"].sum().sort_values(ascending=False).head(8).index
        heatmap_units = (us.groupby("UNIT")["CNT"].sum()
                         .sort_values(ascending=False).head(TOP_N).index.tolist())
        pv = (us[us["UNIT"].isin(heatmap_units) & us["SUBJECT"].isin(ts)]
              .pivot_table(index="SUBJECT", columns="UNIT", values="CNT", fill_value=0))
        if not pv.empty:
            fig3 = px.imshow(pv,
                              color_continuous_scale=[[0,"rgba(0,61,124,0.08)"],[0.4,"#5B9FC8"],[1,NUS_BLUE]],
                              labels=dict(color="Records"), aspect="auto")
            fig3.update_layout(margin=dict(t=10,b=10,l=10,r=10),
                                paper_bgcolor="rgba(0,0,0,0)", height=320,
                                coloraxis_showscale=False,
                                xaxis=dict(tickfont=dict(size=11)),
                                yaxis=dict(tickfont=dict(size=11)))
            st.plotly_chart(fig3, use_container_width=True)
        insight("Darker cells = higher research output in that unit-subject pair. "
                "Use this to match the right NUS unit to the right industry partner by topic.")
