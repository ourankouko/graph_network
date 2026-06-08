import json
import re
import anthropic
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from pyvis.network import Network
from snowflake.snowpark import Session
import networkx as nx


# -----------------------------
# Page setup
# -----------------------------
st.set_page_config(page_title="Graph Network Explorer", layout="wide")

st.title("Graph Network Explorer")
st.write("Explore patent and publication graph networks from Snowflake.")


# -----------------------------
# Snowflake connection
# -----------------------------
@st.cache_resource
def create_session():
    connection_parameters = {
        "account": st.secrets["snowflake"]["account"],
        "user": st.secrets["snowflake"]["user"],
        "password": st.secrets["snowflake"]["password"],
        "role": st.secrets["snowflake"]["role"],
        "warehouse": st.secrets["snowflake"]["warehouse"],
        "database": st.secrets["snowflake"]["database"],
        "schema": st.secrets["snowflake"]["schema"],
    }
    return Session.builder.configs(connection_parameters).create()


session = create_session()

client = anthropic.Anthropic(api_key=st.secrets["anthropic"]["api_key"])

# -----------------------------
# Helper functions
# -----------------------------
@st.cache_data(ttl=3600)
def run_query(sql: str) -> pd.DataFrame:
    return session.sql(sql).to_pandas()


def sql_escape(value: str) -> str:
    return value.replace("'", "''")


def get_node_color(node_type: str, node_name: str, nus_affiliated) -> str:
    node_type = str(node_type).upper()
    node_name = str(node_name).upper()

    if str(nus_affiliated).upper() in ["TRUE", "1", "YES"]:
        return "#C00000"

    if "NATIONAL UNIVERSITY OF SINGAPORE" in node_name:
        return "#C00000"

    if "SUBJECT" in node_type:
        return "#F4B183"

    if "APPLICANT" in node_type:
        return "#9DC3E6"

    if "INSTITUTE" in node_type:
        return "#A9D18E"

    return "#D9D9D9"


def keep_top_communities(df: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    if df.empty:
        return df

    G = nx.Graph()

    for _, row in df.iterrows():
        G.add_edge(row["SOURCE"], row["TARGET"], weight=float(row["WEIGHT"]))

    if G.number_of_edges() == 0:
        return df

    communities = list(nx.community.greedy_modularity_communities(G, weight="weight"))

    community_rows = []
    for i, community in enumerate(communities):
        community_rows.append({
            "community_id": i,
            "nodes": set(community),
            "size": len(community),
        })

    community_rows = sorted(community_rows, key=lambda x: x["size"], reverse=True)
    top_communities = community_rows[:top_n]

    node_to_community = {}
    for community in top_communities:
        for node in community["nodes"]:
            node_to_community[node] = community["community_id"]

    top_nodes = set(node_to_community.keys())

    filtered_df = df[
        df["SOURCE"].isin(top_nodes) & df["TARGET"].isin(top_nodes)
    ].copy()
    filtered_df["CLUSTER"] = filtered_df["SOURCE"].map(node_to_community)

    return filtered_df


def keep_top_n_neighbours(df: pd.DataFrame, search_term: str, top_n: int) -> pd.DataFrame:
    """Keep only the top N neighbours by total edge weight connected to the search term node."""
    if df.empty or not search_term:
        return df

    term = search_term.upper()

    mask = (
        df["SOURCE_NAME"].str.upper().str.contains(term, regex=False) |
        df["TARGET_NAME"].str.upper().str.contains(term, regex=False)
    )
    direct_edges = df[mask].copy()

    if direct_edges.empty:
        return df

    def get_neighbour(row):
        if term in str(row["SOURCE_NAME"]).upper():
            return row["TARGET_NAME"]
        return row["SOURCE_NAME"]

    direct_edges["NEIGHBOUR"] = direct_edges.apply(get_neighbour, axis=1)

    top_neighbours = (
        direct_edges.groupby("NEIGHBOUR")["WEIGHT"]
        .sum()
        .sort_values(ascending=False)
        .head(top_n)
        .index.tolist()
    )

    return df[
        df["SOURCE_NAME"].isin(top_neighbours) |
        df["TARGET_NAME"].isin(top_neighbours)
    ].copy()


def run_similar_no_collab_query(
    institution: str,
    ip_type: str = None,
    category: str = None,
    top_n: int = 20,
    subject_filter: str = None,
) -> pd.DataFrame:
    """
    Find organisations with similar subject interests to the given institution
    that have NOT directly appeared in any edge with it.
    Returns a ranked table with ORG_NAME, ORG_CATEGORY, SHARED_SUBJECTS, TOTAL_WEIGHT.

    Actual edge types in data:
      Applicant_Subject    (patents  — APP:: nodes to SUBJ:: nodes)
      Institute_Subject    (publications — INST:: nodes to SUBJ:: nodes)
      Applicant_Applicant  (patent co-applicants)
      Institution_Institution (publication co-authors)
    IP_TYPE values: 'Patents', 'Publications'
    """
    safe_inst = sql_escape(institution)

    # IP filter — match actual values "Patents" / "Publications"
    ip_filter = f"AND IP_TYPE = '{sql_escape(ip_type)}'" if ip_type and ip_type != "All" else ""

    # Subject edge types depend on ip_type
    if ip_type == "Patents":
        subject_edge_types = "('Applicant_Subject')"
    elif ip_type == "Publications":
        subject_edge_types = "('Institute_Subject')"
    else:
        subject_edge_types = "('Applicant_Subject', 'Institute_Subject')"

    # Direct collab edge types depend on ip_type
    if ip_type == "Patents":
        collab_edge_types = "('Applicant_Applicant')"
    elif ip_type == "Publications":
        collab_edge_types = "('Institution_Institution')"
    else:
        collab_edge_types = "('Applicant_Applicant', 'Institution_Institution')"

    cat_filter = (
        f"AND (SOURCE_CATEGORY = '{sql_escape(category)}' OR TARGET_CATEGORY = '{sql_escape(category)}')"
        if category and category != "All" else ""
    )
    subject_clause = (
        f"AND (SOURCE_NAME ILIKE '%{sql_escape(subject_filter)}%' OR TARGET_NAME ILIKE '%{sql_escape(subject_filter)}%')"
        if subject_filter else ""
    )

    sql = f"""
WITH inst_subjects AS (
    -- Step 1: all subjects the institution is connected to
    SELECT DISTINCT
        CASE WHEN SOURCE_NAME ILIKE '%{safe_inst}%' THEN TARGET ELSE SOURCE END AS SUBJECT_ID
    FROM GRAPH_NETWORK.GRAPH.ALL_EDGES_ENRICHED
    WHERE (SOURCE_NAME ILIKE '%{safe_inst}%' OR TARGET_NAME ILIKE '%{safe_inst}%')
    AND EDGE_TYPE IN {subject_edge_types}
    {ip_filter}
    {subject_clause}
),
org_subject_edges AS (
    -- Step 2: all orgs connected to those same subjects, excluding the institution itself
    SELECT
        CASE WHEN SOURCE_TYPE IN ('Applicant', 'Institutes') THEN SOURCE ELSE TARGET END AS ORG_ID,
        CASE WHEN SOURCE_TYPE IN ('Applicant', 'Institutes') THEN SOURCE_NAME ELSE TARGET_NAME END AS ORG_NAME,
        CASE WHEN SOURCE_TYPE IN ('Applicant', 'Institutes') THEN SOURCE_CATEGORY ELSE TARGET_CATEGORY END AS ORG_CATEGORY,
        CASE WHEN SOURCE IN (SELECT SUBJECT_ID FROM inst_subjects) THEN SOURCE
             ELSE TARGET END AS MATCHED_SUBJECT_ID,
        WEIGHT
    FROM GRAPH_NETWORK.GRAPH.ALL_EDGES_ENRICHED
    WHERE EDGE_TYPE IN {subject_edge_types}
    {ip_filter}
    AND (SOURCE IN (SELECT SUBJECT_ID FROM inst_subjects) OR TARGET IN (SELECT SUBJECT_ID FROM inst_subjects))
    AND SOURCE_NAME NOT ILIKE '%{safe_inst}%'
    AND TARGET_NAME NOT ILIKE '%{safe_inst}%'
    {cat_filter}
),
org_matches AS (
    -- Step 3: aggregate — count distinct shared subjects and total weight per org
    SELECT
        ORG_ID,
        ORG_NAME,
        ORG_CATEGORY,
        COUNT(DISTINCT MATCHED_SUBJECT_ID) AS SHARED_SUBJECTS,
        SUM(WEIGHT) AS TOTAL_WEIGHT
    FROM org_subject_edges
    GROUP BY ORG_ID, ORG_NAME, ORG_CATEGORY
),
direct_collabs AS (
    -- Step 4: any org that has ever appeared in a direct collaboration edge with the institution
    SELECT DISTINCT
        CASE WHEN SOURCE_NAME ILIKE '%{safe_inst}%' THEN TARGET ELSE SOURCE END AS COLLAB_ID
    FROM GRAPH_NETWORK.GRAPH.ALL_EDGES_ENRICHED
    WHERE (SOURCE_NAME ILIKE '%{safe_inst}%' OR TARGET_NAME ILIKE '%{safe_inst}%')
    AND EDGE_TYPE IN {collab_edge_types}
)
-- Step 5: return orgs with shared subjects that never directly collaborated
SELECT
    ORG_ID,
    ORG_NAME,
    ORG_CATEGORY,
    SHARED_SUBJECTS,
    TOTAL_WEIGHT
FROM org_matches
WHERE ORG_ID NOT IN (SELECT COLLAB_ID FROM direct_collabs)
AND SHARED_SUBJECTS > 0
ORDER BY SHARED_SUBJECTS DESC, TOTAL_WEIGHT DESC
LIMIT {top_n}
"""
    return run_query(sql)


def run_similar_no_collab_subject_edges(
    institution: str,
    org_ids: list,
    ip_type: str = None,
    subject_filter: str = None,
) -> pd.DataFrame:
    """
    For a list of matched orgs, fetch their actual connections to shared subjects
    so we can draw org → subject edges in the graph.
    """
    if not org_ids:
        return pd.DataFrame()

    safe_inst = sql_escape(institution)
    ip_filter = f"AND IP_TYPE = '{sql_escape(ip_type)}'" if ip_type and ip_type != "All" else ""

    if ip_type == "Patents":
        subject_edge_types = "('Applicant_Subject')"
    elif ip_type == "Publications":
        subject_edge_types = "('Institute_Subject')"
    else:
        subject_edge_types = "('Applicant_Subject', 'Institute_Subject')"

    subject_clause = (
        f"AND (SOURCE_NAME ILIKE '%{sql_escape(subject_filter)}%' OR TARGET_NAME ILIKE '%{sql_escape(subject_filter)}%')"
        if subject_filter else ""
    )

    # Quote org IDs for SQL IN clause
    quoted_ids = ", ".join(f"'{sql_escape(str(oid))}'" for oid in org_ids)

    sql = f"""
WITH inst_subjects AS (
    SELECT DISTINCT
        CASE WHEN SOURCE_NAME ILIKE '%{safe_inst}%' THEN TARGET ELSE SOURCE END AS SUBJECT_ID,
        CASE WHEN SOURCE_NAME ILIKE '%{safe_inst}%' THEN TARGET_NAME ELSE SOURCE_NAME END AS SUBJECT_NAME
    FROM GRAPH_NETWORK.GRAPH.ALL_EDGES_ENRICHED
    WHERE (SOURCE_NAME ILIKE '%{safe_inst}%' OR TARGET_NAME ILIKE '%{safe_inst}%')
    AND EDGE_TYPE IN {subject_edge_types}
    {ip_filter}
    {subject_clause}
)
SELECT
    CASE WHEN SOURCE IN (SELECT SUBJECT_ID FROM inst_subjects) THEN TARGET ELSE SOURCE END AS ORG_ID,
    CASE WHEN SOURCE IN (SELECT SUBJECT_ID FROM inst_subjects) THEN TARGET_NAME ELSE SOURCE_NAME END AS ORG_NAME,
    CASE WHEN SOURCE IN (SELECT SUBJECT_ID FROM inst_subjects) THEN SOURCE ELSE TARGET END AS SUBJECT_ID,
    CASE WHEN SOURCE IN (SELECT SUBJECT_ID FROM inst_subjects) THEN SOURCE_NAME ELSE TARGET_NAME END AS SUBJECT_NAME,
    WEIGHT
FROM GRAPH_NETWORK.GRAPH.ALL_EDGES_ENRICHED
WHERE EDGE_TYPE IN {subject_edge_types}
{ip_filter}
AND (SOURCE IN (SELECT SUBJECT_ID FROM inst_subjects) OR TARGET IN (SELECT SUBJECT_ID FROM inst_subjects))
AND (SOURCE IN ({quoted_ids}) OR TARGET IN ({quoted_ids}))
"""
    return run_query(sql)


def build_similar_no_collab_graph(results_df: pd.DataFrame, edges_df: pd.DataFrame) -> str:
    """
    Build a bipartite graph: organisations (blue) on the left, shared subject areas (orange) on the right.
    Edges connect orgs to their matching subjects. Node size = number of connections.
    """
    net = Network(
        height="750px",
        width="100%",
        bgcolor="#ffffff",
        font_color="#222222",
        directed=False,
        notebook=False,
        cdn_resources="in_line",
    )
    net.barnes_hut(
        gravity=-15000,
        central_gravity=0.1,
        spring_length=250,
        spring_strength=0.015,
        damping=0.85,
        overlap=0.5,
    )

    added_orgs = set()
    added_subjects = set()

    # Build lookup for org metadata from results_df
    org_meta = {
        str(row["ORG_ID"]): {
            "name": str(row["ORG_NAME"]),
            "category": str(row.get("ORG_CATEGORY", "")),
            "shared": int(row["SHARED_SUBJECTS"]),
            "weight": float(row["TOTAL_WEIGHT"]),
        }
        for _, row in results_df.iterrows()
    }

    for _, row in edges_df.iterrows():
        org_id = str(row["ORG_ID"])
        org_name = str(row["ORG_NAME"])
        subj_id = str(row["SUBJECT_ID"])
        subj_name = str(row["SUBJECT_NAME"])
        weight = float(row["WEIGHT"])

        meta = org_meta.get(org_id, {})

        if org_id not in added_orgs:
            shared = meta.get("shared", 1)
            net.add_node(
                org_id,
                label=org_name,
                title=(
                    f"{org_name}<br>"
                    f"Category: {meta.get('category', '')}<br>"
                    f"Shared subjects: {shared}<br>"
                    f"Total strength: {meta.get('weight', 0):.0f}"
                ),
                color="#9DC3E6",
                value=shared,
            )
            added_orgs.add(org_id)

        if subj_id not in added_subjects:
            net.add_node(
                subj_id,
                label=subj_name,
                title=f"Subject: {subj_name}",
                color="#F4B183",
                value=3,
            )
            added_subjects.add(subj_id)

        net.add_edge(
            org_id,
            subj_id,
            value=weight,
            title=f"Strength: {weight:.0f}",
        )

    net.show_buttons(filter_=["physics"])
    return net.generate_html(notebook=False)


def build_pyvis_graph(df: pd.DataFrame) -> str:
    net = Network(
        height="750px",
        width="100%",
        bgcolor="#ffffff",
        font_color="#222222",
        directed=False,
        notebook=False,
        cdn_resources="in_line",
    )

    net.barnes_hut(
        gravity=-30000,
        central_gravity=0.3,
        spring_length=180,
        spring_strength=0.02,
        damping=0.8,
        overlap=0.5,
    )

    added_nodes = set()

    for _, row in df.iterrows():
        source_id = row["SOURCE"]
        target_id = row["TARGET"]
        source_name = row["SOURCE_NAME"]
        target_name = row["TARGET_NAME"]
        source_type = row["SOURCE_TYPE"]
        target_type = row["TARGET_TYPE"]
        weight = row["WEIGHT"]
        edge_type = row["EDGE_TYPE"]

        if source_id not in added_nodes:
            net.add_node(
                source_id,
                label=source_name,
                title=f"{source_name}<br>Type: {source_type}<br>Category: {row['SOURCE_CATEGORY']}",
                color=get_node_color(source_type, source_name, row["SOURCE_NUS_AFFILIATED"]),
                value=1,
            )
            added_nodes.add(source_id)

        if target_id not in added_nodes:
            net.add_node(
                target_id,
                label=target_name,
                title=f"{target_name}<br>Type: {target_type}<br>Category: {row['TARGET_CATEGORY']}",
                color=get_node_color(target_type, target_name, row["TARGET_NUS_AFFILIATED"]),
                value=1,
            )
            added_nodes.add(target_id)

        net.add_edge(
            source_id,
            target_id,
            value=float(weight),
            title=f"Edge type: {edge_type}<br>Weight: {weight}",
        )

    net.show_buttons(filter_=["physics"])
    html = net.generate_html(notebook=False)
    return html


# -----------------------------
# LLM chat helper
# -----------------------------
SYSTEM_PROMPT = """You are a research collaboration discovery assistant helping users explore a graph network of patents and academic publications.

Based on the user's natural language request, extract their intent and return a JSON object with the following fields:

{
  "query_mode": "<string>",             // REQUIRED. One of:
                                        // "standard"          — normal graph filter mode (default)
                                        // "similar_no_collab" — find orgs with similar research interests that have NOT yet collaborated with the searched institution.
                                        //   Use this when user says things like: "similar interests but no collaboration", "hasn't worked with", "potential new partners", "never collaborated", "who to reach out to", "didn't collaborate"
  "ip_type": "<string or null>",        // "Patents" or "Publications", or null for both. Available: [AVAILABLE_IP_TYPES]
  "edge_type": "<string or null>",      // type of connection, or null for all. Available: [AVAILABLE_EDGE_TYPES]
                                        // Applicant_Applicant = patent co-applicants
                                        // Applicant_Subject = patent-to-subject links
                                        // Institute_Subject = publication-to-subject links
                                        // Institution_Institution = publication co-authors
  "search_term": "<string or null>",    // institution name to focus on, or null
  "category": "<string or null>",       // organisation category, or null for all. Available: [AVAILABLE_CATEGORIES]
  "min_weight": <integer or null>,      // minimum collaboration strength, or null to keep current
  "max_edges": <integer or null>,       // max edges to load (20–1000), or null to keep current.
                                        // For "top N" requests: set max_edges=300 and top_n_nodes=N instead.
  "top_n_nodes": <integer or null>,     // for "top N partners" in standard mode, set to N. Leave null otherwise.
  "top_n_results": <integer or null>,   // for "similar_no_collab" mode only: how many results to return. Default null (will use 20).
  "subject_filter": "<string or null>", // for "similar_no_collab" mode: scope the search to a specific subject area mentioned by the user (e.g. "COMPUTER SCIENCE & INFORMATION SYSTEMS"). Use the exact subject name if mentioned, or null for all subjects.
  "explanation": "<friendly 1-2 sentence explanation of what the results will show>"
}

Rules:
- query_mode is ALWAYS required. Default to "standard" unless the user clearly wants "similar_no_collab".
- Only set other fields the user explicitly implies. Leave others null (do not change).
- ip_type must exactly match one of: [AVAILABLE_IP_TYPES]
- edge_type must exactly match one of: [AVAILABLE_EDGE_TYPES]
- category must exactly match one of: [AVAILABLE_CATEGORIES]
- If the user mentions "NUS" or "National University of Singapore", set search_term to "NATIONAL UNIVERSITY OF SINGAPORE".
- If the user says "reset", "clear", "start over", or "show everything", set query_mode="standard", ip_type=null, edge_type=null, search_term=null, min_weight=1, max_edges=800, category=null, top_n_nodes=null, top_n_results=null.
- For "similar_no_collab" mode, search_term is required.
- Always return ONLY valid JSON. No markdown fences, no extra text."""


def extract_filters_from_llm(
    user_message: str,
    chat_history: list,
    available_ip_types: list,
    available_edge_types: list,
    available_categories: list,
) -> dict: 

    system = SYSTEM_PROMPT.replace(
        "[AVAILABLE_IP_TYPES]", ", ".join(available_ip_types)
    ).replace(
        "[AVAILABLE_EDGE_TYPES]", ", ".join(available_edge_types)
    ).replace(
        "[AVAILABLE_CATEGORIES]", ", ".join(available_categories)
    )

    messages = []
    for turn in chat_history:
        messages.append({"role": turn["role"], "content": turn["content"]})
    messages.append({"role": "user", "content": user_message})

    response = client.messages.create(
        model="claude-sonnet-4-5",
        max_tokens=512,
        system=system,
        messages=messages,
    )

    raw = response.content[0].text.strip()

    # Strip markdown fences if model adds them anyway
    raw = re.sub(r"^```(?:json)?", "", raw).strip()
    raw = re.sub(r"```$", "", raw).strip()

    return json.loads(raw)


def apply_llm_filters(parsed: dict, current_state: dict, available_ip_types: list, available_edge_types: list, available_categories: list) -> dict:
    """Merge LLM-extracted filters onto the current filter state."""
    new_state = current_state.copy()

    if parsed.get("ip_type") is not None:
        val = parsed["ip_type"].upper().rstrip("S")  # "PATENTS"→"PATENT", "PUBLICATIONS"→"PUBLICATION"
        match = next((t for t in available_ip_types if t.upper().rstrip("S") == val), None)
        new_state["ip_type"] = match if match else "All"

    if parsed.get("edge_type") is not None:
        val = parsed["edge_type"].upper().replace("-", "_").replace(" ", "_")
        match = next((t for t in available_edge_types if t.upper().replace("-", "_") == val), None)
        new_state["edge_type"] = match if match else "All"

    if parsed.get("category") is not None:
        val = parsed["category"].upper()
        match = next((c for c in available_categories if c.upper() == val), None)
        new_state["category"] = match if match else "All"

    if parsed.get("search_term") is not None:
        new_state["search_term"] = parsed["search_term"]

    if parsed.get("min_weight") is not None:
        new_state["min_weight"] = max(1, int(parsed["min_weight"]))

    if parsed.get("max_edges") is not None:
        new_state["max_edges"] = max(20, min(1000, int(parsed["max_edges"])))

    new_state["top_n_nodes"] = int(parsed["top_n_nodes"]) if parsed.get("top_n_nodes") is not None else None
    new_state["query_mode"] = parsed.get("query_mode", "standard") or "standard"
    new_state["top_n_results"] = int(parsed["top_n_results"]) if parsed.get("top_n_results") is not None else None
    new_state["subject_filter"] = parsed.get("subject_filter") or None

    return new_state


# -----------------------------
# Load filter options
# -----------------------------
edge_types_df = run_query("""
    SELECT DISTINCT EDGE_TYPE
    FROM GRAPH_NETWORK.GRAPH.ALL_EDGES_ENRICHED
    WHERE EDGE_TYPE IS NOT NULL
    ORDER BY EDGE_TYPE
""")

ip_types_df = run_query("""
    SELECT DISTINCT IP_TYPE
    FROM GRAPH_NETWORK.GRAPH.ALL_EDGES_ENRICHED
    WHERE IP_TYPE IS NOT NULL
    ORDER BY IP_TYPE
""")

categories_df = run_query("""
    SELECT DISTINCT SOURCE_CATEGORY AS CATEGORY
    FROM GRAPH_NETWORK.GRAPH.ALL_EDGES_ENRICHED
    WHERE SOURCE_CATEGORY IS NOT NULL
    UNION
    SELECT DISTINCT TARGET_CATEGORY AS CATEGORY
    FROM GRAPH_NETWORK.GRAPH.ALL_EDGES_ENRICHED
    WHERE TARGET_CATEGORY IS NOT NULL
    ORDER BY CATEGORY
""")

edge_types_raw = edge_types_df["EDGE_TYPE"].tolist()
ip_types_raw = ip_types_df["IP_TYPE"].tolist()
categories_raw = categories_df["CATEGORY"].tolist()

edge_types = ["All"] + edge_types_raw
ip_types = ["All"] + ip_types_raw
categories = ["All"] + categories_raw


# -----------------------------
# Session state initialisation
# -----------------------------
if "filter_state" not in st.session_state:
    st.session_state.filter_state = {
        "ip_type": "All",
        "edge_type": "All",
        "search_term": "",
        "min_weight": 1,
        "max_edges": 800,
        "category": "All",
        "top_n_nodes": None,
        "query_mode": "standard",
        "top_n_results": None,
        "subject_filter": None,
    }

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []  # list of {"role": ..., "content": ...}

if "chat_display" not in st.session_state:
    st.session_state.chat_display = []  # list of {"role": ..., "content": ..., "filters": ...}


# -----------------------------
# Sidebar: manual filters + chat
# -----------------------------
with st.sidebar:
    st.header("⚙️ Manual Filters")
    st.caption("Or just ask the AI assistant below the graph.")

    st.selectbox(
        "Research output type",
        ip_types,
        index=ip_types.index(st.session_state.filter_state["ip_type"]) if st.session_state.filter_state["ip_type"] in ip_types else 0,
        key="sb_ip_type",
        on_change=lambda: st.session_state.filter_state.update({"ip_type": st.session_state.sb_ip_type, "top_n_nodes": None, "query_mode": "standard"}),
    )

    st.selectbox(
        "Connection type",
        edge_types,
        index=edge_types.index(st.session_state.filter_state["edge_type"]) if st.session_state.filter_state["edge_type"] in edge_types else 0,
        key="sb_edge_type",
        on_change=lambda: st.session_state.filter_state.update({"edge_type": st.session_state.sb_edge_type, "top_n_nodes": None, "query_mode": "standard"}),
    )

    st.selectbox(
        "Organisation category",
        categories,
        index=categories.index(st.session_state.filter_state["category"]) if st.session_state.filter_state["category"] in categories else 0,
        key="sb_category",
        on_change=lambda: st.session_state.filter_state.update({"category": st.session_state.sb_category, "top_n_nodes": None, "query_mode": "standard"}),
    )

    st.text_input(
        "Search for an institution or organisation",
        value=st.session_state.filter_state["search_term"],
        placeholder="e.g. NATIONAL UNIVERSITY OF SINGAPORE",
        key="sb_search",
        on_change=lambda: st.session_state.filter_state.update({"search_term": st.session_state.sb_search, "top_n_nodes": None, "query_mode": "standard"}),
    )

    st.number_input(
        "Minimum collaboration strength",
        min_value=1,
        value=st.session_state.filter_state["min_weight"],
        key="sb_min_weight",
        on_change=lambda: st.session_state.filter_state.update({"min_weight": st.session_state.sb_min_weight}),
    )

    st.slider(
        "Maximum connections to load",
        min_value=20,
        max_value=1000,
        value=st.session_state.filter_state["max_edges"],
        step=20,
        key="sb_max_edges",
        on_change=lambda: st.session_state.filter_state.update({"max_edges": st.session_state.sb_max_edges, "top_n_nodes": None}),
    )

    st.divider()
    st.markdown(
        """
        **Legend**  
        🔴 NUS-affiliated  
        🟠 Research subject  
        🔵 Patent applicant  
        🟢 Publication institute  
        ⚪ Other  
        """
    )

# -----------------------------
# Read effective filters (may have been updated by LLM)
# -----------------------------
fs = st.session_state.filter_state
selected_ip_type = fs["ip_type"]
selected_edge_type = fs["edge_type"]
selected_category = fs["category"]
search_term = fs["search_term"]
min_weight = fs["min_weight"]
max_edges = fs["max_edges"]
top_n_nodes = fs.get("top_n_nodes")
query_mode = fs.get("query_mode", "standard")
top_n_results = fs.get("top_n_results") or 20
subject_filter = fs.get("subject_filter")


# -----------------------------
# Main output
# -----------------------------
st.subheader("🗺️ Collaboration Network")

if query_mode == "similar_no_collab":
    # --- Similar interests, no prior collaboration mode ---
    if not search_term.strip():
        st.warning("Please specify an institution to search from. Try asking: _'Show corporations with similar interests to NUS in publications that haven't collaborated with NUS'_")
    else:
        st.markdown(
            f"Showing organisations with **similar research interests** to **{search_term.title()}** "
            f"that have **not yet directly collaborated** with it. "
            "Ranked by number of shared subjects. Hover over nodes for details."
        )
        with st.spinner("Running analysis…"):
            similar_df = run_similar_no_collab_query(
                institution=search_term.strip(),
                ip_type=selected_ip_type if selected_ip_type != "All" else None,
                category=selected_category if selected_category != "All" else None,
                top_n=top_n_results,
                subject_filter=subject_filter,
            )

        if similar_df.empty:
            st.info("No matches found. Try broadening the filters — e.g. remove the category filter or change the output type.")
        else:
            # Fetch org→subject edges for the bipartite graph
            org_ids = similar_df["ORG_ID"].tolist()
            with st.spinner("Loading subject connections…"):
                edges_df = run_similar_no_collab_subject_edges(
                    institution=search_term.strip(),
                    org_ids=org_ids,
                    ip_type=selected_ip_type if selected_ip_type != "All" else None,
                    subject_filter=subject_filter,
                )

            html = build_similar_no_collab_graph(similar_df, edges_df)
            components.html(html, height=780, scrolling=True)
            st.caption(
                f"🔵 Blue nodes = potential partners ({len(similar_df)})  "
                f"🟠 Orange nodes = shared research subjects.  "
                "Hover over any node or edge for details."
            )

            st.subheader("📋 Ranked results")
            display_df = similar_df[["ORG_NAME", "ORG_CATEGORY", "SHARED_SUBJECTS", "TOTAL_WEIGHT"]].copy()
            display_df.columns = ["Organisation", "Category", "Shared Subjects", "Collaboration Strength"]
            display_df.index = range(1, len(display_df) + 1)
            st.dataframe(display_df, use_container_width=True)

else:
    # --- Standard graph mode ---
    st.markdown(
        "Nodes represent institutions, corporations, or research subjects. "
        "Thicker lines indicate stronger collaboration. **Hover over any node or edge** for details."
    )

    where_clauses = [f"WEIGHT >= {min_weight}"]

    if selected_ip_type != "All":
        where_clauses.append(f"IP_TYPE = '{sql_escape(selected_ip_type)}'")

    if selected_edge_type != "All":
        where_clauses.append(f"EDGE_TYPE = '{sql_escape(selected_edge_type)}'")

    if selected_category != "All":
        safe_cat = sql_escape(selected_category)
        where_clauses.append(f"(SOURCE_CATEGORY = '{safe_cat}' OR TARGET_CATEGORY = '{safe_cat}')")

    if search_term.strip():
        safe_search = sql_escape(search_term.strip())
        where_clauses.append(f"(SOURCE_NAME ILIKE '%{safe_search}%' OR TARGET_NAME ILIKE '%{safe_search}%')")

    where_sql = " AND ".join(where_clauses)

    sql = f"""
SELECT
    SOURCE,
    SOURCE_NAME,
    SOURCE_TYPE,
    SOURCE_CATEGORY,
    SOURCE_NUS_AFFILIATED,
    TARGET,
    TARGET_NAME,
    TARGET_TYPE,
    TARGET_CATEGORY,
    TARGET_NUS_AFFILIATED,
    EDGE_TYPE,
    IP_TYPE,
    WEIGHT
FROM GRAPH_NETWORK.GRAPH.ALL_EDGES_ENRICHED
WHERE {where_sql}
ORDER BY WEIGHT DESC
LIMIT {max_edges}
"""

    df = run_query(sql)

    if top_n_nodes and search_term.strip():
        df = keep_top_n_neighbours(df, search_term.strip(), top_n_nodes)
    elif not search_term.strip():
        df = keep_top_communities(df, top_n=10)

    if df.empty:
        st.info(
            "No connections found for the selected filters. "
            "Try asking the AI assistant below."
        )
    else:
        html = build_pyvis_graph(df)
        components.html(html, height=780, scrolling=True)

        n_nodes = pd.concat([df["SOURCE"], df["TARGET"]]).nunique()
        n_edges = len(df)
        st.caption(
            f"Showing {n_nodes:,} institutions across {n_edges:,} connections. "
            "Drag nodes to rearrange, scroll to zoom."
        )

    with st.expander("📊 View connection data"):
        st.dataframe(df, use_container_width=True)

    with st.expander("🔍 View SQL query"):
        st.code(sql, language="sql")

st.divider()

# ---- LLM Chat ----
st.subheader("💬 AI Research Assistant")
st.caption(
    "Ask in plain language to explore the network. Try: "
    "_'Who are the top 10 institutions collaborating with NUS on publications?'_ or "
    "_'Show corporations with similar interests to NUS that haven\\'t collaborated with NUS before'_"
)

# Display chat history
for msg in st.session_state.chat_display:
    if msg["role"] == "user":
        st.chat_message("user").write(msg["content"])
    else:
        with st.chat_message("assistant"):
            st.write(msg["content"])
            if msg.get("filters"):
                with st.expander("Applied filters", expanded=False):
                    st.json(msg["filters"])

# Chat input using st.form — Enter submits, clears after send, renders inline
with st.form(key="chat_form", clear_on_submit=True):
    col1, col2 = st.columns([5, 1])
    with col1:
        user_input = st.text_input(
            "Your question",
            placeholder="e.g. Show me top 10 corporations collaborating with NUS on publications",
            label_visibility="collapsed",
        )
    with col2:
        submitted = st.form_submit_button("Send ➤", use_container_width=True)

if submitted and user_input.strip():
    # Show user message immediately
    st.session_state.chat_display.append({"role": "user", "content": user_input})

    with st.spinner("Thinking…"):
        try:
            parsed = extract_filters_from_llm(
                user_message=user_input,
                chat_history=st.session_state.chat_history,
                available_ip_types=ip_types_raw,
                available_edge_types=edge_types_raw,
                available_categories=categories_raw,
            )

            explanation = parsed.pop("explanation", "Filters updated.")

            new_fs = apply_llm_filters(
                parsed,
                st.session_state.filter_state,
                ip_types_raw,
                edge_types_raw,
                categories_raw,
            )
            st.session_state.filter_state = new_fs

            # Build a readable summary of what changed
            changed = {k: v for k, v in parsed.items() if v is not None}

            # Store in chat histories
            st.session_state.chat_history.append({"role": "user", "content": user_input})
            st.session_state.chat_history.append({"role": "assistant", "content": explanation})

            st.session_state.chat_display.append({
                "role": "assistant",
                "content": explanation,
                "filters": changed if changed else None,
            })

        except Exception as e:
            error_msg = f"Sorry, I couldn't understand that request. Please try rephrasing. (Error: {e})"
            st.session_state.chat_display.append({"role": "assistant", "content": error_msg})

    run_query.clear()
    st.rerun()
