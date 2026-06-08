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
                title=f"{source_name}<br>Type: {source_type}",
                color=get_node_color(source_type, source_name, row["SOURCE_NUS_AFFILIATED"]),
                value=1,
            )
            added_nodes.add(source_id)

        if target_id not in added_nodes:
            net.add_node(
                target_id,
                label=target_name,
                title=f"{target_name}<br>Type: {target_type}",
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
SYSTEM_PROMPT = """You are an assistant helping users explore a graph network of patents and publications.
The graph has filters a user can set. Based on the user's natural language request, extract their intent and return a JSON object with the following fields:

{
  "ip_type": "<string or null>",        // e.g. "PATENT", "PUBLICATION", or null for All
  "edge_type": "<string or null>",      // e.g. "APPLICANT-SUBJECT", or null for All
  "search_term": "<string or null>",    // name to search in source/target, or null
  "min_weight": <integer or null>,      // minimum edge weight, or null to keep current
  "max_edges": <integer or null>,       // max edges to show (20–1000), or null to keep current
  "explanation": "<short human-readable summary of what you understood>"
}

Rules:
- Only set fields the user explicitly or clearly implies. Leave others null (meaning: do not change).
- ip_type must be one of the exact values from this list (case-insensitive match): [AVAILABLE_IP_TYPES]
- edge_type must be one of the exact values from this list: [AVAILABLE_EDGE_TYPES]
- If the user mentions "NUS" or "National University of Singapore", set search_term to "NATIONAL UNIVERSITY OF SINGAPORE".
- If the user says "reset", "clear", or "show everything", return all nulls except set ip_type=null, edge_type=null, search_term=null, min_weight=1, max_edges=800.
- Always return ONLY valid JSON. No markdown fences, no extra text."""


def extract_filters_from_llm(
    user_message: str,
    chat_history: list,
    available_ip_types: list,
    available_edge_types: list,
) -> dict: 

    system = SYSTEM_PROMPT.replace(
        "[AVAILABLE_IP_TYPES]", ", ".join(available_ip_types)
    ).replace(
        "[AVAILABLE_EDGE_TYPES]", ", ".join(available_edge_types)
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


def apply_llm_filters(parsed: dict, current_state: dict, available_ip_types: list, available_edge_types: list) -> dict:
    """Merge LLM-extracted filters onto the current filter state."""
    new_state = current_state.copy()

    if parsed.get("ip_type") is not None:
        val = parsed["ip_type"].upper()
        match = next((t for t in available_ip_types if t.upper() == val), None)
        new_state["ip_type"] = match if match else "All"
    else:
        # null means don't change
        pass

    if parsed.get("edge_type") is not None:
        val = parsed["edge_type"].upper()
        match = next((t for t in available_edge_types if t.upper() == val), None)
        new_state["edge_type"] = match if match else "All"

    if parsed.get("search_term") is not None:
        new_state["search_term"] = parsed["search_term"]

    if parsed.get("min_weight") is not None:
        new_state["min_weight"] = max(1, int(parsed["min_weight"]))

    if parsed.get("max_edges") is not None:
        new_state["max_edges"] = max(20, min(1000, int(parsed["max_edges"])))

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

edge_types_raw = edge_types_df["EDGE_TYPE"].tolist()
ip_types_raw = ip_types_df["IP_TYPE"].tolist()

edge_types = ["All"] + edge_types_raw
ip_types = ["All"] + ip_types_raw


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
    }

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []  # list of {"role": ..., "content": ...}

if "chat_display" not in st.session_state:
    st.session_state.chat_display = []  # list of {"role": ..., "content": ..., "filters": ...}


# -----------------------------
# Sidebar: manual filters + chat
# -----------------------------
with st.sidebar:
    st.header("Filters")

    fs = st.session_state.filter_state

    selected_ip_type = st.selectbox(
        "IP type",
        ip_types,
        index=ip_types.index(fs["ip_type"]) if fs["ip_type"] in ip_types else 0,
        key="sb_ip_type",
    )

    selected_edge_type = st.selectbox(
        "Edge type",
        edge_types,
        index=edge_types.index(fs["edge_type"]) if fs["edge_type"] in edge_types else 0,
        key="sb_edge_type",
    )

    search_term = st.text_input(
        "Search source or target name",
        value=fs["search_term"],
        placeholder="e.g. NATIONAL UNIVERSITY OF SINGAPORE",
        key="sb_search",
    )

    min_weight = st.number_input(
        "Minimum edge weight",
        min_value=1,
        value=fs["min_weight"],
        key="sb_min_weight",
    )

    max_edges = st.slider(
        "Maximum edges to visualise",
        min_value=20,
        max_value=1000,
        value=fs["max_edges"],
        step=20,
        key="sb_max_edges",
    )

    # Keep session state in sync with manual widget changes
    st.session_state.filter_state = {
        "ip_type": selected_ip_type,
        "edge_type": selected_edge_type,
        "search_term": search_term,
        "min_weight": min_weight,
        "max_edges": max_edges,
    }

    st.divider()

# -----------------------------
# Read effective filters (may have been updated by LLM)
# -----------------------------
fs = st.session_state.filter_state
selected_ip_type = fs["ip_type"]
selected_edge_type = fs["edge_type"]
search_term = fs["search_term"]
min_weight = fs["min_weight"]
max_edges = fs["max_edges"]


# -----------------------------
# Build SQL
# -----------------------------
where_clauses = [f"WEIGHT >= {min_weight}"]

if selected_ip_type != "All":
    where_clauses.append(f"IP_TYPE = '{sql_escape(selected_ip_type)}'")

if selected_edge_type != "All":
    where_clauses.append(f"EDGE_TYPE = '{sql_escape(selected_edge_type)}'")

if search_term.strip():
    safe_search = sql_escape(search_term.strip())
    where_clauses.append(
        f"(SOURCE_NAME ILIKE '%{safe_search}%' OR TARGET_NAME ILIKE '%{safe_search}%')"
    )

where_sql = " AND ".join(where_clauses)

sql = f"""
SELECT
    SOURCE,
    SOURCE_NAME,
    SOURCE_TYPE,
    SOURCE_NUS_AFFILIATED,
    TARGET,
    TARGET_NAME,
    TARGET_TYPE,
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

if not search_term.strip():
    df = keep_top_communities(df, top_n=10)


# -----------------------------
# Main output
# -----------------------------
st.subheader("Network graph")
st.markdown(
    """
    **Legend:**  
    🔴 NUS-affiliated node  
    🟠 QS Subject  
    🔵 Patent applicant  
    🟢 Publication institute  
    """
)

if df.empty:
    st.warning("No edges found for the selected filters.")
else:
    html = build_pyvis_graph(df)
    components.html(html, height=780, scrolling=True)

    n_nodes = pd.concat([df["SOURCE"], df["TARGET"]]).nunique()
    n_edges = len(df)

    st.caption(
        f"Displayed {n_nodes:,} nodes and {n_edges:,} edges. "
        "Drag nodes around, zoom, and use the physics controls if needed."
    )

with st.expander("Show edge table"):
    st.dataframe(df, use_container_width=True)

with st.expander("Show SQL"):
    st.code(sql, language="sql")

st.divider()

# ---- LLM Chat ----
st.subheader("💬 Ask the AI assistant")
st.caption("Use natural language to filter the graph, e.g. _'Show NUS patents with weight above 10'_")

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

# Chat input
user_input = st.chat_input("Ask anything about the network…")

if user_input:
    # Show user message immediately
    st.session_state.chat_display.append({"role": "user", "content": user_input})

    with st.spinner("Thinking…"):
        try:
            parsed = extract_filters_from_llm(
                user_message=user_input,
                chat_history=st.session_state.chat_history,
                available_ip_types=ip_types_raw,
                available_edge_types=edge_types_raw,
            )

            explanation = parsed.pop("explanation", "Filters updated.")

            new_fs = apply_llm_filters(
                parsed,
                st.session_state.filter_state,
                ip_types_raw,
                edge_types_raw,
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
            error_msg = f"Sorry, I couldn't parse your request. Error: {e}"
            st.session_state.chat_display.append({"role": "assistant", "content": error_msg})

    st.rerun()
