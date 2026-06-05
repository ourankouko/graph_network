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


# -----------------------------
# Helper functions
# -----------------------------
@st.cache_data(ttl=3600)
def run_query(sql: str) -> pd.DataFrame:
    return session.sql(sql).to_pandas()


def sql_escape(value: str) -> str:
    return value.replace("'", "''")


def get_node_color(node_type: str) -> str:
    node_type = str(node_type).upper()

    if "SUBJECT" in node_type:
        return "#f4b183"
    if "APPLICANT" in node_type:
        return "#9dc3e6"
    if "INSTITUTE" in node_type:
        return "#a9d18e"

    return "#d9d9d9"

def keep_top_communities(df: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    """
    Detect communities from the filtered edge dataframe and keep only
    edges where both source and target are in the top N communities.
    """

    if df.empty:
        return df

    G = nx.Graph()

    for _, row in df.iterrows():
        G.add_edge(
            row["SOURCE"],
            row["TARGET"],
            weight=float(row["WEIGHT"])
        )

    if G.number_of_edges() == 0:
        return df

    communities = list(nx.community.greedy_modularity_communities(G, weight="weight"))

    community_rows = []

    for i, community in enumerate(communities):
        community_rows.append({
            "community_id": i,
            "nodes": set(community),
            "size": len(community)
        })

    community_rows = sorted(
        community_rows,
        key=lambda x: x["size"],
        reverse=True
    )

    top_communities = community_rows[:top_n]

    node_to_community = {}

    for community in top_communities:
        for node in community["nodes"]:
            node_to_community[node] = community["community_id"]

    top_nodes = set(node_to_community.keys())

    filtered_df = df[
        df["SOURCE"].isin(top_nodes) &
        df["TARGET"].isin(top_nodes)
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
        cdn_resources='in_line',
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
                color=get_node_color(source_type),
                value=1,
            )
            added_nodes.add(source_id)

        if target_id not in added_nodes:
            net.add_node(
                target_id,
                label=target_name,
                title=f"{target_name}<br>Type: {target_type}",
                color=get_node_color(target_type),
                value=1,
            )
            added_nodes.add(target_id)

        net.add_edge(
            source_id,
            target_id,
            value=float(weight),
            title=f"Edge type: {edge_type}<br>Weight: {weight}",
        )

    # Useful built-in controls for demo
    net.show_buttons(filter_=["physics"])

    html = net.generate_html(notebook=False)
    return html

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

edge_types = ["All"] + edge_types_df["EDGE_TYPE"].tolist()
ip_types = ["All"] + ip_types_df["IP_TYPE"].tolist()


# -----------------------------
# Sidebar filters
# -----------------------------
with st.sidebar:
    st.header("Filters")

    selected_ip_type = st.selectbox("IP type", ip_types)

    selected_edge_type = st.selectbox("Edge type", edge_types)

    search_term = st.text_input(
        "Search source or target name",
        placeholder="e.g. NATIONAL UNIVERSITY OF SINGAPORE",
    )

    min_weight = st.number_input(
        "Minimum edge weight",
        min_value=1,
        value=1,
    )

    max_edges = st.slider(
        "Maximum edges to visualise",
        min_value=20,
        max_value=1000,
        value=800,
        step=20,
    )


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
    TARGET,
    TARGET_NAME,
    TARGET_TYPE,
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
    st.dataframe(df, width="stretch")

with st.expander("Show SQL"):
    st.code(sql, language="sql")
