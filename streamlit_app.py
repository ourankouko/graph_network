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
st.set_page_config(page_title="Research Collaboration Explorer", layout="wide", initial_sidebar_state="collapsed")

st.title("🔬 Research Collaboration Explorer")
st.write("Discover institutions and corporations with shared research interests for potential collaboration.")


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
        return "#ff9933"

    if "NATIONAL UNIVERSITY OF SINGAPORE" in node_name:
        return "#ff9933"

    if "SUBJECT" in node_type:
        return "#F4B183"

    if "APPLICANT" in node_type:
        return "#9DC3E6"

    if "INSTITUTE" in node_type:
        return "#33cccc"

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


def run_recommendation_query(
    institution: str,
    subject_filter: str = None,
    category: str = "Corporation",
    top_n: int = 3,
) -> pd.DataFrame:
    """
    Find top N recommended industry partners for an institution.
    Returns ALL orgs with shared subjects (both collaborators and non-collaborators),
    flagged with IS_NEW_OPPORTUNITY.
    Aggregates across both Patents and Publications.
    """
    safe_inst = sql_escape(institution)
    subject_clause = (
        f"AND (SOURCE_NAME ILIKE '%{sql_escape(subject_filter)}%' OR TARGET_NAME ILIKE '%{sql_escape(subject_filter)}%')"
        if subject_filter else ""
    )
    cat_filter = (
        f"AND (SOURCE_CATEGORY = '{sql_escape(category)}' OR TARGET_CATEGORY = '{sql_escape(category)}')"
        if category and category != "All" else ""
    )

    sql = f"""
WITH inst_subjects AS (
    SELECT DISTINCT
        CASE WHEN SOURCE_NAME ILIKE '%{safe_inst}%' THEN TARGET ELSE SOURCE END AS SUBJECT_ID,
        CASE WHEN SOURCE_NAME ILIKE '%{safe_inst}%' THEN TARGET_NAME ELSE SOURCE_NAME END AS SUBJECT_NAME
    FROM GRAPH_NETWORK.GRAPH.ALL_EDGES_ENRICHED
    WHERE (SOURCE_NAME ILIKE '%{safe_inst}%' OR TARGET_NAME ILIKE '%{safe_inst}%')
    AND EDGE_TYPE IN ('Applicant_Subject', 'Institute_Subject')
    {subject_clause}
),
org_subject_edges AS (
    SELECT
        CASE WHEN SOURCE_TYPE IN ('Applicant', 'Institutes') THEN SOURCE ELSE TARGET END AS ORG_ID,
        CASE WHEN SOURCE_TYPE IN ('Applicant', 'Institutes') THEN SOURCE_NAME ELSE TARGET_NAME END AS ORG_NAME,
        CASE WHEN SOURCE_TYPE IN ('Applicant', 'Institutes') THEN SOURCE_CATEGORY ELSE TARGET_CATEGORY END AS ORG_CATEGORY,
        CASE WHEN SOURCE IN (SELECT SUBJECT_ID FROM inst_subjects) THEN SOURCE ELSE TARGET END AS MATCHED_SUBJECT_ID,
        CASE WHEN SOURCE IN (SELECT SUBJECT_ID FROM inst_subjects) THEN SOURCE_NAME ELSE TARGET_NAME END AS MATCHED_SUBJECT_NAME,
        IP_TYPE,
        WEIGHT
    FROM GRAPH_NETWORK.GRAPH.ALL_EDGES_ENRICHED
    WHERE EDGE_TYPE IN ('Applicant_Subject', 'Institute_Subject')
    AND (SOURCE IN (SELECT SUBJECT_ID FROM inst_subjects) OR TARGET IN (SELECT SUBJECT_ID FROM inst_subjects))
    AND SOURCE_NAME NOT ILIKE '%{safe_inst}%'
    AND TARGET_NAME NOT ILIKE '%{safe_inst}%'
    {cat_filter}
),
org_patents AS (
    SELECT ORG_ID, ORG_NAME, ORG_CATEGORY,
        COUNT(DISTINCT MATCHED_SUBJECT_ID) AS PATENT_SHARED_SUBJECTS,
        SUM(WEIGHT) AS PATENT_STRENGTH
    FROM org_subject_edges
    WHERE IP_TYPE = 'Patents'
    GROUP BY ORG_ID, ORG_NAME, ORG_CATEGORY
),
org_pubs AS (
    SELECT ORG_ID, ORG_NAME, ORG_CATEGORY,
        COUNT(DISTINCT MATCHED_SUBJECT_ID) AS PUB_SHARED_SUBJECTS,
        SUM(WEIGHT) AS PUB_STRENGTH
    FROM org_subject_edges
    WHERE IP_TYPE = 'Publications'
    GROUP BY ORG_ID, ORG_NAME, ORG_CATEGORY
),
org_matches AS (
    SELECT
        COALESCE(p.ORG_ID, pub.ORG_ID) AS ORG_ID,
        COALESCE(p.ORG_NAME, pub.ORG_NAME) AS ORG_NAME,
        COALESCE(p.ORG_CATEGORY, pub.ORG_CATEGORY) AS ORG_CATEGORY,
        COALESCE(p.PATENT_SHARED_SUBJECTS, 0) AS PATENT_SHARED_SUBJECTS,
        COALESCE(p.PATENT_STRENGTH, 0) AS PATENT_STRENGTH,
        COALESCE(pub.PUB_SHARED_SUBJECTS, 0) AS PUB_SHARED_SUBJECTS,
        COALESCE(pub.PUB_STRENGTH, 0) AS PUB_STRENGTH,
        COALESCE(p.PATENT_SHARED_SUBJECTS, 0) + COALESCE(pub.PUB_SHARED_SUBJECTS, 0) AS TOTAL_SHARED_SUBJECTS,
        COALESCE(p.PATENT_STRENGTH, 0) + COALESCE(pub.PUB_STRENGTH, 0) AS TOTAL_STRENGTH
    FROM org_patents p
    FULL OUTER JOIN org_pubs pub ON p.ORG_ID = pub.ORG_ID
),
direct_collabs AS (
    SELECT
        CASE WHEN SOURCE_NAME ILIKE '%{safe_inst}%' THEN TARGET ELSE SOURCE END AS COLLAB_ID,
        SUM(WEIGHT) AS COLLAB_COUNT
    FROM GRAPH_NETWORK.GRAPH.ALL_EDGES_ENRICHED
    WHERE (SOURCE_NAME ILIKE '%{safe_inst}%' OR TARGET_NAME ILIKE '%{safe_inst}%')
    AND EDGE_TYPE IN ('Applicant_Applicant', 'Institution_Institution')
    GROUP BY 1
),
org_subjects_list AS (
    SELECT ORG_ID,
        LISTAGG(DISTINCT MATCHED_SUBJECT_NAME, ' | ') WITHIN GROUP (ORDER BY MATCHED_SUBJECT_NAME) AS SHARED_SUBJECT_NAMES
    FROM org_subject_edges
    GROUP BY ORG_ID
)
SELECT
    m.ORG_ID,
    m.ORG_NAME,
    m.ORG_CATEGORY,
    m.PATENT_SHARED_SUBJECTS,
    m.PATENT_STRENGTH,
    m.PUB_SHARED_SUBJECTS,
    m.PUB_STRENGTH,
    m.TOTAL_SHARED_SUBJECTS,
    m.TOTAL_STRENGTH,
    CASE WHEN dc.COLLAB_ID IS NULL THEN TRUE ELSE FALSE END AS IS_NEW_OPPORTUNITY,
    COALESCE(dc.COLLAB_COUNT, 0) AS COLLAB_COUNT,
    sl.SHARED_SUBJECT_NAMES
FROM org_matches m
LEFT JOIN direct_collabs dc ON m.ORG_ID = dc.COLLAB_ID
LEFT JOIN org_subjects_list sl ON m.ORG_ID = sl.ORG_ID
WHERE m.TOTAL_SHARED_SUBJECTS > 0
ORDER BY m.TOTAL_SHARED_SUBJECTS DESC, m.TOTAL_STRENGTH DESC
LIMIT {top_n}
"""
    return run_query(sql)


def run_org_collaborators_query(org_ids: list) -> pd.DataFrame:
    """Fetch all collaborators of the recommended orgs (both patents and publications)."""
    if not org_ids:
        return pd.DataFrame()
    quoted = ", ".join(f"'{sql_escape(str(oid))}'" for oid in org_ids)
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
WHERE EDGE_TYPE IN ('Applicant_Applicant', 'Institution_Institution')
AND (SOURCE IN ({quoted}) OR TARGET IN ({quoted}))
ORDER BY WEIGHT DESC
LIMIT 300
"""
    return run_query(sql)


def run_recommendation_subject_edges(
    institution: str,
    org_ids: list,
    subject_filter: str = None,
) -> pd.DataFrame:
    """Fetch shared subject edges between recommended orgs and the institution's subjects."""
    if not org_ids:
        return pd.DataFrame()
    safe_inst = sql_escape(institution)
    subject_clause = (
        f"AND (SOURCE_NAME ILIKE '%{sql_escape(subject_filter)}%' OR TARGET_NAME ILIKE '%{sql_escape(subject_filter)}%')"
        if subject_filter else ""
    )
    quoted = ", ".join(f"'{sql_escape(str(oid))}'" for oid in org_ids)
    sql = f"""
WITH inst_subjects AS (
    SELECT DISTINCT
        CASE WHEN SOURCE_NAME ILIKE '%{safe_inst}%' THEN TARGET ELSE SOURCE END AS SUBJECT_ID,
        CASE WHEN SOURCE_NAME ILIKE '%{safe_inst}%' THEN TARGET_NAME ELSE SOURCE_NAME END AS SUBJECT_NAME
    FROM GRAPH_NETWORK.GRAPH.ALL_EDGES_ENRICHED
    WHERE (SOURCE_NAME ILIKE '%{safe_inst}%' OR TARGET_NAME ILIKE '%{safe_inst}%')
    AND EDGE_TYPE IN ('Applicant_Subject', 'Institute_Subject')
    {subject_clause}
)
SELECT
    CASE WHEN SOURCE IN (SELECT SUBJECT_ID FROM inst_subjects) THEN TARGET ELSE SOURCE END AS ORG_ID,
    CASE WHEN SOURCE IN (SELECT SUBJECT_ID FROM inst_subjects) THEN TARGET_NAME ELSE SOURCE_NAME END AS ORG_NAME,
    CASE WHEN SOURCE IN (SELECT SUBJECT_ID FROM inst_subjects) THEN SOURCE ELSE TARGET END AS SUBJECT_ID,
    CASE WHEN SOURCE IN (SELECT SUBJECT_ID FROM inst_subjects) THEN SOURCE_NAME ELSE TARGET_NAME END AS SUBJECT_NAME,
    IP_TYPE,
    WEIGHT
FROM GRAPH_NETWORK.GRAPH.ALL_EDGES_ENRICHED
WHERE EDGE_TYPE IN ('Applicant_Subject', 'Institute_Subject')
AND (SOURCE IN (SELECT SUBJECT_ID FROM inst_subjects) OR TARGET IN (SELECT SUBJECT_ID FROM inst_subjects))
AND (SOURCE IN ({quoted}) OR TARGET IN ({quoted}))
"""
    return run_query(sql)


def run_titles_for_orgs(
    institution: str,
    org_ids: list,
    subject_filter: str = None,
    max_titles_per_org: int = 5,
) -> pd.DataFrame:
    """
    Fetch representative patent/publication titles for the shared research area
    between the institution and each recommended org.
    Joins ALL_EDGES_ENRICHED_FLAT to INDUSTRY_AGG.PUBLIC.PAT_PUB on UID.
    Returns up to max_titles_per_org titles per org per IP type.
    """
    if not org_ids:
        return pd.DataFrame()

    safe_inst = sql_escape(institution)
    quoted = ", ".join(f"'{sql_escape(str(oid))}'" for oid in org_ids)
    subject_clause = (
        f"AND (e.SOURCE_NAME ILIKE '%{sql_escape(subject_filter)}%' OR e.TARGET_NAME ILIKE '%{sql_escape(subject_filter)}%')"
        if subject_filter else ""
    )

    sql = f"""
WITH inst_subjects AS (
    SELECT DISTINCT
        CASE WHEN SOURCE_NAME ILIKE '%{safe_inst}%' THEN TARGET ELSE SOURCE END AS SUBJECT_ID
    FROM GRAPH_NETWORK.GRAPH.ALL_EDGES_ENRICHED
    WHERE (SOURCE_NAME ILIKE '%{safe_inst}%' OR TARGET_NAME ILIKE '%{safe_inst}%')
    AND EDGE_TYPE IN ('Applicant_Subject', 'Institute_Subject')
),
org_subject_uids AS (
    SELECT DISTINCT
        CASE WHEN e.SOURCE IN (SELECT SUBJECT_ID FROM inst_subjects) THEN e.TARGET ELSE e.SOURCE END AS ORG_ID,
        CASE WHEN e.SOURCE IN (SELECT SUBJECT_ID FROM inst_subjects) THEN e.TARGET_NAME ELSE e.SOURCE_NAME END AS ORG_NAME,
        e.IP_TYPE,
        f.UID
    FROM GRAPH_NETWORK.GRAPH.ALL_EDGES_ENRICHED_FLAT f
    JOIN GRAPH_NETWORK.GRAPH.ALL_EDGES_ENRICHED e
        ON f.SOURCE = e.SOURCE AND f.TARGET = e.TARGET AND f.EDGE_TYPE = e.EDGE_TYPE
    WHERE e.EDGE_TYPE IN ('Applicant_Subject', 'Institute_Subject')
    AND (e.SOURCE IN (SELECT SUBJECT_ID FROM inst_subjects) OR e.TARGET IN (SELECT SUBJECT_ID FROM inst_subjects))
    AND (e.SOURCE IN ({quoted}) OR e.TARGET IN ({quoted}))
    AND e.SOURCE_NAME NOT ILIKE '%{safe_inst}%'
    AND e.TARGET_NAME NOT ILIKE '%{safe_inst}%'
    {subject_clause}
),
ranked AS (
    SELECT
        o.ORG_ID,
        o.ORG_NAME,
        o.IP_TYPE,
        p.TITLE,
        ROW_NUMBER() OVER (PARTITION BY o.ORG_ID, o.IP_TYPE ORDER BY p.TITLE) AS rn
    FROM org_subject_uids o
    JOIN INDUSTRY_AGG.PUBLIC.PAT_PUB p ON o.UID = p.UID
    WHERE p.TITLE IS NOT NULL AND p.TITLE != ''
)
SELECT ORG_ID, ORG_NAME, IP_TYPE, TITLE
FROM ranked
WHERE rn <= {max_titles_per_org}
ORDER BY ORG_ID, IP_TYPE, rn
"""
    return run_query(sql)


def generate_recommendations(
    recs_df: pd.DataFrame,
    institution: str,
    subject_filter: str = None,
    titles_df: pd.DataFrame = None,
) -> str:
    """Call Claude to generate written recommendations from the query results, enriched with actual titles."""
    # Build titles lookup: {org_id: {"Patents": [...], "Publications": [...]}}
    titles_by_org = {}
    if titles_df is not None and not titles_df.empty:
        for _, row in titles_df.iterrows():
            oid = str(row["ORG_ID"])
            ip = str(row["IP_TYPE"])
            title = str(row["TITLE"]).strip().title()
            if oid not in titles_by_org:
                titles_by_org[oid] = {"Patents": [], "Publications": []}
            if ip in titles_by_org[oid]:
                titles_by_org[oid][ip].append(title)

    rows = []
    for _, row in recs_df.iterrows():
        org_id = str(row["ORG_ID"])
        tier = "🆕 New Opportunity" if row["IS_NEW_OPPORTUNITY"] else "🤝 Existing Partner"
        org_titles = titles_by_org.get(org_id, {})

        pat_titles = org_titles.get("Patents", [])
        pub_titles = org_titles.get("Publications", [])

        title_lines = ""
        if pat_titles:
            title_lines += f"  Sample patent titles: {'; '.join(pat_titles[:5])}\n"
        if pub_titles:
            title_lines += f"  Sample publication titles: {'; '.join(pub_titles[:5])}\n"

        rows.append(
            f"- {row['ORG_NAME']} ({row['ORG_CATEGORY']}) [{tier}]\n"
            f"  Patents: {int(row['PATENT_SHARED_SUBJECTS'])} shared subjects, strength {int(row['PATENT_STRENGTH'])}\n"
            f"  Publications: {int(row['PUB_SHARED_SUBJECTS'])} shared subjects, strength {int(row['PUB_STRENGTH'])}\n"
            f"  Shared subjects: {row['SHARED_SUBJECT_NAMES']}\n"
            f"{title_lines}"
        )
    data_str = "\n".join(rows)
    subject_context = f" in {subject_filter}" if subject_filter else ""

    prompt = f"""You are a research collaboration advisor at {institution}.

Based on the data below, write structured recommendations for the top {len(recs_df)} industry partners for {institution}{subject_context}.

Use the tier labels exactly as shown (🆕 New Opportunity or 🤝 Existing Partner).
Be specific and data-driven. Reference actual titles where relevant.
Write in a professional but accessible tone for senior stakeholders.

Data:
{data_str}

Format each recommendation exactly as follows (use markdown):

---
**[Rank]. [Organisation Name]** — [tier label]

**About:** 1-2 sentences on what the organisation does and their research focus, citing specific patent/publication titles as evidence.

**Overlap with {institution}:**
- Patents: [X shared subjects, strength Y] — note key overlapping topics
- Publications: [X shared subjects, strength Y] — note key overlapping topics

**Why collaborate:** 1-2 sentences on why they are a strong candidate, referencing the nature of the overlap and any strategic angle.

**Strategic note:** One sentence on what the tier label means in practice (new opportunity = untapped; existing partner = deepen).
"""

    response = client.messages.create(
        model="claude-sonnet-4-5",
        max_tokens=2000,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.content[0].text.strip()


def build_recommendation_shared_subjects_graph(
    recs_df: pd.DataFrame,
    subject_edges_df: pd.DataFrame,
    institution: str,
) -> str:
    """Graph 1: recommended orgs connected to their shared subject areas."""
    net = Network(
        height="600px", width="100%", bgcolor="#1a1a1a",
        font_color="#ffffff", directed=False, notebook=False, cdn_resources="in_line",
    )
    net.force_atlas_2based(gravity=-50, central_gravity=0.01, spring_length=150, spring_strength=0.08, damping=0.4, overlap=0)

    added_orgs = set()
    added_subjects = set()

    org_meta = {
        str(row["ORG_ID"]): {
            "name": str(row["ORG_NAME"]),
            "category": str(row["ORG_CATEGORY"]),
            "is_new": bool(row["IS_NEW_OPPORTUNITY"]),
            "total": int(row["TOTAL_SHARED_SUBJECTS"]),
            "strength": int(row["TOTAL_STRENGTH"]),
        }
        for _, row in recs_df.iterrows()
    }

    # Aggregate weights per (org, subject) pair so each subject is one node
    agg = (
        subject_edges_df
        .groupby(["ORG_ID", "ORG_NAME", "SUBJECT_ID", "SUBJECT_NAME"], as_index=False)["WEIGHT"]
        .sum()
    )

    for _, row in agg.iterrows():
        org_id = str(row["ORG_ID"])
        org_name = str(row["ORG_NAME"])
        subj_id = str(row["SUBJECT_ID"])
        subj_name = str(row["SUBJECT_NAME"])
        weight = float(row["WEIGHT"])

        meta = org_meta.get(org_id, {})

        if org_id not in added_orgs:
            color = "#ff6b6b" if meta.get("is_new") else "#9DC3E6"
            tier = "🆕 New Opportunity" if meta.get("is_new") else "🤝 Existing Partner"
            net.add_node(
                org_id,
                label=org_name,
                title=f"{org_name}\n{tier}\nCategory: {meta.get('category','')}\nShared subjects: {meta.get('total',0)}\nTotal strength: {meta.get('strength',0)}",
                color=color,
                value=meta.get("total", 1) * 3,
            )
            added_orgs.add(org_id)

        if subj_id not in added_subjects:
            net.add_node(
                subj_id,
                label=subj_name,
                title=f"Subject: {subj_name}",
                color="#F4B183",
                value=2,
            )
            added_subjects.add(subj_id)

        net.add_edge(org_id, subj_id, value=weight, title=f"Strength: {weight:.0f}")

    return net.generate_html(notebook=False)


def build_recommendation_network_graph(
    recs_df: pd.DataFrame,
    collaborators_df: pd.DataFrame,
) -> str:
    """Graph 2: recommended orgs and their existing collaborator network."""
    net = Network(
        height="600px", width="100%", bgcolor="#1a1a1a",
        font_color="#ffffff", directed=False, notebook=False, cdn_resources="in_line",
    )
    net.force_atlas_2based(gravity=-50, central_gravity=0.01, spring_length=150, spring_strength=0.08, damping=0.4, overlap=0)

    org_meta = {
        str(row["ORG_ID"]): {
            "name": str(row["ORG_NAME"]),
            "is_new": bool(row["IS_NEW_OPPORTUNITY"]),
        }
        for _, row in recs_df.iterrows()
    }
    rec_ids = set(org_meta.keys())
    added_nodes = set()

    # Add recommended org nodes first
    for org_id, meta in org_meta.items():
        color = "#ff6b6b" if meta["is_new"] else "#9DC3E6"
        tier = "🆕 New Opportunity" if meta["is_new"] else "🤝 Existing Partner"
        net.add_node(
            org_id,
            label=meta["name"],
            title=f"{meta['name']}\n{tier}",
            color=color,
            value=15,
        )
        added_nodes.add(org_id)

    # Add collaborator edges
    for _, row in collaborators_df.iterrows():
        src = str(row["SOURCE"])
        tgt = str(row["TARGET"])
        src_name = str(row["SOURCE_NAME"])
        tgt_name = str(row["TARGET_NAME"])
        src_cat = str(row["SOURCE_CATEGORY"])
        tgt_cat = str(row["TARGET_CATEGORY"])
        weight = float(row["WEIGHT"])
        ip_type = str(row["IP_TYPE"])

        if src not in added_nodes:
            net.add_node(
                src, label=src_name,
                title=f"{src_name}\nCategory: {src_cat}",
                color="#33cccc" if src not in rec_ids else "#ff6b6b",
                value=5,
            )
            added_nodes.add(src)

        if tgt not in added_nodes:
            net.add_node(
                tgt, label=tgt_name,
                title=f"{tgt_name}\nCategory: {tgt_cat}",
                color="#33cccc" if tgt not in rec_ids else "#9DC3E6",
                value=5,
            )
            added_nodes.add(tgt)

        net.add_edge(src, tgt, value=weight, title=f"Strength: {weight:.0f}\n{ip_type}")

    return net.generate_html(notebook=False)


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
        bgcolor="#1a1a1a",
        font_color="#ffffff",
        directed=False,
        notebook=False,
        cdn_resources="in_line",
    )
    net.force_atlas_2based(
        gravity=-50,
        central_gravity=0.01,
        spring_length=100,
        spring_strength=0.08,
        damping=0.4,
        overlap=0,
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
                title=f"{org_name}\nCategory: {meta.get('category', '—')}\nShared subjects: {shared}\nTotal strength: {meta.get('weight', 0):.0f}",
                color="#9DC3E6",
                value=shared,
            )
            added_orgs.add(org_id)

        if subj_id not in added_subjects:
            net.add_node(
                subj_id,
                label=subj_name,
                title=f"{subj_name}\nResearch subject area",
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

    return net.generate_html(notebook=False)


def inject_layout_controls(html: str) -> str:
    """Replace physics panel with clean preset buttons that don't interfere with scrolling."""
    controls = """
    <div style="padding:8px 12px; display:flex; gap:8px; flex-wrap:wrap; background:#1a1a1a; border-bottom:1px solid #333;">
        <button onclick="setForceAtlas()" style="background:#333;color:#fff;border:1px solid #555;padding:5px 14px;border-radius:4px;cursor:pointer;font-size:12px;">🔄 ForceAtlas2</button>
        <button onclick="setBarnesHut()" style="background:#333;color:#fff;border:1px solid #555;padding:5px 14px;border-radius:4px;cursor:pointer;font-size:12px;">🌐 Barnes-Hut</button>
        <button onclick="freezeGraph()" style="background:#333;color:#fff;border:1px solid #555;padding:5px 14px;border-radius:4px;cursor:pointer;font-size:12px;">❄️ Freeze</button>
        <button onclick="network.fit()" style="background:#333;color:#fff;border:1px solid #555;padding:5px 14px;border-radius:4px;cursor:pointer;font-size:12px;">⊡ Fit to Screen</button>
    </div>
    <script>
    function setForceAtlas() {
        network.setOptions({physics:{enabled:true,solver:'forceAtlas2Based',forceAtlas2Based:{gravity:-50,centralGravity:0.01,springLength:100,springConstant:0.08,damping:0.4,overlap:0}}});
    }
    function setBarnesHut() {
        network.setOptions({physics:{enabled:true,solver:'barnesHut',barnesHut:{gravitationalConstant:-30000,centralGravity:0.3,springLength:180,springConstant:0.02,damping:0.8,avoidOverlap:0.5}}});
    }
    function freezeGraph() {
        network.setOptions({physics:{enabled:false}});
    }
    </script>
    """
    return html.replace("<div id=\"mynetwork\"", controls + "<div id=\"mynetwork\"")


def inject_png_download(html: str, filename: str = "graph.png") -> str:
    """Inject a PNG download button into PyVis HTML using canvas capture."""
    button_js = f"""
    <div style="text-align:right; padding: 6px 12px;">
      <button onclick="downloadPNG()" style="
        background:#0068C9; color:white; border:none;
        padding:7px 16px; border-radius:6px; cursor:pointer; font-size:13px;">
        ⬇️ Download PNG
      </button>
    </div>
    <script>
    function downloadPNG() {{
      var canvas = document.querySelector('canvas');
      if (!canvas) {{ alert('Graph not ready yet — please wait a moment and try again.'); return; }}
      var link = document.createElement('a');
      link.download = '{filename}';
      link.href = canvas.toDataURL('image/png');
      link.click();
    }}
    </script>
    """
    # Inject just before closing </body>
    return html.replace("</body>", button_js + "</body>")


def build_pyvis_graph(df: pd.DataFrame, highlight_term: str = None) -> str:
    net = Network(
        height="750px",
        width="100%",
        bgcolor="#1a1a1a",
        font_color="#ffffff",
        directed=False,
        notebook=False,
        cdn_resources="in_line",
    )

    net.force_atlas_2based(
        gravity=-50,
        central_gravity=0.01,
        spring_length=100,
        spring_strength=0.08,
        damping=0.4,
        overlap=0,
    )

    added_nodes = set()
    term = highlight_term.upper() if highlight_term else None

    for _, row in df.iterrows():
        source_id = row["SOURCE"]
        target_id = row["TARGET"]
        source_name = row["SOURCE_NAME"]
        target_name = row["TARGET_NAME"]
        source_type = row["SOURCE_TYPE"]
        target_type = row["TARGET_TYPE"]
        weight = row["WEIGHT"]
        edge_type = row["EDGE_TYPE"]

        # Determine highlight state
        source_match = term and term in str(source_name).upper()
        target_match = term and term in str(target_name).upper()
        edge_highlighted = source_match or target_match

        if source_id not in added_nodes:
            base_color = get_node_color(source_type, source_name, row["SOURCE_NUS_AFFILIATED"])
            if term:
                color = base_color if source_match else "#E8E8E8"
                border = "#FF6600" if source_match else "#E8E8E8"
                node_size = 20 if source_match else 5
            else:
                color = base_color
                border = base_color
                node_size = 10

            net.add_node(
                source_id,
                label=source_name if (not term or source_match) else "",
                title=f"{source_name}\nType: {source_type}\nCategory: {row['SOURCE_CATEGORY']}",
                color={"background": color, "border": border},
                value=node_size,
            )
            added_nodes.add(source_id)

        if target_id not in added_nodes:
            base_color = get_node_color(target_type, target_name, row["TARGET_NUS_AFFILIATED"])
            if term:
                color = base_color if target_match else "#E8E8E8"
                border = "#FF6600" if target_match else "#E8E8E8"
                node_size = 20 if target_match else 5
            else:
                color = base_color
                border = base_color
                node_size = 10

            net.add_node(
                target_id,
                label=target_name if (not term or target_match) else "",
                title=f"{target_name}\nType: {target_type}\nCategory: {row['TARGET_CATEGORY']}",
                color={"background": color, "border": border},
                value=node_size,
            )
            added_nodes.add(target_id)

        net.add_edge(
            source_id,
            target_id,
            value=float(weight) if edge_highlighted or not term else 0.1,
            color="#FF6600" if edge_highlighted else "#DDDDDD",
            title=f"Connection type: {edge_type}\nStrength: {weight}",
        )

    html = net.generate_html(notebook=False)
    return html


# -----------------------------
# LLM chat helper
# -----------------------------
SYSTEM_PROMPT = """You are a research collaboration discovery assistant helping users explore a graph network of patents and academic publications.

Based on the user's natural language request, first decide if they are:
A) Asking to filter/explore the graph network
B) Asking a general question (about a company, institution, research topic, or concept)

Return a JSON object with the following fields:

{
  "response_type": "<string>",          // REQUIRED. One of:
                                        // "graph_query"    — user wants to filter or explore the graph
                                        // "general_answer" — user wants information about a company, institution, topic, or concept
                                        // "recommendation" — user wants recommended partners/collaborators for an institution
                                        //   Use this when user says things like: "recommend", "who should NUS partner with",
                                        //   "find industry partners", "suggest collaborators", "best partners for",
                                        //   "highly likely to be partners", "who to approach"
  "answer": "<string or null>",         // ONLY for general_answer: a helpful, concise answer (2-4 paragraphs).
                                        // Include: what the org does, their main research/business areas,
                                        // why they might be a good collaboration partner, and any notable facts.
                                        // For graph_query, set this to null.
  "query_mode": "<string>",             // for graph_query only. One of:
                                        // "standard"          — normal graph filter mode (default)
                                        // "similar_no_collab" — find orgs with similar research interests that have NOT yet collaborated with the searched institution.
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
  "top_n_results": <integer or null>,   // for "similar_no_collab" and "recommendation" mode: how many results to return.
                                        // Default is 3 for recommendations, 20 for similar_no_collab if not specified.
  "subject_filter": "<string or null>", // scope the search to a specific QS subject area.
                                        // MUST exactly match one of: [AVAILABLE_SUBJECTS]
                                        // Map natural language to the correct QS subject name e.g.:
                                        //   "AI", "artificial intelligence", "machine learning" → "COMPUTER SCIENCE & INFORMATION SYSTEMS" or "DATA SCIENCE"
                                        //   "biomedical", "life sciences" → "BIOLOGICAL SCIENCES" or "MEDICINE"
                                        //   "engineering" → pick the most specific match e.g. "ENGINEERING - ELECTRICAL & ELECTRONIC"
                                        //   If ambiguous, pick the closest match or leave null for all subjects.
  "explanation": "<friendly 1-2 sentence explanation of what the results will show, or null for general_answer>"
}

Rules:
- response_type is ALWAYS required.
- For "general_answer": fill "answer" with a helpful response, set all filter fields to null.
- For "graph_query": fill filter fields as needed, set "answer" to null.
- query_mode defaults to "standard" for graph_query unless user clearly wants "similar_no_collab".
- ip_type must exactly match one of: [AVAILABLE_IP_TYPES]
- edge_type must exactly match one of: [AVAILABLE_EDGE_TYPES]
- category must exactly match one of: [AVAILABLE_CATEGORIES]
- subject_filter must exactly match one of: [AVAILABLE_SUBJECTS] — never invent a subject name
- If the user mentions "NUS" or "National University of Singapore", set search_term to "NATIONAL UNIVERSITY OF SINGAPORE".
- If the user says "reset", "clear", "start over", or "show everything", set response_type="graph_query", query_mode="standard", ip_type=null, edge_type=null, search_term=null, min_weight=1, max_edges=200, category=null, top_n_nodes=null, top_n_results=null.
- For "similar_no_collab" mode, search_term is required.
- Always return ONLY valid JSON. No markdown fences, no extra text."""


def extract_filters_from_llm(
    user_message: str,
    chat_history: list,
    available_ip_types: list,
    available_edge_types: list,
    available_categories: list,
    available_subjects: list,
) -> dict: 

    system = SYSTEM_PROMPT.replace(
        "[AVAILABLE_IP_TYPES]", ", ".join(available_ip_types)
    ).replace(
        "[AVAILABLE_EDGE_TYPES]", ", ".join(available_edge_types)
    ).replace(
        "[AVAILABLE_CATEGORIES]", ", ".join(available_categories)
    ).replace(
        "[AVAILABLE_SUBJECTS]", ", ".join(available_subjects)
    )

    messages = []
    for turn in chat_history:
        messages.append({"role": turn["role"], "content": turn["content"]})
    messages.append({"role": "user", "content": user_message})

    response = client.messages.create(
        model="claude-sonnet-4-5",
        max_tokens=1024,
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

subjects_df = run_query("""
    SELECT DISTINCT NODE_NAME AS SUBJECT_NAME
    FROM GRAPH_NETWORK.GRAPH.ALL_NODES
    WHERE NODE_TYPE = 'Subject'
    AND NODE_NAME != 'NO SUBJECT DETECTED'
    ORDER BY SUBJECT_NAME
""")
subjects_raw = subjects_df["SUBJECT_NAME"].tolist()

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
        "max_edges": 200,
        "category": "All",
        "top_n_nodes": None,
        "query_mode": "standard",
        "top_n_results": None,
        "subject_filter": None,
        "has_queried": False,
        "recommendation_data": None,
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
        on_change=lambda: (st.session_state.filter_state.update({"ip_type": st.session_state.sb_ip_type, "top_n_nodes": None, "query_mode": "standard", "has_queried": True, "recommendation_data": None}), run_query.clear()),
    )

    subject_options = ["All"] + subjects_raw
    current_subject = st.session_state.filter_state.get("subject_filter") or "All"
    st.selectbox(
        "Research subject",
        subject_options,
        index=subject_options.index(current_subject) if current_subject in subject_options else 0,
        key="sb_subject_filter",
        on_change=lambda: (st.session_state.filter_state.update({"subject_filter": st.session_state.sb_subject_filter if st.session_state.sb_subject_filter != "All" else None, "top_n_nodes": None, "query_mode": "standard", "has_queried": True}), run_query.clear()),
    )

    st.selectbox(
        "Organisation category",
        categories,
        index=categories.index(st.session_state.filter_state["category"]) if st.session_state.filter_state["category"] in categories else 0,
        key="sb_category",
        on_change=lambda: (st.session_state.filter_state.update({"category": st.session_state.sb_category, "top_n_nodes": None, "query_mode": "standard", "has_queried": True}), run_query.clear()),
    )

    st.text_input(
        "Search for an institution or organisation",
        value=st.session_state.filter_state["search_term"],
        placeholder="e.g. NATIONAL UNIVERSITY OF SINGAPORE",
        key="sb_search",
        on_change=lambda: (st.session_state.filter_state.update({"search_term": st.session_state.sb_search, "top_n_nodes": None, "query_mode": "standard", "has_queried": True}), run_query.clear()),
    )

    st.number_input(
        "Minimum collaboration strength",
        min_value=1,
        value=st.session_state.filter_state["min_weight"],
        key="sb_min_weight",
        on_change=lambda: (st.session_state.filter_state.update({"min_weight": st.session_state.sb_min_weight, "has_queried": True}), run_query.clear()),
    )

    st.slider(
        "Maximum connections to load",
        min_value=20,
        max_value=1000,
        value=st.session_state.filter_state["max_edges"],
        step=20,
        key="sb_max_edges",
        on_change=lambda: (st.session_state.filter_state.update({"max_edges": st.session_state.sb_max_edges, "top_n_nodes": None, "has_queried": True}), run_query.clear()),
    )

    st.divider()
    st.caption("Use the AI assistant below the graph to explore the network.")

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
has_queried = fs.get("has_queried", False)


# -----------------------------
# Two column layout
# -----------------------------
# -----------------------------
# Two column layout
# -----------------------------
chat_col, graph_col = st.columns([2, 3], gap="large")

with graph_col:
    st.subheader("🗺️ Collaboration Network")
    st.markdown(
        "<span style='font-size:14px'>"
        "<span style='color:#ff9933'>■</span> NUS-affiliated &nbsp;|&nbsp; "
        "<span style='color:#ff6b6b'>■</span> New opportunity &nbsp;|&nbsp; "
        "<span style='color:#9DC3E6'>■</span> Patent applicant / existing partner &nbsp;|&nbsp; "
        "<span style='color:#33cccc'>■</span> Publication institute"
        "<br>"
        "<span style='color:#FFD700'>■</span> Patent subject &nbsp;|&nbsp; "
        "<span style='color:#F4B183'>■</span> Publication subject &nbsp;|&nbsp; "
        "<span style='color:#D9D9D9'>■</span> Other"
        "</span>",
        unsafe_allow_html=True,
    )

    if not has_queried:
        st.info(
            "👋 **Welcome to the Research Collaboration Explorer!**\n\n"
            "Use the AI assistant on the left to get started. Here are some things you can ask:\n\n"
            "- _'Recommend industry partners for NUS in computer science'_\n"
            "- _'Show NUS patent partners in engineering'_\n"
            "- _'Show corporations with similar research interests to NUS that haven\\'t collaborated before'_\n\n"
            "Or use the manual filters in the sidebar."
        )

    elif query_mode == "recommendation":
        # --- Recommendation mode ---
        rec_data = fs.get("recommendation_data")
        if not rec_data:
            st.info("Ask the AI assistant for recommendations — try: _'Recommend industry partners for NUS in AI'_")
        else:
            recs_df = pd.DataFrame(rec_data["recs_df"])
            institution = rec_data["institution"]
            subject_filter = rec_data.get("subject_filter")
            org_ids = recs_df["ORG_ID"].tolist()

            subject_context = f" in {subject_filter}" if subject_filter else ""
            st.markdown(f"Showing supporting evidence for recommended industry partners for **{institution.title()}**{subject_context}.")

            tab1, tab2 = st.tabs(["📚 Shared Research Subjects", "🌐 Partner Industry Network"])

            with tab1:
                st.markdown(
                    "<span style='font-size:15px'>"
                    "<span style='color:#ff6b6b'>■</span> New opportunity &nbsp;|&nbsp; "
                    "<span style='color:#9DC3E6'>■</span> Existing partner &nbsp;|&nbsp; "
                    "<span style='color:#F4B183'>■</span> Shared research subjects"
                    "</span>",
                    unsafe_allow_html=True,
                )
                with st.spinner("Loading shared subjects graph…"):
                    subj_edges_df = run_recommendation_subject_edges(institution, org_ids, subject_filter)
                if subj_edges_df.empty:
                    st.info("No subject edges found.")
                else:
                    html1 = build_recommendation_shared_subjects_graph(recs_df, subj_edges_df, institution)
                    html1 = inject_layout_controls(inject_png_download(html1, "shared_subjects.png"))
                    components.html(html1, height=620, scrolling=True)
                    col1, col2 = st.columns([1, 5])
                    with col1:
                        st.download_button(
                            label="⬇️ Download CSV",
                            data=subj_edges_df[["ORG_NAME", "SUBJECT_NAME", "IP_TYPE", "WEIGHT"]].to_csv(index=False).encode("utf-8"),
                            file_name="shared_subjects.csv",
                            mime="text/csv",
                            key="dl_shared_subjects",
                        )

            with tab2:
                st.markdown(
                    "<span style='font-size:15px'>"
                    "<span style='color:#ff6b6b'>■</span> New opportunity &nbsp;|&nbsp; "
                    "<span style='color:#9DC3E6'>■</span> Existing partner &nbsp;|&nbsp; "
                    "<span style='color:#33cccc'>■</span> Their existing collaborators"
                    "</span>",
                    unsafe_allow_html=True,
                )
                with st.spinner("Loading industry network graph…"):
                    collab_df = run_org_collaborators_query(org_ids)
                if collab_df.empty:
                    st.info("No collaborator data found.")
                else:
                    html2 = build_recommendation_network_graph(recs_df, collab_df)
                    html2 = inject_layout_controls(inject_png_download(html2, "industry_network.png"))
                    components.html(html2, height=620, scrolling=True)
                    col1, col2 = st.columns([1, 5])
                    with col1:
                        st.download_button(
                            label="⬇️ Download CSV",
                            data=collab_df[["SOURCE_NAME", "SOURCE_CATEGORY", "TARGET_NAME", "TARGET_CATEGORY", "EDGE_TYPE", "IP_TYPE", "WEIGHT"]].to_csv(index=False).encode("utf-8"),
                            file_name="industry_network.csv",
                            mime="text/csv",
                            key="dl_industry_network",
                        )

            st.subheader("📋 Summary table")
            summary_df = recs_df[[
                "ORG_NAME", "ORG_CATEGORY",
                "PATENT_SHARED_SUBJECTS", "PATENT_STRENGTH",
                "PUB_SHARED_SUBJECTS", "PUB_STRENGTH",
                "TOTAL_SHARED_SUBJECTS", "TOTAL_STRENGTH",
                "IS_NEW_OPPORTUNITY", "COLLAB_COUNT",
            ]].copy()
            summary_df.columns = [
                "Organisation", "Category",
                "Patent Shared Subjects", "Patent Count",
                "Publication Shared Subjects", "Publication Count",
                "Total Shared Subjects", "Total Count",
                "New Opportunity", "Existing Collaborations",
            ]
            # Only show collaboration count for existing partners
            summary_df["Existing Collaborations"] = summary_df.apply(
                lambda r: int(r["Existing Collaborations"]) if not r["New Opportunity"] else "—",
                axis=1,
            )
            summary_df.index = range(1, len(summary_df) + 1)
            st.dataframe(summary_df, use_container_width=True)

            col1, col2 = st.columns([1, 5])
            with col1:
                st.download_button(
                    label="⬇️ Download CSV",
                    data=summary_df.to_csv().encode("utf-8"),
                    file_name="recommendations.csv",
                    mime="text/csv",
                )

    elif query_mode == "similar_no_collab":
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
                org_ids = similar_df["ORG_ID"].tolist()
                with st.spinner("Loading subject connections…"):
                    edges_df = run_similar_no_collab_subject_edges(
                        institution=search_term.strip(),
                        org_ids=org_ids,
                        ip_type=selected_ip_type if selected_ip_type != "All" else None,
                        subject_filter=subject_filter,
                    )

                html = build_similar_no_collab_graph(similar_df, edges_df)
                html = inject_layout_controls(inject_png_download(html, filename="potential_partners.png"))
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

                col1, col2 = st.columns([1, 5])
                with col1:
                    st.download_button(
                        label="⬇️ Download CSV",
                        data=display_df.to_csv().encode("utf-8"),
                        file_name="potential_partners.csv",
                        mime="text/csv",
                    )

    else:
        # --- Standard graph mode ---
        st.markdown(
            "Nodes represent institutions, corporations, or research subjects. "
            "Thicker lines indicate stronger collaboration. **Hover over any node or edge** for details."
        )

        where_clauses = [f"WEIGHT >= {min_weight}"]

        if selected_ip_type != "All":
            where_clauses.append(f"IP_TYPE = '{sql_escape(selected_ip_type)}'")

        if subject_filter:
            safe_subj = sql_escape(subject_filter)
            where_clauses.append(f"(SOURCE_NAME ILIKE '%{safe_subj}%' OR TARGET_NAME ILIKE '%{safe_subj}%')")

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
                "Try asking the AI assistant."
            )
        else:
            highlight_term = st.text_input(
                "🔍 Highlight a node",
                placeholder="Type a node name to highlight it in the graph…",
                key="highlight_input",
            )

            html = build_pyvis_graph(df, highlight_term=highlight_term.strip() if highlight_term else None)
            html = inject_layout_controls(inject_png_download(html, filename="collaboration_network.png"))
            components.html(html, height=780, scrolling=True)

            n_nodes = pd.concat([df["SOURCE"], df["TARGET"]]).nunique()
            n_edges = len(df)
            st.caption(
                f"Showing {n_nodes:,} institutions across {n_edges:,} connections. "
                "Drag nodes to rearrange, scroll to zoom."
            )

        with st.expander("📊 View connection data"):
            st.dataframe(df, use_container_width=True)
            col1, col2 = st.columns([1, 5])
            with col1:
                st.download_button(
                    label="⬇️ Download CSV",
                    data=df.to_csv(index=False).encode("utf-8"),
                    file_name="collaboration_network.csv",
                    mime="text/csv",
                )

        with st.expander("🔍 View SQL query"):
            st.code(sql, language="sql")

with chat_col:
    st.subheader("💬 AI Research Assistant")
    st.caption(
        "Ask in plain language to explore the network. Try: "
        "_'Recommend industry partners for NUS in computer science'_ or "
        "_'Show NUS patent partners in engineering'_"
    )

    chat_container = st.container(height=400)
    with chat_container:
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
                placeholder="e.g. Recommend industry partners for NUS in AI",
                label_visibility="collapsed",
            )
        with col2:
            submitted = st.form_submit_button("Send ➤", use_container_width=True)

if submitted and user_input.strip():
    st.session_state.chat_display.append({"role": "user", "content": user_input})

    with st.spinner("Thinking…"):
        try:
            parsed = extract_filters_from_llm(
                user_message=user_input,
                chat_history=st.session_state.chat_history,
                available_ip_types=ip_types_raw,
                available_edge_types=edge_types_raw,
                available_categories=categories_raw,
                available_subjects=subjects_raw,
            )

            response_type = parsed.get("response_type", "graph_query")

            if response_type == "general_answer":
                # --- General knowledge answer — don't touch filters ---
                answer = parsed.get("answer", "I'm not sure about that. Try rephrasing your question.")
                st.session_state.chat_history.append({"role": "user", "content": user_input})
                st.session_state.chat_history.append({"role": "assistant", "content": answer})
                st.session_state.chat_display.append({
                    "role": "assistant",
                    "content": answer,
                    "filters": None,
                })

            elif response_type == "recommendation":
                # --- Recommendation mode — fetch data, generate written recs ---
                search_term_val = parsed.get("search_term", "NATIONAL UNIVERSITY OF SINGAPORE") or "NATIONAL UNIVERSITY OF SINGAPORE"
                subject_val = parsed.get("subject_filter")
                # Only use category if explicitly specified by user, otherwise show all
                category_val = parsed.get("category") or None
                top_n_val = int(parsed.get("top_n_results") or 3)

                with st.spinner("Fetching partner data from Snowflake…"):
                    recs_df = run_recommendation_query(
                        institution=search_term_val,
                        subject_filter=subject_val,
                        category=category_val,
                        top_n=top_n_val,
                    )

                if recs_df.empty:
                    answer = "No matching organisations found. Try broadening the subject area or category."
                else:
                    org_ids = recs_df["ORG_ID"].tolist()
                    with st.spinner("Fetching relevant titles…"):
                        titles_df = run_titles_for_orgs(
                            institution=search_term_val,
                            org_ids=org_ids,
                            subject_filter=subject_val,
                        )

                    with st.spinner("Generating recommendations…"):
                        rec_text = generate_recommendations(
                            recs_df,
                            search_term_val,
                            subject_val,
                            titles_df=titles_df,
                        )

                    # Store recommendation data for graph rendering
                    st.session_state.filter_state["recommendation_data"] = {
                        "recs_df": recs_df.to_dict("records"),
                        "institution": search_term_val,
                        "subject_filter": subject_val,
                        "category": category_val,
                    }
                    st.session_state.filter_state["has_queried"] = True
                    st.session_state.filter_state["query_mode"] = "recommendation"

                    answer = rec_text

                st.session_state.chat_history.append({"role": "user", "content": user_input})
                st.session_state.chat_history.append({"role": "assistant", "content": answer})
                st.session_state.chat_display.append({
                    "role": "assistant",
                    "content": answer,
                    "filters": None,
                })

            else:
                # --- Graph query — update filters as normal ---
                explanation = parsed.pop("explanation", "Filters updated.")
                parsed.pop("answer", None)
                parsed.pop("response_type", None)

                new_fs = apply_llm_filters(
                    parsed,
                    st.session_state.filter_state,
                    ip_types_raw,
                    edge_types_raw,
                    categories_raw,
                )
                new_fs["has_queried"] = True
                st.session_state.filter_state = new_fs

                changed = {k: v for k, v in parsed.items() if v is not None}

                st.session_state.chat_history.append({"role": "user", "content": user_input})
                st.session_state.chat_history.append({"role": "assistant", "content": explanation})
                st.session_state.chat_display.append({
                    "role": "assistant",
                    "content": explanation,
                    "filters": changed if changed else None,
                })

                run_query.clear()

        except Exception as e:
            error_msg = f"Sorry, I couldn't understand that request. Please try rephrasing. (Error: {e})"
            st.session_state.chat_display.append({"role": "assistant", "content": error_msg})

    st.rerun()
