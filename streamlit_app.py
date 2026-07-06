import streamlit as st

st.set_page_config(
    page_title="NUS Research Intelligence",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="auto",
)

pg = st.navigation([
    st.Page("pages/1_Research_Collaboration_Explorer.py", title="Potential Collaborator Finder", icon="🧲"),
    st.Page("pages/2_Industry_Intelligence_Dashboard.py", title="Industry Collaboration Overview", icon="📊"),
])
pg.run()
