# app.py

import streamlit as st
from src.memory.candidate_store import load_candidates
from src.dashboard.feed import build_emerging_feed
from src.dashboard.gemini_explainer import generate_human_cluster_title, explain_cluster_with_gemini
from src.dashboard.graph import build_cluster_graph
from src.dashboard.search import search_clusters_hybrid
from src.embeddings.embedding_model import EmbeddingModel
from streamlit.components.v1 import html

st.set_page_config(page_title="Weak Signal Engine", layout="wide")

# Initialize embedding model (reused across search operations)
@st.cache_resource
def get_embedding_model():
    return EmbeddingModel()

embedding_model = get_embedding_model()

st.title("📡 Weak Signal Engine — Emerging Trends")

# === INTENT-BASED SEARCH ===
st.subheader("🔍 Search Emerging Signals")

col1, col2 = st.columns([4, 1])

with col1:
    search_query = st.text_input(
        "What trend are you looking for?",
        placeholder="e.g., AWS Trainium3, compute, AI power, data centers",
        key="search_input"
    )

with col2:
    st.write("")  # Spacing
    st.write("")  # Spacing
    search_button = st.button("🔎 Search", use_container_width=True)

# Perform search
if search_button and search_query:
    candidates = load_candidates()
    
    if candidates:
        with st.spinner("Searching across all clusters..."):
            results = search_clusters_hybrid(
                query=search_query,
                clusters=candidates,
                embedding_model=embedding_model,
                min_final_score=0.35
            )
        
        if results:
            st.success(f"Found {len(results)} matching clusters")
            
            for idx, result in enumerate(results):
                with st.container():
                    # Generate title
                    signal_texts = [s['text'] for s in result["signals"]]
                    cluster_id = result["cluster_id"]
                    title = generate_human_cluster_title(signal_texts, cluster_id=cluster_id)
                    
                    # Cluster type badge
                    badge_color = "🔥" if result["cluster_type"] == "Active" else "🌱"
                    
                    st.markdown(f"### {badge_color} {title}")
                    
                    # Display scores in 5 columns
                    col_a, col_b, col_c, col_d, col_e = st.columns(5)
                    with col_a:
                        st.metric("Type", result["cluster_type"])
                    with col_b:
                        st.metric("Final Score", f"{result['final_score']:.2%}")
                    with col_c:
                        st.metric("Semantic", f"{result['semantic_score']:.2%}")
                    with col_d:
                        st.metric("Lexical", f"{result['lexical_score']:.2%}")
                    with col_e:
                        st.metric("Signals", result["signal_count"])
                    
                    # Expandable signal list
                    with st.expander(f"View {result['signal_count']} signals"):
                        for sig in result["signals"]:
                            st.markdown(f"- {sig['text']}")
                    
                    st.divider()
        else:
            st.info(f"No clusters found matching '{search_query}'. Try broader terms or lower the similarity threshold.")
    else:
        st.warning("No clusters available. Run main.py first.")

st.divider()

# === EXISTING CONTENT ===
candidates = load_candidates()

if not candidates:
    st.warning("No candidate clusters found. Run main.py first.")
    st.stop()

ACTIVE_MIN = 3

active_clusters = [
    c for c in candidates if c["signal_count"] >= ACTIVE_MIN
]

candidate_clusters = [
    c for c in candidates if c["signal_count"] < ACTIVE_MIN
]

st.subheader("🔥 Active Emerging Clusters")

if active_clusters:
    feed = build_emerging_feed(active_clusters)

    for item in feed:
        with st.container():
            # Generate meaningful cluster label using Gemini
            cluster_data = next(c for c in active_clusters if c["cluster_id"] == item["cluster_id"])
            signal_texts = [s['text'] for s in cluster_data["signals"]]
            
            # Use cluster_id as stable cache key
            cluster_id = cluster_data["cluster_id"]
            label = generate_human_cluster_title(signal_texts, cluster_id=cluster_id)
            st.markdown(f"### 📈 {label}")

            st.write(
                f"**Size:** {item['signal_count']}  |  "
                f"**Emergence:** {item['emergence_level']}  |  "
                f"**Growth:** {item['growth_ratio']:.2f}"
            )

            # Expandable signals
            with st.expander(f"Show {item['signal_count']} signals"):
                for s in cluster_data["signals"]:
                    st.markdown(f"- {s['text']}")

            # Cluster Explainer Chat
            st.markdown("#### 💬 Ask about this cluster")
            
            # Create unique key for each cluster's chat
            chat_key = f"chat_{cluster_data['cluster_id']}"
            question_key = f"question_{cluster_data['cluster_id']}"
            
            # Common questions as buttons
            col1, col2, col3 = st.columns(3)
            with col1:
                if st.button("Why is this emerging?", key=f"why_{cluster_data['cluster_id']}"):
                    st.session_state[question_key] = "Why is this emerging?"
            with col2:
                if st.button("Who should care?", key=f"who_{cluster_data['cluster_id']}"):
                    st.session_state[question_key] = "Who should care about this?"
            with col3:
                if st.button("What could happen next?", key=f"next_{cluster_data['cluster_id']}"):
                    st.session_state[question_key] = "What could happen next?"
            
            # Custom question input
            user_question = st.text_input(
                "Or ask your own question:",
                key=question_key,
                placeholder="e.g., What are the risks?"
            )
            
            # Generate response
            if user_question:
                with st.spinner("Generating explanation..."):
                    response = explain_cluster_with_gemini(signal_texts, user_question)
                    st.info(response)

            st.divider()
else:
    st.info("No active clusters yet.")

st.subheader("🕸 Cluster Relationship Graph")

if active_clusters:
    # Add labels to clusters for graph
    for c in active_clusters:
        signal_texts = [s['text'] for s in c["signals"]]
        cluster_id = c["cluster_id"]
        c["label"] = generate_human_cluster_title(signal_texts, cluster_id=cluster_id)

    # Build and display graph
    build_cluster_graph(active_clusters)
    html(open("cluster_graph.html").read(), height=650)
else:
    st.info("Need active clusters to show relationships.")

st.subheader("📄 Cluster Details")

if active_clusters:
    selected_cluster = st.selectbox(
        "Select a cluster to view full signals",
        active_clusters,
        format_func=lambda c: c["label"]
    )

    st.markdown(f"**{selected_cluster['label']}** ({selected_cluster['signal_count']} signals)")

    for s in selected_cluster["signals"]:
        st.markdown(f"**• {s['text']}**")
        st.divider()
else:
    st.info("No active clusters to show details.")

st.subheader("🌱 Candidate Clusters (Incubating)")

if candidate_clusters:
    for c in candidate_clusters:
        # Generate label for candidate cluster
        signal_texts = [s['text'] for s in c["signals"]]
        cluster_id = c["cluster_id"]
        c["label"] = generate_human_cluster_title(signal_texts, cluster_id=cluster_id)
        
        with st.expander(f"{c['label']} (size={c['signal_count']})"):
            for s in c["signals"]:
                st.markdown(f"- {s['text']}")
else:
    st.info("No candidate clusters yet.")