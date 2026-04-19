"""
Interface web Streamlit pour le chatbot RAG.
Lance avec : streamlit run src/app.py
"""

import streamlit as st
from config import (
    get_collection,
    retrieve_context,
    generate_answer,
    CHROMA_DIR,
    COLLECTION_NAME,
    EMBEDDING_MODEL,
    LLM_MODEL,
    N_RESULTS,
)


@st.cache_resource
def get_collection_cached():
    """Charge la collection ChromaDB (mis en cache pour eviter de recharger a chaque interaction)."""
    return get_collection()


def build_sources_text(contexts):
    """Construit le texte des sources pour l'affichage."""
    source_text = ""
    for ctx in contexts:
        source_text += f"- **{ctx['source']}** — page {ctx['page']}\n"
    return source_text


# Interface Streamlit

st.set_page_config(page_title="Assistant de cours RAG", layout="wide")
st.title("Assistant de cours RAG")
st.caption(f"Propulse par **{LLM_MODEL}** (local) et **ChromaDB** — 100% prive, aucune donnee ne sort de ta machine.")

if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    st.header("A propos")
    st.markdown(f"""
    **Pipeline RAG** :
    1. Extraction PDF + PPT
    2. Embeddings `{EMBEDDING_MODEL}`
    3. Base vectorielle ChromaDB
    4. Retrieval top-{N_RESULTS} par similarite cosinus
    5. Generation par LLM local
    """)
    st.header("Configuration")
    st.markdown(f"- **LLM** : `{LLM_MODEL}`\n- **Embedding** : `{EMBEDDING_MODEL.split('/')[-1]}`\n- **Top-k** : {N_RESULTS}")
    if st.button("Effacer l'historique"):
        st.session_state.messages = []
        st.rerun()

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and "sources" in msg:
            with st.expander("Sources utilisees"):
                st.markdown(build_sources_text(msg["sources"]))

if question := st.chat_input("Pose ta question sur les cours..."):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Recherche des passages pertinents..."):
            collection = get_collection_cached()
            contexts = retrieve_context(collection, question)

        with st.spinner(f"Generation de la reponse avec {LLM_MODEL}..."):
            answer = generate_answer(question, contexts)

        st.markdown(answer)

        with st.expander("Sources utilisees"):
            st.markdown(build_sources_text(contexts))

        st.session_state.messages.append({
            "role": "assistant",
            "content": answer,
            "sources": contexts,
        })