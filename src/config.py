"""
Configuration et fonctions partagées pour le pipeline RAG.
"""

import chromadb
from chromadb.utils import embedding_functions
import ollama


CHROMA_DIR = "chroma_db"
COLLECTION_NAME = "cours_ia"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "qwen2.5:7b"
N_RESULTS = 4


def get_collection():
    """Charge la collection ChromaDB déjà indexée."""
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODEL
    )
    return client.get_collection(name=COLLECTION_NAME, embedding_function=embedding_fn)


def retrieve_context(collection, question, n_results=N_RESULTS):
    """Retourne les n passages les plus pertinents pour la question."""
    results = collection.query(query_texts=[question], n_results=n_results)
    return [
        {
            "content": doc,
            "source": meta["source_file"],
            "page": meta["page_number"],
            "title": meta["title"],
        }
        for doc, meta in zip(results["documents"][0], results["metadatas"][0])
    ]


def build_prompt(question, contexts):
    """Construit le prompt système avec les extraits de cours et la question."""
    context_text = ""
    for i, ctx in enumerate(contexts, start=1):
        context_text += f"\n--- Extrait [{i}] ---\n"
        context_text += f"Source : {ctx['source']}, page {ctx['page']}\n"
        context_text += f"Contenu : {ctx['content']}\n"

    return f"""Tu es un assistant pedagogique qui repond a des questions en se basant UNIQUEMENT sur les extraits de cours fournis ci-dessous.

REGLES IMPORTANTES :
1. Reponds uniquement a partir des extraits fournis. N'invente rien.
2. Si l'information n'est pas dans les extraits, dis clairement : "Cette information ne se trouve pas dans les documents fournis."
3. Cite tes sources en fin de phrase avec le format [source: nom_fichier, page X].
4. Reponds de maniere claire et pedagogique, en francais.
5. Si plusieurs extraits donnent des informations complementaires, combine-les intelligemment.

EXTRAITS DE COURS :
{context_text}

QUESTION DE L'UTILISATEUR :
{question}

REPONSE :"""


def generate_answer(question, contexts):
    """Appelle le LLM local via Ollama et retourne la reponse."""
    response = ollama.chat(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": build_prompt(question, contexts)}],
        options={"temperature": 0.2},
    )
    return response["message"]["content"]
