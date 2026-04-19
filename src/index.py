"""
Indexation vectorielle des elements extraits.
Transforme chaque slide/page en embedding et les stocke dans ChromaDB.
"""

import json
import chromadb
from chromadb.utils import embedding_functions


EXTRACTED_FILE = "extracted_data.json"
CHROMA_DIR = "chroma_db"
COLLECTION_NAME = "cours_ia"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def load_extracted_data(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def build_document_text(item):
    """Combine titre + contenu + notes pour enrichir la recherche vectorielle."""
    parts = []
    if item.get("title") and item["title"] != "(sans titre)":
        parts.append(f"Titre : {item['title']}")
    if item.get("content"):
        parts.append(item["content"])
    if item.get("notes"):
        parts.append(f"Notes : {item['notes']}")
    return "\n\n".join(parts)


def index_all():
    """Cree ou recrée la collection ChromaDB avec tous les elements."""
    print(f"Chargement de {EXTRACTED_FILE}...")
    items = load_extracted_data(EXTRACTED_FILE)
    print(f"{len(items)} elements a indexer.\n")

    client = chromadb.PersistentClient(path=CHROMA_DIR)
    embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODEL
    )

    # Supprimer la collection existante pour repartir propre
    try:
        client.delete_collection(name=COLLECTION_NAME)
        print("Ancienne collection supprimee.")
    except Exception:
        pass

    collection = client.create_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_fn,
        metadata={"hnsw:space": "cosine"},
    )

    documents, metadatas, ids = [], [], []
    for i, item in enumerate(items):
        doc_text = build_document_text(item)
        if len(doc_text.strip()) < 20:
            continue
        documents.append(doc_text)
        metadatas.append({
            "source_file": item["source_file"],
            "source_type": item["source_type"],
            "page_number": item["page_number"],
            "title": item["title"],
        })
        ids.append(f"doc_{i}")

    print(f"Indexation de {len(documents)} documents...")
    print("(Le premier run telecharge le modele d'embedding, patience)\n")

    batch_size = 50
    for start in range(0, len(documents), batch_size):
        end = min(start + batch_size, len(documents))
        collection.add(
            documents=documents[start:end],
            metadatas=metadatas[start:end],
            ids=ids[start:end],
        )
        print(f"  Batch {start // batch_size + 1} : {end}/{len(documents)} indexes")

    print(f"\nIndexation terminee.")
    print(f"Collection '{COLLECTION_NAME}' : {collection.count()} documents.")
    print(f"Base stockee dans : ./{CHROMA_DIR}/")


def test_search(query, n_results=3):
    """Teste une recherche dans la collection."""
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODEL
    )
    collection = client.get_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_fn,
    )

    results = collection.query(query_texts=[query], n_results=n_results)
    print(f"\nRecherche : '{query}'\n" + "=" * 70)
    for i, (doc, meta, dist) in enumerate(zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    )):
        print(f"\n[{i+1}] {meta['source_file']} - {meta['title']} "
              f"(page {meta['page_number']}, distance={dist:.3f})")
        print("-" * 70)
        print(f"{doc[:300].replace(chr(10), ' ')}...")


if __name__ == "__main__":
    index_all()
    print("\n" + "=" * 70)
    print("TEST DE RECHERCHE")
    print("=" * 70)
    test_search("Comment fonctionne l'algorithme k-NN ?")
