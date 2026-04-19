"""
Chatbot RAG en ligne de commande.
Combine recherche vectorielle (ChromaDB) et generation par LLM local (Ollama).
"""

from config import get_collection, retrieve_context, generate_answer, LLM_MODEL


def ask(question):
    """Prend une question, recupere le contexte, appelle le LLM et affiche la reponse."""
    print(f"\n[Question] {question}\n")

    collection = get_collection()
    contexts = retrieve_context(collection, question)

    print(f"[Retrieval] {len(contexts)} passages trouves :")
    for i, ctx in enumerate(contexts, start=1):
        print(f"  [{i}] {ctx['source']} - page {ctx['page']}")

    print(f"\n[LLM] Generation avec {LLM_MODEL}...\n")
    answer = generate_answer(question, contexts)

    print("=" * 70)
    print("REPONSE")
    print("=" * 70)
    print(answer)
    print("=" * 70)
    print("\nSources utilisees :")
    for ctx in contexts:
        print(f"  - {ctx['source']} (page {ctx['page']})")

    return answer


if __name__ == "__main__":
    print("=" * 70)
    print(f"Chatbot RAG - Assistant de cours (modele : {LLM_MODEL})")
    print("=" * 70)
    print("Pose tes questions (tape 'exit' pour quitter)\n")

    while True:
        question = input("Ta question > ").strip()
        if not question:
            continue
        if question.lower() in ("exit", "quit", "q"):
            print("Au revoir !")
            break
        try:
            ask(question)
        except Exception as e:
            print(f"\nErreur : {e}\n")
        print()