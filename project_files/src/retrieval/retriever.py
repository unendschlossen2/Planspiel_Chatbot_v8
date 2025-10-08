from typing import List, Dict, Any, Optional
import chromadb
from sentence_transformers import SentenceTransformer
import numpy as np

def embed_query(
        model: SentenceTransformer,
        query_text: str,
        query_prefix: str = "query: ",
        normalize_embeddings: bool = True
) -> np.ndarray:
    if not query_text.strip():
        print("Leerer Anfragetext empfangen. Gebe Zero-Embedding zurück oder verlasse mich auf Modellverhalten.")
        pass

    prefixed_query = f"{query_prefix}{query_text}"
    # print(f"Präfixierte Anfrage für Embedding: '{prefixed_query}'")

    try:
        query_embedding = model.encode(
            prefixed_query,
            normalize_embeddings=normalize_embeddings,
            convert_to_numpy=True
        )
        return query_embedding
    except Exception as e:
        print(f"Fehler beim Embedden der Anfrage '{query_text}': {e}")
        raise

def query_vector_store(
        collection: chromadb.Collection,
        query_embedding: np.ndarray,
        top_k: int,
        filter_metadata: Optional[Dict[str, str]] = None
) -> List[Dict[str, Any]]:
    if query_embedding is None or query_embedding.size == 0:
        print("Leeres Anfrage-Embedding empfangen. Gebe keine Ergebnisse zurück.")
        return []
    if top_k <= 0:
        print(f"top_k Wert {top_k} ist ungültig. Gebe keine Ergebnisse zurück.")
        return []

    print(f"Durchsuche Kollektion '{collection.name}' nach den Top {top_k} Ergebnissen.")

    try:
        results = collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=min(top_k, collection.count()),
            include=['documents', 'metadatas', 'distances'],
            where=filter_metadata
        )
    except Exception as e:
        print(f"Fehler beim Durchsuchen der ChromaDB Kollektion '{collection.name}': {e}")
        return []

    retrieved_chunks = []
    if not results or not results.get('ids') or not results['ids'][0]:
        print("Keine Ergebnisse in ChromaDB für die Anfrage gefunden.")
        return []

    ids_list = results['ids'][0]
    documents_list = results['documents'][0] if results['documents'] else [None] * len(ids_list)
    metadatas_list = results['metadatas'][0] if results['metadatas'] else [None] * len(ids_list)
    distances_list = results['distances'][0] if results['distances'] else [None] * len(ids_list)

    for i, item_id in enumerate(ids_list):
        retrieved_chunks.append({
            "id": item_id,
            "document": documents_list[i] if documents_list else None,
            "metadata": metadatas_list[i] if metadatas_list else None,
            "distance": distances_list[i] if distances_list else None,
        })

    print(f"{len(retrieved_chunks)} Chunks aus dem Vektor-Speicher abgerufen.")
    return retrieved_chunks