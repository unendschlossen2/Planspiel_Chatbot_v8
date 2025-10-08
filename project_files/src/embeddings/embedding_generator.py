from typing import List, Dict, Any, Optional
from sentence_transformers import SentenceTransformer
import numpy as np

def load_embedding_model(model_id: str, device: str) -> SentenceTransformer:
    print(f"Lade Embedding-Modell '{model_id}' auf Gerät '{device}'.")
    try:
        model = SentenceTransformer(model_id, device=device)
        print(f"Modell '{model_id}' erfolgreich geladen.")
        return model
    except Exception as e:
        print(f"Fehler beim Laden des Satz-Embedding-Modells '{model_id}': {e}")
        raise

def generate_embeddings_for_corpus(
        model: SentenceTransformer,
        text_corpus: List[str],
        batch_size: int = 32,
        show_progress_bar: bool = True,
        normalize_embeddings: bool = True
) -> np.ndarray:
    if not text_corpus:
        print("Textkorpus ist leer. Gebe leeres NumPy-Array für Embeddings zurück.")
        return np.array([])

    print(f"Generiere Embeddings für {len(text_corpus)} Dokumente (Batch-Größe: {batch_size}, Normalisierung: {normalize_embeddings})...")
    try:
        embeddings = model.encode(
            text_corpus,
            batch_size=batch_size,
            show_progress_bar=show_progress_bar,
            normalize_embeddings=normalize_embeddings,
            convert_to_numpy=True
        )
        print(f"{len(embeddings)} Embeddings erfolgreich generiert.")
        return embeddings
    except Exception as e:
        print(f"Fehler während der Embedding-Generierung: {e}")
        raise

def embed_chunks(
        chunks_data: List[Dict[str, Any]],
        model_id: str,
        device: str,
        batch_size: int = 32,
        normalize_embeddings: bool = True,
        preloaded_model: Optional[SentenceTransformer] = None
) -> List[Dict[str, Any]]:
    if not chunks_data:
        print("Leere Liste von Chunks empfangen. Keine Embeddings zu generieren.")
        return []

    model_to_use: SentenceTransformer
    if preloaded_model:
        print(f"Verwende vorab geladenes Embedding-Modell für {len(chunks_data)} Chunks.")
        model_to_use = preloaded_model
    else:
        print(f"Kein vorab geladenes Modell bereitgestellt, lade Modell '{model_id}' für {len(chunks_data)} Chunks.")
        model_to_use = load_embedding_model(model_id=model_id, device=device)

    texts_to_embed_map = []
    for i, chunk in enumerate(chunks_data):
        content = chunk.get("content", "")
        if content.strip():
            texts_to_embed_map.append((i, content))
        else:
            chunk['embedding'] = np.array([])

    if not texts_to_embed_map:
        print("Alle Chunk-Inhalte sind nach der ersten Prüfung leer oder nur Whitespace. Keine Embeddings über leere Arrays hinaus generiert.")
        return chunks_data

    original_indices = [item[0] for item in texts_to_embed_map]
    actual_texts_to_embed = [item[1] for item in texts_to_embed_map]

    embeddings_array = generate_embeddings_for_corpus(
        model=model_to_use,
        text_corpus=actual_texts_to_embed,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=normalize_embeddings
    )

    for i, original_idx in enumerate(original_indices):
        if i < len(embeddings_array):
            chunks_data[original_idx]['embedding'] = embeddings_array[i]
        else:
            # This case should ideally not happen if logic is correct
            print(f"Fehler: Nichtübereinstimmung in der Embedding-Anzahl für Chunk-Index {original_idx}. Weise leeres Embedding zu.")
            chunks_data[original_idx]['embedding'] = np.array([])

    print("Hinzufügen von Embeddings zu Chunk-Daten abgeschlossen.")
    return chunks_data