from typing import List, Dict, Any, Optional
from sentence_transformers import SentenceTransformer
import numpy as np
import torch

def load_embedding_model(model_id: str, device: str) -> SentenceTransformer:
    """Loads a SentenceTransformer model onto a specified device."""
    print(f"Attempting to load embedding model '{model_id}' onto device '{device}'.")
    try:
        model = SentenceTransformer(model_id, device=device)
        # Verify the model's device to confirm it loaded correctly
        actual_device = model.device
        print(f"Model '{model_id}' successfully loaded on device '{actual_device}'.")
        if str(actual_device) != device:
            print(f"Warning: Model loaded on device '{actual_device}' but device '{device}' was requested.")
        return model
    except Exception as e:
        print(f"Error loading sentence embedding model '{model_id}': {e}")
        raise

def generate_embeddings_for_corpus(
        model: SentenceTransformer,
        text_corpus: List[str],
        batch_size: int = 32,
        show_progress_bar: bool = True,
        normalize_embeddings: bool = True
) -> np.ndarray:
    """Generates embeddings for a list of texts using the provided model."""
    if not text_corpus:
        print("Text corpus is empty. Returning empty NumPy array for embeddings.")
        return np.array([])

    print(f"Generating embeddings for {len(text_corpus)} documents (Batch size: {batch_size}, Normalization: {normalize_embeddings})...")
    try:
        embeddings = model.encode(
            text_corpus,
            batch_size=batch_size,
            show_progress_bar=show_progress_bar,
            normalize_embeddings=normalize_embeddings,
            convert_to_numpy=True
        )
        print(f"{len(embeddings)} embeddings generated successfully.")
        return embeddings
    except Exception as e:
        print(f"Error during embedding generation: {e}")
        raise

def embed_chunks(
        chunks_data: List[Dict[str, Any]],
        model_id: str,
        device: str,
        batch_size: int = 32,
        normalize_embeddings: bool = True,
        preloaded_model: Optional[SentenceTransformer] = None
) -> List[Dict[str, Any]]:
    """Embeds the content of chunks using a SentenceTransformer model."""
    if not chunks_data:
        print("Received an empty list of chunks. No embeddings to generate.")
        return []

    model_to_use: SentenceTransformer
    if preloaded_model:
        print(f"Using pre-loaded embedding model for {len(chunks_data)} chunks.")
        model_to_use = preloaded_model
        # Ensure the preloaded model is on the correct device
        if str(model_to_use.device) != device:
            model_to_use.to(device)
            print(f"Moved pre-loaded model to device '{model_to_use.device}'.")
    else:
        print(f"No pre-loaded model provided, loading model '{model_id}' for {len(chunks_data)} chunks.")
        model_to_use = load_embedding_model(model_id=model_id, device=device)

    # Map original chunk indices to texts that need embedding
    texts_to_embed_map = []
    for i, chunk in enumerate(chunks_data):
        content = chunk.get("content", "")
        if content.strip():
            texts_to_embed_map.append((i, content))
        else:
            # Assign empty embedding for chunks with no content
            chunk['embedding'] = np.array([])

    if not texts_to_embed_map:
        print("All chunk contents are empty or whitespace. No embeddings generated.")
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

    # Assign generated embeddings back to the original chunks
    for i, original_idx in enumerate(original_indices):
        if i < len(embeddings_array):
            chunks_data[original_idx]['embedding'] = embeddings_array[i]
        else:
            # This case should ideally not happen if logic is correct
            print(f"Error: Mismatch in embedding count for chunk index {original_idx}. Assigning empty embedding.")
            chunks_data[original_idx]['embedding'] = np.array([])

    print("Finished adding embeddings to chunk data.")
    return chunks_data


def embed_query(model: SentenceTransformer, query_text: str) -> np.ndarray:
    """Generates an embedding for a single query text."""
    return model.encode(
        query_text,
        normalize_embeddings=True,
        convert_to_numpy=True
    )


def unload_embedding_model(model: SentenceTransformer):
    """Unloads the SentenceTransformer model and clears VRAM."""
    print(f"Unloading embedding model from device '{model.device}'...")
    # Move model to CPU before deleting to ensure CUDA memory is released
    model.to('cpu')
    del model
    # Clear CUDA cache and run garbage collection
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    print("Embedding model unloaded and memory cleared.")