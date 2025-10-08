from typing import List, Dict, Any
import chromadb
import chromadb.errors
import numpy as np
import re

def _sanitize_id_part(part: str) -> str:
    part = str(part)
    part = re.sub(r'\s+', '_', part)
    part = re.sub(r'[^a-zA-Z0-9_.-]+', '', part)
    return part[:60]

def _prepare_data_for_chroma(
        chunks_with_embeddings: List[Dict[str, Any]]
) -> tuple[List[str], List[list | np.ndarray], List[Dict[str, Any]], List[str]]:
    ids_list: List[str] = []
    embeddings_list: List[list | np.ndarray] = []
    metadatas_list: List[Dict[str, Any]] = []
    documents_list: List[str] = []
    processed_ids = set()
    valid_chunks_processed_count = 0

    for i, chunk in enumerate(chunks_with_embeddings):
        embedding_vector = chunk.get('embedding')
        content_text = chunk.get('content', "")

        is_embedding_valid = False
        if isinstance(embedding_vector, np.ndarray) and embedding_vector.size > 0:
            is_embedding_valid = True
        elif isinstance(embedding_vector, list) and len(embedding_vector) > 0:
            is_embedding_valid = True

        if not is_embedding_valid:
            print(
                f"Chunk (Index {i}) von '{chunk.get('source_filename', 'unbekannte_quelldatei')}' "
                f"mit Titel '{chunk.get('header_text', 'N/A')[:30]}...' hat kein gültiges Embedding. Überspringe."
            )
            continue

        if not content_text.strip() and not chunk.get('header_text', '').strip():
            print(
                f"Chunk (Index {i}) von '{chunk.get('source_filename', 'unbekannte_quelldatei')}' "
                f"hat keinen Inhalt und keine Überschrift. Überspringe."
            )
            continue

        sfn_part = _sanitize_id_part(chunk.get('source_filename', f'sfn_unbekannt'))
        obh_part = _sanitize_id_part(chunk.get('original_base_header', f'obh_unbekannt'))
        ht_part = _sanitize_id_part(chunk.get('header_text', f'ht_unbekannt'))

        base_id = f"{sfn_part}_{obh_part}_{ht_part}_{valid_chunks_processed_count}"
        unique_id = base_id
        id_suffix_counter = 0
        while unique_id in processed_ids:
            id_suffix_counter += 1
            unique_id = f"{base_id}_dup{id_suffix_counter}"

        processed_ids.add(unique_id)
        ids_list.append(unique_id)

        if isinstance(embedding_vector, np.ndarray):
            embeddings_list.append(embedding_vector.tolist())
        else:
            embeddings_list.append(embedding_vector)

        documents_list.append(content_text)

        metadata_item = {
            key: value for key, value in chunk.items()
            if key not in ['content', 'embedding'] and isinstance(value, (str, int, float, bool))
        }
        metadatas_list.append(metadata_item)
        valid_chunks_processed_count +=1

    if valid_chunks_processed_count == 0 and len(chunks_with_embeddings) > 0:
        print("Keine gültigen Chunks mit Embeddings zur Vorbereitung für ChromaDB gefunden.")
        return [], [], [], []

    return ids_list, embeddings_list, metadatas_list, documents_list

def create_and_populate_vector_store(
        chunks_with_embeddings: List[Dict[str, Any]],
        db_path: str,
        collection_name: str,
        force_rebuild_collection: bool = True
) -> chromadb.Collection | None:

    print(f"Initialisiere ChromaDB PersistentClient unter Pfad: {db_path}")
    try:
        client = chromadb.PersistentClient(path=db_path)
        existing_collections_objects = client.list_collections()
        existing_collection_names = [col.name for col in existing_collections_objects]
        print(f"Verfügbare Kollektionen in DB unter '{db_path}': {existing_collection_names}")
    except Exception as e:
        print(f"Fehler bei der Initialisierung des ChromaDB-Clients oder beim Auflisten der Kollektionen unter {db_path}: {e}")
        return None

    if force_rebuild_collection:
        print(f"Force_rebuild_collection ist True für Kollektion '{collection_name}'. Stelle einen frischen Build sicher.")
        if collection_name in existing_collection_names:
            try:
                print(f"Kollektion '{collection_name}' existiert. Lösche sie für den Rebuild.")
                client.delete_collection(name=collection_name)
                print(f"Kollektion '{collection_name}' erfolgreich gelöscht.")
            except Exception as e_del:
                print(f"Fehler beim Löschen der existierenden Kollektion '{collection_name}': {e_del}. Breche Rebuild ab, um Dateninkonsistenz zu vermeiden.")
                return None
        else:
            print(f"Kollektion '{collection_name}' existiert nicht. Kein Löschen erforderlich.")

        try:
            collection = client.create_collection(name=collection_name, metadata={"hnsw:space": "cosine"})
            print(f"Neue leere Kollektion '{collection_name}' erfolgreich erstellt.")

        except Exception as e_create:
            print(f"Allgemeiner Fehler während der Erstellung der Kollektion '{collection_name}' für den Rebuild: {e_create}")
            return None

        if not chunks_with_embeddings and collection is not None:
            print("Rebuild ist True, aber keine Chunks zum Füllen bereitgestellt. Kollektion wird leer sein.")
            return collection

        ids, embeddings, metadatas, documents = _prepare_data_for_chroma(chunks_with_embeddings)
        if not ids:
            print(f"Keine gültigen Daten aus Chunks vorbereitet. Kollektion '{collection_name}' wird in diesem Durchlauf nicht gefüllt.")
            return collection

        try:
            print(f"Füge {len(ids)} Elemente zur neu erstellten ChromaDB-Kollektion '{collection_name}' hinzu...")
            collection.add(ids=ids, embeddings=embeddings, metadatas=metadatas, documents=documents)
            print(f"Erfolgreich {collection.count()} Elemente zur Kollektion '{collection_name}' hinzugefügt.")
            return collection
        except Exception as e_add:
            print(f"Fehler beim Hinzufügen von Daten zur Kollektion '{collection_name}' während des Rebuilds: {e_add}")
            return collection

    else:
        print(f"Force_rebuild_collection ist False. Versuche, existierende Kollektion '{collection_name}' zu laden.")
        if collection_name in existing_collection_names:
            try:
                collection = client.get_collection(name=collection_name)
                print(f"Existierende Kollektion '{collection_name}' mit {collection.count()} Elementen erfolgreich geladen. Daten werden nicht erneut hinzugefügt.")
                return collection
            except Exception as e_get:
                print(f"Ein Fehler trat auf beim Versuch, die existierende Kollektion '{collection_name}' zu laden: {e_get}")
                return None
        else:
            print(f"Kollektion '{collection_name}' existiert nicht. "
                  f"Bitte mit force_rebuild_collection=True ausführen, um sie zuerst zu erstellen und zu füllen.")
            return None