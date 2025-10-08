import re
import time
from typing import Set
from helper.file_loader import *
from helper.load_gpu import *
from preprocessing.text_cleanup import *
from processing.chunking import *
from embeddings.embedding_generator import load_embedding_model, embed_chunks
from vector_store.vector_store_manager import create_and_populate_vector_store
from retrieval.retriever import embed_query, query_vector_store
from generation.llm_answer_generator import *
from retrieval.reranker import load_reranker_model, gap_based_rerank_and_filter
from generation.query_expander import expand_user_query
from generation.conversation_handler import condense_conversation, force_unload_ollama_model
from helper.settings import settings
import gc
import torch


def main():

    try:
        processing_device_obj = load_gpu()
        processing_device_str = str(processing_device_obj)
    except RuntimeError as e:
        print(f"GPU Ladefehler: {e}. Wechsle zu CPU.")
        processing_device_str = "cpu"
    print(f"Verwende Gerät: {processing_device_str} für Modell-Operationen.")

    loaded_embedding_model = None
    loaded_reranker_model = None

    if not settings.system.low_vram_mode:
        print("Standard-Modus: Lade Embedding- und Reranker-Modelle für die gesamte Sitzung.")
        try:
            loaded_embedding_model = load_embedding_model(settings.models.embedding_id, processing_device_str)
            if settings.pipeline.use_reranker:
                loaded_reranker_model = load_reranker_model(settings.models.reranker_id, device=processing_device_str)
        except Exception as e:
            print(f"KRITISCH: Fehler beim Laden der permanenten Modelle: {e}")
            return
    else:
        print("Low-VRAM-Modus ist AKTIVIERT. Modelle werden pro Anfrage geladen und entladen.")

    all_files_chunks_with_embeddings = []
    if settings.database.force_rebuild:
        print("Modus: Vollständige Datenverarbeitung und Neuaufbau der Vektor-Speicher-Kollektion AKTIVIERT.")
        input_dir = get_input_files_directory()
        file_list = load_markdown_directory(input_dir)

        if not file_list:
            print(f"Keine Markdown-Dateien in {input_dir} für den Neuaufbau der Kollektion gefunden.")
        else:
            print(f"Gefundene Dateien: {[file_item['source'] for file_item in file_list]}")
            print("-" * 30)
            for file_item in file_list:
                print(f"Verarbeite Datei: {file_item['source']} für den Neuaufbau...")
                normalized_content = normalize_markdown_whitespace(file_item["content"])
                if not normalized_content.strip():
                    print(f"  Datei '{file_item['source']}' ist leer. Überspringe.")
                    print("-" * 30); continue
                print(f"  Länge des normalisierten Inhalts: {len(normalized_content)}")
                print(f"  Starte Chunking für {file_item['source']}...")
                chunks = split_markdown_by_headers(
                    markdown_text=normalized_content,
                    source_filename=file_item["source"],
                    split_level=settings.processing.initial_split_level,
                    max_chars_per_chunk=settings.processing.max_chars_per_chunk,
                    min_chars_per_chunk=settings.processing.min_chars_per_chunk
                )
                print(f"  Chunking abgeschlossen. {len(chunks)} Chunks generiert.")
                if chunks:
                    print(f"  Starte Embedding-Generierung für {len(chunks)} Chunks...")
                    try:
                        chunks_with_embeddings_for_file = embed_chunks(
                            chunks_data=chunks,
                            model_id=settings.models.embedding_id,
                            device=processing_device_str,
                            preloaded_model=loaded_embedding_model,
                            normalize_embeddings=True
                        )
                        print(f"  Embedding-Generierung für {file_item['source']} abgeschlossen.")
                        all_files_chunks_with_embeddings.extend(chunks_with_embeddings_for_file)
                    except Exception as e:
                        print(f"  Fehler während der Embedding-Generierung für {file_item['source']}: {e}")
                print("-" * 30)
            print(f"Dateiverarbeitung abgeschlossen. Gesamtanzahl Chunks mit Embeddings: {len(all_files_chunks_with_embeddings)}")
    else:
        print(f"Modus: Datenverarbeitung wird übersprungen. Versuche, existierenden Vektor-Speicher zu laden: '{settings.database.collection_name}'.")

    if settings.database.force_rebuild and not all_files_chunks_with_embeddings:
        print("Neuaufbau angefordert, aber keine Chunks verarbeitet. Vektor-Speicher könnte leer sein, falls erstellt.")

    print(f"Stelle Vektor-Speicher-Kollektion '{settings.database.collection_name}' unter '{settings.database.persist_path}' sicher...")
    try:
        data_to_populate = all_files_chunks_with_embeddings if settings.database.force_rebuild else []
        db_collection = create_and_populate_vector_store(
            chunks_with_embeddings=data_to_populate,
            db_path=settings.database.persist_path,
            collection_name=settings.database.collection_name,
            force_rebuild_collection=settings.database.force_rebuild
        )
        if db_collection:
            print(f"Vektor-Speicher bereit. Kollektion '{db_collection.name}' enthält {db_collection.count()} Elemente.")
        else:
            print(f"Fehler beim Einrichten/Laden der Kollektion '{settings.database.collection_name}'. Fortfahren nicht möglich.")
            return
    except Exception as e:
        print(f"Vektor-Speicher-Fehler für '{settings.database.collection_name}': {e}")
        return

    if db_collection:
        print(f"\n--- TOPSIM RAG Chatbot Bereit (Modell: {settings.models.ollama_llm}) ---")
        if settings.pipeline.use_reranker and (loaded_reranker_model or settings.system.low_vram_mode):
            print(f"--- Reranking AKTIVIERT mit Modell: {settings.models.reranker_id} ---")
        else:
            print(f"--- Reranking DEAKTIVIERT ---")
        print("Geben Sie Ihre Anfrage ein oder 'quit' zum Beenden.")

        last_query = None
        last_response = ""

        while True:
            try:
                user_query_text = input("\nAnfrage: ").strip()
            except KeyboardInterrupt:
                print("\nChatbot wird aufgrund einer Tastaturunterbrechung beendet...")
                break
            if not user_query_text:
                continue
            if user_query_text.lower() in ['quit', 'exit', 'beenden']:
                print("Chatbot wird beendet.")
                break
            try:
                if settings.system.low_vram_mode:
                    print("\nLow-VRAM-Modus: Lade temporäre Modelle...")
                    try:
                        loaded_embedding_model = load_embedding_model(settings.models.embedding_id,
                                                                      processing_device_str)
                        if settings.pipeline.use_reranker:
                            loaded_reranker_model = load_reranker_model(settings.models.reranker_id,
                                                                        device=processing_device_str)
                    except Exception as e:
                        print(f"Fehler beim Laden der temporären Modelle: {e}")
                        continue

                if not loaded_embedding_model:
                    print("Fehler: Embedding-Modell ist nicht verfügbar.")
                    continue

                condensed_query = user_query_text
                if settings.pipeline.enable_conversation_memory:
                    condensed_query = condense_conversation(
                        model_name=settings.models.condenser_model_id,
                        last_query=last_query,
                        last_response=last_response,
                        new_query=user_query_text
                    )

                expanded_query_for_retrieval = condensed_query
                if settings.pipeline.enable_query_expansion:
                    expanded_query_for_retrieval = expand_user_query(
                        user_query=condensed_query,
                        model_name=settings.models.query_expander_id,
                        char_threshold=settings.pipeline.query_expansion_char_threshold
                    )

                print(f"Verarbeite Anfrage: '{expanded_query_for_retrieval}'...")

                query_embedding_vector = embed_query(loaded_embedding_model, expanded_query_for_retrieval)

                current_retrieval_top_k = settings.pipeline.retrieval_top_k if settings.pipeline.use_reranker and loaded_reranker_model else settings.pipeline.default_retrieval_top_k

                print(f"Rufe Top-{current_retrieval_top_k} Dokumente aus dem Vektor-Speicher ab.")
                retrieved_docs_initial = query_vector_store(db_collection, query_embedding_vector, current_retrieval_top_k)

                final_docs_for_llm = []
                if settings.pipeline.use_reranker and loaded_reranker_model:
                    if retrieved_docs_initial:
                        reranked_and_filtered_docs = gap_based_rerank_and_filter(
                            user_query=expanded_query_for_retrieval,
                            initial_retrieved_docs=retrieved_docs_initial,
                            reranker_model=loaded_reranker_model,
                            min_absolute_rerank_score_threshold=settings.pipeline.min_absolute_score_threshold,
                            min_chunks_to_llm=settings.pipeline.min_chunks_to_llm,
                            max_chunks_to_llm=settings.pipeline.max_chunks_to_llm,
                            min_chunks_for_gap_detection=settings.pipeline.min_chunks_for_gap_detection,
                            gap_detection_factor=settings.pipeline.gap_detection_factor,
                            small_epsilon=settings.pipeline.small_epsilon
                        )
                        final_docs_for_llm = reranked_and_filtered_docs
                    else:
                        print("Keine Dokumente vom initialen Retrieval für Reranking erhalten.")
                else:
                    final_docs_for_llm = retrieved_docs_initial

                if not final_docs_for_llm:
                    print("Konnte keine relevanten Dokumente im Handbuch für Ihre Anfrage finden (nach ggf. Reranking).")
                    continue

                if settings.system.low_vram_mode:
                    print("\nLow-VRAM-Modus: Entlade temporäre Modelle...")
                    loaded_embedding_model = None
                    loaded_reranker_model = None
                    gc.collect()
                    torch.cuda.empty_cache()
                    print("Temporäre Modelle entladen und VRAM freigegeben.")


                print(f"Generiere Antwort mit {settings.models.ollama_llm} basierend auf {len(final_docs_for_llm)} abgerufenen/gefilterten Chunk(s)...")

                # --- FINALE LOGIK: ZITATE ERSETZEN UND ANZEIGEN ---

                llm_answer_generator, citation_map = generate_llm_answer(
                    user_query=condensed_query,
                    retrieved_chunks=final_docs_for_llm,
                    ollama_model_name=settings.models.ollama_llm,
                    ollama_options=settings.models.ollama_options,
                )

                raw_response_with_citations = "".join([chunk for chunk in llm_answer_generator])

                used_source_ids: Set[int] = set()
                citation_regex = re.compile(r'\[Source ID: (\d+)]')

                matches = citation_regex.finditer(raw_response_with_citations)
                for match in matches:
                    source_id = int(match.group(1))
                    used_source_ids.add(source_id)

                # ERSETZT [Source ID: 1] mit [1] anstatt es zu löschen
                formatted_response = citation_regex.sub(r'[\1]', raw_response_with_citations).strip()

                print(f"\nAssistent: ", end="", flush=True)
                for char in formatted_response:
                    print(char, end="", flush=True)
                    time.sleep(0.01)
                print()

                if used_source_ids and citation_map:
                    print("\n**Quellen:**")
                    for source_id in sorted(list(used_source_ids)):
                        if source_id in citation_map:
                            source_info = citation_map[source_id]
                            print(f"- [{source_id}] {source_info['filename']}, Abschnitt: *{source_info['header']}*")

                last_query = user_query_text
                last_response = formatted_response # Speichert die Antwort mit den [1]-Tags

                # --- ENDE DER FINALEN LOGIK ---


            except Exception as e:
                print(f"\nFehler während der Anfrageverarbeitung oder LLM-Generierung: {e}")
                print("Assistent: Entschuldigung, bei der Bearbeitung Ihrer Anfrage ist ein Fehler aufgetreten.")
                last_query, last_response = None, ""

            finally:
                if settings.system.low_vram_mode:
                    print("\nLow-VRAM-Modus: Entlade Modelle...")

                    force_unload_ollama_model(settings.models.ollama_llm)
                    force_unload_ollama_model(settings.models.condenser_model_id)
                    force_unload_ollama_model(settings.models.query_expander_id)

                    gc.collect()
                    torch.cuda.empty_cache()


    else:
        print("Chatbot kann nicht gestartet werden: DB-Kollektion oder Embedding-Modell nicht verfügbar.")

    print("Ausführung des Hauptskripts beendet.")

if __name__ == "__main__":
    main()