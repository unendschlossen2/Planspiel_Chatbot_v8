import sys
if "torch.classes" in sys.modules:
    del sys.modules["torch.classes"]
import streamlit as st
import time
import re
import os
from typing import Set

# --- Import all your project's modules ---
from helper.settings import settings
from helper.load_gpu import load_gpu
from embeddings.embedding_generator import load_embedding_model, embed_chunks
from retrieval.retriever import embed_query, query_vector_store
from retrieval.reranker import load_reranker_model, gap_based_rerank_and_filter
from vector_store.vector_store_manager import create_and_populate_vector_store
from generation.conversation_handler import condense_conversation
from generation.query_expander import expand_user_query
from generation.llm_answer_generator import generate_llm_answer
from helper.file_loader import load_markdown_directory
from processing.chunking import split_markdown_by_headers
from preprocessing.text_cleanup import normalize_markdown_whitespace


# --- Caching Functions for Model and DB Loading ---

@st.cache_resource
def load_processing_device():
    """Lädt das Verarbeitungsgerät (GPU oder CPU) nur einmal."""
    try:
        device = str(load_gpu())
        # st.success(f"GPU erfolgreich erkannt und geladen: {device}")
        return device
    except RuntimeError as e:
        # st.warning(f"GPU Ladefehler: {e}. Wechsle zu CPU.")
        return "cpu"

@st.cache_resource
def load_models(device):
    """Lädt alle Modelle einmalig."""
    with st.spinner("Lade Embedding- und Reranker-Modelle..."):
        embedding_model = load_embedding_model(settings.models.embedding_id, device)
        reranker_model = None
        if settings.pipeline.use_reranker:
            reranker_model = load_reranker_model(settings.models.reranker_id, device=device)
    return embedding_model, reranker_model

@st.cache_resource
def setup_database_connection():
    """Stellt die Verbindung zur Vektor-Datenbank her."""
    db_collection = create_and_populate_vector_store(
        chunks_with_embeddings=[],
        db_path=settings.database.persist_path,
        collection_name=settings.database.collection_name,
        force_rebuild_collection=False
    )
    return db_collection

def rebuild_database_from_source_files(device, embedding_model):
    """Baut die Datenbank neu auf und nutzt dabei das bereits geladene Embedding-Modell."""
    with st.spinner("Lösche alte Datenbank-Kollektion..."):
        create_and_populate_vector_store(
            chunks_with_embeddings=[],
            db_path=settings.database.persist_path,
            collection_name=settings.database.collection_name,
            force_rebuild_collection=True
        )

    with st.spinner("Lade Quelldateien und starte Chunking..."):
        input_dir = os.path.join("project_files", "input_files")
        file_list = load_markdown_directory(input_dir)
        all_files_chunks_with_embeddings = []

        for file_item in file_list:
            normalized_content = normalize_markdown_whitespace(file_item["content"])
            chunks = split_markdown_by_headers(
                markdown_text=normalized_content,
                source_filename=file_item["source"],
                split_level=settings.processing.initial_split_level,
                max_chars_per_chunk=settings.processing.max_chars_per_chunk,
                min_chars_per_chunk=settings.processing.min_chars_per_chunk
            )

            chunks_with_embeddings_for_file = embed_chunks(
                chunks_data=chunks,
                model_id=settings.models.embedding_id,
                device=device,
                preloaded_model=embedding_model,
                normalize_embeddings=True
            )
            all_files_chunks_with_embeddings.extend(chunks_with_embeddings_for_file)

    with st.spinner("Fülle die Vektor-Datenbank mit neuen Daten..."):
        create_and_populate_vector_store(
            chunks_with_embeddings=all_files_chunks_with_embeddings,
            db_path=settings.database.persist_path,
            collection_name=settings.database.collection_name,
            force_rebuild_collection=True
        )

# --- Main Chatbot Logic ---

def get_bot_response(user_query: str, chat_history: list):
    """Enthält die vollständige RAG-Pipeline-Logik."""
    device = st.session_state.device
    embedding_model, reranker_model = st.session_state.embedding_model, st.session_state.reranker_model
    db_collection = setup_database_connection()

    if not db_collection:
        return "Fehler: Datenbank-Kollektion konnte nicht geladen werden.", ""

    last_query = None
    last_response = None

    if len(chat_history) >= 2:
        if "keine relevanten Informationen" not in chat_history[-1]['content']:
            last_query = chat_history[-2]['content']
            last_response = chat_history[-1]['content']

    condensed_query = user_query
    if settings.pipeline.enable_conversation_memory and last_query and last_response:
        with st.spinner("Verarbeite Konversationskontext..."):
            condensed_query = condense_conversation(
                model_name=settings.models.condenser_model_id,
                last_query=last_query,
                last_response=last_response,
                new_query=user_query
            )

    expanded_query = condensed_query
    if settings.pipeline.enable_query_expansion and len(condensed_query) < settings.pipeline.query_expansion_char_threshold:
        with st.spinner("Erweitere kurze Anfrage..."):
            expanded_query = expand_user_query(
                user_query=condensed_query,
                model_name=settings.models.query_expander_id,
                char_threshold=settings.pipeline.query_expansion_char_threshold
            )

    with st.spinner(f"Suche nach Dokumenten für: '{expanded_query}'..."):
        query_embedding = embed_query(embedding_model, expanded_query)
        retrieved_docs = query_vector_store(db_collection, query_embedding, settings.pipeline.retrieval_top_k)

        final_docs = retrieved_docs
        if settings.pipeline.use_reranker and reranker_model and retrieved_docs:
            final_docs = gap_based_rerank_and_filter(
                user_query=expanded_query,
                initial_retrieved_docs=retrieved_docs,
                reranker_model=reranker_model,
                min_absolute_rerank_score_threshold=settings.pipeline.min_absolute_score_threshold,
                min_chunks_to_llm=settings.pipeline.min_chunks_to_llm,
                max_chunks_to_llm=settings.pipeline.max_chunks_to_llm
            )

    if not final_docs:
        return "Ich konnte leider keine relevanten Informationen zu Ihrer Anfrage im Handbuch finden.", ""

    with st.spinner("Generiere Antwort..."):
        llm_answer_generator, citation_map = generate_llm_answer(
            user_query=condensed_query,
            retrieved_chunks=final_docs,
            ollama_model_name=settings.models.ollama_llm,
            ollama_options=settings.models.ollama_options,
        )
        raw_response = "".join([chunk for chunk in llm_answer_generator])

    used_source_ids: Set[int] = set()
    citation_regex = re.compile(r'\[Source ID: (\d+)]')

    matches = citation_regex.finditer(raw_response)
    for match in matches:
        used_source_ids.add(int(match.group(1)))

    formatted_response = citation_regex.sub(r'[\1]', raw_response).strip()

    sources_text = ""
    if used_source_ids and citation_map:
        sources_list = []
        for source_id in sorted(list(used_source_ids)):
            if source_id in citation_map:
                info = citation_map[source_id]
                sources_list.append(f"- [{source_id}] **{info['filename']}**, Abschnitt: *{info['header']}*")
        sources_text = "\n".join(sources_list)

    return formatted_response, sources_text

# --- Helper function for logging ---
def log_app_state():
    """Gibt den aktuellen Zustand der Anwendung in der Konsole aus."""
    print("\n--- ANWENDUNGSZUSTAND ---")
    try:
        db_collection = setup_database_connection()
        if db_collection:
            print(f"Datenbank-Status: Kollektion '{db_collection.name}' geladen mit {db_collection.count()} Elementen.")
        else:
            print("Datenbank-Status: Kollektion nicht geladen.")
    except Exception as e:
        print(f"Datenbank-Status: Fehler beim Zugriff auf die DB - {e}")

    print("\n[Konfiguration]")
    print(f"  LLM: {settings.models.ollama_llm}")
    print(f"  Reranker: {'Aktiv' if settings.pipeline.use_reranker else 'Inaktiv'}")
    print(f"  Konversationsgedächtnis: {'Aktiv' if settings.pipeline.enable_conversation_memory else 'Inaktiv'}")

    print("\n[Session-Zustand]")
    print(f"  Anzahl Nachrichten im Chat-Verlauf: {len(st.session_state.get('messages', []))}")
    print("--- ENDE ZUSTAND ---\n")

# --- Streamlit UI ---

st.set_page_config(page_title="TOPSIM RAG Chatbot", page_icon="🤖", layout="wide")
st.title("🤖 TOPSIM RAG Chatbot")

# Initialisierung im Session State
if "device" not in st.session_state:
    st.session_state.device = load_processing_device()
if "messages" not in st.session_state:
    st.session_state.messages = []
if "rebuild_triggered" not in st.session_state:
    st.session_state.rebuild_triggered = False
if "models_loaded" not in st.session_state:
    embedding_model, reranker_model = load_models(st.session_state.device)
    st.session_state.embedding_model = embedding_model
    st.session_state.reranker_model = reranker_model
    st.session_state.models_loaded = True

def trigger_rebuild():
    st.session_state.rebuild_triggered = True

def exit_app():
    print("\n[AKTION] 'App beenden' geklickt. Beende den Prozess.")
    # This is a clean way to stop the Streamlit server process
    os._exit(0)

with st.sidebar:
    st.header("Steuerung")
    if st.button("Neuer Chat"):
        print("\n[AKTION] 'Neuer Chat' geklickt. Chat-Verlauf wird zurückgesetzt.")
        st.session_state.messages = []
        st.rerun()

    st.button("Datenbank neu aufbauen", on_click=trigger_rebuild)
    st.button("Zustand in Konsole ausgeben", on_click=log_app_state)
    st.button("App beenden", on_click=exit_app, type="primary")

    st.header("Konfiguration")
    st.info(f"**LLM:** `{settings.models.ollama_llm}`")
    st.info(f"**Reranker:** `{'Aktiviert' if settings.pipeline.use_reranker else 'Deaktiviert'}`")
    st.info(f"**Gerät:** `{st.session_state.device}`")

if st.session_state.rebuild_triggered:
    st.cache_resource.clear()
    rebuild_database_from_source_files(st.session_state.device, st.session_state.embedding_model)
    st.session_state.rebuild_triggered = False
    st.success("Datenbank erfolgreich neu aufgebaut! Die Seite wird neu geladen.")
    time.sleep(2)
    st.rerun()

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Stellen Sie Ihre Frage an das TOPSIM Handbuch..."):
    print(f"\n[BENUTZEREINGABE] Neue Anfrage erhalten: '{prompt}'")
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        response_text, sources = get_bot_response(prompt, st.session_state.messages)

        placeholder = st.empty()
        full_response = ""
        for char in response_text:
            full_response += char
            placeholder.markdown(full_response + "▌")
            time.sleep(0.01)
        placeholder.markdown(full_response)

        if sources:
            with st.expander("**Quellen**"):
                st.markdown(sources)

    full_bot_message = response_text + (f"\n\n**Quellen**\n\n{sources}" if sources else "")
    st.session_state.messages.append({"role": "assistant", "content": full_bot_message})
