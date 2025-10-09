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
from helper.logger import SessionLogger
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
    """
    Führt den vollständigen Prozess zum Neuaufbau der Vektor-Datenbank aus den Quelldateien durch.
    Diese Funktion ist eine direkte Integration der Logik aus dem alten `temp_main.py`.
    """
    st.info("Starte den Neuaufbau der Vektor-Datenbank. Dieser Vorgang kann einige Minuten dauern.")

    # 1. Alte Kollektion löschen
    with st.spinner("Schritt 1/4: Lösche alte Datenbank-Kollektion..."):
        create_and_populate_vector_store(
            chunks_with_embeddings=[],
            db_path=settings.database.persist_path,
            collection_name=settings.database.collection_name,
            force_rebuild_collection=True
        )
        st.success("Schritt 1/4: Alte Datenbank-Kollektion erfolgreich gelöscht.")

    # 2. Quelldateien laden und verarbeiten
    with st.spinner("Schritt 2/4: Lade und verarbeite Quelldateien..."):
        input_dir = os.path.join("project_files", "input_files")
        file_list = load_markdown_directory(input_dir)
        if not file_list:
            st.error("Keine Markdown-Dateien im `input_files`-Verzeichnis gefunden. Abbruch.")
            return

        all_chunks = []
        progress_bar = st.progress(0)
        for i, file_item in enumerate(file_list):
            normalized_content = normalize_markdown_whitespace(file_item["content"])
            if not normalized_content.strip():
                continue
            chunks = split_markdown_by_headers(
                markdown_text=normalized_content,
                source_filename=file_item["source"],
                split_level=settings.processing.initial_split_level,
                max_chars_per_chunk=settings.processing.max_chars_per_chunk,
                min_chars_per_chunk=settings.processing.min_chars_per_chunk
            )
            all_chunks.extend(chunks)
            progress_bar.progress((i + 1) / len(file_list), text=f"Verarbeite: {file_item['source']}")
        st.success(f"Schritt 2/4: Quelldateien erfolgreich verarbeitet. {len(all_chunks)} Chunks erstellt.")

    # 3. Embeddings generieren
    with st.spinner(f"Schritt 3/4: Generiere Embeddings für {len(all_chunks)} Chunks..."):
        chunks_with_embeddings = embed_chunks(
            chunks_data=all_chunks,
            model_id=settings.models.embedding_id,
            device=device,
            preloaded_model=embedding_model,
            normalize_embeddings=True
        )
        st.success("Schritt 3/4: Embeddings erfolgreich generiert.")

    # 4. Datenbank füllen
    with st.spinner("Schritt 4/4: Fülle die Vektor-Datenbank mit neuen Daten..."):
        create_and_populate_vector_store(
            chunks_with_embeddings=chunks_with_embeddings,
            db_path=settings.database.persist_path,
            collection_name=settings.database.collection_name,
            force_rebuild_collection=False # Die Kollektion wurde bereits in Schritt 1 neu erstellt
        )
        st.success("Schritt 4/4: Vektor-Datenbank erfolgreich gefüllt.")

# --- Main Chatbot Logic ---

def get_bot_response(
    user_query: str,
    chat_history: list,
    temperature: float,
    use_reranker: bool,
    retrieval_top_k: int,
    enable_query_expansion: bool,
    enable_conversation_memory: bool
):
    """Enthält die vollständige RAG-Pipeline-Logik."""
    device = st.session_state.device
    embedding_model, reranker_model = st.session_state.embedding_model, st.session_state.reranker_model
    db_collection = setup_database_connection()

    if not db_collection:
        return "Fehler: Datenbank-Kollektion konnte nicht geladen werden.", "", user_query

    last_query = None
    last_response = None

    if len(chat_history) >= 2:
        if "keine relevanten Informationen" not in chat_history[-1]['content']:
            last_query = chat_history[-2]['content']
            last_response = chat_history[-1]['content']

    condensed_query = user_query
    if enable_conversation_memory and last_query and last_response:
        with st.spinner("Verarbeite Konversationskontext..."):
            condensed_query = condense_conversation(
                last_query=last_query,
                last_response=last_response,
                new_query=user_query,
                condenser_model_path=settings.models.condenser_model_path,
                condenser_generation_config=settings.models.condenser_generation_config,
            )

    expanded_query = condensed_query
    if enable_query_expansion and len(condensed_query) < settings.pipeline.query_expansion_char_threshold:
        with st.spinner("Erweitere kurze Anfrage..."):
            expanded_query = expand_user_query(
                user_query=condensed_query,
                char_threshold=settings.pipeline.query_expansion_char_threshold,
                query_expander_model_path=settings.models.query_expander_model_path,
                query_expander_generation_config=settings.models.query_expander_generation_config,
            )

    with st.spinner(f"Suche nach Dokumenten für: '{expanded_query}'..."):
        query_embedding = embed_query(embedding_model, expanded_query)
        retrieved_docs = query_vector_store(db_collection, query_embedding, retrieval_top_k)

        final_docs = retrieved_docs
        if use_reranker and reranker_model and retrieved_docs:
            final_docs = gap_based_rerank_and_filter(
                user_query=expanded_query,
                initial_retrieved_docs=retrieved_docs,
                reranker_model=reranker_model,
                min_absolute_rerank_score_threshold=settings.pipeline.min_absolute_score_threshold,
                min_chunks_to_llm=settings.pipeline.min_chunks_to_llm,
                max_chunks_to_llm=settings.pipeline.max_chunks_to_llm
            )

    if not final_docs:
        return "Ich konnte leider keine relevanten Informationen zu Ihrer Anfrage im Handbuch finden.", "", user_query

    # Erstelle eine Kopie der Generation-Config, um sie für diese Anfrage zu modifizieren
    generation_config = settings.models.llm_generation_config.copy()
    generation_config["temperature"] = temperature

    with st.spinner("Generiere Antwort..."):
        llm_answer_generator, citation_map = generate_llm_answer(
            user_query=condensed_query,
            retrieved_chunks=final_docs,
            llm_model_path=settings.models.llm_model_path,
            llm_generation_config=generation_config,
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

    # Gib die finale Anfrage zurück, damit sie geloggt werden kann
    return formatted_response, sources_text, expanded_query

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
    print(f"  LLM: {settings.models.llm_model_id} ({settings.models.llm_model_path})")
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
if "session_logger" not in st.session_state:
    st.session_state.session_logger = SessionLogger()

# Initialisierung für UI-gesteuerte Pipeline-Parameter
if "temperature" not in st.session_state:
    st.session_state.temperature = settings.models.llm_generation_config.get("temperature", 0.2)
if "retrieval_top_k" not in st.session_state:
    st.session_state.retrieval_top_k = settings.pipeline.retrieval_top_k
if "use_reranker" not in st.session_state:
    st.session_state.use_reranker = settings.pipeline.use_reranker
if "enable_query_expansion" not in st.session_state:
    st.session_state.enable_query_expansion = settings.pipeline.enable_query_expansion
if "enable_conversation_memory" not in st.session_state:
    st.session_state.enable_conversation_memory = settings.pipeline.enable_conversation_memory

def trigger_rebuild():
    st.session_state.rebuild_triggered = True

def exit_app():
    print("\n[AKTION] 'App beenden' geklickt. Beende den Prozess.")
    # This is a clean way to stop the Streamlit server process
    os._exit(0)

# --- UI Logic ---

# Prüfen, ob die Datenbank existiert oder ein Neuaufbau erzwungen wird
db_collection = setup_database_connection()
if not db_collection or st.session_state.rebuild_triggered:
    if not db_collection:
        st.warning("Keine bestehende Datenbank gefunden. Starte den initialen Aufbau...")
    elif st.session_state.rebuild_triggered:
        st.info("Manueller Neuaufbau der Datenbank wurde ausgelöst...")


    st.cache_resource.clear() # Cache leeren, um Neuladen zu erzwingen
    rebuild_database_from_source_files(st.session_state.device, st.session_state.embedding_model)
    st.session_state.rebuild_triggered = False
    st.success("Datenbank erfolgreich (neu) aufgebaut! Die Seite wird neu geladen.")
    time.sleep(3)
    st.rerun()

# Erstelle die Tabs
tab_chat, tab_settings = st.tabs(["💬 Chat", "⚙️ Einstellungen"])

with tab_chat:
    with st.sidebar:
        st.header("⚙️ Generation Parameters")
        st.session_state.temperature = st.slider(
            "LLM Temperature", min_value=0.0, max_value=1.0,
            value=st.session_state.temperature, step=0.05,
            help="Controls the randomness of the model's output. Lower values make it more deterministic."
        )
        st.session_state.retrieval_top_k = st.slider(
            "Retrieval Top-K", min_value=1, max_value=20,
            value=st.session_state.retrieval_top_k, step=1,
            help="Number of initial documents to retrieve from the vector store."
        )
        st.session_state.use_reranker = st.toggle(
            "Use Reranker", value=st.session_state.use_reranker,
            help="Enable or disable the reranking step to improve document relevance."
        )
        st.session_state.enable_query_expansion = st.toggle(
            "Enable Query Expansion", value=st.session_state.enable_query_expansion,
            help="Automatically expand short queries to improve retrieval."
        )
        st.session_state.enable_conversation_memory = st.toggle(
            "Enable Conversation Memory", value=st.session_state.enable_conversation_memory,
            help="Allow the chatbot to remember the context of the last interaction."
        )

    # Haupt-Chat-Interface
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Stellen Sie Ihre Frage an das TOPSIM Handbuch..."):
        print(f"\n[BENUTZEREINGABE] Neue Anfrage erhalten: '{prompt}'")
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            response_text, sources, final_query = get_bot_response(
                user_query=prompt,
                chat_history=st.session_state.messages,
                temperature=st.session_state.temperature,
                use_reranker=st.session_state.use_reranker,
                retrieval_top_k=st.session_state.retrieval_top_k,
                enable_query_expansion=st.session_state.enable_query_expansion,
                enable_conversation_memory=st.session_state.enable_conversation_memory,
            )

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
        # Log the complete interaction
        st.session_state.session_logger.log_interaction(
            user_query=prompt,
            final_prompt=final_query,
            assistant_response=full_bot_message
        )
        st.session_state.messages.append({"role": "assistant", "content": full_bot_message})

with tab_settings:
    st.header("App-Steuerung")

    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("Neuer Chat", use_container_width=True):
            print("\n[AKTION] 'Neuer Chat' geklickt. Chat-Verlauf wird zurückgesetzt.")
            st.session_state.messages = []
            st.rerun()
    with col2:
        st.button("Datenbank neu aufbauen", on_click=trigger_rebuild, use_container_width=True)
    with col3:
        st.button("App beenden", on_click=exit_app, type="primary", use_container_width=True)

    st.divider()

    st.header("Sitzungsprotokolle (Logs)")

    log_files = SessionLogger.list_log_files()
    if not log_files:
        st.info("Es sind noch keine Log-Dateien vorhanden.")
    else:
        selected_logs = st.multiselect(
            "Wählen Sie die zu löschenden Protokolle aus:",
            options=log_files,
            format_func=lambda path: path.name
        )

        col_del1, col_del2 = st.columns(2)
        with col_del1:
            if st.button("Ausgewählte Logs löschen", disabled=not selected_logs, use_container_width=True):
                SessionLogger.delete_log_files(selected_logs)
                st.success(f"{len(selected_logs)} Log(s) gelöscht.")
                st.rerun()
        with col_del2:
            if st.button("Alle Logs löschen", type="primary", use_container_width=True):
                SessionLogger.delete_log_files(log_files)
                st.success("Alle Logs wurden gelöscht.")
                st.rerun()

        st.subheader("Alle Protokolle")
        for log in log_files:
            with st.expander(f"{log.name} ({os.path.getsize(log)} Bytes)"):
                try:
                    with open(log, "r", encoding="utf-8") as f:
                        st.text(f.read())
                except Exception as e:
                    st.error(f"Konnte Log-Datei nicht lesen: {e}")
