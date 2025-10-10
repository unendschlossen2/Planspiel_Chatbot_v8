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
from helper.model_manager import ModelManager
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
        return device
    except RuntimeError as e:
        st.warning(f"GPU Ladefehler: {e}. Wechsle zu CPU.")
        return "cpu"

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

def rebuild_database_from_source_files(device: str):
    """
    Führt den vollständigen Prozess zum Neuaufbau der Vektor-Datenbank aus den Quelldateien durch.
    Nutzt den ModelManager, um das Embedding-Modell zu laden.
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
        embedding_model = st.session_state.model_manager.get_model("embedding")
        chunks_with_embeddings = embed_chunks(
            chunks_data=all_chunks,
            model_id=settings.models.embedding_id,
            device=device,
            preloaded_model=embedding_model,
            normalize_embeddings=True
        )
        st.success("Schritt 3/4: Embeddings erfolgreich generiert.")
        # Nach Gebrauch sofort entladen, da dies ein einmaliger Prozess ist
        st.session_state.model_manager.unload_model("embedding")

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
    """
    Enthält die vollständige RAG-Pipeline-Logik und nutzt den ModelManager,
    um Modelle basierend auf der ausgewählten VRAM-Strategie zu laden und zu entladen.
    """
    model_manager = st.session_state.model_manager
    strategy = model_manager.get_vram_strategy()

    db_collection = setup_database_connection()
    if not db_collection:
        return "Fehler: Datenbank-Kollektion konnte nicht geladen werden.", "", user_query

    last_query, last_response = (chat_history[-2]['content'], chat_history[-1]['content']) if len(chat_history) >= 2 and "keine relevanten Informationen" not in chat_history[-1]['content'] else (None, None)

    # --- Conversation Condensing ---
    condensed_query = user_query
    if enable_conversation_memory and last_query and last_response:
        with st.spinner("🔄 Verarbeite Konversationskontext..."):
            condenser_model = model_manager.get_model("condenser")
            if condenser_model:
                condensed_query = condense_conversation(
                    last_query=last_query, last_response=last_response, new_query=user_query,
                    condenser_model=condenser_model,
                    condenser_generation_config=settings.models.condenser_generation_config,
                )
            if strategy == "Aggressive": model_manager.unload_model("condenser")

    # --- Query Expansion ---
    expanded_query = condensed_query
    if enable_query_expansion and len(condensed_query) < settings.pipeline.query_expansion_char_threshold:
        with st.spinner("🔍 Erweitere Anfrage..."):
            expander_model = model_manager.get_model("expander")
            if expander_model:
                expanded_query = expand_user_query(
                    user_query=condensed_query, char_threshold=settings.pipeline.query_expansion_char_threshold,
                    expander_model=expander_model,
                    query_expander_generation_config=settings.models.query_expander_generation_config,
                )
            if strategy == "Aggressive": model_manager.unload_model("expander")

    # --- Retrieval ---
    with st.spinner(f"📚 Suche in Dokumenten für: '{expanded_query}'..."):
        embedding_model = model_manager.get_model("embedding")
        if not embedding_model:
             return "Fehler: Embedding-Modell konnte nicht geladen werden. Der Vektor-Datenbank-Aufbau ist möglicherweise fehlgeschlagen.", "", user_query
        query_embedding = embed_query(embedding_model, expanded_query)
        retrieved_docs = query_vector_store(db_collection, query_embedding, retrieval_top_k)
        if strategy in ["Aggressive", "Balanced"]:
            model_manager.unload_model("embedding")

    # --- Reranking ---
    final_docs = retrieved_docs
    if use_reranker and retrieved_docs:
        with st.spinner("⚖️ Bewerte Dokumenten-Relevanz neu..."):
            reranker_model = model_manager.get_model("reranker")
            if reranker_model:
                final_docs = gap_based_rerank_and_filter(
                    user_query=expanded_query, initial_retrieved_docs=retrieved_docs, reranker_model=reranker_model,
                    min_absolute_rerank_score_threshold=settings.pipeline.min_absolute_score_threshold,
                    min_chunks_to_llm=settings.pipeline.min_chunks_to_llm, max_chunks_to_llm=settings.pipeline.max_chunks_to_llm
                )
            if strategy in ["Aggressive", "Balanced"]:
                model_manager.unload_model("reranker")

    if not final_docs:
        return "Ich konnte leider keine relevanten Informationen zu Ihrer Anfrage im Handbuch finden.", "", user_query

    # --- Answer Generation ---
    generation_config = settings.models.llm_generation_config.copy()
    generation_config["temperature"] = temperature
    with st.spinner("✍️ Generiere Antwort..."):
        llm_model = model_manager.get_model("llm")
        if llm_model:
            llm_answer_generator, citation_map = generate_llm_answer(
                user_query=condensed_query, retrieved_chunks=final_docs, llm_model=llm_model,
                llm_generation_config=generation_config
            )
            raw_response = "".join([chunk for chunk in llm_answer_generator])
        else:
            raw_response = "Fehler: Das Haupt-Antwortmodell konnte nicht geladen werden."
            citation_map = {}

        # In Balanced/Aggressive mode, unload all generative models after use
        if strategy != "Performance":
            model_manager.unload_model("llm")
            if enable_conversation_memory: model_manager.unload_model("condenser")
            if enable_query_expansion: model_manager.unload_model("expander")

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

def chat_page():
    """Rendert die Haupt-Chat-Seite."""
    st.title("🤖 TOPSIM RAG Chatbot")

    # --- Callbacks für UI-Feedback ---
    def on_setting_change(setting_name: str, session_key: str, is_toggle: bool = False):
        new_value = st.session_state[session_key]
        if is_toggle:
            display_value = "Aktiviert" if new_value else "Deaktiviert"
        else:
            display_value = new_value
        st.toast(f"{setting_name} auf '{display_value}' gesetzt.", icon="✅")

    with st.expander("⚙️ Generation Parameters"):
        st.slider(
            "LLM Temperature", min_value=0.0, max_value=1.0,
            key="temperature", step=0.05,
            help="Controls the randomness of the model's output. Lower values make it more deterministic.",
            on_change=on_setting_change, args=("Temperatur", "temperature")
        )
        st.slider(
            "Retrieval Top-K", min_value=1, max_value=20,
            key="retrieval_top_k", step=1,
            help="Number of initial documents to retrieve from the vector store.",
            on_change=on_setting_change, args=("Retrieval Top-K", "retrieval_top_k")
        )
        st.toggle(
            "Use Reranker", key="use_reranker",
            help="Enable or disable the reranking step to improve document relevance.",
            on_change=on_setting_change, args=("Reranker", "use_reranker", True)
        )
        st.toggle(
            "Enable Query Expansion", key="enable_query_expansion",
            help="Automatically expand short queries to improve retrieval.",
            on_change=on_setting_change, args=("Anfrage-Erweiterung", "enable_query_expansion", True)
        )
        st.toggle(
            "Enable Conversation Memory", key="enable_conversation_memory",
            help="Allow the chatbot to remember the context of the last interaction.",
            on_change=on_setting_change, args=("Konversationsgedächtnis", "enable_conversation_memory", True)
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
        st.session_state.session_logger.log_interaction(
            user_query=prompt,
            final_prompt=final_query,
            assistant_response=full_bot_message
        )
        st.session_state.messages.append({"role": "assistant", "content": full_bot_message})

def settings_page():
    """Rendert die Einstellungs- und Verwaltungsseite."""
    st.title("⚙️ Einstellungen & Verwaltung")

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

    st.header("VRAM-Verwaltungsstrategie")

    def on_vram_strategy_change():
        st.session_state.model_manager.set_vram_strategy(st.session_state.vram_strategy)
        st.toast(f"VRAM-Strategie auf '{st.session_state.vram_strategy}' geändert. Alle Modelle wurden entladen.", icon="🧠")

    st.radio(
        "Wählen Sie eine Strategie:",
        ["Performance", "Balanced", "Aggressive"],
        key="vram_strategy",
        on_change=on_vram_strategy_change,
        help=(
            "- **Performance:** Lädt alle Modelle beim ersten Bedarf und behält sie im Speicher.\n"
            "- **Balanced:** Gruppiert Modelle (Retrieval/Generation) und entlädt sie, wenn sie nicht mehr benötigt werden.\n"
            "- **Aggressive:** Lädt jedes Modell nur für seine spezifische Aufgabe und entlädt es sofort wieder."
        )
    )

    st.divider()

    st.header("Modell-Status")
    status = st.session_state.model_manager.get_loaded_models_status()
    cols = st.columns(len(status))
    for i, (model_name, is_loaded) in enumerate(status.items()):
        with cols[i]:
            st.metric(
                label=model_name.capitalize(),
                value="Geladen" if is_loaded else "Entladen",
                delta="Aktiv" if is_loaded else "Inaktiv",
                delta_color="normal" if is_loaded else "off"
            )

    st.divider()

    st.header("Sitzungsprotokolle (Logs)")
    log_files = SessionLogger.list_log_files()
    if not log_files:
        st.info("Es sind noch keine Log-Dateien vorhanden.")
    else:
        selected_logs = st.multiselect(
            "Wählen Sie die zu löschenden Protokolle aus:",
            options=log_files,
            format_func=lambda path: path.name,
            key="log_multiselect"  # Verhindert Duplizierung
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

# --- App Initialisierung & Haupt-Seiten-Rendering ---

st.set_page_config(page_title="TOPSIM RAG Chatbot", page_icon="🤖", layout="wide")

# Initialisierung im Session State
if "device" not in st.session_state:
    st.session_state.device = load_processing_device()
if "messages" not in st.session_state:
    st.session_state.messages = []
if "rebuild_triggered" not in st.session_state:
    st.session_state.rebuild_triggered = False
if "session_logger" not in st.session_state:
    st.session_state.session_logger = SessionLogger()
if "model_manager" not in st.session_state:
    st.session_state.model_manager = ModelManager(settings_models=settings.models, device=st.session_state.device)
if "vram_strategy" not in st.session_state:
    st.session_state.vram_strategy = "Performance" # Default

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
    os._exit(0)

# Prüfen, ob die Datenbank existiert oder ein Neuaufbau erzwungen wird
db_collection = setup_database_connection()
if not db_collection or st.session_state.rebuild_triggered:
    if not db_collection:
        st.warning("Keine bestehende Datenbank gefunden. Starte den initialen Aufbau...")
    elif st.session_state.rebuild_triggered:
        st.info("Manueller Neuaufbau der Datenbank wurde ausgelöst...")

    st.cache_resource.clear()
    st.session_state.model_manager.unload_all_models() # Ensure a clean slate before rebuild
    rebuild_database_from_source_files(st.session_state.device)
    st.session_state.rebuild_triggered = False
    st.success("Datenbank erfolgreich (neu) aufgebaut! Die Seite wird neu geladen.")
    time.sleep(3)
    st.rerun()

# Seiten-Navigation in der Sidebar
with st.sidebar:
    st.title("Navigation")
    page = st.radio("Wählen Sie eine Seite:", ["Chat", "Einstellungen"], key="navigation_radio")

# Rendere die ausgewählte Seite
if page == "Chat":
    chat_page()
elif page == "Einstellungen":
    settings_page()
