import sys
if "torch.classes" in sys.modules:
    del sys.modules["torch.classes"]
import streamlit as st
import time
import re
import os
from typing import Set, List

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
    """Loads the processing device and its display name only once."""
    try:
        device, device_name = load_gpu()
        return str(device), device_name
    except RuntimeError as e:
        st.warning(f"GPU loading error: {e}. Switching to CPU.")
        return "cpu", "CPU"

@st.cache_resource
def setup_database_connection():
    """Establishes the connection to the vector database."""
    db_collection = create_and_populate_vector_store(
        chunks_with_embeddings=[],
        db_path=settings.database.persist_path,
        collection_name=settings.database.collection_name,
        force_rebuild_collection=False
    )
    return db_collection

def rebuild_database_from_source_files(device: str):
    """
    Executes the complete process of rebuilding the vector database from source files.
    Uses the ModelManager to load the embedding model.
    """
    st.info("Starting the vector database rebuild. This process may take a few minutes.")

    with st.spinner("Step 1/4: Deleting old database collection..."):
        create_and_populate_vector_store(
            chunks_with_embeddings=[],
            db_path=settings.database.persist_path,
            collection_name=settings.database.collection_name,
            force_rebuild_collection=True
        )
        st.success("Step 1/4: Old database collection successfully deleted.")

    with st.spinner("Step 2/4: Loading and processing source files..."):
        input_dir = os.path.join("project_files", "input_files")
        file_list = load_markdown_directory(input_dir)
        if not file_list:
            st.error("No Markdown files found in the `input_files` directory. Aborting.")
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
            progress_bar.progress((i + 1) / len(file_list), text=f"Processing: {file_item['source']}")
        st.success(f"Step 2/4: Source files successfully processed. {len(all_chunks)} chunks created.")

    with st.spinner(f"Step 3/4: Generating embeddings for {len(all_chunks)} chunks..."):
        embedding_model = st.session_state.model_manager.get_model("embedding")
        chunks_with_embeddings = embed_chunks(
            chunks_data=all_chunks,
            model_id=settings.models.embedding_id,
            device=device,
            preloaded_model=embedding_model,
            normalize_embeddings=True
        )
        st.success("Step 3/4: Embeddings successfully generated.")
        st.session_state.model_manager.unload_model("embedding")

    with st.spinner("Step 4/4: Populating the vector database with new data..."):
        create_and_populate_vector_store(
            chunks_with_embeddings=chunks_with_embeddings,
            db_path=settings.database.persist_path,
            collection_name=settings.database.collection_name,
            force_rebuild_collection=False
        )
        st.success("Step 4/4: Vector database successfully populated.")

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
    Contains the complete RAG pipeline logic and uses the ModelManager
    to load and unload models based on the selected VRAM strategy.
    """
    model_manager = st.session_state.model_manager
    strategy = model_manager.get_vram_strategy()
    db_collection = setup_database_connection()

    if not db_collection:
        return "Error: Database collection could not be loaded.", "", user_query

    last_query, last_response = (chat_history[-2]['content'], chat_history[-1]['content']) if len(chat_history) >= 2 and "no relevant information" not in chat_history[-1]['content'] else (None, None)

    condensed_query = user_query
    if enable_conversation_memory and last_query and last_response:
        with st.spinner("🔄 Processing conversation context..."):
            condenser_model = model_manager.get_model("condenser")
            if condenser_model:
                condensed_query = condense_conversation(
                    last_query=last_query, last_response=last_response, new_query=user_query,
                    condenser_model=condenser_model,
                    condenser_generation_config=settings.models.condenser_generation_config,
                )
            if strategy == "Aggressive": model_manager.unload_model("condenser")

    expanded_query = condensed_query
    if enable_query_expansion and len(condensed_query) < settings.pipeline.query_expansion_char_threshold:
        with st.spinner("🔍 Expanding query..."):
            expander_model = model_manager.get_model("expander")
            if expander_model:
                expanded_query = expand_user_query(
                    user_query=condensed_query, char_threshold=settings.pipeline.query_expansion_char_threshold,
                    expander_model=expander_model,
                    query_expander_generation_config=settings.models.query_expander_generation_config,
                )
            if strategy == "Aggressive": model_manager.unload_model("expander")

    with st.spinner(f"📚 Searching documents for: '{expanded_query}'..."):
        embedding_model = model_manager.get_model("embedding")
        if not embedding_model:
             return "Error: Embedding model could not be loaded. Vector database build may have failed.", "", user_query
        query_embedding = embed_query(embedding_model, expanded_query)
        retrieved_docs = query_vector_store(db_collection, query_embedding, retrieval_top_k)
        if strategy in ["Aggressive", "Balanced"]:
            model_manager.unload_model("embedding")

    final_docs = retrieved_docs
    if use_reranker and retrieved_docs:
        with st.spinner("⚖️ Re-ranking document relevance..."):
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
        return "I could not find any relevant information for your query in the manual.", "", user_query

    generation_config = settings.models.llm_generation_config.copy()
    generation_config["temperature"] = temperature
    with st.spinner("✍️ Generating answer..."):
        llm_model = model_manager.get_model("llm")
        if llm_model:
            llm_answer_generator, citation_map = generate_llm_answer(
                user_query=condensed_query, retrieved_chunks=final_docs, llm_model=llm_model,
                llm_generation_config=generation_config
            )
            raw_response = "".join([chunk for chunk in llm_answer_generator])
        else:
            raw_response = "Error: The main response model could not be loaded."
            citation_map = {}

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
        sources_list = [f"- [{source_id}] **{citation_map[source_id]['filename']}**, Section: *{citation_map[source_id]['header']}*" for source_id in sorted(list(used_source_ids)) if source_id in citation_map]
        sources_text = "\n".join(sources_list)

    return formatted_response, sources_text, expanded_query

# --- App Initialization & State Management ---

st.set_page_config(page_title="TOPSIM RAG Chatbot", page_icon="🤖", layout="wide")

# --- Callbacks ---
def on_setting_change(setting_name: str, session_key: str, is_toggle: bool = False):
    """Generic callback to show a toast message when a setting changes."""
    new_value = st.session_state[session_key]
    display_value = "Aktiviert" if new_value else "Deaktiviert" if is_toggle else new_value
    st.toast(f"{setting_name} auf '{display_value}' gesetzt.", icon="✅")

def on_vram_strategy_change():
    """Callback for when the VRAM strategy is changed."""
    st.session_state.model_manager.set_vram_strategy(st.session_state.vram_strategy)
    st.toast(f"VRAM-Strategie auf '{st.session_state.vram_strategy}' geändert. Alle Modelle wurden entladen.", icon="🧠")

def trigger_rebuild():
    st.session_state.rebuild_triggered = True

def exit_app():
    print("\n[ACTION] 'Exit App' clicked. Terminating the process.")
    os._exit(0)

# --- Session State Initialization ---
if "device" not in st.session_state:
    st.session_state.device, st.session_state.device_name = load_processing_device()
if "messages" not in st.session_state:
    st.session_state.messages = []
if "rebuild_triggered" not in st.session_state:
    st.session_state.rebuild_triggered = False
if "session_logger" not in st.session_state:
    st.session_state.session_logger = SessionLogger()
if "model_manager" not in st.session_state:
    st.session_state.model_manager = ModelManager(settings_models=settings.models, device=st.session_state.device)

# --- UI and Pipeline Parameter Initialization ---
VRAM_OPTIONS = ["Performance", "Balanced", "Aggressive"]
if "vram_strategy" not in st.session_state:
    st.session_state.vram_strategy = VRAM_OPTIONS[0]

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

# --- Database Initialization Check ---
db_collection = setup_database_connection()
if not db_collection or st.session_state.rebuild_triggered:
    if not db_collection:
        st.warning("No existing database found. Starting the initial build...")
    elif st.session_state.rebuild_triggered:
        st.info("Manual database rebuild has been triggered...")

    st.cache_resource.clear()
    st.session_state.model_manager.unload_all_models()
    rebuild_database_from_source_files(st.session_state.device)
    st.session_state.rebuild_triggered = False
    st.success("Database successfully (re)built! The page will now reload.")
    time.sleep(3)
    st.rerun()

def reset_chat():
    """Clears the chat history and reruns the app to reflect the change."""
    st.session_state.messages = []
    st.rerun()

# --- UI LAYOUT ---

# --- Left Sidebar ---
with st.sidebar:
    st.title("⚙️ Generation Parameters")
    st.slider(
        "LLM Temperature", min_value=0.0, max_value=1.0,
        value=st.session_state.temperature,
        key="temperature", step=0.05,
        on_change=on_setting_change,
        args=("Temperatur", "temperature"),
        help="Controls the randomness of the model's output. Lower values make it more deterministic."
    )
    st.slider(
        "Retrieval Top-K", min_value=1, max_value=20,
        value=st.session_state.retrieval_top_k,
        key="retrieval_top_k", step=1,
        on_change=on_setting_change,
        args=("Retrieval Top-K", "retrieval_top_k"),
        help="Number of initial documents to retrieve from the vector store."
    )
    st.toggle("Use Reranker", value=st.session_state.use_reranker, key="use_reranker", on_change=on_setting_change, args=("Reranker", "use_reranker", True), help="Enable or disable the reranking step to improve document relevance.")
    st.toggle("Enable Query Expansion", value=st.session_state.enable_query_expansion, key="enable_query_expansion", on_change=on_setting_change, args=("Anfrage-Erweiterung", "enable_query_expansion", True), help="Automatically expand short queries to improve retrieval.")
    st.toggle("Enable Conversation Memory", value=st.session_state.enable_conversation_memory, key="enable_conversation_memory", on_change=on_setting_change, args=("Konversationsgedächtnis", "enable_conversation_memory", True), help="Allow the chatbot to remember the context of the last interaction.")

    st.divider()

    st.title("📦 Gerätestatus & Modelle")
    st.info(f"**Gerät:** {st.session_state.device_name}")
    st.radio(
        "VRAM-Verwaltungsstrategie:", VRAM_OPTIONS,
        key="vram_strategy",
        on_change=on_vram_strategy_change,
        index=VRAM_OPTIONS.index(st.session_state.vram_strategy),
        help=(
            "- **Performance:** Lädt alle Modelle beim ersten Bedarf und behält sie im Speicher.\n"
            "- **Balanced:** Gruppiert Modelle (Retrieval/Generation) und entlädt sie, wenn sie nicht mehr benötigt werden.\n"
            "- **Aggressive:** Lädt jedes Modell nur für seine spezifische Aufgabe und entlädt es sofort wieder."
        )
    )
    # Display Model Status Metrics
    status = st.session_state.model_manager.get_loaded_models_status()
    for model_name, is_loaded in status.items():
        st.metric(
            label=model_name.capitalize(),
            value="Geladen" if is_loaded else "Entladen",
            delta="Aktiv" if is_loaded else "Inaktiv",
            delta_color="normal" if is_loaded else "off"
        )


# --- Main Content Area (3 Columns) ---
main_col, right_sidebar = st.columns([2, 1])

with main_col:
    st.title("🤖 TOPSIM RAG Chatbot")
    # Chat Interface
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Stellen Sie Ihre Frage an das TOPSIM Handbuch..."):
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

with right_sidebar:
    st.title("📋 Logs & Steuerung")

    # App Control Buttons
    st.button("Neuer Chat", on_click=reset_chat, use_container_width=True)
    st.button("Datenbank neu aufbauen", on_click=trigger_rebuild, use_container_width=True)
    st.button("App beenden", on_click=exit_app, type="primary", use_container_width=True)

    st.divider()

    # Session Logs
    st.subheader("Sitzungsprotokolle")
    log_files = SessionLogger.list_log_files()
    if not log_files:
        st.info("Keine Log-Dateien vorhanden.")
    else:
        selected_logs = st.multiselect(
            "Wählen Sie die zu löschenden Protokolle aus:",
            options=log_files,
            format_func=lambda path: path.name,
        )

        if st.button("Ausgewählte Logs löschen", disabled=not selected_logs, use_container_width=True):
            SessionLogger.delete_log_files(selected_logs)
            st.success(f"{len(selected_logs)} Log(s) gelöscht.")
            st.rerun()

        for log in log_files:
            with st.expander(f"{log.name} ({os.path.getsize(log)} Bytes)"):
                try:
                    with open(log, "r", encoding="utf-8") as f:
                        st.text(f.read())
                except Exception as e:
                    st.error(f"Konnte Log-Datei nicht lesen: {e}")