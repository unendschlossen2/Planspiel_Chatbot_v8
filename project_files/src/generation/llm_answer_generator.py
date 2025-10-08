from typing import List, Dict, Any, Optional, Generator, Tuple
from llama_cpp import Llama
import streamlit as st
import os

# --- Model Caching ---
@st.cache_resource
def load_llm_model(model_path: str, generation_config: Dict[str, Any]):
    """
    Lädt das Llama.cpp Modell und speichert es im Streamlit Cache.
    """
    if not os.path.exists(model_path):
        st.error(f"Modelldatei nicht gefunden unter: {model_path}")
        return None
    try:
        print(f"Lade lokales LLM von: {model_path}")
        # Wichtige Parameter aus der Config extrahieren
        llm = Llama(
            model_path=model_path,
            n_ctx=generation_config.get("n_ctx", 4096),
            n_gpu_layers=generation_config.get("n_gpu_layers", -1), # -1 für maximale GPU-Auslagerung
            verbose=generation_config.get("verbose", False)
        )
        print("Lokales LLM erfolgreich geladen.")
        return llm
    except Exception as e:
        st.error(f"Fehler beim Laden des lokalen LLM: {e}")
        return None

# --- Unveränderte Hilfsfunktion ---
def format_retrieved_context(retrieved_chunks: List[Dict[str, Any]]) -> Tuple[str, Dict[int, Dict[str, str]]]:
    if not retrieved_chunks:
        return "Kein relevanter Kontext wurde aus dem Handbuch abgerufen.", {}

    context_str = ""
    citation_map = {}
    for i, chunk in enumerate(retrieved_chunks):
        source_id = i + 1
        context_str += f"[Source ID: {source_id}]\n"
        metadata = chunk.get('metadata', {})
        document_content = chunk.get('document', "N/A")
        source_filename = metadata.get('source_filename', 'Unbekannte Quelle')
        header_text = metadata.get('header_text', 'N/A')
        context_str += f"  Quelldatei: {source_filename}\n"
        context_str += f"  Spezifische Überschrift: {header_text}\n"
        context_str += f"  Inhalt: {document_content}\n---\n"
        citation_map[source_id] = {
            "filename": source_filename,
            "header": header_text
        }
    return context_str, citation_map

# --- Angepasste Hauptfunktion ---
def generate_llm_answer(
        user_query: str,
        retrieved_chunks: List[Dict[str, Any]],
        llm_model_path: str,
        llm_generation_config: Dict[str, Any],
        prompt_template_str: Optional[str] = None
) -> Tuple[Generator[str, None, None], Dict[int, Dict[str, str]]]:

    formatted_context, citation_map = format_retrieved_context(retrieved_chunks)

    if prompt_template_str:
        current_prompt_template = prompt_template_str
    else:
        # Ein robustes Standard-Template
        current_prompt_template = """Du bist ein hilfreicher Assistent. Deine Aufgabe ist es, die folgende Benutzerfrage ausschließlich basierend auf den unten stehenden Kontext-Schnipseln aus einem Handbuch zu beantworten.
- Zitiere relevante Quellen direkt in deiner Antwort, indem du `[Source ID: X]` am Ende des Satzes oder Absatzes verwendest, der die Information aus der Quelle enthält.
- Wenn die Antwort nicht in den Kontext-Schnipseln enthalten ist, antworte: 'Ich konnte leider keine relevanten Informationen zu Ihrer Anfrage im Handbuch finden.'

Kontext-Schnipsel:
{context}

Benutzerfrage:
{query}

Antwort:"""

    full_prompt = current_prompt_template.format(context=formatted_context, query=user_query)

    # Lade das Modell aus dem Cache (oder erstelle es, falls es nicht vorhanden ist)
    llm = load_llm_model(llm_model_path, llm_generation_config)

    if llm is None:
        def error_generator():
            yield "Fehler: Das Sprachmodell konnte nicht geladen werden. Bitte überprüfen Sie den Pfad und die Konfiguration."
        return error_generator(), {}

    try:
        print(f"Sende Anfrage an lokales LLM (Streaming aktiviert)...")
        # Extrahiere Generierungsparameter für den Aufruf
        generation_params = {
            "temperature": llm_generation_config.get("temperature", 0.2),
            "max_tokens": llm_generation_config.get("max_tokens", 2048),
        }

        response_generator = llm.create_chat_completion(
            messages=[
                {"role": "system", "content": "Du bist ein hilfreicher Assistent."},
                {"role": "user", "content": full_prompt}
            ],
            stream=True,
            **generation_params
        )

        def content_generator():
            for chunk in response_generator:
                delta = chunk.get('choices', [{}])[0].get('delta', {})
                content = delta.get('content')
                if content:
                    yield content

        return content_generator(), citation_map

    except Exception as e:
        error_message = f"\nFehler bei der Interaktion mit dem lokalen LLM: {e}"
        def error_generator():
            yield error_message
        return error_generator(), {}