from typing import List, Dict, Any, Optional, Generator, Tuple
from llama_cpp import Llama
import streamlit as st
import os
from helper.file_loader import load_prompt_template

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
        llm_model: Llama,
        llm_generation_config: Dict[str, Any],
        prompt_template_str: Optional[str] = None
) -> Tuple[Generator[str, None, None], Dict[int, Dict[str, str]]]:

    formatted_context, citation_map = format_retrieved_context(retrieved_chunks)

    # Lade das System-Prompt aus der externen Datei
    system_prompt = load_prompt_template("answer_generator.txt")
    if not system_prompt:
        st.error("Konnte das System-Prompt für den Antwort-Generator nicht laden. Fällt auf Standard-Prompt zurück.")
        system_prompt = "Du bist ein hilfreicher Assistent." # Fallback

    user_prompt_template = """Kontext-Schnipsel:
{context}

Benutzerfrage:
{query}

Antwort:"""
    full_user_prompt = user_prompt_template.format(context=formatted_context, query=user_query)

    if llm_model is None:
        def error_generator():
            yield "Fehler: Das Haupt-Sprachmodell wurde nicht korrekt übergeben."
        return error_generator(), {}

    try:
        print(f"Sende Anfrage an lokales LLM (Streaming aktiviert)...")
        # Extrahiere Generierungsparameter für den Aufruf
        generation_params = {
            "temperature": llm_generation_config.get("temperature", 0.2),
            "max_tokens": llm_generation_config.get("max_tokens", 2048),
        }

        response_generator = llm_model.create_chat_completion(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": full_user_prompt}
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