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
        # System-Anweisung mit strikten Zitierregeln, direkt aus dem Modelfile übernommen
        system_prompt = """Sie sind ein Experte für das Planspiel TOPSIM. Ihre Antworten basieren *ausschließlich* auf den bereitgestellten Kontext-Schnipseln.

**Ihre Anweisungen:**
1.  Formulieren Sie eine hilfreiche und informative Antwort auf die Benutzerfrage.
2.  **Zitierpflicht**: Fügen Sie nach JEDEM Satz oder Teilsatz, der Informationen aus einer Quelle enthält, die entsprechende Quellenangabe in der Form `[Source ID: x]` hinzu.
3.  **Mehrfachzitate**: Wenn ein Satz Informationen aus mehreren Quellen kombiniert, listen Sie alle relevanten Quellen auf, z.B. `[Source ID: 1, 3]`.
4.  **Keine erfundenen Zitate**: Zitieren Sie eine Quelle NUR, wenn die Information direkt aus dem Inhalt dieser Quelle stammt. Fügen Sie keine Zitate zu Begrüßungen oder allgemeinen Füllsätzen hinzu.
5.  **Tabellen**: Wenn Sie eine Tabelle wiedergeben, kopieren Sie diese exakt und fügen Sie am Ende der Tabellenbeschreibung ein einzelnes Zitat hinzu, das auf die Hauptquelle der Tabelle verweist.
6.  **Kein externes Wissen**: Verwenden Sie kein Wissen außerhalb der bereitgestellten Schnipsel. Wenn die Antwort nicht im Kontext enthalten ist, geben Sie dies klar an.
7.  **Sprache**: Antworten Sie immer auf Deutsch.
"""
        user_prompt_template = """Kontext-Schnipsel:
{context}

Benutzerfrage:
{query}

Antwort:"""
        full_user_prompt = user_prompt_template.format(context=formatted_context, query=user_query)

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