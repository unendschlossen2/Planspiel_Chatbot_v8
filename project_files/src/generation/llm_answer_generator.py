from typing import List, Dict, Any, Optional, Generator, Tuple
import ollama
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def format_retrieved_context(retrieved_chunks: List[Dict[str, Any]]) -> Tuple[str, Dict[int, Dict[str, str]]]:
    if not retrieved_chunks:
        return "Kein relevanter Kontext wurde aus dem Handbuch abgerufen.", {}

    context_str = ""
    citation_map = {}
    for i, chunk in enumerate(retrieved_chunks):
        source_id = i + 1
        # Maschinenlesbarer Tag für das LLM
        context_str += f"[Source ID: {source_id}]\n"

        metadata = chunk.get('metadata', {})
        document_content = chunk.get('document', "N/A")

        source_filename = metadata.get('source_filename', 'Unbekannte Quelle')
        header_text = metadata.get('header_text', 'N/A')

        context_str += f"  Quelldatei: {source_filename}\n"
        context_str += f"  Spezifische Überschrift: {header_text}\n"
        context_str += f"  Inhalt: {document_content}\n---\n"

        # Befüllen der Citation Map
        citation_map[source_id] = {
            "filename": source_filename,
            "header": header_text
        }

    return context_str, citation_map

def generate_llm_answer(
        user_query: str,
        retrieved_chunks: List[Dict[str, Any]],
        ollama_model_name: str,
        ollama_host: Optional[str] = None,
        ollama_options: Optional[Dict[str, Any]] = None,
        prompt_template_str: Optional[str] = None
) -> Tuple[Generator[str, None, None], Dict[int, Dict[str, str]]]:

    formatted_context, citation_map = format_retrieved_context(retrieved_chunks)

    if prompt_template_str:
        current_prompt_template = prompt_template_str
    else:
        # Vereinfachtes Template, da die Hauptanweisungen im Modelfile stehen
        current_prompt_template = """Kontext-Schnipsel:
{context}

Benutzerfrage:
{query}

Antwort:"""

    full_prompt = current_prompt_template.format(context=formatted_context, query=user_query)

    try:
        client_args = {}
        if ollama_host:
            client_args['host'] = ollama_host

        client = ollama.Client(**client_args)

        print(f"Sende Anfrage an Ollama-Modell '{ollama_model_name}' (Streaming aktiviert)...")
        response_generator = client.chat(
            model=ollama_model_name,
            messages=[{'role': 'user', 'content': full_prompt}],
            stream=True,
            options=ollama_options
        )

        def content_generator():
            for chunk in response_generator:
                if 'content' in chunk['message']:
                    yield chunk['message']['content']

        # Gibt den Generator und die Citation Map zurück
        return content_generator(), citation_map

    except ollama.ResponseError as e:
        error_message = f"\nOllama API Antwortfehler für Modell '{ollama_model_name}': {e.status_code} - {e.error}"
        def error_generator():
            yield error_message
        return error_generator(), {}
    except Exception as e:
        error_message = f"\nFehler bei der Interaktion mit dem Ollama-Modell '{ollama_model_name}': {e}"
        def error_generator():
            yield error_message
        return error_generator(), {}