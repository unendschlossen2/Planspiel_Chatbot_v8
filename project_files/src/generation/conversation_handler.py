from typing import Optional, Dict, Any
from llama_cpp import Llama
import streamlit as st
import os

# --- Model Caching ---
@st.cache_resource
def load_condenser_model(model_path: str, generation_config: Dict[str, Any]):
    """
    Lädt das Llama.cpp Modell für den Konversations-Verdichter und speichert es im Cache.
    """
    if not os.path.exists(model_path):
        st.warning(f"Condenser-Modelldatei nicht gefunden unter: {model_path}. Konversationsgedächtnis wird deaktiviert.")
        return None
    try:
        print(f"Lade Condenser-LLM von: {model_path}")
        llm = Llama(
            model_path=model_path,
            n_ctx=generation_config.get("n_ctx", 2048),
            n_gpu_layers=generation_config.get("n_gpu_layers", -1),
            verbose=generation_config.get("verbose", False),
            logits_all=True # Notwendig für manche Modelle, um Logits zu erzeugen
        )
        print("Condenser-LLM erfolgreich geladen.")
        return llm
    except Exception as e:
        st.error(f"Fehler beim Laden des Condenser-LLM: {e}")
        return None

# --- Hauptfunktion ---
def condense_conversation(
        last_query: Optional[str],
        last_response: Optional[str],
        new_query: str,
        condenser_model_path: str,
        condenser_generation_config: Dict[str, Any]
) -> str:
    """
    Verdichtet den Konversationsverlauf zu einer eigenständigen Anfrage, falls die neue Anfrage eine Folgefrage ist.
    Nutzt ein lokales Llama.cpp Modell.
    """
    if not last_query or not last_response:
        return new_query

    print("Analysiere Konversationsverlauf auf Folgefragen...")

    # Lade das Condenser-Modell aus dem Cache
    condenser_llm = load_condenser_model(condenser_model_path, condenser_generation_config)
    if condenser_llm is None:
        print("Condenser-Modell nicht verfügbar, überspringe Verdichtung.")
        return new_query

    # Erstelle den Prompt für das Condenser-Modell
    prompt = f"""Du bist ein KI-Assistent, der darauf spezialisiert ist, eine Konversation in eine einzelne, eigenständige Suchanfrage umzuformulieren.
- Wenn die "Neue Anfrage" eine direkte Folgefrage zur "Letzten Antwort" ist, kombiniere die Informationen zu einer neuen, klaren Frage.
- Wenn die "Neue Anfrage" ein völlig neues Thema ist, gib NUR die "Neue Anfrage" unverändert zurück.

Letzte Anfrage: {last_query}
Letzte Antwort: {last_response}
Neue Anfrage: {new_query}

Eigenständige Suchanfrage:"""

    try:
        generation_params = {
            "temperature": condenser_generation_config.get("temperature", 0.0),
            "max_tokens": condenser_generation_config.get("max_tokens", 256),
        }

        response = condenser_llm.create_chat_completion(
            messages=[
                {"role": "system", "content": "Formuliere die Konversation in eine eigenständige Suchanfrage um."},
                {"role": "user", "content": prompt}
            ],
            **generation_params
        )

        condensed_query = response['choices'][0]['message']['content'].strip()

        if condensed_query != new_query:
            print(f"Folgefrage erkannt. Neue, eigenständige Anfrage: '{condensed_query}'")
        else:
            print("Neues Thema erkannt. Verwende Anfrage wie sie ist.")

        return condensed_query

    except Exception as e:
        print(f"Fehler während der Konversationsverdichtung mit lokalem LLM: {e}. Verwende ursprüngliche neue Anfrage.")
        return new_query