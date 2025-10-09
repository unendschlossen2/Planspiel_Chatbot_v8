from typing import Dict, Any
from llama_cpp import Llama
import streamlit as st
import os
from helper.file_loader import load_prompt_template

# --- Model Caching ---
@st.cache_resource
def load_query_expander_model(model_path: str, generation_config: Dict[str, Any]):
    """
    Lädt das Llama.cpp Modell für den Anfrage-Erweiterer und speichert es im Cache.
    """
    if not os.path.exists(model_path):
        st.warning(f"Query Expander-Modelldatei nicht gefunden unter: {model_path}. Anfrage-Erweiterung wird deaktiviert.")
        return None
    try:
        print(f"Lade Query Expander-LLM von: {model_path}")
        llm = Llama(
            model_path=model_path,
            n_ctx=generation_config.get("n_ctx", 2048),
            n_gpu_layers=generation_config.get("n_gpu_layers", -1),
            verbose=generation_config.get("verbose", False)
        )
        print("Query Expander-LLM erfolgreich geladen.")
        return llm
    except Exception as e:
        st.error(f"Fehler beim Laden des Query Expander-LLM: {e}")
        return None

# --- Hauptfunktion ---
def expand_user_query(
        user_query: str,
        char_threshold: int,
        query_expander_model_path: str,
        query_expander_generation_config: Dict[str, Any]
) -> str:
    """
    Erweitert eine kurze Benutzeranfrage zu einer vollständigen Frage mithilfe eines lokalen LLM.
    """
    if len(user_query) >= char_threshold:
        return user_query

    print(f"Kurze Anfrage erkannt (unter {char_threshold} Zeichen). Erweitere '{user_query}'...")

    # Lade das Query Expander-Modell aus dem Cache
    expander_llm = load_query_expander_model(query_expander_model_path, query_expander_generation_config)
    if expander_llm is None:
        print("Query Expander-Modell nicht verfügbar, überspringe Erweiterung.")
        return user_query

    # Lade das Prompt-Template aus der externen Datei
    prompt_template = load_prompt_template("query_expander.txt")
    if not prompt_template:
        st.error("Konnte das Prompt-Template für den Query Expander nicht laden. Überspringe Erweiterung.")
        return user_query

    # Fülle die Platzhalter im Template
    prompt = prompt_template.format(user_query=user_query)

    try:
        generation_params = {
            "temperature": query_expander_generation_config.get("temperature", 0.7),
            "max_tokens": query_expander_generation_config.get("max_tokens", 128),
        }

        response = expander_llm.create_chat_completion(
            messages=[
                {"role": "system", "content": "Formuliere Stichworte in eine vollständige Frage um."},
                {"role": "user", "content": prompt}
            ],
            **generation_params
        )

        expanded_query = response['choices'][0]['message']['content'].strip()
        print(f"Erweiterte Anfrage: '{expanded_query}'")
        return expanded_query
    except Exception as e:
        print(f"Fehler während der Anfrage-Erweiterung mit lokalem LLM: {e}. Verwende die ursprüngliche Anfrage.")
        return user_query