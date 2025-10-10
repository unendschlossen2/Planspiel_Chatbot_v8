from typing import Dict, Any
from llama_cpp import Llama
import streamlit as st
import os
from helper.file_loader import load_prompt_template

# --- Hauptfunktion ---
def expand_user_query(
        user_query: str,
        char_threshold: int,
        expander_model: Llama,
        query_expander_generation_config: Dict[str, Any]
) -> str:
    """
    Erweitert eine kurze Benutzeranfrage zu einer vollständigen Frage mithilfe eines lokalen LLM.
    """
    if len(user_query) >= char_threshold:
        return user_query

    print(f"Kurze Anfrage erkannt (unter {char_threshold} Zeichen). Erweitere '{user_query}'...")

    if expander_model is None:
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

        response = expander_model.create_chat_completion(
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