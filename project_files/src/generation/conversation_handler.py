from typing import Optional, Dict, Any
from llama_cpp import Llama
import streamlit as st
import os
from helper.file_loader import load_prompt_template

# --- Hauptfunktion ---
def condense_conversation(
        last_query: Optional[str],
        last_response: Optional[str],
        new_query: str,
        condenser_model: Llama,
        condenser_generation_config: Dict[str, Any]
) -> str:
    """
    Verdichtet den Konversationsverlauf zu einer eigenständigen Anfrage, falls die neue Anfrage eine Folgefrage ist.
    Nutzt ein lokales Llama.cpp Modell.
    """
    if not last_query or not last_response:
        return new_query

    print("Analysiere Konversationsverlauf auf Folgefragen...")

    if condenser_model is None:
        print("Condenser-Modell nicht verfügbar, überspringe Verdichtung.")
        return new_query

    # Lade das Prompt-Template aus der externen Datei
    prompt_template = load_prompt_template("conversation_condenser.txt")
    if not prompt_template:
        st.error("Konnte das Prompt-Template für den Conversation Condenser nicht laden. Überspringe Verdichtung.")
        return new_query

    # Fülle die Platzhalter im Template
    prompt = prompt_template.format(last_query=last_query, last_response=last_response, new_query=new_query)

    try:
        generation_params = {
            "temperature": condenser_generation_config.get("temperature", 0.0),
            "max_tokens": condenser_generation_config.get("max_tokens", 256),
        }

        response = condenser_model.create_chat_completion(
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