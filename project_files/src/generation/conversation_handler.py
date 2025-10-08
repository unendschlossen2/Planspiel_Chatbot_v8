import subprocess
import sys
import time
from typing import Optional
import ollama

def force_unload_ollama_model(model_name: str):
    """
    Forcefully unloads a model by calling the Ollama CLI directly.
    This is the most reliable method to ensure VRAM is freed.
    """
    try:
        command = ["ollama", "stop", model_name] # Note: Using "stop" as requested.
        # The official command might be different,
        # but we will use what you confirmed works.

        # Platform-specific setup to hide the console window on Windows
        startupinfo = None
        if sys.platform == "win32":
            startupinfo = subprocess.STARTUPINFO()
            startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW

        # Execute the command silently
        subprocess.run(
            command,
            check=False, # Don't raise an error if the command fails
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            startupinfo=startupinfo
        )
        print(f"CLI unload command sent for model: {model_name}")

    except FileNotFoundError:
        # This happens if 'ollama' is not in the system's PATH for the script
        print(f"WARNUNG: Der Befehl 'ollama' wurde nicht gefunden. Das Modell '{model_name}' konnte nicht entladen werden.")
    except Exception as e:
        # Catch any other potential errors from subprocess
        print(f"Fehler beim Ausführen des unload-Befehls für '{model_name}': {e}")

def condense_conversation(
        model_name: str,
        last_query: Optional[str],
        last_response: Optional[str],
        new_query: str,
        ollama_host: Optional[str] = None
) -> str:
    """
    Condenses the conversation history into a standalone query if the new query is a follow-up.
    """
    # If there is no history, there's nothing to condense.
    if not last_query or not last_response:
        return new_query

    print("Analysiere Konversationsverlauf auf Folgefragen...")

    # Construct the prompt for the condenser model
    prompt = f"""Letzte Anfrage: {last_query}
Letzte Antwort: {last_response}
Neue Anfrage: {new_query}"""

    try:
        client_args = {}
        if ollama_host:
            client_args['host'] = ollama_host
        client = ollama.Client(**client_args)

        response = client.chat(
            model=model_name,
            messages=[{'role': 'user', 'content': prompt}],
            stream=False,
        )
        condensed_query = response['message']['content'].strip()

        if condensed_query != new_query:
            print(f"Folgefrage erkannt. Neue, eigenständige Anfrage: '{condensed_query}'")
        else:
            print("Neues Thema erkannt. Verwende Anfrage wie sie ist.")

        return condensed_query

    except Exception as e:
        print(f"Fehler während der Konversationsverdichtung: {e}. Verwende ursprüngliche neue Anfrage.")
        return new_query