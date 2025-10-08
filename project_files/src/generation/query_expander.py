from typing import Optional
import ollama
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def expand_user_query(
        user_query: str,
        model_name: str,
        char_threshold: int,
        ollama_host: Optional[str] = None
) -> str:
    """
    Erweitert eine kurze Benutzeranfrage oder ein Stichwort zu einer vollständigen Frage mithilfe eines spezialisierten, schnellen LLM.
    Die Erweiterung wird nur ausgelöst, wenn die Anfrage unterhalb des Zeichenlimits liegt.
    """
    if len(user_query) >= char_threshold:
        return user_query

    print(f"Kurze Anfrage erkannt (unter {char_threshold} Zeichen). Erweitere '{user_query}'...")

    try:
        client_args = {}
        if ollama_host:
            client_args['host'] = ollama_host
        client = ollama.Client(**client_args)

        response = client.chat(
            model=model_name,
            messages=[{'role': 'user', 'content': user_query}],
            stream=False,
        )
        expanded_query = response['message']['content'].strip()
        print(f"Erweiterte Anfrage: '{expanded_query}'")
        return expanded_query
    except Exception as e:
        print(f"Fehler während der Anfrage-Erweiterung: {e}. Verwende die ursprüngliche Anfrage.")
        return user_query