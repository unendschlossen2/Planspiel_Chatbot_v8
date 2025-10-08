import huggingface_hub
from huggingface_hub import login
from typing import Optional

def attempt_huggingface_login(token: Optional[str] = None) -> bool:
    if not huggingface_hub.whoami():
        if token:
            try:
                login(token=token, add_to_git_credential=False)
                print("Hugging Face Login mit bereitgestelltem Token erfolgreich.")
                return True
            except Exception as e:
                print(f"Hugging Face Login mit bereitgestelltem Token fehlgeschlagen: {e}")
                return False
        else:
            print("Kein Hugging Face Token bereitgestellt. Überspringe tokenbasierten Login.")
            print("Sie können sich interaktiv anmelden, falls in einer passenden Umgebung, oder durch Setzen der HUGGING_FACE_HUB_TOKEN Umgebungsvariable.")
            return False
    else:
        print("Bereits bei Hugging Face angemeldet.")
        return True