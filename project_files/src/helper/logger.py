import os
from pathlib import Path
from datetime import datetime
from typing import List

class SessionLogger:
    """
    Eine Klasse zur Verwaltung von Sitzungsprotokollen für den Chatbot.

    Jede Instanz dieser Klasse repräsentiert eine einzelne App-Sitzung und
    ist an eine eindeutige, mit Zeitstempel versehene Protokolldatei gebunden.
    """

    def __init__(self, log_dir: str = "project_files/logs"):
        self.log_dir = Path(log_dir)
        self._ensure_log_dir_exists()

        # Erstelle einen eindeutigen Dateinamen für diese spezifische Sitzung
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.session_log_file = self.log_dir / f"session_log_{timestamp}.txt"

        print(f"Logger initialisiert. Protokolle werden in '{self.session_log_file}' gespeichert.")

    def _ensure_log_dir_exists(self):
        """Stellt sicher, dass das Protokollverzeichnis existiert."""
        self.log_dir.mkdir(parents=True, exist_ok=True)

    def log_interaction(self, user_query: str, final_prompt: str, assistant_response: str):
        """
        Protokolliert eine einzelne Benutzer-Assistent-Interaktion.
        """
        try:
            with open(self.session_log_file, "a", encoding="utf-8") as f:
                f.write("="*50 + "\n")
                f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"User Query: {user_query}\n")
                f.write(f"Final Prompt to LLM: {final_prompt}\n")
                f.write(f"Assistant Response:\n{assistant_response}\n")
                f.write("="*50 + "\n\n")
        except Exception as e:
            print(f"FEHLER beim Schreiben in die Protokolldatei '{self.session_log_file}': {e}")

    @staticmethod
    def list_log_files(log_dir: str = "project_files/logs") -> List[Path]:
        """
        Listet alle .txt-Protokolldateien im angegebenen Verzeichnis auf.
        """
        log_path = Path(log_dir)
        if not log_path.exists():
            return []
        # Sortiert die Dateien, sodass die neuesten zuerst erscheinen
        return sorted(log_path.glob("*.txt"), key=os.path.getmtime, reverse=True)

    @staticmethod
    def delete_log_files(files_to_delete: List[Path]):
        """
        Löscht eine Liste von Protokolldateien.
        """
        for f in files_to_delete:
            try:
                os.remove(f)
                print(f"Protokolldatei gelöscht: {f.name}")
            except Exception as e:
                print(f"FEHLER beim Löschen der Protokolldatei '{f.name}': {e}")