import os
from typing import List, Dict, Optional
from pathlib import Path

def get_input_files_directory() -> str:
    script_file_path = Path(os.path.abspath(__file__))
    src_dir = script_file_path.parent

    if src_dir.name == "src":
        project_root = src_dir.parent
    else:
        project_root = src_dir.parent.parent

    input_dir = project_root / "input_files"
    return str(input_dir)

def ensure_directory_exists(directory_path: str) -> None:
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        print(f"Verzeichnis erstellt: {directory_path}")

def load_markdown_file(file_path: str) -> Optional[Dict[str, str]]:
    if not file_path.endswith(".md"):
        print(f"Datei ist keine Markdown-Datei: {file_path}")
        return None
    if not os.path.exists(file_path):
        print(f"Datei nicht gefunden: {file_path}")
        return None

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        print(f"Markdown-Datei geladen: {os.path.basename(file_path)}")
        return {"source": os.path.basename(file_path), "content": content,}
    except Exception as e:
        print(f"Fehler beim Laden der Markdown-Datei {file_path}: {e}")
        return None

def load_markdown_directory(directory_path: str) -> List[Dict[str, str]]:
    documents = []
    if not os.path.isdir(directory_path):
        print(f"Verzeichnis nicht gefunden: {directory_path}")
        return documents

    # print(f"Lade Markdown-Dateien aus Verzeichnis: {directory_path}")
    for filename in os.listdir(directory_path):
        if filename.endswith(".md"):
            file_path = os.path.join(directory_path, filename)
            doc = load_markdown_file(file_path)
            if doc:
                documents.append(doc)
    print(f"{len(documents)} Markdown-Dokumente geladen.")
    return documents