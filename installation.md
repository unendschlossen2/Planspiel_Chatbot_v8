# Installationsanleitung für den TOPSIM RAG Chatbot (Native LLM Version)

Diese Anleitung führt Sie durch die Einrichtung des Chatbots mit einer lokalen, GPU-beschleunigten Sprachmodell-Engine (via `llama-cpp-python`).

## Vorbereitung: Systemanforderungen

- **Python:** Version 3.9 oder höher.
- **GPU:**
    - **NVIDIA:** Eine CUDA-fähige GPU mit installierten CUDA-Treibern und dem CUDA Toolkit.
    - **AMD (Windows):** Eine ROCm-fähige GPU mit installiertem ROCm SDK (Beta für Windows). Dies ist für fortgeschrittene Benutzer.
- **Git:** Für das Klonen des Repositories.

## Schritt 1: Projekt-Setup und Virtuelle Umgebung

Zuerst richten wir den Projektordner und eine isolierte Python-Umgebung ein.

1.  **Projektordner:** Platzieren Sie den gesamten Projektordner an einem geeigneten Ort (z.B. `C:\Projekte\TOPSIM_Chatbot`).
2.  **Terminal öffnen:** Öffnen Sie ein Terminal (CMD, PowerShell oder Windows Terminal) und navigieren Sie in diesen Projektordner.
    ```bash
    cd C:\Projekte\TOPSIM_Chatbot
    ```
3.  **Virtuelle Umgebung erstellen:**
    ```bash
    python -m venv venv
    ```
4.  **Virtuelle Umgebung aktivieren:**
    ```bash
    .\venv\Scripts\activate
    ```
    Im Terminal sollte nun `(venv)` vor dem Pfad stehen.

## Schritt 2: Python-Abhängigkeiten installieren

Die Installation ist in zwei Phasen unterteilt: Zuerst die GPU-spezifische Bibliothek, dann die restlichen Pakete.

### 2.1 GPU-beschleunigtes LLM-Backend installieren (WICHTIG!)

Dies ist der kritischste Schritt. Wählen Sie den Befehl, der zu Ihrer GPU passt.

**Für NVIDIA (CUDA):**
Führen Sie diesen Befehl aus, um `llama-cpp-python` mit CUDA-Unterstützung zu kompilieren.
```bash
set CMAKE_ARGS="-DLLAMA_CUBLAS=on"
set FORCE_CMAKE=1
pip install --upgrade --force-reinstall llama-cpp-python --no-cache-dir
```

**Für AMD (ROCm auf Windows - Beta):**
Dieser Prozess ist experimentell. Stellen Sie sicher, dass Ihr ROCm-SDK korrekt installiert ist.
```bash
set CMAKE_ARGS="-DLLAMA_HIPBLAS=on"
set FORCE_CMAKE=1
pip install --upgrade --force-reinstall llama-cpp-python --no-cache-dir
```

### 2.2 Standard-Pakete installieren

Nachdem die GPU-Bibliothek installiert ist, installieren Sie die restlichen Abhängigkeiten aus der `requirements.txt`-Datei.
```bash
pip install -r requirements.txt
```

## Schritt 3: Lokales Sprachmodell herunterladen und konfigurieren

Der Chatbot benötigt ein Sprachmodell im GGUF-Format.

1.  **Modell herunterladen:**
    - Besuchen Sie eine Plattform wie [Hugging Face](https://huggingface.co/models?search=gguf).
    - Suchen Sie nach einem passenden Modell (z.B. `Mistral-7B-Instruct`, `Llama-3-8B-Instruct`) im GGUF-Format.
    - Laden Sie eine quantisierte Version herunter (z.B. eine mit `Q4_K_M` im Namen für einen guten Kompromiss aus Leistung und Größe).
2.  **Modell platzieren:**
    - Erstellen Sie einen Ordner für Ihre Modelle, z.B. `C:\Models`.
    - Platzieren Sie die heruntergeladene `.gguf`-Datei in diesem Ordner.
3.  **Konfiguration anpassen:**
    - Öffnen Sie die Datei `project_files/src/helper/config.yaml`.
    - Suchen Sie den Abschnitt `models` und ändern Sie den Wert von `llm_model_path`, sodass er auf Ihre heruntergeladene Modelldatei verweist.
    ```yaml
    models:
      # ... andere Einstellungen
      llm_model_path: "C:/Models/Ihr- heruntergeladenes-Modell.gguf" # WICHTIG: Pfad anpassen!
      # ... andere Einstellungen
    ```

## Schritt 4: Erster Start und Datenbankaufbau

Beim ersten Start muss der Chatbot die Wissensdatenbank verarbeiten und eine Vektor-Datenbank erstellen.

1.  **Konfiguration prüfen:** Öffnen Sie die `config.yaml` und stellen Sie sicher, dass `force_rebuild` auf `true` gesetzt ist.
    ```yaml
    database:
      # ...
      force_rebuild: true
    ```
2.  **Datenbank aufbauen:** Führen Sie die App aus. Dies wird eine Weile dauern, da die Quelldateien verarbeitet und in die Vektor-Datenbank eingebettet werden.
    ```bash
    streamlit run project_files/src/app.py
    ```
    Warten Sie, bis im Terminal keine neuen Meldungen mehr erscheinen und die App im Browser geladen ist.
3.  **Konfiguration zurücksetzen:** Ändern Sie `force_rebuild` in der `config.yaml` zurück auf `false`, damit die Datenbank bei zukünftigen Starts nicht jedes Mal neu erstellt wird.

## Schritt 5: Chatbot ausführen

Nachdem die Ersteinrichtung abgeschlossen ist, können Sie den Chatbot jederzeit mit diesem Befehl starten:
```bash
streamlit run project_files/src/app.py
```
Ein Browser-Tab öffnet sich automatisch unter `http://localhost:8501`.