# Installationsanleitung für den TOPSIM RAG Chatbot (Native LLM Version)

Diese Anleitung führt Sie durch die Einrichtung des Chatbots mit einer lokalen, GPU-beschleunigten Sprachmodell-Engine (via `llama-cpp-python`).

## Vorbereitung: Systemanforderungen

- **Python:** Version 3.9 oder höher.
- **GPU:**
    - **NVIDIA:** Eine CUDA-fähige GPU. **WICHTIG:** Das NVIDIA CUDA Toolkit muss auf Ihrem System installiert sein.
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
set CMAKE_ARGS="-DGGML_CUDA=on"
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

## Schritt 3: Lokale Sprachmodelle herunterladen und konfigurieren

Der Chatbot benötigt drei separate Sprachmodelle im GGUF-Format: ein Hauptmodell für die Antworten, ein kleines Modell zur Gesprächsverdichtung und ein kleines Modell zur Anfrageerweiterung.

1.  **Modelle herunterladen:**
    - Besuchen Sie eine Plattform wie [Hugging Face](https://huggingface.co/models?search=gguf).
    - **Haupt-LLM:** Suchen Sie nach einem leistungsstarken Modell (z.B. `Mistral-7B-Instruct`, `Llama-3-8B-Instruct`).
    - **Condenser & Expander LLMs:** Suchen Sie nach sehr kleinen, schnellen Modellen (z.B. `TinyLlama-1.1B`, `Qwen2-0.5B`). Für diese Aufgaben ist Geschwindigkeit wichtiger als maximale Qualität.
    - Laden Sie für alle Modelle quantisierte Versionen herunter (z.B. mit `Q4_K_M` im Namen).
2.  **Modelle platzieren:**
    - Erstellen Sie einen Ordner für Ihre Modelle, z.B. `C:\Models`.
    - Platzieren Sie die drei heruntergeladenen `.gguf`-Dateien in diesem Ordner.
3.  **Konfiguration anpassen:**
    - Öffnen Sie die Datei `project_files/src/helper/config.yaml`.
    - Suchen Sie den Abschnitt `models` und ändern Sie die Pfade `llm_model_path`, `condenser_model_path` und `query_expander_model_path`, sodass sie auf Ihre heruntergeladenen Modelldateien verweisen.
    ```yaml
    models:
      # ...
      llm_model_path: "C:/Models/Ihr-Haupt-Modell.gguf"
      # ...
      condenser_model_path: "C:/Models/Ihr-Condenser-Modell.gguf"
      # ...
      query_expander_model_path: "C:/Models/Ihr-Expander-Modell.gguf"
      # ...
    ```

## Schritt 4: Erster Start und Datenbankaufbau

Beim ersten Start muss der Chatbot die Wissensdatenbank verarbeiten und eine Vektor-Datenbank erstellen.

1.  **Konfiguration prüfen:** Öffnen Sie die `config.yaml` und stellen Sie sicher, dass `force_rebuild` auf `true` gesetzt ist.
2.  **Datenbank aufbauen:** Führen Sie die App aus.
    ```bash
    streamlit run project_files/src/app.py
    ```
3.  **Konfiguration zurücksetzen:** Ändern Sie `force_rebuild` in der `config.yaml` zurück auf `false`, damit die Datenbank bei zukünftigen Starts nicht jedes Mal neu erstellt wird.

## Schritt 5: Chatbot ausführen

Nachdem die Ersteinrichtung abgeschlossen ist, können Sie den Chatbot jederzeit mit diesem Befehl starten:
```bash
streamlit run project_files/src/app.py
```
Ein Browser-Tab öffnet sich automatisch unter `http://localhost:8501`.