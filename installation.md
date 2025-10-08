Vorbereitung: Basis-Modelle herunterladen

Bevor Sie die spezialisierten Chatbot-Modelle erstellen, müssen die Basis-LLMs von Ollama heruntergeladen werden. Die hier verwendeten Modelle sind gemma3n:e4b und qwen3:1.7b. Führen Sie die folgenden Befehle in Ihrem Terminal aus:

ollama pull gemma3n:e4b
ollama pull qwen3:1.7b

Schritt 1: Projekt-Setup und Virtuelle Umgebung

Zuerst richten wir den Projektordner und eine isolierte Python-Umgebung ein, um Konflikte mit anderen Projekten zu vermeiden.

Projektordner: Platzieren Sie den gesamten Projektordner an einem geeigneten Ort auf Ihrem Computer (z. B. C:\Projekte\TOPSIM_Chatbot).

Terminal öffnen: Öffnen Sie ein Terminal (CMD, PowerShell oder Windows Terminal) und navigieren Sie in diesen Projektordner.
cd C:\Projekte\TOPSIM_Chatbot

Virtuelle Umgebung erstellen:
python -m venv venv

Virtuelle Umgebung aktivieren:
.\venv\Scripts\activate

Im Terminal sollte nun (venv) vor dem Pfad stehen.

Schritt 2: Ollama installieren und einrichten

Ollama ist die Software, die die lokalen Sprachmodelle ausführt.

Ollama herunterladen: Besuchen Sie ollama.com und laden Sie die Windows-Version herunter.

Installation: Führen Sie das Installationsprogramm aus. Ollama wird als Hintergrunddienst eingerichtet.

Überprüfung: Öffnen Sie ein neues Terminal und geben Sie "ollama --version" ein, um zu bestätigen, dass die Installation erfolgreich war.

Schritt 3: Python-Abhängigkeiten installieren

Wir installieren nun alle notwendigen Python-Pakete, einschließlich des wichtigen Fixes für pydantic.

Standard-Pakete installieren: Stellen Sie sicher, dass Ihre virtuelle Umgebung aktiv ist, und führen Sie den folgenden Befehl aus:
pip install -r requirements.txt

Pydantic-Konflikt beheben (Wichtig!): Um den bekannten "NameError: name 'sys' is not defined"-Fehler zu beheben, müssen wir bestimmte Versionen von pydantic und typer erzwingen. Führen Sie die folgenden Befehle nacheinander aus:
pip uninstall -y pydantic pydantic_core typer

Warten Sie, bis der Befehl abgeschlossen ist, und führen Sie dann den nächsten aus:
pip install pydantic==2.5.3

Und schließlich:
pip install typer==0.9.0

Schritt 4: Lokale LLM-Modelle erstellen

Der Chatbot verwendet mehrere spezialisierte Modelle. Wir müssen Ollama anweisen, diese aus den Konfigurationsdateien (Modelfile) zu erstellen.

Haupt-Expertenmodell erstellen:
ollama create topsim-expert-v1 -f ./TopsimExpert.Modelfile

Konversations-Verdichter erstellen:
ollama create condenser-v1 -f ./ConversationCondenser.Modelfile

Anfrage-Erweiterer erstellen:
ollama create query-expander-v1 -f ./QueryExpander.Modelfile

Schritt 5: Erster Start und Datenbankaufbau

Beim ersten Start muss der Chatbot die Wissensdatenbank (Topsim Handbuch Markdown.md) verarbeiten und eine Vektor-Datenbank erstellen.

Konfiguration prüfen: Öffnen Sie die Datei config.yaml im src/helper/-Verzeichnis. Stellen Sie sicher, dass force_rebuild auf true gesetzt ist.

Datenbank aufbauen: Führen Sie das Hauptskript aus. Dies wird eine Weile dauern.
python src/temp_main.py

Warten Sie, bis der Prozess abgeschlossen ist und die Meldung "TOPSIM RAG Chatbot Bereit" erscheint. Sie können das Programm danach mit "quit" beenden.

Konfiguration zurücksetzen: Ändern Sie force_rebuild in der config.yaml zurück auf false, damit die Datenbank bei zukünftigen Starts nicht jedes Mal neu erstellt wird.

Schritt 6: Chatbot ausführen

Sie können den Chatbot nun entweder über das Terminal oder über die grafische Benutzeroberfläche starten.

Terminal-Version (CLI):
python src/temp_main.py

Grafische Benutzeroberfläche (GUI):
streamlit run app.py

Ein Browser-Tab öffnet sich automatisch unter http://localhost:8501.

** KI-Generiert