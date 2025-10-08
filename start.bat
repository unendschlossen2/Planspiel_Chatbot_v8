@echo off
TITLE TOPSIM RAG Chatbot Launcher

echo Aktiviere die virtuelle Python-Umgebung...
call .\\venv\\Scripts\\activate.bat

if not defined VIRTUAL_ENV (
    echo FEHLER: Konnte die virtuelle Umgebung nicht aktivieren.
    pause
    exit /b
)

echo Virtuelle Umgebung erfolgreich aktiviert.
echo.

echo Starte die grafische Benutzeroberflaeche (GUI)...
echo Schliessen Sie dieses Fenster, um den Chatbot zu beenden.
streamlit run ./project_files/src/app.py

echo Deaktiviere die virtuelle Umgebung...
call .\\venv\\Scripts\\deactivate.bat

echo.
echo Launcher beendet.
pause
