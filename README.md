# Anleitung Installation
Hinweis: Kommandos am besten kopieren und nicht abtippen.

Es werden mindestens 10 GB freier Speicher benötigt.
Eine schnelle & stablie Internetverbindung sind empfehlenswert.

Optional: Ollama installieren (https://ollama.com/) und Modell herunterladen (`ollama run gemma3` oder anderes Modell, s. https://github.com/ollama/ollama?tab=readme-ov-file#model-library)

## Windows

### Git installieren
Falls noch nicht vorhanden
- von https://git-scm.com/downloads herunterladen
- installieren, die Standardeinstellungen sollten OK sein

### IDE installieren
Wir empfehlen VS Code
- auf https://code.visualstudio.com/download herunterladen
- installieren, die Standardeinstellungen sollten OK sein

### Dieses Projekt clonen
- VS Code öffnen
- "Clone Git Repository..." auswählen
- als Url "https://github.com/Tryner/aml.git" angeben und clonen
- Ordner auswählen in den das Repo geclont werden soll
- Projekt öffnen und den Autoren vertrauen

### Python installieren
Falls noch nicht vorhanden
- auf https://www.python.org/downloads/ herunterladen
- beide Haken setzten und installieren

Installation überprüfen
- VS öffnen
- Terminal -> New Terminal
    - Terminal findet ihr ganz oben, irgendwo auf der linken Seite, zwischen "Run" und "Help". Ggf. ist "Terminal" hinter drei Punkten oder horizontalen Strichen versteckt.
    - Es sollte sich unten ein Terminal öffnen.
- `python --version` ausführen
- Falls das Kommando nicht erkannt wird neustarten und erneut "python --version" versuchen
    - Wurden wiklich beide Haken bei der Installation gesetzt?

### uv installieren
- Windows Powershell öffnen
- `powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"` ausführen
    - (siehe https://docs.astral.sh/uv/getting-started/installation/#standalone-installer)

### Python Umgebung einrichten
Terminal öffnen (siehe oben)
- Terminal sollte im Ordner "aml" geöffnet sein
- venv erstellen `uv venv --python 3.13`
- venv aktivieren `.venv\Scripts\activate`
    - Links vor "C:" sollte jetzt "aml" stehen
- Abhängigkeiten instllieren `uv pip sync requirements.txt`
    - Dies kann etwas dauern
- Projekt installieren `uv pip install -e . --no-deps`

### Vorbereitungsnotebook ausführen
"vorbereitung.ipynb" ausführen
- "Run All" auswählen
- Kernel auswählen
    - "Python Environments..." -> "aml [...]"

- Es zu einer Fehlermeldung und folgendem Hinweis kommen:
> Microsoft Visual C++ Redistributable is not installed, this may lead to the DLL load failure.
> It can be downloaded at https://aka.ms/vs/16/release/vc_redist.x64.exe

- Die betreffende .exe nachinstallieren, Restart (Kernel) und wieder Run All.
- Die Ausführung kann etwas dauern
- Wenn ihr eine Nvidia-GPU habt, sollte diese in der Zweiten Zelle erkannt werden
- Wenn "Success!" ausgegeben wird ist alles gut gelaufen.
- Fertig

## Andere Betriebssysteme
Ist für Windows getestet, funktioniert auf anderen Betriebssystem ehr einfacher.

### Dinge installieren
Genau so wie für Windows müsst ihr ein paar Dinge installieren (falls noch nicht vorhanden)
- Git (https://git-scm.com/downloads)
- Python (https://www.python.org/downloads/)
- uv (https://docs.astral.sh/uv/getting-started/installation/)
- VS Code (https://code.visualstudio.com/download)

### Projekt clonen
Genau wie bei Windows

### Python Umgebung einrichten
Ähnlich wie bei Windows

Terminal in VS Code öffnen
- Terminal sollte im Ordner "aml" geöffnet sein
- venv erstellen `uv venv --python 3.13`
- venv aktivieren `source .venv\bin\activate`
    - Die venv "aml" sollte in der Kommandozeile auftauchen
- Abhängigkeiten instllieren `uv pip sync requirements.txt`
    - Dies kann etwas dauern
- Projekt installieren `uv pip install -e . --no-deps`

### Vorbereitungsnotebook ausführen
Wie bei Windows
- Bei Macbooks sollte "Apple NPU available: True" bei der zweiten Zelle ausgegeben werden.