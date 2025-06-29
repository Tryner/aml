Python installieren (aktuelle Version)
Git installieren
VS Code (IDE) installieren
uv installieren

Projekt clonen
Projekt mit VS Code öffnen

venv erstellen "uv venv --python 3.13" (Welche Python version?)
venv aktivieren ".venv\Scripts\activate"
Abhängigkeiten instllieren "uv pip sync requirements.txt"
Projekt installieren "uv pip install -e . --no-deps"

Vorbereitung.ipynb ausführen
Hierbei kann es zu einer Fehlermeldung und folgendem Hinweis kommen:
> Microsoft Visual C++ Redistributable is not installed, this may lead to the DLL load failure.
> It can be downloaded at https://aka.ms/vs/16/release/vc_redist.x64.exe
Die betreffende .exe nachinstallieren, restart kernel und wieder run all.