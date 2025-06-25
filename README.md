# Windows
Ich empfehele die Kommandos aus den Anführungszeichen zu kopieren und einzufügen, abtippen führt schnell zu Fehlern.

## Python 3.12 installieren
Hinweis: Python 3.13 funktioniert nicht. Alte Python Version ggf. vorher deinstallieren.

Python Installer downloaden: https://www.python.org/downloads/release/python-3127/ (Die Downloads sind relativ weit unten unter "Files". "Windows installer (64-bit)" auswählen.)

Python Installer ausführen. Im ersten Menü beide Haken (Use admin privileges [...] & Add python.exe to PATH) setzen und "Install Now".

Erfolg überprüfen: Cmd.exe öffnen (Mit Windows nach cmd oder Eingabeaufforderung suchen) und "python --version" ausführen (kopieren, einfügen und "Enter" drücken) -> sollte "Python 3.12.7" ausgeben

Bei Problemen:
- Nachschauen, ob Python istalliert wurde
- Windows neu starten (hilft tatsächlich häufig).
- Deinstallieren, neu starten, installieren, neu starten.
- Python manuell zu PATH hinzufügen

## Git - dieses Repository clonen
Git installieren: https://git-scm.com/downloads (Standardeinstellungen sind ok)

Einen Ordner anlegen oder auswählen, in den das Repo geclont werden soll.

In dem Ordner Rechtsclick -> open in Git Bash (Windows 11: ist möglicherweise hinter "weitere Optionen" oder so versteckt)

In Git Bash: "git clone https://github.com/Tryner/aml_wa24.git" ausführen.

Jetzt sollte in dem gewählten Ordner ein neuer Ordner "aml_wa24" auftauchen.

## VS Code installieren & Projekt in VS Code öffnen
VS Code installieren: https://code.visualstudio.com/Download (Standardeinstellungen sind ok)

Projekt in VS Code öffnen: File -> Open Folder -> Ordner aml_wa24 auswählen

Falls gefragt wird ob ihr den Autoren vertraut: Ja.

Unten rechts sollte vorgeschlagen werden Extensions zu installieren, dies akzeptieren. Ggf. einen kleinen Moment warten.

## Notwendige packages installieren
In VS Code Terminal öffnen: Terminal -> new Terminal (Terminal findet ihr irgendwo oben links, ggf. unter ...)

Dies sollte eine Kommandozeile öffnen. In dieser Folgendes ausführen:
- venv erstellen: "python -m venv venv"
- Hinweis: Sollte "We noticed a new environment has been created. Do you want to select it for the workspace folder?" aufploppen könnt ihr "Yes" auswählen.
- Hinweis: Sollte "python" nicht als Kommando erkannt werden VS Code einmal schließen und neu öffnen.
- venv aktivieren: "venv\Scripts\activate.bat"
- Am Anfang der Commandozeile sollte nun "(venv)" stehen. Falls nicht: nicht fortfahren.
- pip aktualisieren: "python -m pip install --upgrade pip"
- pip tools installieren: "python -m pip install pip-tools"
- Packages installieren: "pip-sync requirements.txt". 
- Der vorherige Schritt kann durchaus über 15 min dauern. Es werden mehrere Gigabyte heruntergeladen. Auch ohne Download dauert die Installation mehr als eine Minute. 

## Vorbereitungs Notebook ausführen
Wenn alles geklappt hat solltet ihr jetzt das Vorbereitungsnotebook ausführen können.

Das Notebook findet ihr in dem Ordner "notebooks" und heißt "vorbereitung.ipynb". Den File-Explorer findet ihr in VS-Code auf der linken Seite, ganz oben direkt unter dem VS Code Logo.

"Run All" drücken.

Es sollte ein Dialog aufpoppen wo ihr den Kernel auswählen müsst. Hier zuert "Python" und dann "venv" (ggf. mit Stern) auswählen. Die Ausführung dauert etwas, es werden mehrere Gigabyte heruntergeladen.

Es sollten ein Haufen Zahlen, vermutlich Ladebalken und am Ende "Success" ausgegeben werden.

Fertig!

# Linux
Ihr müsst grundsätzlich die selben Schritte befolgen wie für Windows. Ist teilweise etwas tricky und kommt auf eure Distro an. Bekommt ihr irgendwie hin.

- Python 3.12 installieren (Soll etwas schwiergig sein, bei meinem aktuellen Ubuntu LTS wird Python 3.12 mitgeliefert. Ggf. müsst ihr Python 3.12 parallel zu der bestehenden Python Version installieren. Sonst könnt ihr auch einfach versuchen mit der bestehenden Python Version zu arbeiten.)
- Github Repository clonen
- Projekt mit VS Code öffnen
- Requirements installieren (Die requirements.txt funktioniert nur für Windows, stattdessen "make update" ausführen. Vermutlich müsst ihr für hdbscan noch Python dev tools nachinstallieren.)
- Notebook vorbereitung.ipynb ausführen

# MacOS
Sollte analog zu Linux funktionieren. Hab ich nicht und kann euch folglich wenig weiterhelfen. Wenn es nicht funktioniert müsst ihr wohl Colab nutzen.
