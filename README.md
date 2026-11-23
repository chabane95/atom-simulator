# atom-simulator

Vollständige Wasserstofforbitale, goldene 3D-Punktwolken und **Bohmscher Teilchenfluss**.
Die sichtbare Dichte und die Bewegung werden im eigenen Physikkern berechnet.
Der Renderer übernimmt ausschließlich Darstellung, Licht, Kamera und Ausschnitt.

![Goldene Orbitaldarstellung](docs/preview.png)

## Starten und einstellen

Unter Windows `Start.cmd` doppelt anklicken. Die bereits eingerichtete Python-Umgebung
wird benutzt. Für eine frische Installation: Python 3.11 oder neuer und OpenGL 3.3.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[all]"
.\.venv\Scripts\python.exe -m atom_simulator view
```

**Die festen Einstellungen stehen in [config.py](src/atom_simulator/config.py).**
Dort lassen sich `N`, `L`, `M`, Punktzahl, Farben, Auflösung, Dauer und die vier
Produktionsfälle direkt bearbeiten. Die Ausgangswerte sind ausdrücklich aufgeschrieben.
Für einen anderen Zustand ist kein Umbau des Programms nötig.

| Eingabe im Fenster | Wirkung |
| --- | --- |
| Maus ziehen / Mausrad | Ansicht drehen / zoomen |
| Leertaste | Bohmschen Fluss pausieren / fortsetzen |
| A | Langsame Kameradrehung ein-/ausschalten |
| C | Geometrischen Ausschnitt umschalten |
| F | Vorzeichen von m und damit Flussrichtung wechseln |
| B | Heller / schwarzer Hintergrund |
| R | Kamera und Zeit zurücksetzen |
| 1 / 2 / 3 / 4 | Die vier unten aufgeführten Flussfälle |
| Esc | Schließen |

Alternativ lassen sich Werte beim Start überschreiben:

```powershell
.\.venv\Scripts\python.exe -m atom_simulator view --n 3 --l 2 --m 2 --time-rate 14
```

Zulässig sind `1 <= n <= 12`, `0 <= l < n`, `-l <= m <= l`.
Bei `m=0` stehen die Bohmschen Positionen in diesem spinlosen Eigenzustandsmodell still.
Jedes Orbital wird auf seinen eigenen Radius eingepasst: Bildschirmgrößen verschiedener
Zustände sind kein gemeinsamer physikalischer Größenmaßstab.

## Fertige Medien und erneutes Rendern

Die lokalen Medien stehen unter `outputs/`; dieser große Ordner wird nicht zu GitHub hochgeladen.

| Ordner / Datei | Inhalt |
| --- | --- |
| `outputs/Bilder_gesichert_2026-11-23/` | 13 unveränderte Ansichten des akzeptierten ersten Prototyps |
| `outputs/Orbitalgalerie/index.html` | Offline-Galerie mit Filtern und Links zu 4K-Originalen |
| `outputs/Orbitalgalerie/n_01/` bis `n_12/` | 728 PNGs: 364 verschiedene Dichten, jeweils voll / geschnitten |
| `outputs/bohm_clips/01_2p_m_plus1.mp4` | n=2, l=1, m=+1 |
| `outputs/bohm_clips/02_2p_m_minus1.mp4` | Gleiche Dichte, entgegengesetzter Fluss |
| `outputs/bohm_clips/03_3d_m_plus2.mp4` | n=3, l=2, m=+2 |
| `outputs/bohm_clips/04_6d_m_plus1.mp4` | n=6, l=2, m=+1, mit radialen Knoten |
| `outputs/bohm_clips/Bohmscher_Fluss_Alle_Faelle.mp4` | Alle vier Fälle nacheinander, 88 Sekunden |

Jeder Einzelclip dauert 22 Sekunden und enthält drei ruhige Ansichten mit zwei
weichen Kamerafahrten. Dazwischen bleibt die Kamera stehen, während die Punkte fließen.
Ausgabe: 1920 × 1080, 30 Bilder/s, H.264, ohne Ton; intern doppelte Auflösung zur Kantenglättung.
Die Einzelclips sind für den Schnitt gedacht; der Gesamtclip enthält dieselbe Reihenfolge.

```powershell
.\.venv\Scripts\python.exe -m atom_simulator clips
.\.venv\Scripts\python.exe -m atom_simulator gallery
```

Eine kleine Galerie zum Testen: `gallery --max-n 3`. Die Galerie setzt einen abgebrochenen
Lauf fort, wenn Bild, Metadaten und Thumbnail vollständig zur Konfiguration passen.
Die beiden Vorzeichen von m haben dieselbe Dichte; die Galerie speichert daher m >= 0.

Eigene Bilder und Clips:

```powershell
.\.venv\Scripts\python.exe -m atom_simulator image --n 6 --l 2 --m 0 --output outputs/6d.png
.\.venv\Scripts\python.exe -m atom_simulator render --n 2 --l 1 --m 1 --duration 30 --output outputs/mein-fluss.mp4
.\.venv\Scripts\python.exe -m atom_simulator render --motion camera --n 6 --l 2 --m 0 --output outputs/kamerafahrt.mp4
```

Weitere Optionen: `--no-cutaway`, `--background black`, `--no-labels`, `--points`,
`--seed`, `--width`, `--height`, `--fps`, `--time-rate`, `--supersample 1|2|3`.
Bei schwächerer Hardware helfen 100.000 Punkte, 1280 × 720 und Supersampling 1.
Auf macOS/Linux lautet der Python-Pfad `.venv/bin/python`; ein Desktop-/GL-Kontext wird benötigt.

## Ein expliziter Physikkern, wenige Abhängigkeiten

**[physics.py](src/atom_simulator/physics.py)** enthält die gesamte Physik:

- Quantenzahlen und Naturkonstanten;
- eigene Rekursionen für Laguerre- und zugeordnete Legendre-Polynome;
- normierte Radial- und Winkelwellenfunktionen;
- Trapezintegration, kumulative Verteilungen und Punkt-Sampling;
- Wahrscheinlichkeitsdichte, Bohmsches Geschwindigkeitsfeld und exakte Zeitentwicklung.

Der Kern braucht nur **NumPy** als externes Paket. Es gibt keine SciPy-Abhängigkeit,
keinen externen Differentialgleichungs-Solver und keine verschachtelten Python-Schleifen.
Zwischenschritte und Datenstrukturen sind explizit und typisiert, damit die Berechnung
später leicht nach C++ übertragen werden kann.

Die Grundinstallation hat drei direkte Pakete: NumPy für Arrays, ModernGL für die
Grafikkarte und Pillow für Bilder/Beschriftung. **PyAV ist nur für Videos nötig, GLFW
nur für das Fenster.** `pip install -e .` installiert die Basis; `[video]`, `[viewer]`
oder `[all]` ergänzen die benötigten Ein-/Ausgabefunktionen. ModernGL zieht seinen
kleinen Kontextadapter `glcontext` automatisch hinzu.

| Datei | Aufgabe |
| --- | --- |
| `physics.py` | Alle physikalischen Rechnungen einschließlich Sampling und Bohmfluss |
| `config.py` | Feste, direkt bearbeitbare Szenenwerte und Fälle |
| `camera.py` | Kamera und Übergangszeiten |
| `renderer.py`, `shaders.py` | Zeichnen der vom Physikkern gelieferten Positionen |
| `viewer.py` | Fenster und Eingaben |
| `export.py`, `gallery.py` | Clips, Bilder, Metadaten und Übersicht |
| `cli.py` | Programmeinstieg und Parameterprüfung |

## Was genau bewegt sich?

Für die komplexen Eigenzustände gilt `psi = R_nl * Y_lm`. In natürlichen Einheiten
`r / a_mu` und `tau = t / t_mu` lautet das Bohmsche Feld

$$
\mathbf v=m\frac{(-y,x,0)}{x^2+y^2},\qquad
\Delta\varphi=\frac{m\tau}{x^2+y^2}.
$$

Der Physikkern rotiert jede Anfangsposition mit ihrem eigenen berechneten Winkel.
Radius und Höhe bleiben erhalten; es gibt keine Euler-Integrationsfehler und keine
individuelle Geschwindigkeitsdeckelung. Ein einziger Zeitfaktor gilt pro Clip.
2p mit m=+1 und m=-1 benutzt identische Anfangspunkte, Kamera und Zeitlupe.
Die helleren Punkte sind ausgewählte Proben derselben Wolke; ihre kurzen Spuren folgen
exakt derselben Gleichung und denselben Sichtbarkeitsregeln.

Die Wolke beschreibt ein Ensemble möglicher Konfigurationen **eines Elektrons**.
Bohmsche Trajektorien sind eine ausdrücklich gewählte Interpretation des
Wahrscheinlichkeitsstroms. Das Modell ist nichtrelativistisch und spinlos; es stellt
die Relativkoordinate dar. Die Schnittansicht entfernt einen geometrischen Keil aus
der Anzeige und verändert die Wellenfunktion nicht. Gold, Licht und Kugelgröße sind
Darstellungsmittel und keine lineare Dichteskala.

Sehr nahe der Achse können Bahnen bei endlicher Bildrate zeitlich unterabgetastet
sein. Exportmetadaten nennen Winkel pro Frame und betroffene Anteile. Die hervorgehobenen
Proben werden aus zeitlich gut auflösbaren Bahnen ausgewählt; ihre Geschwindigkeiten
werden nicht verändert. Die Zielrestwahrscheinlichkeit außerhalb des Radialgrenzwerts
ist ungefähr `1e-8`, numerisch aus der Gitter-CDF bestimmt.

Einheiten, Herleitung, Prüfungen und Primärquellen: [docs/physics.md](docs/physics.md).
Die visuelle Referenz ist minutephysics' [A Better Way To Picture Atoms](https://www.youtube.com/watch?v=W2Xb2GFK2yc);
Code und Bilder dieses Projekts sind eigenständig erstellt.

## Prüfen

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
.\.venv\Scripts\python.exe -m ruff check .
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe tools/verify_media.py
```

Die Tests laufen ohne SciPy. Sie prüfen analytische Referenzlösungen, Normierung,
Knoten, Momente, Punktverteilungen und die Bohmsche Bewegung einschließlich Strom,
Richtungsumkehr, Invarianten und stationärer Dichte. GitHub Actions führt die CPU-Tests
unter Windows/Linux aus. Das Medienwerkzeug dekodiert sämtliche Bilder und Videoframes
und prüft Galerieabdeckung, Indizes, Auflösungen, Laufzeiten und Zeitstempel.
