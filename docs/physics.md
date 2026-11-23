# Physikalisches Modell und Bildinterpretation

Der Simulator zeigt die räumliche Wahrscheinlichkeitsdichte eines einzelnen Elektrons im Wasserstoffatom. Die goldenen Kugelpunkte sind Stichproben aus dieser Verteilung: anschaulich mögliche Ortsmessergebnisse an wiederholt gleich präparierten Atomen. Ihre Anzahl ist keine Elektronenzahl. Kugelradius, Goldfarbe, Beleuchtung und Verdeckung dienen der Darstellung; sie sind keine physikalischen Eigenschaften des Elektrons. Grundlage dieser Interpretation ist die [Born-Regel](https://www.damtp.cam.ac.uk/user/tong/qm/qmhtml/S1.html#S1.SS1).

## Koordinate, Einheiten und Zustand

Die dargestellte Koordinate ist die Relativposition $\mathbf r=\mathbf r_e-\mathbf r_p$. Die Schwerpunktbewegung ist abgetrennt. Für das Coulombproblem eines Elektrons und eines Protons gilt

$$
\mu=\frac{m_em_p}{m_e+m_p},\qquad
a_\mu=\frac{4\pi\varepsilon_0\hbar^2}{\mu e^2}
=a_0\frac{m_e}{\mu}.
$$

Alle gespeicherten Punktkoordinaten werden in $a_\mu$ angegeben. Verwendet wird $m_p/m_e=1836.152673426$, der [CODATA-2022-Wert von NIST](https://physics.nist.gov/cgi-bin/cuu/Value?mpsme). Damit ist $a_\mu/a_0\approx1.000544617$. Ein dargestellter Ursprung markiert den Ursprung der Relativkoordinate; seine grafische Größe ist kein maßstäblicher Protonradius.

Zulässig sind $1\le n\le12$, $0\le\ell<n$ und $-\ell\le m\le\ell$, jeweils ganzzahlig. Es werden vollständige Wasserstoff-Eigenfunktionen berechnet:

$$
\psi_{n\ell m}(r,\theta,\varphi)
=R_{n\ell}(r)Y_\ell^m(\theta,\varphi).
$$

Mit $s=r/a_\mu$, $\xi=2s/n$ und
$R_{n\ell}(r)=a_\mu^{-3/2}\widetilde R_{n\ell}(s)$ lautet der normierte Radialteil

$$
\widetilde R_{n\ell}(s)=
\sqrt{\left(\frac2n\right)^3\frac{(n-\ell-1)!}{2n(n+\ell)!}}
\,e^{-\xi/2}\xi^\ell L_{n-\ell-1}^{(2\ell+1)}(\xi),
\qquad
\int_0^\infty s^2|\widetilde R_{n\ell}(s)|^2\,ds=1.
$$

Diese Normalisierung entspricht Gleichung (107) von [Lyu, Daniel und Freericks, *Cartesian Operator Factorization Method for Hydrogen*](https://doi.org/10.3390/atoms10010014). Die verallgemeinerten Laguerre-Polynome werden direkt über ihre Dreitermrekurrenz ausgewertet; dafür ist kein SciPy-Aufruf nötig. Andere Lehrbücher verwenden teilweise anders normierte Laguerre-Polynome und dadurch andere Vorfaktoren.

## Komplexe Winkelbasis und Knoten

Standard ist die komplexe Basis $Y_\ell^m$, einschließlich Condon–Shortley-Phase, mit Polarwinkel $\theta$ von der positiven z-Achse und Azimut $\varphi$ in der xy-Ebene. Die zugeordneten Legendre-Funktionen werden ebenfalls per Rekurrenz berechnet und mit dem Normierungsfaktor sowie $e^{im\varphi}$ kombiniert. Definition und Konvention stehen in [NIST DLMF 14.30](https://dlmf.nist.gov/14.30), die verwendeten Rekurrenzen in [DLMF 18.9](https://dlmf.nist.gov/18.9.T1) und [DLMF 14.10](https://dlmf.nist.gov/14.10.E3).

Aus $Y_\ell^m\propto P_\ell^m(\cos\theta)e^{im\varphi}$ folgt: Die Dichte ist azimutunabhängig, und $m$ sowie $-m$ liefern dasselbe Dichtebild. Die vertrauten reellen $p_x,p_y$-Orbitale entstehen erst durch bestimmte Überlagerungen und sind hier keine stillschweigende Voreinstellung.

Die Dichtegalerie braucht deshalb nur $m\ge0$: Bis $n=12$ sind das $\sum_{n=1}^{12}n(n+1)/2=364$ Zustände beziehungsweise 728 Bilder bei zwei Ansichten je Zustand. Beide Vorzeichen bleiben für die Strömung relevant. Jedes Orbital wird für die Darstellung separat eingepasst; gleich große Bildschirmbilder bedeuten daher keine gleichen physikalischen Ausdehnungen.

Es gibt $n-\ell-1$ positive radiale Nullstellen, also kugelförmige Knotenflächen. Der Ursprung wird dabei nicht mitgezählt. Bei komplexem $Y_\ell^m$ besitzt der Polarwinkelanteil $\ell-|m|$ Nullstellen im offenen Winkelbereich. Der Faktor $e^{im\varphi}$ erzeugt keine azimutalen Knotenebenen. Für $|m|>0$ verschwindet die Amplitude zusätzlich auf der z-Achse; eine solche Linie ist keine zusätzliche Knotenfläche. Die pauschale Aussage „$\ell$ sichtbare Winkelknotenflächen“ wäre daher für diese Basis irreführend. Diese Unterscheidung folgt unmittelbar aus der [DLMF-Definition](https://dlmf.nist.gov/14.30.E1).

Für den ursprünglichen, gesicherten Beispielzustand $(n,\ell,m)=(6,2,0)$ ergeben sich drei Radialknoten bei
$r/a_\mu\approx10.42919,\ 21.96074,\ 39.61007$.
Die Winkelnullstellen liegen bei $\theta\approx54.73561^\circ$ und $125.26439^\circ$, weil $Y_2^0\propto3\cos^2\theta-1$. Diese Zahlen sind aus den obigen Formeln berechnet, keine zusätzlichen Modellparameter.

## Stichproben und numerische Darstellung

Das Volumenelement muss beim Sampling berücksichtigt werden. Die radiale Ziehungsverteilung ist $p_s(s)=s^2|\widetilde R(s)|^2$, nicht allein $|\widetilde R(s)|^2$. Mit $u=\cos\theta$ gilt $p_u(u)=2\pi|Y_\ell^m(\arccos u,0)|^2$; $\varphi$ ist gleichverteilt in $[0,2\pi)$. Der zusätzliche Sinusfaktor ist beim Wechsel von $\theta$ zu $u$ bereits enthalten.

Eine endliche Stichprobe und ein endlicher numerischer Radius bilden die Verteilung nur näherungsweise ab. Eine äußere Abschneidegrenze ist keine physikalische Atomoberfläche. Zufallsseed, Punktzahl, Grenzradius und numerisch erfasste Wahrscheinlichkeitsmasse gehören deshalb zu den Reproduktionsdaten. Größere Kugelpunkte können schmale Knotenbereiche optisch überdecken; Bildhelligkeit ist wegen Beleuchtung und Überlagerung kein linearer Dichtewert.

Die numerische Ziehung invertiert getrennte monotone CDFs für Radius und $u$. Dafür werden 65.537 radial quadratisch verteilte Gitterpunkte bis $s_{\rm int}=4n^2+20n$ und 131.073 gleichmäßig verteilte $u$-Werte verwendet. Trapezsummen liefern die CDFs; ihre Normen werden mit Toleranz $2\times10^{-7}$ geprüft. Der Darstellungsradius ist das aus der normierten Radial-CDF interpolierte Quantil $1-10^{-8}$. Die Zielrestwahrscheinlichkeit von $10^{-8}$ ist eine numerische Schätzung, keine unabhängig berechnete oder zertifizierte Fehlerschranke. Die Stichprobe ist auf den numerisch erfassten Bereich bedingt. Der feste Seed macht dieselbe Stichprobe in derselben Softwareumgebung reproduzierbar; er macht die Monte-Carlo-Näherung nicht exakt.

Für einen Energieeigenzustand gilt $\Psi(t)=\psi e^{-iE_nt/\hbar}$, also bleibt $|\Psi(t)|^2$ konstant. Eine Dichteansicht kann dieselbe Stichprobe eingefroren zeigen. Der Bohm-Modus bewegt dagegen die Stichproben entlang des unten definierten Geschwindigkeitsfeldes und erhält dieselbe stationäre Verteilung. Beide Darstellungen beschreiben dieselbe Dichte; Gold kodiert weder Vorzeichen noch komplexe Phase. Siehe [Tong: Wellenfunktion und Wahrscheinlichkeitsstrom](https://www.damtp.cam.ac.uk/user/tong/qm/qmhtml/S1.html).

## Bohm-Fluss im spinlosen Modell

Verwendet wird die skalare, nichtrelativistische Führungsgleichung für die **Relativkoordinate**:

$$
\frac{d\mathbf r}{dt_{\rm phys}}
=\frac{\mathbf j}{|\psi|^2}
=\frac{\hbar}{\mu}\operatorname{Im}\frac{\nabla_{\mathbf r}\psi}{\psi}.
$$

Das ist eine Modellierung Bohmscher Bahnen und keine Aufnahme oder Rekonstruktion experimentell gemessener Elektronenwege. Die spinlose Führungsgleichung und ihre Abgrenzung von Pauli-/Dirac-Strömen werden in der Einleitung von [Colijn und Vrscay, *Spin-dependent Bohm trajectories for Pauli and Dirac eigenstates of hydrogen*](https://arxiv.org/pdf/quant-ph/0304198) ausdrücklich unterschieden. Insbesondere lässt sich das hier erhaltene Ruhen für $m=0$ nicht unverändert auf eine vollständige Theorie mit Elektronenspin übertragen.

Für dimensionslose Koordinaten $\mathbf q=\mathbf r/a_\mu=(x,y,z)$ und die Zeit

$$
\tau=\frac{t_{\rm phys}}{t_\mu},\qquad
t_\mu=\frac{\mu a_\mu^2}{\hbar}\approx2.4202\times10^{-17}\,\mathrm{s}
$$

vereinfacht sich die Bewegung außerhalb der Knoten zu

$$
\frac{d\mathbf q}{d\tau}
=\frac{m}{\varpi^2}(-y,x,0),\qquad
\varpi^2=x^2+y^2.
$$

$\varpi$ ist der Abstand zur z-Achse, nicht der Abstand zum Kern. Die Zeiteinheit ergibt sich aus der [NIST-Konstantentabelle](https://physics.nist.gov/cuu/Constants/Table/allascii.txt) mit der reduzierten Masse. Da $\varpi$ und $z$ entlang einer Bahn konstant bleiben, genügt die exakte Rotation

$$
\varphi(\tau)=\varphi(0)+\frac{m\tau}{\varpi^2},\qquad
r(\tau)=r(0),\quad z(\tau)=z(0).
$$

Es wird kein numerischer Zeitschritt zur Integration einer Kraftgleichung benötigt. Für $m>0$ läuft die Bewegung von der positiven x- zur positiven y-Richtung; aus Blickrichtung der positiven z-Achse zum Ursprung erscheint sie gegen den Uhrzeigersinn. $m\mapsto-m$ kehrt die Bewegung bei unveränderter Dichte um. Für $m=0$ bleibt die Stichprobe in diesem Modell stehen. Radiale und polare Knoten werden nicht durchquert. An Knoten ist $\mathbf j/|\psi|^2$ undefiniert; exakt solche Positionen haben unter der kontinuierlichen Born-Verteilung Wahrscheinlichkeit null und sind keine gültigen Anfangspunkte einer Bahn. Die Implementierung weist erkannte Knoten zurück; Gleitkommaarithmetik kann nicht jede mathematisch exakte Nullstelle identifizieren.

Die vielen Punkte repräsentieren unabhängige Anfangskonfigurationen desselben Ein-Elektronen-Zustands. Die stationäre Dichte bleibt unter diesem Fluss erhalten, obwohl sich einzelne Punkte bewegen. Eine endliche Punktzahl verursacht weiterhin Stichprobenrauschen. Heller markierte Tracer und kurze Spuren heben einzelne dieser Modellbahnen hervor; sie sind keine zusätzlichen Elektronen und verändern die zugrunde liegende Wolke nicht.

### Filmzeit und zeitliche Auflösung

Pro Clip gilt ein einziger konstanter Faktor $\kappa$ mit $\tau=\kappa t_{\rm Video}$. Somit entspricht eine Videosekunde $\kappa t_\mu$ physikalischen Sekunden. Alle Punkte und ihre Spuren verwenden denselben Faktor. Es gibt keine individuelle Geschwindigkeitsnormierung und keine Deckelung schneller Bahnen. Die automatische räumliche Einpassung eines Orbitals verändert diesen physikalischen Zeitmaßstab nicht. Werden für verschiedene Zustände unterschiedliche $\kappa$ verwendet, erlauben ihre scheinbaren Filmgeschwindigkeiten keinen unmittelbaren Geschwindigkeitsvergleich.

Nahe der z-Achse wird $|d\varphi/d\tau|=|m|/\varpi^2$ groß. Auch eine exakte Bahnberechnung kann bei endlicher Bildrate zeitliches Aliasing zeigen: Punkte können zwischen Bildern springen oder scheinbar rückwärts laufen. Maßgebend ist der Winkelabstand $|m|\kappa/(\varpi^2\,\mathrm{fps})$ pro Bild. Ein kleinerer gemeinsamer Zeitfaktor oder eine höhere Bildrate verbessert die Auflösung, ohne die Bewegungsgesetze zu verändern. Tracerauswahl, Spurlänge und Zeitfaktor dienen der Lesbarkeit; sie müssen von der unveränderten Bewegungsgleichung getrennt bleiben.

Die Demonstrationsfälle ergänzen sich:

| Zustand | Aussage |
| --- | --- |
| $2p,\ m=+1$ | Strömung bei stationärer Dichte; keine Radialknoten |
| $2p,\ m=-1$ | Gleiche Dichte, umgekehrte Richtung; gleicher Seed und Zeitfaktor wie beim positiven Fall |
| $3d,\ m=2$ | Weitere komplexe Winkelverteilung; bei gleichem $\varpi$ doppelte Winkelgeschwindigkeit gegenüber $m=1$ |
| $6d,\ m=1$ | Drei Radialknoten und die Äquatorebene als polare Knotenfläche |
| $6d,\ m=0$ | Optionale ruhende Kontrolle des spinlosen Modells |

## Schnittansicht und Modellgrenzen

Die optionale **Schnittansicht** blendet Punkte in einem geometrisch definierten Raumteil aus, damit innere Radialknoten sichtbar werden. Sie erzeugt kein physikalisches Loch, verändert den Zustand nicht und verteilt die verbleibenden Punkte nicht neu. Die Maske wirkt auf die aktuellen Positionen; Bahnen laufen rechnerisch auch durch den ausgeblendeten Bereich weiter. Eine feste Schnittöffnung bleibt im Koordinatensystem des Atoms fest und dreht sich nicht mit den Punkt-IDs. Der entfernte Bereich muss in der Anzeige als Schnittansicht gekennzeichnet bleiben. Beim Umschalten werden dieselben Anfangspunkte und dieselbe aktuelle Zeit verwendet.

Das Modell ist nichtrelativistische Coulomb-Quantenmechanik ohne äußere Felder. Spin, Fein- und Hyperfeinstruktur, Lamb-Verschiebung, endliche Kernausdehnung und Strahlungsübergänge sind nicht enthalten. Sehr große Führungsgeschwindigkeiten nahe der Achse sind innerhalb dieses nichtrelativistischen Modells zu verstehen und kein Nachweis physikalisch überlichtschneller Elektronen. Galerie, Dichteansicht und Bohm-Fluss ändern diese Modellgrenzen nicht.
