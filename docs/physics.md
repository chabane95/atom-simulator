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

Diese Normalisierung verwendet die verallgemeinerten Laguerre-Polynome in der [SciPy-Konvention](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.eval_genlaguerre.html); sie entspricht Gleichung (107) von [Lyu, Daniel und Freericks, *Cartesian Operator Factorization Method for Hydrogen*](https://doi.org/10.3390/atoms10010014). Andere Lehrbücher verwenden teilweise anders normierte Laguerre-Polynome und dadurch andere Vorfaktoren.

## Komplexe Winkelbasis und Knoten

Standard ist die komplexe Basis $Y_\ell^m$, einschließlich Condon–Shortley-Phase, mit Polarwinkel $\theta$ von der positiven z-Achse und Azimut $\varphi$ in der xy-Ebene. In SciPy heißt der Aufruf `sph_harm_y(l, m, theta, phi)`: dessen erster Parameter bezeichnet $\ell$, nicht die Hauptquantenzahl $n$. Siehe [SciPy](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.sph_harm_y.html) und [NIST DLMF 14.30](https://dlmf.nist.gov/14.30).

Aus $Y_\ell^m\propto P_\ell^m(\cos\theta)e^{im\varphi}$ folgt: Die Dichte ist azimutunabhängig, und $m$ sowie $-m$ liefern dasselbe Dichtebild. Die vertrauten reellen $p_x,p_y$-Orbitale entstehen erst durch bestimmte Überlagerungen und sind hier keine stillschweigende Voreinstellung.

Es gibt $n-\ell-1$ positive radiale Nullstellen, also kugelförmige Knotenflächen. Der Ursprung wird dabei nicht mitgezählt. Bei komplexem $Y_\ell^m$ besitzt der Polarwinkelanteil $\ell-|m|$ Nullstellen im offenen Winkelbereich. Der Faktor $e^{im\varphi}$ erzeugt keine azimutalen Knotenebenen. Für $|m|>0$ verschwindet die Amplitude zusätzlich auf der z-Achse; eine solche Linie ist keine zusätzliche Knotenfläche. Die pauschale Aussage „$\ell$ sichtbare Winkelknotenflächen“ wäre daher für diese Basis irreführend. Diese Unterscheidung folgt unmittelbar aus der [DLMF-Definition](https://dlmf.nist.gov/14.30.E1).

Für den Startzustand $(n,\ell,m)=(6,2,0)$ ergeben sich drei Radialknoten bei
$r/a_\mu\approx10.42919,\ 21.96074,\ 39.61007$.
Die Winkelnullstellen liegen bei $\theta\approx54.73561^\circ$ und $125.26439^\circ$, weil $Y_2^0\propto3\cos^2\theta-1$. Diese Zahlen sind aus den obigen Formeln berechnet, keine zusätzlichen Modellparameter.

## Stichproben, Zeit und Schnittansicht

Das Volumenelement muss beim Sampling berücksichtigt werden. Die radiale Ziehungsverteilung ist $p_s(s)=s^2|\widetilde R(s)|^2$, nicht allein $|\widetilde R(s)|^2$. Mit $u=\cos\theta$ gilt $p_u(u)=2\pi|Y_\ell^m(\arccos u,0)|^2$; $\varphi$ ist gleichverteilt in $[0,2\pi)$. Der zusätzliche Sinusfaktor ist beim Wechsel von $\theta$ zu $u$ bereits enthalten.

Eine endliche Stichprobe und ein endlicher numerischer Radius bilden die Verteilung nur näherungsweise ab. Eine äußere Abschneidegrenze ist keine physikalische Atomoberfläche. Zufallsseed, Punktzahl, Grenzradius und numerisch erfasste Wahrscheinlichkeitsmasse gehören deshalb zu den Reproduktionsdaten. Größere Kugelpunkte können schmale Knotenbereiche optisch überdecken; Bildhelligkeit ist wegen Beleuchtung und Überlagerung kein linearer Dichtewert.

Die numerische Ziehung invertiert getrennte monotone CDFs für Radius und $u$. Der Grenzradius wird über die radiale Restwahrscheinlichkeit bestimmt; standardmäßig bleiben $10^{-8}$ außerhalb. Die Stichprobe ist auf den erfassten Bereich bedingt, also auf ungefähr 99,999999 % der Gesamtwahrscheinlichkeit. Radiale und polare Knoten werden in die Integrationsgitter aufgenommen. Der feste Seed macht dieselbe Stichprobe in derselben Softwareumgebung reproduzierbar; er macht die Monte-Carlo-Näherung nicht exakt.

Für einen Energieeigenzustand gilt $\Psi(t)=\psi e^{-iE_nt/\hbar}$, also bleibt $|\Psi(t)|^2$ konstant. Die einmal gezogenen Punkte bleiben deshalb während der Präsentation fest; die Kamera bewegt sich. Das ist keine simulierte Elektronenbahn und keine Integration Bohmscher Trajektorien. Gold kodiert weder Vorzeichen noch komplexe Phase. Stationäre Dichte bedeutet allgemein auch nicht verschwindenden Wahrscheinlichkeitsstrom; diese Größen werden unterschieden. Siehe [Tong: Wellenfunktion und Wahrscheinlichkeitsstrom](https://www.damtp.cam.ac.uk/user/tong/qm/qmhtml/S1.html).

Die optionale **Schnittansicht** blendet Punkte in einem geometrisch definierten Raumteil aus, damit innere Radialknoten sichtbar werden. Sie erzeugt kein physikalisches Loch, verändert den Zustand nicht und verteilt die verbleibenden Punkte nicht neu. Der entfernte Bereich muss in der Anzeige als Schnittansicht gekennzeichnet bleiben. Beim Umschalten wird dieselbe gespeicherte Punktwolke verwendet.

Das Modell ist nichtrelativistische Coulomb-Quantenmechanik ohne äußere Felder. Spin, Fein- und Hyperfeinstruktur, Lamb-Verschiebung, endliche Kernausdehnung und Strahlungsübergänge sind nicht enthalten. Die Kamerazeit von 20 Sekunden ist Präsentationszeit und keine atomare dynamische Zeitskala.
