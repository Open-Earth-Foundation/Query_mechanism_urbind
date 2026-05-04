![img-0.jpeg](img-0.jpeg)

# München elektrisiert
## Abschlussbericht

Gefördert durch:

Bundesministerium für Wirtschaft und Klimaschutz

aufgrund eines Beschlusses des Deutschen Bundestages

München elektrisiert

3
.
# München elektrisiert

Ermittlung und Bewertung der Netzbelastung durch Ladeinfrastruktur im Verteilnetz der Stadt München
Herausgeber:

![img-1.jpeg](img-1.jpeg)

Forschungsstelle für
Energiewirtschaft e.V.
Am Blütenanger 71, 80995 München
+49 (0) 89 158121-0
Mail: info@ffe.de
Web: www.ffe.de

Abschlussbericht zum Projekt: München elektrisiert
Ermittlung und Bewertung der Netzbelastung durch Ladeinfrastruktur im Verteilnetz der Stadt München

Veröffentlicht am:
3

FfE-Nummer:
DLR-02

Projektleitung:
Andreas Weiß
Mathias Müller

Projektpartner:
Referat für Klima- und Umweltschutz der Landeshauptstadt München
TUM – Lehrstuhl für Fahrzeugtechnik
TUM – Lehrstuhl für Verkehrstechnik

Bearbeiter/in:
Florian Biedenbach
Janis Reinhard
Yannic Blume (geb. Schulze)
Valerie Ziemsky
Jeremias Hawran
Adrian Ostermann
Michael Ebner

Assoziierte Partner:
HWK für München und Oberbayern
IHK München und Oberbayern

Gefördert durch:
Bundesministerium
für Wirtschaft
und Klimaschutz

Förderkennzeichen:
MZ18010B

aufgrund eines Beschlusses
des Deutschen Bundestages
# Inhalt

Kurzzusammenfassung 7
Motivation und Zielsetzung 10
Methodik 11

Identifikation charakteristischer Untersuchungsgebiete 11
Identifikation relevanter Eingangsparameter 11
Korrelationsanalyse der Parameter zur Auswahl signifikanter Ausprägungen 12
Statistische Clustering zur Unterteilung in Netzelastungsgebiete 12
Modellierung der Netztopologien 15
Modellierung der aktuellen und zukünftigen Netzelastung 16
Modellierung regionaler Gebäude inklusive Wohn- und Gewerbeeinheiten 16
Verteilung von Photovoltaikanlagen und Hausspeichersystemen 18
Verteilung von Power-to-Heat Systemen 19
Verteilung von Elektrostraßenfahrzeugen 20
Elektrifizierung in München bis 2030 20
Primäre Erweiterungen im Modell GridSim 22
Integrierter Haushaltslastgenerator zur Abbildung von Ladelastgängen privat genutzter Elektrofahrzeuge 22
Ladelastgänge gewerblich genutzter Elektrofahrzeuge 24
Lastgenerator für öffentliches Laden 26

Entwicklung der Netzelastung im mittleren Verteilnetz der Stadt München 28

Simulation der Netzelastung im Status quo 28
Simulation der zukünftigen Netzelastung durch ungesteuerte Ladevorgänge 31
Simulation der zukünftigen Netzelastung durch gesteuerte Ladevorgänge 33
Auswirkungen auf Stakeholder und Handlungsempfehlungen 38

Energiewirtschaftliche Bewertung der Entwicklung der Elektromobilität 41

Bewertung von Ladevorgängen zukünftiger Elektromobilität in München aus energiewirtschaftlicher Perspektive 41
Entwicklung des Summenlastgangs der Stadt München 41
Bewertung der resultierenden Emissionen 43
Bewertung ausgewählter Geschäftsmodelle 44
Spitzenlastbegrenzung 45
Eigenverbrauchsoptimiertes Laden 46
Flottenmanagement 48
Lokale Netzdienstleistung 49
Fazit zu den Geschäftsmodellen...52
Veröffentlichung von Ladelastgängen...53
Förderung und Projektpartner...57
Literatur...58
Anhang...62
# 1 Kurzzusammenfassung

Die mit der Energiewende implizierte Verkehrswende und daraus resultierende, zunehmende Elektrifizierung des Verkehrssektors wird die Belastung der Verteilnetze in naher Zukunft insbesondere in urbanen Regionen stark beeinflussen. Zur Beschleunigung des Hochlaufs der Elektromobilität wurde im Rahmen des Projekts „München elektrisiert“ die Installation von Ladeinfrastruktur im Großraum München, sowohl im privaten als auch im öffentlichen Raum, gefördert. Im Kontext der begleitenden Forschung, durch zwei Lehrstühle der TUM und durch die FfE, wurde dieser Hochlauf in München aus verschiedenen Perspektiven wissenschaftlich analysiert und bewertet. Die FfE untersuchte dabei, wie sich die Integration von Ladeinfrastruktur in größerem Umfang auf die aktuelle und zukünftige Belastung im Münchner Verteilnetz auswirkt.

In einer Metropole wie München, zeigen sich aufgrund des historischen Wachstums sozidemographisch und infrastrukturell deutliche Unterschiede im Stadtgebiet. Diese Unterschiede sind folglich auch bei der Charakteristik der Verteilnetzbelastung im Status quo ersichtlich und wirken sich auch auf die regionale Entwicklung der Elektrifizierung von Mobilitäts- und Wärmesektor aus. Zur Bewertung der daraus resultierenden Unterschiede in der Münchner Netzbelastung durch die zukünftige Elektromobilität wurden im Rahmen des Projekts zahlreiche Modellierungen, Simulation und Analyse durchgeführt.

Die Analyse des Verteilnetzes in einem großflächigen, infrastrukturell diversen Untersuchungsgebiet wie der Stadt München (310,7 km² Gesamtfläche) erfordert fundierte Methoden und Vorgehensweisen. Zur Eingrenzung des Untersuchungsraumes wurde im Projekt mittels eines Clusterverfahrens zunächst die Charakteristik des Stadtgebiets im Kontext der Netzbelastung analysiert und auf ausgewählte, repräsentative Untersuchungsräume eingegrenzt. Auf Basis dieser Selektion an Untersuchungsräumen wurde, in Abstimmung mit dem Verteilnetzbetreiber, eine Auswahl an Postleitzahlengebieten getroffen und regionstypische Verteilnetztopologien modelliert. Zur simulativen Bewertung des Status quo der Netzbelastung in diesen Gebieten wurde zusätzlich der gesamte Gebäudebestand Münchens, einschließlich der für die Verteilnetzebene relevanten, elektrischen Komponenten, modelliert. In Monte-Carlo-Simulationen konnte die Netzbelastung im Status quo analysiert und bewertet werden. Die Simulationen zeigen deutliche Unterschiede in den Regionen, wobei über alle Untersuchungsgebiete hinweg im Status quo deutliche Kapazitäten zur Integration sektorgekoppelter Komponenten besteht.

Zur Simulation des Hochlaufs der Elektromobilität wurde das Stützjahr 2030 gewählt und die modellierten Gebäude mittels Szenarien in dieses Jahr projiziert. In diesem Prozess wurden, neben des Hochlaufs der Elektromobilität, auch weitere Entwicklungen, wie der Ausbau von Photovoltaik-Anlagen und Wärmepumpen, berücksichtigt. Neben dieser grundsätzlichen Verteilung der Komponenten über Szenarien wurden diverse, punktuelle Erweiterungen mit Fokus auf Elektromobilität im Stromnetz- und Energiesystem-Modell für Verteilnetze der FfE „GridSim“ vorgenommen. Diese Erweiterungen umfassen unter anderem die Berechnung des Mobilitätsbedarfs und -verhaltens privater Haushalte und gewerblich genutzter Fahrzeuge, Module zur Erzeugung realitätsnaher Belastung öffentlicher Ladeinfrastruktur sowie die Implementierung regelungsbasierter Lade- und Lastmanagementstrategien.

Ausgehend von der Analyse des Status quo, wurden diverse Entwicklungsszenarien von batterieelektrischen Fahrzeugen und weiteren Komponenten simuliert und analysiert. Auch

www.ffe.de
die Auswirkungen der Variation verschiedener Lade- und Betriebsstrategien, sowie Extremausprägungen des Ladens in hoher Konzentration wurden in verschiedenen Sensitivitätsanalysen simuliert und bezüglich des wirtschaftlichen und netzentlastenden Potentials bewertet.

Die Simulationen zeigen eine deutliche Veränderung der Netzbelastung, wobei in den Untersuchungsgebieten regionale Unterschiede auftreten. Im Hinblick auf den gewählten Zeithorizont (Stützjahr 2030) tritt ein vorhersehbares Ergebnis ein, wonach der größte Anstieg in der Netzbelastung in den suburbanen Stadtrandgebieten auftritt. Dies ist zurückzuführen auf die in diesen Regionen günstigen Bedingungen für die Elektrifizierung. Die dort dominierenden Gebäudetypen in Form von Ein- und Zweifamilienhäusern gestatten, im Gegensatz zu komplexen Eigentumsstrukturen in Mehrfamilienhäusern im Stadtzentrum, eine einfachere Installation von PV-Anlagen und Wärmepumpen, was bei der Verteilung der Komponenten in den Szenarien durch die Modelle berücksichtigt wurde. Auch die Verteilung von Elektrofahrzeugen wird in naher Zukunft zunächst in Stadtrandgebieten erwartet und entsprechend modelliert. In den Simulationen wurden primär die Charakteristik der Netzbelastung sowie potenzielle Grenzwertverletzungen analysiert. Eine Überschreitung der Netzbetriebsgrenzen findet in den Simulationen des mittleren Verteilnetzes in keinem der Szenarien statt.

Im Rahmen der Simulation verschiedener Sensitivitäten des Flotten- und Lastmanagements zeigt sich, dass die Begrenzung der Leistung am Hausanschluss nur marginale Auswirkungen auf die Mobilität der Elektrofahrzeuge hat, jedoch individuell zu deutlich verminderten Lasten respektive Kosten für den Hausanschlussnehmer führt, sofern dieser einen Leistungspreis zahlt. Verschiedene Priorisierungslogiken für Flotten vermindern die Einschränkungen der Elektromobilisten, wobei bereits eine einfache First-Come-First-Served-Logik zur deutlichen Verbesserung beiträgt und zeigt bei kleinen Flotten keine signifikanten Nachteile im Gegensatz zu einer komplexeren Logik, wie z. B. der Priorisierung nach Ladezustand. Aus Perspektive der Netzbetreiber wirken die Lastmanagementstrategien mit Fokus auf der Reduktion der Last am Hausanschluss nur geringfügig auf die Lastspitze im Verteilnetz aus, da sich die Spitzenlastzeitpunkte der individuellen Gebäude nur selten überschneiden.

Auch aus energiewirtschaftlicher Perspektive wurden die zukünftigen Ladevorgänge in München analysiert. Hierzu wurde auf Basis von Zukunftsszenarien der zukünftig Gesamtlastgang Münchens, einschließlich der Lastgänge individueller Sektoren (Haushalte, Industrie, Verkehr und Gewerbe), prognostiziert. Die Verschneidung mit Emissionszeitreihen ermöglichte die Prognose, Analyse und Bewertung der perspektivisch resultierenden Emissionen. Die Ergebnisse verdeutlichen, dass die CO2-Emissionen in München trotz der steigenden Gesamtlast im Jahr 2030, aufgrund des deutlich größeren Anteils an erneuerbaren Energien im Strommix, sinken werden.

Weiterführend wurden im Projekt verschiedene Geschäftsmodelle im Kontext der Einbindung von Elektrofahrzeugen in die energiewirtschaftliche Wertschöpfungskette analysiert und bewertet. Aus grundlegenden Use-Cases wurden Geschäftsmodelle aus Sicht der beteiligten Akteure in Form von Business Model Canvas' entwickelt. Im Anschluss wurden diese mittels einer SWOT-Analyse bewertet. Die analysierten Geschäftsmodelle beruhen dabei alle auf Form des flexiblen Ladeverhaltens zur Reduktion, Verschiebung, oder Vermeidung von Netzbelastung. Hier zeigt sich, dass „behind-the-meter“ Geschäftsmodelle, also diejenigen, welche ausschließlich hinter dem Netzanschlusspunkt stattfinden, gegenüber „in-front-of-the-meter“ Geschäftsmodelle geringeren Unsicherheiten aufweisen.

www.ffe.de
Kurzzusammenfassung
Mit dem Ziel der öffentlichen Publikation von Ladelastgängen wurden im Projekt über 2.000 Ladevorgänge aufgezeichnet, extrahiert, analysiert und einschließlich diverser, elektrotechnischer Kenngrößen unter freier Lizenz auf der Open-Data-Plattform der FfE (opendata.ffe.de) veröffentlicht. Insgesamt wurden mehrere Open-Source Datensätze und ein Modell geschaffen, welche für verschiedene Zwecke genutzt werden können. Die Datensätze umfassen z. B. repräsentative, synthetische Jahresprofile (elektrische Haushaltslastgänge, Mobilitätsprofile der Agenten und thermischer Lastgang) von verschiedenen Haushaltstypen in Deutschland. Dies ermöglicht die Simulation von individuellen Haushalten einschließlich individueller Lastspitzen oder der Verknüpfung mit Elektromobilität und stellt dadurch gegenüber der üblich genutzten Standardlastprofile, in diversen Anwendungen ein signifikantes Upgrade dar. Aufgrund des Wettbewerbs bei Errichtung und Betrieb öffentlicher Ladeinfrastruktur sind Daten zur Auslastung, selbst für wissenschaftliche Zwecke, häufig nicht verfügbar. Das implementierte und veröffentlichte Modell zur Modellierung von Ladelastgängen öffentlicher Ladeinfrastruktur schließt ebenfalls eine Lücke in der Modelllandschaft. Es ermöglicht dem Nutzer die Erzeugung von individuellen Ladelastgängen entsprechend definierter Charakteristik und dient gemeinsam mit der zugehörigen Publikation als methodisch fundierte Quantifizierung für das Problem der Nichtverfügbarkeit dieser Daten.

www.ffe.de
9
# 2 Motivation und Zielsetzung

Die Stadt München präsentiert sich bezüglich des Klimaschutzes seit Jahren als ambitionierte Kommune. Bereits im Jahr 2008 wurde ein gesamtstädtisches Klimaschutzkonzept mit dem Zeithorizont 2030 verabschiedet, welches unter anderem die Reduzierung der pro-Kopf- $\mathrm{CO}_{2}$-Emissionen um $50\%$ gegenüber dem Bezugsjahr 1990 vorsah. /RGU-01 09/ Insbesondere im Verkehrssektor besteht diesbezüglich akuter Handlungsbedarf, so wird an den Münchner Hauptverkehrsstraßen der Immissionsgrenzwert für Stickstoffdioxid (Jahresmittelgrenzwert von $40~\mu \mathrm{g} / \mathrm{m}^3$) z. T. deutlich überschritten. /LHM-01 19/ Zur Verminderung dieser Emissionen finden in München verschiedene Aktivitäten mit dem Ziel der Verlagerung des Verkehrs auf umweltfreundlichere Verkehrsmittel und der Diffusion diverser, energieeffizienterer Technologien statt /ÖKO-02 17/. Auf Bundesebene wurde im November 2017 durch das „Sofortprogramm Saubere Luft 2017-2020“ ein Maßnahmenpaket aufgelegt, welches unter anderem durch die Elektrifizierung des Verkehrs die Luftreinheit in städtischen Gebieten verbessern soll. Zur Umsetzung dieses Vorhabens in der Landeshauptstadt München wurde als Teil dieses Sofortprogramms das Förderprojekt „München elektrisiert“ (M⁰) geschaffen. Dieses Projekt setzte sich insbesondere den großflächigen Ausbau von Ladeinfrastruktur (LIS) zum Ziel, um die Wende hin zur Elektromobilität im privaten und gewerblichen Sektor weiter zu beschleunigen. /FFE-34 19/

Die Integration einer großen Anzahl verschieden genutzter Ladepunkte in die bestehende Infrastruktur, stellt die Verteilnetzbetreiber (VNB) in der Netzplanung und dem Netzbetrieb vor neue Herausforderungen. Historisch gewachsene Verteilnetzstrukturen müssen neben den stromeinspeisenden Erneuerbaren-Energien-Anlagen auch der Belastung neuer Verbraucher, wie Wärmepumpen (WP) und den dezentral ladenden, batterieelektrischen Elektrofahrzeugen (EFZ) standhalten. Bei der Integration der EFZ ist die resultierende Netzbelastung im Wesentlichen von zwei zentralen Faktoren abhängig:

Der Anzahl an EFZ bzw. den resultierenden Ladeleistungsprofile und die aus dem Ladeverhalten der Fahrzeuge resultierenden Ladegleichzeitigkeiten
Die bestehende, individuelle Netzbelastungssituation im jeweiligen Verteilnetz abseits der Elektromobilität /FGH-01 18/

Im wissenschaftlichen Teilvorhaben der FfE liegt der Fokus auf der Analyse der zukünftigen Netzbelastung im Verteilnetz der Stadt München. Als erstes Projektziel galt die Ermittlung und Beurteilung des Status quo der Netzbelastung, um die Belastbarkeit und akuten Handlungsbedarf im Bereich der Verteilnetze zu identifizieren. In nächster Instanz sollten die Auswirkungen unterschiedlicher Ladeinfrastrukturausprägungen und deren Betriebsweisen auf die Verteilnetze mit Hilfe von Lastflussberechnungen ermittelt werden, um mittelfristig unzulässige Zustände der Netzbelastung zu vermeiden und um Handlungsempfehlungen abzuleiten.

www.ffe.de
Motivation und Zielsetzung
# 3 Methodik

Mit dem übergeordneten Ziel, die Auswirkungen des Ausbaus an LIS im Stadtgebiet Münchens simulativ zu beurteilen, musste zunächst eine entsprechende Datenbasis geschaffen werden. Die dabei notwendige Datengrundlage beschränkt sich im Wesentlichen auf die topologische Struktur der Verteilnetze und die zugehörige individuelle Belastung der angeschlossenen Betriebsmittel. In der räumlichen Dimension der Großstadt München hätte dies einer Detailanalyse eines über $300\,\mathrm{km}^2$ großen Versorgungsgebiets mit ca. 12.000 km Leitungslänge entsprochen /SWM-02 18/. Auch die beschränkte Verfügbarkeit detaillierter Daten (Netztopologien, aktuelle und zukünftige Netzbelastung) stellte eine Hürde bei der Beurteilung der Netzbelastung dar. Es musste ein Verfahren entwickelt werden, um mit beschränkter Datenbasis essentielle Aussagen hinsichtlich der aktuellen und zukünftigen Netzbelastung im Verteilnetz zu treffen.

## 3.1 Identifikation charakteristischer Untersuchungsgebiete

In Relation zur Netzbelastung in ländlichen Gebieten, in welchen sich die Versorgungsaufgabe primär durch die Integration einer großen Anzahl dezentrale Erzeugungsanlagen verändert, liegt die Aufgabe der VNB im urbanen Raum primär auf der Versorgung der Lasten. Urbane Netztopologien weisen aufgrund infrastruktureller Planungsgrundsätze im Mittel eine deutlich homogenere Struktur als ländliche Verteilnetze auf, jedoch ergeben sich auch in urbanen Verteilnetzgebieten lokale Unterschiede, welche von einer Vielzahl infrastruktureller Ausprägungen abhängig sind.

Grundlegend unterscheidet sich die Netzbelastung in städtischen Gebieten durch verschiedene Kriterien, wobei insbesondere die Flächennutzung z. B. als Wohnraum oder Gewerbeflächen ein ausschlaggebendes Unterscheidungskriterium ist. Jedoch bestehen auch innerhalb dieser Unterscheidung bestehen Abhängigkeiten von den verschiedenen Charakteristiken der in den Gebieten verorteten Betriebsmittel und angeschlossenen Komponenten. Wie in den nachfolgenden Kapiteln beschrieben, wurden in erster Instanz eine Selektion relevanter Charakteristiken der Netzbelastungen in einer statistischen Analyse von Metadaten identifiziert und das Stadtgebiet München räumlich geclustert. Das Ziel dieses Verfahrens war die Identifikation repräsentativer Netzbelastungsgebiete, wobei das Stadtgebiet in geeignet große Rasterzellen unterteilt und jeder Zelle jeweils die beiden, statistisch extremsten Ausprägungen aus einer Auswahl an Charakteristiken zugeordnet wurden. Entsprechend der resultierenden Ausprägungspaare konnten repräsentative Netzbelastungsgebiete für weitere Analysen selektiert werden. Eine detailliertere Beschreibung der nachfolgend erörterten Methode wurde ebenfalls im Fachartikel /FFE-97 19/ publiziert.

## 3.1.1 Identifikation relevanter Eingangsparameter

Zur Definition von Netzbelastungsgebieten wurde, durch Analyse verschiedenartiger Metadaten, eine Auswahl an energetischen, soziodemographischen und infrastrukturellen Parametern getroffen, welche die Netzbelastung im Untersuchungsgebiet besonders beeinflussen. Im Rahmen dieser Auswahl war die räumliche Auflösung dieser Metadaten ein ausschlaggebendes Kriterium. Für die qualitative Beurteilung der Charakteristik der Netzbelastung mussten die Daten in einer räumlichen Auflösung erhoben werden, welche sich

www.ffe.de

Identifikation charakteristischer Untersuchungsgebiete
mit den räumlichen Dimensionen der in der Region vorherrschenden Verteilnetzstrukturen vereinen lies. Des Weiteren musste die räumliche Vergleichbarkeit der selektierten Gebiete gewährleistet sein. Dies konnte sowohl durch die Auswahl einer flächendeckend äquivalenten Gebietsgröße, als auch durch die Normierung der Parametersätze auf die jeweiligen Gebietsflächen erfolgen. Der primäre Anteil an erhobenen Eingangsdaten wurde in einem Raster mit räumlicher Auflösung von 1 ha (10.000 m²) erhoben. Für die Vereinheitlichung der Eingangsdaten und der Einteilung des Stadtgebiets wurde das 1 km² Raster als Referenz definiert, woraus sich unter Berücksichtigung der Stadtgrenzgebiete 318 Raster-Zellen ergaben. Die Wahl dieses Rasters als Referenz begründete sich aufgrund der guten Eignung zur Vereinigung mit den räumlichen Dimensionen von Verteilnetzstrukturen und der Option einer einfachen Hochskalierung der Daten im vorwiegenden 1 ha Raster, wodurch der Einfluss der Vielzahl punktuelle Extremausprägungen des 1 ha Rasters vermindert wurde. Im Zuge der Datenerhebung wurden primär öffentliche Datenquellen berücksichtigt, wobei diverse Datensätze bereits in Vorarbeiten in der FfE-internen Datenbank „Regionalisiertes Energiesystemmodell" (FREM) aufbereitet und hinterlegt wurden /FFE-21 17/.

## 3.1.2 Korrelationsanalyse der Parameter zur Auswahl signifikanter Ausprägungen

Das selektierte Verfahren zur statistischen Clusterung der Metadaten zielte im Wesentlichen darauf ab, in jedem Gebiet in der selektierten, räumlichen Auflösung zwei Ausprägungen zu identifizieren, welche die Charakteristik der Zelle aus Netzbelastungssicht besonders gut beschreiben. Zur Einschränkung der Vielzahl an kombinatorischen Möglichkeiten und zur Auswahl repräsentativer Ausprägungen, wurden die im Rahmen der Datenerhebung selektierten Parametersätze, ohne großen Informationsverlust, auf eine definierte Anzahl an Parametern reduziert. Dieser Reduktionsschritt wurde im gewählten Verfahren durch eine Korrelationsanalyse und einen nachgelagerten Auswahlprozess vollzogen.

Die Auswahl der statistischen Repräsentanten erfolgte durch die Berechnung der empirischen Korrelationskoeffizienten, welche als Maßzahlen für die Stärke der linearen Zusammenhänge von jeweils zwei auf Korrelation untersuchten Parametersätzen dienen. Im Ablauf des Auswahlprozesses wurde das definierte Kriterium und damit die für eine Repräsentation notwendige Korrelation zweier Parameter von 100 % Korrelation iterativ vermindert. Beispielsweise diente der Parameter der Bruttogrundfläche der Gebäude aufgrund der Korrelation mit z. B. dem Strom- und Wärmebedarf der Personenhaushalte als statistischer Repräsentant dieser Parameter. Diese Auswahl begründete sich aus der betragsgrößten Korrelation zu allen weiteren, korrelierenden Parametersätzen (maximale, absolute Summe der Korrelationskoeffizienten).

## 3.1.3 Statistische Clusterung zur Unterteilung in Netzbelastungsgebiete

Das Ziel der nachfolgenden, statistischen Analyse war die Selektion von jeweils zwei charakteristischen Ausprägungen jeder Rasterzelle. Dazu wurden für jeden Parametersatz der Median, das obere und untere Quartil sowie der Interquartilsabstand (IQR) ermittelt. Letzterer wurde der Standardabweichung vorgezogen, da dieser in Relation einem robusteren Maß gegen „Ausreißern" entspricht. Es erfolgte eine Normierung der Parametersätze auf den jeweiligen IQR, um den Datenpunkt jedes Gebiets statistisch vergleichbar zu machen (vgl. exemplarische Abbildung 3-1).

www.ffe.de
Methodik
www.ffe.de
Identifikation charakteristischer Untersuchungsgebiete

![img-2.jpeg](img-2.jpeg)
Abbildung 3-1: Exemplarische Darstellung der Selektion von charakteristischen Parameterkombinationen

Als Resultat konnten für jedes Gebiet die beiden Parameter identifiziert werden, welche statistisch die größte Abweichung zum Median aufwiesen und somit die Zelle charakteristisch prägen. Im dargestellten Beispiel weist die selektierte Raster-Zelle die beiden größten Abweichungen sowohl in negative Richtung (PV Potenzial) als auch in positive Richtung (EFZ Bestand) auf, weshalb die Raster-Zelle durch diese Ausprägungen charakterisiert wird.

Zur Auswahl repräsentativer Cluster, wurde die Häufigkeit der resultierenden Parameterkombinationen ausgewertet. In einem iterativen Prozess wurden repräsentative Cluster identifiziert und die Raster-Zellen nach Möglichkeit diesen Clustern zugeordnet. Als Resultat dieses mehrstufigen Zuordnungsprozesses ergab sich die in Abbildung 3-2 dargestellte quantitative Verteilung der Cluster, wobei 310 der 318 Raster-Zellen im Stadtgebiet den repräsentativen Clustern zugeordnet werden konnten.

![img-3.jpeg](img-3.jpeg)
Abbildung 3-2: Selektion relevanter Parameterkombinationen in Abhängigkeit der resultierenden Häufigkeit nach iterativer Zuordnung

3
Aus dieser quantitativen Verteilung der repräsentativen Cluster erfolgte für das untersuchte Stadtgebiet Münchens eine erneute Selektion an Clustern, welche sich bezüglich des Vorhabens, der Beurteilung der Netzbelastung durch eine ausgeprägte LIS, besonders eigneten. Den verbleibenden Clustern (in Abbildung 3-2 grau dargestellt) konnte, mit den vorhandenen Daten und der Anwendung dieser Methode, keine Signifikanz hinsichtlich potenzieller Veränderungen der zukünftigen Netzbelastung identifiziert werden. Bei räumlicher Verortung der Cluster auf die definierten $1\mathrm{km}^2$ Zellen resultiert die in Abbildung 3-3 dargestellte Verteilung. Die farblich weniger transparenten Raster-Zellen entsprechen denen, welche bereits in erster Iteration den repräsentativen Clustern zugeordnet wurden, während die transparenter dargestellten Raster-Zellen den im iterativen Prozess sekundär zugeordneten Raster-Zellen entsprechen. Die in dieser Darstellung fehlenden Raster-Zellen, konnten auch im iterativen Zuordnungsprozess keinem repräsentativen Cluster zugeordnet werden.

![img-4.jpeg](img-4.jpeg)
Abbildung 3-3: Auswahl flächiger, gleich charakterisierter Untersuchungsgebiete

Im Folgenden wurden in der Abstimmung mit dem zuständigen VNB, aus den im Raster verorteten Clustern repräsentative Belastungsgebiete selektiert (vgl. rote Ellipsen in Abbildung 3-3). In diesem Auswahlprozess wurden Gebiete ausgewählt, welche einem möglichst großflächigen Verbund mehrerer Raster-Zellen des äquivalenten Clusters entsprachen und sich in ein Postleitzahlengebiet einordnen ließen. Des Weiteren wurden dabei sowohl vorrangig die Raster-Zellen berücksichtigt, welche in erster Iteration dem jeweiligen Cluster zugeordnet wurden, als auch Raster-Zellen mit besonders prägnanter Ausprägung im spezifischen Cluster.

www.ffe.de

Methodik
Modellierung der Netztopologien

Für die Berechnung und Bewertung der Netzteilastung in der Niederspannungsebene der Untersuchungsgebiete wurden charakteristische Netztopologien modelliert. Die selektierten, repräsentativen Belastungsgebieten wurden dazu im Abstimmungsprozesses mit dem verantwortlichen VNB anschließend in Postleitzahlengebiete verortet, für welche vom VNB jeweils statistische Netz-Parameter übergegeben wurden. Auf Basis dieser statistischen Daten konnten im folgenden regionstypische Verteilnetztopologien für die einzelnen Gebiete erzeugt werden. Als Restriktion war dabei zu berücksichtigen, dass die übergebenen Daten jeweils einer statistischen Auswertung eines gesamten Postleitzahlengebiets entsprachen und somit keine Unterscheidung zwischen den Transformatoren oder Zählpunkten innerhalb der Gebiete vorgenommen wurden. Die erzeugten Typnetze wurden somit z. B. auch durch größere Gebäudeanschlüsse mit eigenem Transformator beeinflusst.

Aus den übergebenen Parametern wurden im Folgenden für jedes Untersuchungsgebiet das Niederspannungs-Typnetz, im notwendigen OpenDSS-Standard, in ein entsprechendes Knoten-Kanten-Modell übersetzt. Die Modellierung der Niederspannungstransformatoren erfolgte entsprechend der übergebenen Kennwerte, wobei für fehlende Parameter plausible Ersatzwerte aus weiteren Transformatordaten gebildet wurden. Die Stränge der Netztopologien wurden in diesem Modellierungsschritt sternförmig, abgehend von der Unterspannungsseite des Transformators aufgespannt. Die Netzverknüpfungspunkte inklusive der jeweiligen Hausanschlussleitung wurden homogen über sämtliche Stränge verteilt. Bei Typnetzen in welchen die Anzahl der HA nicht ganzzahlig durch die Anzahl der vom Transformator abgehenden Stränge geteilt werden konnte, wurden die verbleibenden HA auf die jeweils ersten Stränge verteilt. Es resultierten die in nachfolgender Abbildung 3-4 dargestellten Niederspannungs-Typnetz-Topologien. Bereits die Architektur der Topologien verdeutlicht die infrastrukturellen Unterschiede. So weisen die suburbanen Regionen in Relation zu den Gebieten im Stadtzentrum eine deutlich höhere Anzahl an HA je Strang und Netzgebiet auf.

![img-5.jpeg](img-5.jpeg)
Abbildung 3-4: Darstellung der mittleren Niederspannungsverteilnetztopologien

www.ffe.de

Modellierung der Netztopologien
Modellierung der aktuellen und zukünftigen Netebelastung

Die im Projekt analysierten Szenarien fokussieren sich vorrangig auf die Entwicklung der Elektromobilität. Als relevante Stützstelle, neben dem Status quo, wurde das zum damals von der Bundesregierung ausgewiesene Ziel von bundesweit zehn Millionen EFZ im Jahr 2030 anvisiert /BRD-01 20/. Dazu wurde ein Konzept entwickelt und umgesetzt, um die Charakteristik und den Grad Netzelastung in GridSim, über eine Schnittstelle mit dem regionalisierten Energiesystemmodell der FfE „FREM“ abzubilden. Zur Realisierung dieses Vorhabens wurde der gesamte Gebäudebestand der Stadt München im Status quo modelliert und relevante Komponenten detailliert oder anteilig über Quoten in diesen Gebäuden integriert. Es wurde die Belastung durch die Haushalte/Wohneinheiten (WE) und Gewerbe-Einheiten (GHD-Einheiten) berücksichtigt, wobei zusätzlich noch die Energie erzeugenden oder verbrauchenden Netzkomponenten Photovoltaik-Anlagen, Hausspeichersysteme (HSS), die Power-to-Heat (PTH) Systeme Wärmepumpen oder Elektrospeicherheizungen (ESH) sowie EFZ berücksichtigt wurden. Diese Komponenten wurden über Szenarien anschließend konsistent in das anvisierte Zieljahr 2030 projiziert.

Die Eingangsdaten zur Erzeugung der Gebäudestrukturen, stammen dabei vorwiegend aus Open-Source Daten. Ihre Erhebung und Verarbeitung der Daten zum Status quo und den Zukunftsszenarien wurde im Rahmen von Tagungsbeiträgen in den Paper /FFE-02 20/ und /FFE-04 21/ publiziert.

## 3.3.1 Modellierung regionaler Gebäude inklusive Wohn- und Gewerbeeinheiten

Zur Abbildung der regionalen, infrastrukturellen Gegebenheiten wurde in erster Instanz aus frei verfügbaren Open-Street-Map (OSM) Daten der Gebäudebestand im Untersuchungsgebiet extrahiert und abgebildet. Die in den OSM-Daten hinterlegte Klassifizierung der Gebäude, ermöglicht die Untergliederung in Wohngebäude verschiedenen Typs (Ein- und Mehrfamilienhäuser, Nicht-Wohngebäude). Dieser Gebäudebestand wurde im Anschluss mit Daten aus dem Zensus 2011 verschritten, um die Anzahl der jeweiligen WE der Gebäude festzulegen /DESTATIS-15 13/. Für die Simulationen im Projekt wurde eine über die Szenarien gleichbleibende Gebäudestruktur und somit konstante Verteilung von Wohn- und Gewerbeeinheiten angenommen. Demographischer Zuwachs und damit verbundene Neubauten wurden nicht modelliert. Für das gesamte Untersuchungsgebiet Münchens resultierte eine fixe Auswahl an Gebäuden inklusive der Verteilung an Haushalten. /FFE-02 20/

Die realitätsnahe Modellierung des Gewerbesektors war aufgrund der Datenverfügbarkeit nur bedingt möglich. Die Verteilung und Modellierung von Gewerbeeinheiten erfolgte hier über gewerbliche Standardlastprofile. Als Grundlage der Verteilung von Gewerbeeinheiten diente die Aufschlüsselung einer kommunalen, sektorenscharfen Verbrauchsentwicklung (private Haushalte, GHD und Industrie), in welcher München ein Gesamtenergiebedarf des GHD-Sektors ermittelte wurde. Dieser wurde entsprechend einer definierten Verteilung den verschiedenen GHD-Standardlastprofilen zugeordnet und über statistisch gemittelte Jahresenergiebedarfe in eine Gesamtanzahl an GHD-Einheiten umgerechnet (vgl. Tabelle 3-1). Die Verteilung der gewerblichen Standardlastprofile erfolgte über die energetische Verteilung aus ca. 1.000 Gewerbe-Standardlastprofilen eines realen Referenzgebietes.

www.ffe.de

Methodik
Tabelle 3-1: Auswahl und Verteilung gewerblicher Standardlastprofile

|   | G0 | G1 | G2 | G3 | G4 | G5 | G6  |
| --- | --- | --- | --- | --- | --- | --- | --- |
|  Energetische Verteilung zu Anteil in % | 21,56 | 49,77 | 1,22 | 12,32 | 11,31 | 0,98 | 2,83  |
|  Typ. Jahresenergiebedarf je Einheit in kWh | 8.297,8 | 7.236,5 | 17.967,5 | 2.333,5 | 8.541,2 | 11.576,4 | 18.535,6  |
|  Anzahl der SLPs in München | 105.090 | 278.177 | 2.746 | 213.539 | 53.558 | 3.425 | 6.175  |

Entsprechend der bereits im Clusterverfahren genutzten Energiebedarfsverteilungen wurde die berechnete Anzahl an GHD-Einheiten anteilig über das 1 ha-Raster Münchens auf die jeweiligen Gebäude verteilt (vgl. Abbildung 3-5). Der selektierte Ansatz war durch die fehlende Genauigkeit in diesem Sektor deutlich beeinflusst. Insbesondere nicht verfügbare Lastprofile von Gewerbeeinheiten mit registrierender Leistungsmessung (RLM) und die aus realen Profilen resultierenden Lastspitzen vermindern die Qualität der Ergebnisse in Bezug auf individuelle HA.

![img-6.jpeg](img-6.jpeg)

Abbildung 3-5: Verteilung des Jahresenergiebedarfs im Sektor GHD in München

Für die verschiedenen Untersuchungsgebiete resultiert die in nachfolgender Abbildung 3-6 dargestellte Zusammensetzung an Gebäuden. Gebäude welche nach OSM mit industrieller Nutzung oder ohne Nutzung klassifiziert sind, wurden nicht berücksichtigt. Dies folgt der Annahme, dass industriell genutzte Gebäude über individuelle, ausreichend groß dimensionierte Transformatoren an das Netz angeschlossen sind und Gebäude ohne Nutzung nicht an das Netz angeschlossen sind bzw. dieses nur marginal belasten (z. B. freistehende Garagen).

www.ffe.de

Modellierung der aktuellen und zukünftigen Netzbelastung
![img-7.jpeg](img-7.jpeg)

Abbildung 3-6: Anteile der Gebäudenutzung in den Untersuchungsgebieten

Auch in der Gebäudezusammensetzung zeigen sich regional deutliche Unterschiede. In zentral gelegenen Gebieten wie der Altstadt nehmen Gebäude mit Gewerbeeinheiten den mehrheitlichen Anteil ein, wobei diese vorwiegend in Mischnutzung vorliegen. In suburbanen Gebieten dominiert der Anteil an Wohngebäuden und dort liegt eine deutlichere Trennung zwischen Wohn- und Gewerbegebäuden vor. Dies deckte sich mit den Erwartungen und den modellierten Netztopologien, wonach in zentralen Gebieten vorwiegend hohe Gebäude mit vielen Einheiten an einer geringen Anzahl an Netzknoten angeschlossen sind und im suburbanen Raum die Siedlungsstruktur zur Ein- und Reihenhaussiedlung tendiert. Eine tiefergehende Auswertung findet sich in der Publikation /FFE-45 21/.

## 3.3.2 Verteilung von Photovoltaikanlagen und Hausspeichersystemen

Die Modellierung der PV-Anlagen im Status quo basiert auf realen Daten, welche im Marktstammdatenregister hinterlegt sind /BNETZA-06 19/. Bei Verteilung von PV-Anlagen im Zukunftsszenario wurde die auf /FERS-01 19/ basierende rasterbasierte Modellierung der Gebäude-Photovoltaik für eine Gitterweite von 100 m durchgeführt, wobei für die zukünftige Betrachtung eine bundesweite Ausbaustufe von 52 GW installierter Leistung an Gebäude-Photovoltaik kalkuliert wurde. Die Leistung der Bestandsanlagen wurde dabei von der modellierten zukünftigen PV-Leistung je Rasterzelle abgezogen. Unter der Annahme, dass die Leistung von PV-Anlagen perspektivisch steigt und zukünftige zu installierende Gebäude-Photovoltaik eine durchschnittliche Leistung von 20 kW besitzt, wurde die zu verteilende Anzahl an PV-Anlagen je Raster-Zelle bestimmt und zufällig auf Gebäude ohne bereits bestehende PV-Anlagen verteilt. Der Leistungszuwachs für zukünftige PV-Anlagen begründet sich aus der Annahme, dass die Betreiber der PV-Anlagen in zukünftig aufgrund steigender Strompreise und dem Wegfall der EEG-Vergütung, eine eigenverbrauchoptimierte Betriebsweise anstreben und dementsprechend die verfügbaren Dachflächen noch weiter nutzen. Auch der Hochlauf von PV-Anlagen auf gewerblichen Dachflächen wurde antizipiert. Die Modellierung zeigt in Relation zum Status quo über die gesamte Fläche Münchens einen deutlichen Ausbau der PV-Anlagen (vgl. Abbildung 3-7). Der Großteil der Rasterzellen weist mindestens eine PV-Anlage auf. Freiflächen-PV-Anlagen wurden in der Modellierung nicht berücksichtigt, da diese im Stadtgebiet Münchens keinen großen Einfluss haben. /FFE-04 21/

www.ffe.de

Methodik
Die Verortung der Hausspeichersysteme basiert in beiden Szenarien auf /SCHM-01 18/. Für die Untersuchung wurde für 2030 eine Gesamtzahl von ca. 530.000 HSS in Deutschland kalkuliert. Bei der Verteilung der Anlagen galt die Restriktion, dass nur Gebäude mit einer PV-Anlagen, ein HSS erhalten.

![img-8.jpeg](img-8.jpeg)
Verwaltungsgrenzen:  $\odot$  Großbasis OE / BKG 2017 |  $\odot$  OpenStreetMap-Ritwirkende

Abbildung 3-7: Prognostizierte Entwicklung der PV-Anlagenleistung

## 3.3.3 Verteilung von Power-to-Heat Systemen

Bei der Modellierung von Power-to-Heat-Systemen im Status quo wurde die Restriktion vorgegeben, dass ein Gebäude nur einen Heizungsstandard besitzen kann. Für die prozentuale Verteilung von Elektrospeicherheizungen je Gebäudetyp, wurde zunächst die in FREM hinterlegten Verteilungen der Wohneinheiten nach Heizstruktur, Gebäudetyp und Baualter je Gemeinde /FFE-21 17/ sowie die im Rahmen von /FFE-30 18/ ermittelten Fernwärmepotenzialgebiete herangezogen, welche den Wärmeverbrauch je  $100\mathrm{m}^2$ -Raster-Zelle beinhalten. In der Modellierung wurde angenommen, dass sich die Wahrscheinlichkeit für die Versorgung eines Ein- bzw. Mehrfamilienhauses durch eine ESH um die Hälfte reduziert, falls dieses in einem Fernwärmepotenzialgebiet mit einem Wärmeverbrauch bei über 4.000 MWh/ha liegt. Die Verteilung der WP basiert ebenfalls auf der identischen Verteilung der WE nach Heizstruktur, Gebäudetyp und Baualter /FFE-21 17/. Hieraus ergab sich die Restriktion, dass im Status quo keine WP auf Mehrfamilienhäuser verteilt wurden. Die Verteilung der PtH-Systeme auf die Gebäude erfolgte zufallsbasiert, entsprechend individueller Wahrscheinlichkeiten je 1 ha Raster-Zelle.

Die Modellierung der ESH für das zukünftige Szenario basiert auf der gleichen Annahme wie beim Status quo, so gilt z. B. weiterhin die Restriktion, dass ein Gebäude nur eine Heizungsart besitzen kann. Jedoch wurde basierend auf den Ergebnissen von /FFE-144 19/ eine bundesweite, jährliche Abnahmerate gegenüber dem Status quo von  $1,10\%$  für Einfamilienhäusern und  $0,90\%$  für Mehrfamilienhäusern angenommen, wodurch keine neuen ESH verteilt und bestehende ESH zurückgebaut wurden. Die Verteilung der WP folgt für das zukünftige Szenario ebenfalls den gleichen Annahmen wie beim Status quo. Der bundesweite Anteil der WP-Durchdringung wurde im Szenario für das Jahr 2030 mit  $15\%$  prognostiziert.

www.ffe.de

Modellierung der aktuellen und zukünftigen Netzbelastung
Verteilung von Elektrostraßenfahrzeugen

Die Verteilung der EFZ beruht auf der Modellierung für die Szenario-Entwicklung des Netzentwicklungsplans /FFE-142 19/. Hierbei wurden für verschiedene zukünftige Entwicklungspfade sowohl für privat als auch für gewerblich EFZ Prognosen sowie Regionalisierungen durchgeführt. Im Projekt wurden mehrere regionalisierte EFZ-Szenarien und deren Auswirkungen auf die Netzbelastung analysiert. Die EFZ wurden in der Verteilung zu 86 % in private und zu 14 % in gewerblich genutzte Elektrofahrzeuge klassifiziert. Die Verteilung der privat genutzten EFZ für die Szenarien lag in einem Raster von 1 ha vor, während die gewerblichen EFZ auf Landkreisebene regionalisiert waren. Die Verteilung der Fahrzeuge erfolgte auf der Ebene der Wohn- bzw. GHD-Einheiten, wobei diesen auch mehrere Fahrzeuge zugewiesen werden konnten. Insgesamt wurden drei Szenarien definiert, wobei neben dem Status quo und dem damaligen Ziel der Bundesregierung von zehn Millionen EFZ in 2030 (Referenzszenario), auch ein für das Zieljahr extremeres Szenario mit bundesweit 20 Millionen EFZ verwendet wurde. Die bundesweit modellierten Fahrzeuge wurden entsprechend der definierten Methode für jedes Szenario auf München regionalisiert und auf die Wohn- und GHD-Einheiten nach der kalkulierten Wahrscheinlichkeiten verteilt. Nachfolgende Abbildung 3-8 verdeutlicht die Entwicklung der Durchdringungsgrade auf Ebene der Wohn- und Gewerbe-Einheiten. Auch hier werden regionale Unterschiede deutlich, wobei EFZ an WE aufgrund der Verteilungslogik priorisiert in den suburbanen Regionen verteilt wurden, während sich die Durchdringungsgrade bei EFZ an GHD-Einheiten aufgrund der Datengrundlage und Verteilungslogik nur marginal unterscheiden.

![img-9.jpeg](img-9.jpeg)
Abbildung 3-8: Durchdringung der Elektrofahrzeuge bezogen auf Gewerbe- und Wohneinheiten

# 3.3.5 Elektrifizierung in München bis 2030

Insgesamt zeigt sich in den modellierten Szenarien ein im Kontext der durch Elektrifizierung neu ins Verteilnetz eingebundenen Komponenten deutlich verändertes Stadtbild. Besonders gravierend ist dabei der prognostizierte Zuwachs von Elektrofahrzeugen, welcher von einer marginalen Anzahl auf eine durchschnittliche Hausanschluss-Durchdringung von ca. 30 % anwächst, sofern das Ziel der Bundesregierung (10 Mio. EFZ bis 2030) erreicht wird. Nachfolgende Abbildung 3-9 verdeutlicht die Verteilung und Entwicklung der Komponenten in den individuellen Untersuchungsgebieten. Es zeigen sich deutliche Unterschiede hinsichtlich der Durchdringungsgrade zwischen zentral gelegenen Gebieten (Altstadt, Milbertshofen) und Gebieten am Stadtrand (Pasing, Feldmoching).

www.ffe.de

Methodik
![img-10.jpeg](img-10.jpeg)

Abbildung 3-9: Hausanschluss-Durchdringungsgrade verschiedener Komponenten im Status quo und für das Szenario 2030

Die im Projekt modellierten Entwicklungen decken wesentliche Faktoren im Kontext der zukünftigen Netzelastung im Niederspannungsnetz ab, jedoch wird diese auch durch weitere Veränderungen beeinflusst werden, welche im Rahmen des Projekts nicht modelliert werden konnten. Der demographische Wandel und der damit verbundenen Um- bzw. Neubauten von Gebäuden oder ganzen Gebäudeblöcken wurde, aufgrund der Komplexität in der Prognose, nicht berücksichtigt. Auch im Wärmesektor wurden Vereinfachungen getroffen, wobei die im Kontext der Wärmewende zwingend notwendige Gebäudesanierung vernachlässigt wurde. Ebenso wurden keine Gebäudekühlung durch z. B. Klimaanlagen oder die perspektivisch mögliche Wärmeversorgung durch dezentrale Blockheizkraftwerke modelliert. Darüber hinaus wurden auch im Mobilitätssektor relevante Entwicklungen vernachlässigt, wie die Elektrifizierung von Nutzfahrzeugen, Pendlerströme aus dem Umland Münchens und verkehrstechnische Nebenverbraucher, wie z. B. Ampeln, Laternen oder Oberleitungstransportverkehr.

www.ffe.de

Modellierung der aktuellen und zukünftigen Netzelastung
Primäre Erweiterungen im Modell GridSim

Zur zeitreihenbasierten Simulation der Netzbelastung wurde neben der Entwicklung der Szenarien auch die Simulationsumgebung, das Modell GridSim signifikant weiterentwickelt. Der Fokus lag dabei besonders auf der detailierten Modellierung verschiedener Arten der (Elektro-)Mobilitätsnutzung, welche den Energie- und Ladebedarf signifikant beeinflussen. Bei der Implementierung wurde der Realitätsbezug der entwickelten Mobilitäts-Module kontinuierlich geprüft und validiert. Die dabei im Modell implementierten Logikbausteine, Automatisierungsmechanismen und Visualisierungen ermöglichte zusätzlich die vereinfachte Analyse sowie Bewertung der in den Simulationen resultierenden Netzbelastung.

## 3.4.1 Integrierter Haushaltslastgenerator zur Abbildung von Ladelastgängen privat genutzter Elektrofahrzeuge

Der elektrische Verbrauch durch private Haushalte ist eine zentrale Versorgungsaufgabe der Verteilnetzbetreiber. Haushalte werden dabei im Kontext netztechnischer Simulationen häufig durch Standardlastprofile (H0-Profil) abgebildet, wodurch Lastspitzen durch individuelle Haushaltsverbraucher geglättet werden. Auch die (Elektro-)Mobilität privater Haushalte wird dabei meistens nur über Statistik abgebildet, wodurch die zeitliche Komponente des Energieverbrauchs vernachlässigt und dieser nur bilanziell realitätsnah dargestellt wird. Um Spitzenlasten durch private Haushalte und deren zukünftiger, verknüpfter Elektrofahrzeuge der Realität entsprechend abbilden zu können, wurde im Projekt ein neues Modul implementiert. Elektrische und thermische Lastgänge sowie das Mobilitätsverhalten privater Haushalte werden nun über einen „integrierten Lastgenerator“ erzeugt. Dabei handelt es sich um ein bottom-up Modell, welches alle Bewohner einer simulierten Siedlung abbildet, ihnen Aktivitäten zuweist und diese mit elektrischen Geräten, Warmwasserverbräuchen und der Nutzung elektrischer Fahrzeuge gekoppelt. Eine ausführliche Beschreibung des Modells und der Implementierung ist in /FFE-61 20/ zu finden. Eine Übersicht des integrierten Lastgenerators und dessen Modellkomponenten zeigt Abbildung 3-10.

![img-11.jpeg](img-11.jpeg)
Abbildung 3-10: Übersichtsschaubild integrierter Haushaltslastgenerator

www.ffe.de

Methodik
Zunächst werden den Bewohnern der Gebäude (im Folgenden als Agenten bezeichnet) im Aktivitätsgenerator mithilfe einer modifizierten Markow-Kette für jeden Zeitschritt Aktivitäten zugewiesen. Es handelt sich um einen modifizierten Markow-Prozess, da die Zuweisung der Aktivitäten nicht in jedem Zeitschritt erfolgen, sondern für jede Aktivität zusätzlich eine Aktivitätsdauer bestimmt wird. Erst nachdem eine Aktivität abgeschlossen ist, wird eine Neue bestimmt. Die dafür verwendeten Wahrscheinlichkeiten für Aktivitäten und Aktivitätsdauern wurden auf Basis einer Zeitverwendungserhebung des statistischen Bundesamts von 2012/2013 bestimmt /DESTATIS-05 16/. Es wird zwischen 20 verschiedenen Aktivitäten unterschieden, deren Zuweisung abhängig von Bewohnertyp, Typtag und Tageszeit ist. Die Aktivität Arbeitsweg wird nachträglich eingefügt, um eine konstante Dauer dieser Aktivität zu erreichen. Anschließend werden den Bewohnern für jeden Zeitschritt Orte zugewiesen. Diese Zuweisung erfolgt lediglich auf Basis der Aktivitäten. Einigen Aktivitäten sind eindeutig Orten zugewiesen z. B. findet die Aktivität Schlafen immer zuhause statt. Andere Aktivitäten können an verschiedenen Orten stattfinden z. B. Essen. In diesen Fällen erfolgt die Zuweisung des Ortes in Abhängigkeit des vorherigen und nachfolgenden Ortes. Es wird zwischen den Orten „Zuhause“, „Arbeit“, „sonstiger Ort“, „Arbeitsweg“ und „sonstiger Weg“ unterschieden. Die Ergebnisse des Aktivitätsgenerators sind Aktivitäts- und Mobilitätprofile aller in der simulierten Siedlung lebenden Agenten. Im elektrischen Lastgenerator werden die Aktivitäten über Einschaltwahrscheinlichkeiten mit elektrischen Verbrauchern gekoppelt. Es gibt z. B. eine hohe Einschaltwahrscheinlichkeit des Herdes bei der Zubereitung von Mahlzeiten. Andere elektrische Verbraucher, wie z. B. die Beleuchtung sind nur von der Anwesenheit der Agenten abhängig. Im thermischen Modell werden die Aktivitäten über Wahrscheinlichkeiten mit Warmwasserverbräuchen gekoppelt. Die Heizwärme wird über die Wohnfläche und die spezifische Heizwärme berechnet.

Die Datengrundlage für das Mobilitätsmodell bildet die Studie Mobilität in Deutschland (MiD) von 2016 /INFAS-01 19/. Daraus werden Wahrscheinlichkeiten für Modal Split (Entscheidung über PKW-Nutzung) von Arbeitswegen und sonstigen Wegen, sowie mittlere Distanzen und Geschwindigkeiten bestimmt. Das Modell generiert aus den Mobilitätsprofilen der Agenten mithilfe der MiD Daten Mobilitätsprofile und Wegdistanzen der Fahrzeuge. Diese werden in Fahrtenbüchern gespeichert. Abbildung 3-11 zeigt den Ablauf des Modells.

![img-12.jpeg](img-12.jpeg)
Abbildung 3-11: Übersichtsschaubild Mobilitätsmodell

www.ffe.de

Primäre Erweiterungen im Modell GridSim
Das Modell wird nur für Haushalte aufgerufen, die ein EFZ besitzen, da die Mobilitätsnutzung bei Agenten ohne EFZ keinen direkten Einfluss auf die Netebelastung im Verteilnetz des Wohnorts hat. Ob ein Agent mit dem EFZ zur Arbeit fährt und die Distanz des Arbeitsweges wird einmalig zu Beginn der Simulation für jeden erwerbstätigen Agenten bestimmt und jeder Aktivität „Arbeitsweg“ zugeordnet. Anschließend wird über den Modal Split für jeden „sonstigen Weg“ entschieden, ob dafür das EFZ genutzt wird. Sind einem Haushalt mehr Agenten mit Fahrerlaubnis als EFZ verordnet, koordiniert das Modell die Nutzung des PKWs. Dabei liegt die Priorität auf dem Fahrer mit der höheren Fahrtdauer innerhalb des Simulationszeitraums. Die Bestimmung der Distanz von „sonstigen Wegen“ erfolgt über die durchschnittliche Geschwindigkeit der Fahrt. Hierfür wird deterministisch eine reale Geschwindigkeit aus der MiD mit der gleichen Fahrtdauer gezogen und darüber die Distanz der Fahrt berechnet. Ergebnis des Mobilitätsmodells sind Mobilitätsprofile der EFZ und deren Fahrtenbücher.

Die Validierung des generierten Mobilitätsverhaltens erfolgte anhand verschiedener Kenngrößen auf Deutschlandebene, da für München keine speziellen Referenzwerte abgeleitet werden konnten. Dafür wurde eine für Deutschland repräsentative Siedlung erzeugt und hierfür das Mobilitätsverhalten mithilfe des integrierten Lastgenerators generiert. Die Ergebnisse wurden mit denen der MiD verglichen. Die Jahresfahrleistung lag in der Validierungssimulation im Durchschnitt bei 10.700 km. Zum Vergleich liegt die MiD bei 14.700 km. Die Jahresfahrleistung wird also deutlich unterschätzt. Dies lässt sich darüber erklären, dass kurze Fahrten überrepräsentiert und sehr lange Fahrten z. B. Urlaubsfahrten deutlich unterrepräsentiert sind. Diese Abweichung wurde im Projekt toleriert, da sie hauptsächlich durch fehlende Langstreckenfahrten verursacht wird, welche jedoch beim Ladeverhalten zu Hause eine untergeordnete Rolle einnehmen. Für eine umfangreiche Validierung des Modells wird auf /FFE-61 20/ verwiesen.

Einige beispielhafte Last- und Mobilitätsprofile, welche durch den Lastgenerator erzeugt wurden, sind im JSON-Format unter opendata.ffe.de unter freier Lizenz verfügbar.

Elektrische Lastprofile: Hyperlink-Elektrische-Lastprofile

Thermische Lastprofile: Hyperlink-Thermische-Lastprofile

Mobilitätsprofile: Hyperlink-Mobilitätsprofile

## 3.4.2 Ladelastgänge gewerblich genutzter Elektrofahrzeuge

Neben den Ladevorgängen am Wohnort wird perspektivisch auch das Laden am Arbeitsplatz an Signifikanz gewinnen, weshalb im Projekt auch die gewerbliche Mobilität modelliert wurde. Diese und der daraus resultierende Ladebedarf unterteilt sich dabei in zwei Nutzungsarten: Dem Laden von Gewerbefahrzeugen und dem Laden von Mitarbeiterfahrzeugen am Arbeitsplatz.

Die Modellierung gewerblich genutzter EFZ basiert primär auf den Basisdaten der Studie „Kraftfahrzeugverkehr in Deutschland“ von 2010 (KiD) /VIP-01 12/. Nach Filterung der Eingangsdaten entsprechend verschiedenen Kriterien wurden zunächst Tagesfahrprofile ausgewählt, welche unter Berücksichtigung verschiedener Restriktionen zu Wochen- und schließlich Jahresfahrprofilen verknüpft wurden. Unter Zuhilfenahme eines Verbrauchsmodells und der Annahme eines spezifischen Ladeverhaltens werden Ladelastgänge modelliert. /FFE-23 18/ Durch Zuordnung von EFZ zu den individuellen Gewerbeeinheiten (vgl. Kapitel 3.3.4) und deren Parametrierung nach definierten Kriterien (vgl. Kapitel 4), konnte die

www.ffe.de

Methodik
durch gewerbliche EFZ zusätzlich resultierende Belastung im Rahmen der Simulationen berechnet werden. Dabei konnte das Ladeverhalten der Fahrzeuge durch individuelle Ladesteuerungen und Restriktionen auf HA-Ebene auf verschiedene Arten beeinflusst werden. /FFE-118 20/

Um die erzeugten Mobilitätsprofile für gewerbliche Mobilität zu validieren wurden die Ergebnisse des Modells mit den Eingangsdaten aus der KiD sowie einem Datensatz aus dem Projekt REM 2030 /ISI-30 15/ verglichen. Der REM-Datensatz wurde dafür zunächst aufbereitet, um ihn mit dem Modell vergleichbar zu machen. Tabelle 3-2 zeigt berechnete Kenngrößen der Fahrprofile im Vergleich. Es zeigt sich, dass lediglich geringe Abweichungen zwischen den Eingangsdaten und dem Modell auftreten. Dagegen sind die Abweichung des Modells zum REM-Datensatz deutlicher. Im Modell stehen im Durchschnitt mehr Fahrzeuge ungenutzt am Gewerbe. Auch werden im Vergleich mit dem REM-Datensatz mehr als doppelt so viele Business-to-Business-Fahrten pro Tag durchgeführt. Dies zeigt sich auch im Unterschied der Tagesfahrleistung.

Tabelle 3-2: Vergleich relevanter Kennwerte zur gewerblichen Mobilität

|  Kenngröße |  | KiD | Modell | REM  |
| --- | --- | --- | --- | --- |
|  Anteil der fahrenden EFZ | % | 58,20 | 58,40 | 75,70  |
|  Tagesfahrleistung | km | 55,20 | 50,40 | 80,0  |
|  Abwesenheitsdauer pro Tag | min | 176,0 | 167,80 | 222,0  |
|  Verkehrsbeteiligungsdauer pro Tag | min | 59,60 | 54,10 | 87,70  |
|  Anzahl der Fahrten pro Tag | / | 0,86 | 0,74 | 2,03  |

Auch bei der Verteilung des prozentualen Anteils anwesender Fahrzeuge im Tagesverlauf unterscheiden sich die modellierten Fahrprofile und die aus dem KiD-Datensatz deutlich vom REM-Datensatz. Die Verteilung von KiD und modellierter Fahrprofile sind dabei nahezu deckungsgleich. So sind im Zeitraum zwischen 0 bis 5 Uhr fast alle Fahrzeuge anwesend, während im REM-Datensatz deutlich mehr Fahrzeuge abwesend sind. Um 10 Uhr sind laut KiD und Modell am meisten Fahrzeuge abwesend. Im REM-Datensatz wird das Maximum erst an abwesenden Fahrzeugen erst gegen 12 Uhr erreicht. In der zweiten Hälfte des Tages werden die Abweichungen noch deutlicher. Insgesamt zeigt der Vergleich, wie diverser Fahrprofile gewerblich genutzter Fahrzeuge sind. Das implementierte Modul bildet die Statistik entsprechend dem sehr umfassenden KiD-Datensatz sehr gut ab, weshalb die Abweichungen gegenüber dem REM-Datensatz toleriert wurden.

Zur Erzeugung der Fahrprofile von am Gewerbestandort ladenden, privaten Mitarbeiter-Fahrzeugen wurde der integrierte Haushaltslastgenerator zur Abbildung von Ladelastgängen privat genutzter EFZ genutzt (vgl. Kapitel 3.4.1 verwendet. Bei der Erzeugung der Mobilitätsprofile wird eine virtuelle Siedlung aus Einfamilienhäusern erstellt, wobei die Anzahl der WE der Anzahl zu erstellenden Fahrprofile entspricht. Diese WE wird anteilig ein zufälliger Haushaltstyp nach /DESTATIS-10 15/ und /DESTATIS-15 13/ zugeordnet. Der Anteil der Rentner wurde herausgerechnet und anschließend zufällig ein bis zwei EFZ zugeteilt, wobei nur WE mit zwei erwerbstätigen Personen zwei EFZ bekommen können. Die Zuteilung von zwei EFZ erfolgte, da private Haushalte in Deutschland oft zwei Fahrzeuge besitzen, wobei der Zweitwagen die Auslastung des anderen Fahrzeuges beeinflusst /INFAS-01 18/. Durch Anwendung des Haushaltslastganggenerators mit der Restriktion von Pendler-EFZ werden

www.ffe.de

Primäre Erweiterungen im Modell GridSim
5
Fahrprofile bezogen auf den Ladepunkt „Arbeitgeber" erstellt. Dadurch kann das Fahrzeug nur am Arbeitsplatz geladen werden. Wird ein EFZ von mehreren Haushaltsmitgliedern für Fahrten zu verschiedenen Arbeitgebern genutzt, wird der Standort mit der maximalen Standzeit als ausschließlicher Ladeort berücksichtigt. Bei Haushalten mit zwei Pendler-EFZ, wird nur das mit der längeren Standzeit am Arbeitsplatz selektiert. /JOOSS-01 21/

Neben der Nutzungsart wurden zusätzlich, für Gewerbestandorte besonders relevante, Steuerungsalgorithmen zur Spitzenlastkappung umgesetzt. Implementierte Restriktionen auf HA-Ebene (statisches und dynamisches Lastmanagement) wurden einschließlich verschiedener Priorisierungslogiken implementiert, um das gesteuerte Laden und dessen Auswirkungen auf EFZ und Verteilnetz im Rahmen von Fallstudien zu analysieren /FFE-118 20/ /FFE-13 22/. Als Priorisierungslogiken wurden dabei „First Come First Served", „Priorisierung nach state of charge (SOC)" und „Priorisierung nach SOC und Abfahrtszeit" umgesetzt. Die Ergebnisse der dazu durchgeführten Fallstudien verdeutlichen, dass dynamisches Lastmanagement mit verschiedenen Priorisierungslogiken an Gewerbestandorten möglich ist, ohne die Elektrofahrzeugnutzer signifikant zu beeinflussen. /FFE-118 20/

## 3.4.3 Lastgenerator für öffentliches Laden

Als Ergänzung zu den Modellen des privaten Ladens und des Ladens am Arbeitsplatz, wurde innerhalb des Projektes ein Modell entwickelt, das die elektrische Last durch öffentliches Laden abbildet. Hierfür wurde ein stochastischer Modellierungsansatz gewählt, welcher schematisch in nachfolgender Abbildung 3-12 dargestellt ist.

![img-13.jpeg](img-13.jpeg)
Abbildung 3-12: Schematischer Aufbau des Lastgenerators für öffentliches Laden

Auf der Basis empirischer Daten wird das Ladeverhalten an öffentlichen Ladepunkten über Wahrscheinlichkeitsverteilungen berechnet. Jeder Ladevorgang wird durch den Zeitpunkt des Ladebeginns, die Ansteckdauer und die geladene Energiemenge modelliert, welche entsprechend der kalkulierten Wahrscheinlichkeiten unter Zuhilfenahme von Zufallszahlen berechnet werden. Die Datengrundlage für diese Wahrscheinlichkeiten bildet ein aufbereiteter Datensatz des zentralen Datenmonitorings des Förderprogramms Elektromobilität vor Ort (kurz: ZDM) /NOW-02 20/. Das ZDM erfasste mit Hilfe von Datenloggern Betriebs- und

www.ffe.de
Methodik
Ladedaten von EFZ und Ladestationen. Über den Zeitraum von 2013 bis 2020 wurden über 400.000 Ladevorgänge an verschiedenen Ladestationen erfasst. Die Aufbereitung und Analyse des Datensatzes, die Beschreibung der Modellierung sowie die Anwendung des Modells wird in /FFE-36 22/ detailiert beschrieben.

Aufbauend auf den Wahrscheinlichkeiten werden synthetische Ladevorgänge einzelner Ladepunkte erzeugt, welche wiederum zu Ladesäulen aggregiert werden. Das Modell wurde ausgiebig getestet und mit den Ergebnissen einschlägiger Literatur validiert. Ein Auszug daraus ist in Tabelle 3-3: dargestellt. Diese zeigt die Mittelwerte für die tägliche Ansteckzeit in Stunden, die geladenen Energiemengen pro Ladevorgang und die Anzahl Ladevorgänge pro Tag und Ladesäule bzw. Ladepunkt gegenübergestellt.

Tabelle 3-3: Vergleich relevanter Kennwerte zum öffentlichen Laden

|   | MUC | NOW | Modell  |
| --- | --- | --- | --- |
|  Durchschnittliche Ansteckzeit in h | 8,83 | 4,50 | 4,48  |
|  Durchschnittliche geladene Energiemengen pro Ladevorgang in kWh | 14,00 | 9,79 | 9,49  |
|  Durchschnittliche Anzahl Ladevorgänge pro LS pro Tag | 1,44 | 0,40 | 0,68  |
|  Durchschnittliche Anzahl Ladevorgänge pro LP pro Tag | 0,72 | 0,20 | 0,34  |

Es zeigt sich, dass die synthetischen Lastgänge nach Filterung die Eingangsdaten (NOW) trotz kleiner Abweichungen weiterhin gut abbilden. Auch wurden durchschnittliche Werte für München mit in die Validierung einbezogen. Es zeigt sich, dass aus dem Modell geringere Ansteckzeiten und Energiemengen für die Ladevorgänge resultieren. Auch die Anzahl Ladevorgänge pro Tag wird für München im Modell deutlich unterschätzt. Die Eingangsdaten für das Modell wurden im Vorfeld nicht regionalisiert. Außerdem wurden Eingangsdaten von 2012 bis 2020 verwendet. In diesem Zeitraum sind die Energiemengen pro Ladevorgang und auch die Ansteckhäufigkeit aufgrund der zunehmenden Verbreitung von EFZ deutlich gestiegen. Beides erklärt die Abweichungen zwischen Modell und den Werten für München (vgl. auch Analysen in /FFE-36 22/). Die Abweichungen wurden für die Simulationen toleriert bzw. durch Korrekturfaktoren an die Münchner Werte angenähert.

Im Kontext der Modellierung öffentlicher Ladelastgänge wurde deutlich, dass diese Art der Modellierung in der Literatur nur selten durchgeführt wurde und dass die Verfügbarkeit von Datensätzen öffentlicher Ladepunkte, aufgrund des Wettbewerbs bei der Erschließung hoch frequentierter Ladepunkt-Standort, sehr beschränkt ist. Das im Projekt umgesetzte Modul wurde einschließlich der berechneten Wahrscheinlichkeiten öffentlich unter freier Lizenz publiziert (Hyperlink-PublicChagingLoadTool) und schließt dadurch eine Lücke in der Modelllandschaft.

www.ffe.de
Primäre Erweiterungen im Modell GridSim
# 4 Entwicklung der Netzbelastung im mittleren Verteilnetz der Stadt München

Nach Modellierung der Szenarien und notwendigen Erweiterungen im Modell wurden die Simulationen und Analysen zur Entwicklung der Netzbelastung im mittleren Verteilnetz der Stadt München durchgeführt. Entsprechend der Anzahl der Hausanschlussknoten der zur Simulation selektierten Typnetze wurden reproduzierbar zufällig Gebäude inklusive der jeweils modellierten Komponenten aus dem Gesamtbestand des Untersuchungsgebietes gezogen. Diese Selektion wurde anschließend entsprechend der DIN 18015-1 „Leistungsbedarf für Wohngebäude“ geprüft, um eine unrealistische Zusammensetzung an gezogenen Gebäuden auszuschließen. Die dadurch nach Netzplanungsnormen realistische, zufällige Verteilung, wurde einschließlich der modellierten Komponenten in Datenbankstrukturen abgelegt und somit eine Verteilung definiert. Dieser Prozess wurde mehrfach wiederholt, bis für jedes Netzgebiet 50 Verteilungen vorlagen, welche im Anschluss in Monte-Carlo-Simulationen evaluiert wurden /FFE-02 20/.

Die Möglichkeiten, Restriktionen und resultierenden Fehler dieser Art der Modellierung und Simulation wurden im Rahmen der Publikation „Analyse methodischer Modellierungsansätze im Kontext von Verteilnetzsimulationen“ mit weiteren Modellierungsansätzen verglichen und validiert. /FFE-61 21/

## 4.1 Simulation der Netzbelastung im Status quo

Die Simulationen zum Status quo der Netzbelastung stellten im Projekt die grundlegende Referenz dar, an welcher die Auswirkungen von Szenarien zu zukünftigen Ladevorgängen der Elektromobilität gemessen wurde. Die modellierten Gebäudestrukturen wurden nach Import und Verteilung auf die Hausanschlussknoten, durch die vorgegebenen Rahmenparameter dimensioniert und für die Simulation aufbereitet. Die grundlegende Struktur der Gebäude mit den Wohn- und GHD-Einheiten definierte dabei die Größe des Gebäudes, aus welchem individuelle Wärmebedarfe ermittelt wurden. Die Erzeugung des elektrischen Lastgangs der WE wurde über den integrierten Lastganggenerator /FFE-61 20/ vorgenommen. Die Lastgänge der GHD-Einheiten wurden aus den hinterlegten Jahresenergiebedarfen und zugehörigen Standardlastprofilen erzeugt. Photovoltaikanlagen (PV-Anlagen) und EFZ wurden den entsprechenden Wohn- und GHD-Einheiten in den Gebäuden zugeordnet und charakterisiert. Die WP und ESH wurden über die Wärmebedarfe der jeweiligen Gebäude dimensioniert. Die Dimensionierung der Hausspeichersysteme erfolgte entsprechend der PV-Anschlussleistung. Die Erzeugung der Lastgänge flexibler Komponenten erfolgte basierend auf den definierten Eigenschaften und Ladesteuerungen. Folgende Eigenschaften waren dabei ausschlaggebend:

WP &amp; ESH: Die Größe und zugeordnete Baualtersklasse des Gebäudes bestimmten den Wärmelastgang, woraus durch spezifische Eigenschaften die notwendige Leistung der WP/ESH definiert wurde. In den Simulationen folgte die WP dem thermischen Lastgang, wodurch ein wärmegeführter, elektrischer Lastgang resultierte. ESH hatten definierte Zeitfenster (Freigabe-/Sperrzeiten), in welchen diese

www.ffe.de

Entwicklung der Netzbelastung im mittleren Verteilnetz der Stadt München
zur Ladung berechtigt waren. Entsprechend dem Wärmebedarf des nächsten Tages, wurde die notwendige Energie in diesen Zeitfenstern geladen.

PV &amp; HSS: Der Lastgang für PV-Anlagen resultierte aus dem definierten Wetterjahr und den damit verbundenen Einstrahlungsdaten in München. Über die kalkulierte Dimensionierung resultierten die individuellen Erzeugungsgänge. HSS waren auf die Maximierung des Eigenverbrauchs geregelt, woraus Ladung/Entladung und der zugehörige Lastgang resultierte.
EFZ: Die Lastgänge für EFZ wurden in den Simulationen durch das Mobilitätsprofil und die Nennleistung am zugeordneten Ladepunkt bestimmt. Die Mobilitätsprofile unterschieden sich dabei je nach primärer Nutzung und dem definierten Ladeort (Laden von privaten/gewerblichen EFZ an jeweiliger Wohn- oder GHD-Einheit). Die Fahrzeuge luden im Status quo un gesteuert und direkt nach der Ankunft bis der maximale SOC erreicht war oder das Fahrzeug erneut losfuhr.

Die Ergebnisse zeigen, entsprechend der Abbildung 3-9 dargestellten Zusammensetzung der in den Untersuchungsgebieten verorteten Gebäude, eine ebenso individuelle Charakteristik der Netzbelastung. In den durchgeführten Jahressimulationen treten in den mittleren Verteilnetzen der ausgewählten Untersuchungsgebiete keine Netzüberlastungen auf. Die in München üblich verbauten 630 kVA Transformatoren werden in den mittleren Verteilnetzen der Untersuchungsgebiete nicht überlastet. Die in Abbildung 4-1 dargestellte, sortierte Jahresdauerlinie der Wirkleistung an den Ortsnetzstationen, verdeutlicht die verschiedenen hohe Belastung im mittleren Verteilnetz über den Untersuchungszeitraum von einem Jahr. Es wird deutlich, dass insbesondere in Stadtrandgebieten eine erhöhte Belastung auftritt.

![img-14.jpeg](img-14.jpeg)
Abbildung 4-1: Jahresdauerlinie der Residuallast an der Ortsnetzstation im mittleren Verteilnetz der Untersuchungsgebiete

Die in Abbildung 4-1 dargestellten Zeitpunkte hoher Belastung, treten in den Simulationen besonders in Zeitfenstern aktiver PtH-Systeme auf. Dies wird exemplarisch für das Untersuchungsgebiet Pasing in Abbildung 4-2 verdeutlicht. Durch den Kontrast in der gewählten Heatmap-Darstellung werden die zeitlichen Unterschiede der Residuallast an der

www.ffe.de

Simulation der Netzbelastung im Status quo
9
Ortsnetzstation deutlich, wobei die Zeitachse in Mitteleuropäischer Zeit (UTC + 1) dargestellt ist, wodurch die in Lokalzeit resultierende Belastung während der Sommerzeit, um eine Stunde verschoben ist. Auffällig in der Darstellung ist der starke Einfluss der PtH-Systeme, welche vorwiegend am Abend in den Wintermonaten zu einer hohen Belastung führt. Die klaren, horizontalen Kanten in der Darstellung entsprechen den Start- und Endzeitpunkten der ESH, welche im Modell so betrieben wurden, dass diese teils vorwärts (ab 22:00 Uhr) oder teils rückwärts (bis 6:00 Uhr) den Wärmebedarf des Gebäudes für die kommenden 24 h laden. Neben dem starken Einfluss der PtH-Systeme ist im vertikalen Muster (abwechselnd dunklere, dick und heller, dünn dargestellte Phasen über Tags) auch die Mehrbelastung an Wochentagen durch gewerbliche Last erkennbar.

![img-15.jpeg](img-15.jpeg)
Abbildung 4-2: Residuallast an der Ortsnetzstation im mittleren Verteilnetz des Untersuchungsgebiets Pasing

Auch bei der Auslastung der Leitungen treten im Status quo nur in einzeln simulierten Verteilungen geringfügig Überlastungen auf. Im Verteilungsmittel resultiert bezüglich des zulässigen Nennstroms eine maximale Auslastung deutlich unter 100 %. Das im Niederspannungsnetz zulässige Spannungsband wird im jeweiligen Verteilungsmittel in den fünf repräsentativen Untersuchungsgebieten nicht verletzt. Weitere, detailliertere Informationen zur Modellierung und den Status quo-Simulationen sind in der Publikation /FFE-02 20/ beschrieben.

www.ffe.de

Entwicklung der Netzbelastung im mittleren Verteilnetz der Stadt München
# 4.2 Simulation der zukünftigen Netzbelastung durch ungesteuerte Ladevorgänge

Ausgehend von der in Kapitel 4.1 beschriebenen Referenzsimulation wurden die modellierten Gebäudestrukturen entsprechend der in Kapitel 3.3 definierten Szenarien in die Zukunft projiziert und erneut Jahressimulationen durchgeführt. In erster Instanz wurden dabei nur die Anzahlen der individuellen Komponenten verändert. Deren Verteilung auf Gebäude und deren Betriebsweisen ist dabei konsistent zum Referenzszenario, so besitzt ein Haushalt, welcher im Referenzszenario bereits ein EFZ besessen hat, auch im Zukunftsszenario ein EFZ.

Im Fokus der Untersuchung stand besonders die resultierenden Änderungen in der Netzbelastung im mittleren Verteilnetz. Die in Abbildung 4-3 dargestellten Jahresdauerlinien verdeutlichen erneut die Dimensionen der Residuallast über das Jahr für die verschiedenen Untersuchungsgebiete im Extremszenario (20 Mio. EFZ in DE). Im Vergleich zur Referenz (in Abbildung 4-2) zeigt einen deutlichen Anstieg der Spitzenlast über alle Untersuchungsgebiete. Beim Mittelwert der Residuallast zeigt sich keine signifikante Änderung, wohingegen die Minimalwerte aufgrund des prognostizierten PV-Zubaus niedriger ist, wodurch im am Stadtrand gelegenen Untersuchungsgebiet Feldmoching Zeitpunkte negativer Residuallast (Rückspeisung in das Mittelspannungsnetz) auftreten.

![img-16.jpeg](img-16.jpeg)
Abbildung 4-3: Jahresdauerlinie der Residuallast an der Ortsnetzstation im mittleren Verteilnetz der Untersuchungsgebiete im Extremszenario

Bei der Analyse der Residuallast über die Zeit werden signifikante Unterschiede durch die Projektion der Untersuchungsgebiete in das Zukunftsszenario deutlich. Die in Abbildung 4-4 dargestellte Heatmap zeigt die Änderung der Residuallast im Extremszenario gegenüber dem Status quo des Untersuchungsgebiets Pasing. Die deutlichste Veränderung ist in den Abendstunden im Winter zu erkennen (dunkelblau gekennzeichnete Bereiche), in welchen durch den zeitgleichen Betrieb von WP und dem ungesteuerten Laden von EFZ, eine starke Mehrbelastung auftritt. Im betrachteten Untersuchungsgebiet verändert sich die auf HA bezogene Durchdringung der wärmegeführt betriebenen WP im Mittel von $6,73\%$ auf

www.ffe.de

Simulation der zukünftigen Netzbelastung durch ungesteuerte Ladevorgänge
,04 %, während die Durchdringung der ungesteuert ladenden EFZ im Mittel von 0,59 % auf 56,63 % ansteigt. Besonders der starke Hochlauf der EFZ wirkt sich dabei auf die Netzbelastung aus, wodurch z. B. die maximal gleichzeitige Last im Verteilungsmittel von 11,48 kW auf 141,62 kW ansteigt.

Neben den Zeitbereichen, in welchen ein erhöhte Netzbelastung auftritt, resultieren weitere Zeitbereiche in welchen eine Verminderung der Belastung zu verzeichnen ist (hellrot gekennzeichnete Bereiche in Abbildung 4-4). Einerseits sorgt der Zubau von PV-Anlagen (2,39 % auf 6,41 % Durchdringung) für eine deutliche Verminderung der Residuallast zur Mittagszeit in den Sommermonaten. Andererseits sorgt auch der Rückbau der leistungsstarken ESH (5,99 % auf 4,88 % Durchdringung) für eine Entlastung des Verteilnetzes in den Nachtstunden, bzw. für eine Verschiebung dieser Last in die Abendstunden, da ESH in der Modellierung teils durch wärmebedarfsgeführte WP ersetzt wurden.

![img-17.jpeg](img-17.jpeg)
Abbildung 4-4 Differenz der Residuallast aus Extremszenario abzüglich Status quo an der Ortsnetzstation im mittleren Verteilnetz des Untersuchungsgebiets Pasing

Auch die Leitungsauslastung steigt durch die neuen Komponenten und deren Gleichzeitigkeit deutlich an, der Vergleich der Leitungsauslastung im Verteilungsmittel vom Extremszenario gegenüber dem Status quo weist im Maximum einen Anstieg von 39,57 % auf 50,69 % auf. Somit treten im mittleren Verteilnetz auch in den modellierten Zukunftsszenarien keine Leitungsüberlastungen auf. In individuellen Verteilungen treten aufgrund des Hochlaufs von EFZ und WP teils deutliche Überlastungen auf, welche aufgrund des Simulationsdesigns jedoch nur bedingt repräsentativ sind (detaillierte Informationen zu diesen Restriktionen in /FFE-61 21/). Auch hinsichtlich dem Spannungsband resultieren in den Simulationen zu den projizierten Zukunftsszenarien keine Über- und Unterschreitungen der zulässigen Spannungsgrenzwerte. Weitere, detailliertere Informationen zur Modellierung und Simulationen zur Darstellung der projizierten Zukunftsszenarien sind in der Publikation /FFE-45 21/ beschrieben. Weitere Simulationen zur Darstellung der projizierten Zukunftsszenarien sind in der Publikation /FFE-04 21/ beschrieben, in welcher der Fokus der Analysen insbesondere auf die Interaktion zwischen PV-Anlagen und EFZ gesetzt wurde.

www.ffe.de

Entwicklung der Netzbelastung im mittleren Verteilnetz der Stadt München
Simulation der zukünftigen Netzbelastung durch gesteuerte Ladevorgänge

Ausgehend von den Zukunftsszenarien wird in diesem Kapitel die Netzbelastungen bei gesteuerter Ladung von EFZ untersucht. Bei den Analysen lag der Fokus auf den Verteilnetzgebieten der Regionen „Pasing“ und „Altstadt“, wobei insbesondere die Zeitpunkte der höchsten Belastung (Spitzenlast) analysiert und für verschiedene Ladestrategien verglichen wurden. Da im Status quo nahezu keine EFZ modelliert wurden, beschränkt sich die Analyse auf die Simulationen der Referenz- und Extremszenarien. Die gesteuerte Ladung der EFZ umfasst die Ladestrategien statisches und dynamisches Lastmanagement, sowie die Optimierung entsprechend einer Kostenfunktion. Die Ladsteuerung erfolgt auf HA-Ebene und alle EFZ, sowohl private als auch gewerbliche, werden in die Ladestrategie einbezogen. Die Implementierung der Ladestrategien sowie deren Erprobung in GridSim ist detailiert in /FFE-13 22/ beschrieben. Folgende Sensitivitäten wurden simuliert und analysiert:

Ungesteuertes Laden (Szenarien: „Ext ungest.“, „Ref ungest.“):
Die EFZ werden nach Ankunft am Hauptladeort an den zugewiesenen Ladepunkten angesteckt und bis zur Abfahrt oder Vollladung geladen. Die Simulation wurde mit dieser Ladestrategie für das Referenzszenario und das Extremszenario durchgeführt.

Statisches Lastmanagement (Szenario: „Ext stat.“):
Die EFZ werden nach Ankunft am Hauptladeort an den zugewiesenen Ladepunkten angesteckt und Laden bei Verfügbarkeit mit voller Ladeleistung. Über alle EFZ am HA besteht eine statische Leistungsbegrenzung (3 kW x Anzahl der EFZ am HA), welche die Leistung der „Flotte“ begrenzt. Die Priorisierung der EFZ erfolgt nach „First Come, First Served“-Prinzip (Fahrzeug mit jeweils frühster Ankunftszeit wird priorisiert). Die Simulation wurde mit dieser Ladestrategie für das Extremszenario durchgeführt.

Dynamisches Lastmanagement (Szenarien: „Ext dyn.“, „Ref dyn.“):
Die EFZ werden nach Ankunft am Hauptladeort an den zugewiesenen Ladepunkten angesteckt und Laden bei Verfügbarkeit mit voller Ladeleistung. Über alle EFZ am HA besteht eine dynamische Leistungsbegrenzung, welche die maximale Leistung der „Flotte“ begrenzt (Jahreshöchstlast des HA ohne EFZ abzüglich aktueller Residuallast). Es wird nach „First Come, First Served“ priorisiert. Die Simulation wurde mit dieser Ladestrategie für das Referenzszenario und das Extremszenario durchgeführt.

Optimiertes Lastmanagement (Szenario: „Ext Opt.“):
Die Zielfunktion der Optimierung ist es, die Ladeleistung der EFZ soweit abzusenken, dass die Jahresleistungsspitze respektive der resultierenden Kosten minimiert wird. Die Entscheidung der Priorisierung ladender EFZ am Standort erfolgt entsprechend der Kostenfunktion. Nebenbedingung ist hierbei das Erreichen eines definierten SOCs je Fahrzeug zum Zeitpunkt der Abfahrt (Energiebedarf der nächsten Fahrt soll gedeckt werden). Die Simulation mit dieser Ladestrategie wurde für das Extremszenario durchgeführt.

Abbildung 4-5 verdeutlicht exemplarisch die Unterschiede der Leistungsbegrenzung bei den verschiedenen Ladestrategien. Der Hausanschluss besteht aus Wohneinheiten mit einer Jahresmaximallast von $20\,\mathrm{kW}$ und besitzt drei Wallboxen mit jeweils $11\,\mathrm{kW}$ installierter Leistung. Zum Zeitpunkt der Betrachtung ist das erste Fahrzeug nicht anwesend ($P_{\mathrm{soll}} = 0\,\mathrm{kW}$) und das dritte Fahrzeug lädt in einem hohen SOC-Bereich in welchem Fahrzeugseitig nicht mit

www.ffe.de

Simulation der zukünftigen Netzbelastung durch gesteuerte Ladevorgänge
maximaler Ladeleistung geladen wird ($P_{\mathrm{soll}} = 6\,\mathrm{kW}$). Die Wohneinheiten benötigen in diesem Zeitschritt eine Leistung von $13\,\mathrm{kW}$ und in der Ankunftshierarchie steht das zweite Fahrzeug bereits länger am Standort als das erste. Im ungesteuerten Fall liegt der theoretisch maximale Leistungsbezug am Hausanschluss bei $53\,\mathrm{kW}$ im Zeitschritt des maximalen Leistungsbezugs durch die Wohneinheiten sowie dem Laden aller drei Fahrzeuge mit maximaler Ladeleistung. Ohne Restriktionen werden im aktuellen Zeitschritt keine Leistungen abgeregt, woraus eine Residuallast von $30\,\mathrm{kW}$ resultiert. Im Fall des statischen Lastmanagements ist die theoretisch maximale Leistung des Hausanschlusses auf $29\,\mathrm{kW}$ begrenzt (20 kW durch die Wohneinheiten und 3x 3kW durch die EFZ). Im konkreten Zeitschritt stehen noch 16 kW an Ladeleistung zur Verfügung, woraus eine Reduktion der Leistung am dritten Fahrzeug um 1 kW resultiert, da dieses später am Standort angekommen ist. Im Fall des dynamischen Lastmanagements ist die theoretisch maximale Leistung des Hausanschlusses auf 20 kW, gleich der maximalen Leistung der Wohneinheiten begrenzt. Im konkreten Zeitschritt führt dies am Hausanschluss zu einer Reduktion der Ladeleistung um 10 kW (1 kW bei Fahrzeug 2 und 6 kW bei Fahrzeug 3). Im Fall des optimierten Lastmanagements wurde das Jahresoptimum entsprechend der Kostenfunktion und Nebenbedingungen auf 26 kW identifiziert. Im konkreten Zeitschritt führt die Optimierung entsprechend der notwendigen SOCs bei Abfahrt zu einer bedarfsgerechten Reduktion der Ladeleistung beider Fahrzeuge (3 kW bei Fahrzeug 2 und 1 kW bei Fahrzeug 3).

![img-18.jpeg](img-18.jpeg)

Abbildung 4-5: Exemplarische Darstellung der Leistungsbegrenzung bei verschiedenen Ladestrategien

Die in den Simulationen resultierenden Jahresleistungsspitzen auf Ebene des Verteilnetzes der beiden primär analysierten Regionen sind in Abbildung 4-6 dargestellt. Diese zeigt, wie sich die maximale Belastung für die verschiedenen Szenarien im Jahr anteilig aus der Last der Komponenten zusammensetzt. Die Reduktion der Ladeleistung nach unterschiedlichen Prämissen führt zu unterschiedlichen Zeitpunkten maximaler Last. Im ungesteuerten Fall des Extremszenarios sind in der Altstadt mit $47\,\mathrm{kW}$ ca. $20\%$ und in Pasing mit $97\,\mathrm{kW}$ ca. $29\%$ der Leistungsspitze auf EFZ zurückzuführen. Der Großteil der Leistungsspitze wird also von den übrigen, in diesem Szenario unflexiblen Lasten verursacht. Die Abbildung verdeutlicht, dass alle erprobten Ladestrategien das Ziel einer Absenkung der Jahresleistungsspitze erreichen, jedoch auf Netzebene einen nur sehr geringen Effekt aufweisen. Durch den höheren Anteil der EFZ an der Leistungsspitze in Pasing sind erwartungsgemäß auch die Effekte der Leistungsreduktion durch Lastmanagementstrategien deutlicher.

www.ffe.de

Entwicklung der Netzbelastung im mittleren Verteilnetz der Stadt München
![img-19.jpeg](img-19.jpeg)
Abbildung 4-6: Aufteilung der Jahresleistungsspitze auf Netzebene für Altstadt (links) und Pasing (rechts)

![img-20.jpeg](img-20.jpeg)

Um die Auswirkungen der Lastmanagementstrategien zu verdeutlichen, wurden die Szenarien der Monte-Carlo-Simulationen auch für individuelle Verteilungen analysiert. In Abbildung 4-7 ist die Jahresleistungsspitze ausgewählter HA einer Verteilung für die unterschiedlichen Ladestrategien für Pasing für das Extremszenario abgebildet. Aufgrund der Reduktion der Ladeleistung nach unterschiedlichen Prämissen resultiert die Lastspitze auch an den Haushalten zu unterschiedlichen Zeitpunkten im Jahr. Die gestapelten Balken enthalten die Anteile der Komponenten an der Leistung am HA. Die Jahresleistungsspitze tritt über alle HA im Fall der ungesteuerten Ladung auf (Balken B).

![img-21.jpeg](img-21.jpeg)

![img-22.jpeg](img-22.jpeg)
Abbildung 4-7: Jahresleistungsspitzen auf HA-Ebene für Pasing im Extremszenario

Da beim dynamischen Lastmanagement (Balken C), die verfügbare Leistung immer so gesetzt wird, dass die Leistung am HA ohne EFZ (Balken A) nicht überschritten werden kann, besitzen die Balken A und C aller HA den gleichen Wert. Da diese Grenze hart definiert wird, ist auf HA-Ebene das dynamische Lastmanagement am effektivsten. Die Balken D zeigen die Lastspitze für statisches Lastmanagement, welches der simplen Logik einer Begrenzung um $3\mathrm{kW}$ je Fahrzeug entspricht, wodurch keine Anpassung an die unflexible Last der restlichen Verbraucher stattfindet. Durch die Definition der verfügbaren Leistung unabhängig von der aktuellen statischen Last, ist diese Strategie auf HA-Ebene weniger effektiv. Im Fall der Optimierung (Balken E) ist das Erreichen eines definierten SOC bei Abfahrt eine vorgegebene Randbedingung. Dies hat zur Folge, dass die Ladeleistung je nach Fahrverhalten weniger weit

www.ffe.de

Simulation der zukünftigen Netzbelastung durch gesteuerte Ladevorgänge
gesenkt werden kann, als bei den anderen Lademanagementstrategien. Die Auswertung verdeutlicht, dass die Ladesteuerungen besonders an HA mit vielen EFZ einen starken Einfluss haben. Die Auswertungen zeigen, dass trotz signifikanter Reduktion der Leistungsspitze an den individuellen HA, die Jahresspitzenleistung am Transformator durch die Ladestrategien nur marginal reduziert wird. Dieses Ergebnis wird durch die in Abbildung 4-8 dargestellte Verteilung der für HA individuellen Zeitpunkte der Jahresspitzenlast verdeutlicht.

![img-23.jpeg](img-23.jpeg)
Abbildung 4-8: Jahresdauerline auf Netzebene für Altstadt (links) und Pasing (rechts) bei ungesteuertem Laden im Extremszenario

![img-24.jpeg](img-24.jpeg)

In dieser Darstellung ist jeweils das Extremszenario bei ungesteuerter Ladung für eine einzelne Verteilung abgebildet. Neben der Leistungsspitze am Transformator (schwarzer Quader), zeigen die Diagramme auch die Zeitpunkte der Jahresspitzenleistung der HA (schwarze Punkte) und die Lasten der einzelnen HA unter der Jahresdauerlinie als gestapeltes Flächendiagramm. Die Streuung der Zeitpunkte der Leistungsspitzen der HA verdeutlicht, dass im ungesteuerten Fall ein großer Teil nicht zu den jeweiligen Zeitpunkten der Leistungsspitzen auf Netzebene beitragen. Es wird deutlich, dass Maxima auf HA-Ebene nicht mit Maxima auf Netzebene gleichzusetzen sind und dass sich das hausanschlussscharfe Lastmanagement nur bedingt auf die Leistungsspitze auf Netzebene auswirkt.

Auch auf das resultierende Ladeverhalten wirken sich die Ladestrategien unterschiedlich aus. Nachfolgende Tabelle 4-1 verdeutlicht die Auswirkungen auf dieser Ebene für verschiedene Szenarien im mittleren Verteilnetz der Untersuchungsgebiete.

Tabelle 4-1: Maximal gleichzeitige Ladeleistung der EFZ und mittlere geladene Energie pro EFZ pro Woche im mittleren Verteilnetz der Untersuchungsgebiete

|   | Altstadt |   | Pasing  |   |
| --- | --- | --- | --- | --- |
|   |  Leistung | Energie | Leistung | Energie  |
|  Status quo | 0,00 kW | 0,00 kWh | 11,48 kW | 18,80 kWh  |
|  Referenzszenario ungest. | 34,87 kW | 29,29 kWh | 87,83 kW | 27,40 kWh  |
|  Referenz dyn. LM | 31,96 kW | 29,29 kWh | 70,55 kW | 27,38 kWh  |
|  Extremszenario ungest. | 73,41 kW | 29,20 kWh | 141,62 kW | 27,75 kWh  |
|  Extremszenario stat. LM | 28,21 kW | 29,13 kWh | 66,15 kW | 27,64 kWh  |
|  Extremszenario dyn. LM | 67,01 kW | 29,20 kWh | 115,42 kW | 27,64 kWh  |
|  Extremszenario Opt. | 69,99 kW | 28,17 kWh | 132,96 kW | 27,04 kWh  |

www.ffe.de
Entwicklung der Netzbelastung im mittleren Verteilnetz der Stadt München
Aus der Tabelle wird deutlich, dass der Einfluss verschiedener Ladestrategien auf die EFZ im statistischen Mittel über das gesamte Jahr teils nur geringfügige Differenzen aufweist. Die maximal gleichzeitige Ladeleistung aller EFZ steigt erwartungsgemäß mit der Anzahl an modellierten EFZ und sinkt bei der Anwendung einer leistungsbegrenzenden Ladesteuerung gegenüber den Szenarien mit ungesteuerten Ladevorgängen. Die Anzahl der vollen Ladezyklen bleibt über das gesamte Jahr betrachtet in vergleichbaren Szenarien nahezu konstant. In den Szenarien steigt die Anzahl der EFZ von Status quo über das Referenzszenario bis zum Extremszenario kontinuierlich an, wobei sich im Gebiet „Altstadt“ die Anzahl der Vollzyklen reduziert, während die Anzahl in „Pasing“ ab dem Referenzszenario in etwa konstant bleibt. Hier wird der Einfluss einer geringen Anzahl an EFZ deutlich, wobei im Referenzszenario in „Pasing“ mit einer Durchdringung von 30,3 % bzgl. HA bereits ein Wert erreicht ist, in welchem im Mittel das durchschnittliche EFZ der modellierten Fahrzeugklassen (Klein-/Kompaktklasse 34 kWh (53 %), Mittelklasse 44 kWh (15 %) und Oberklasse 105 kWh (32 %)) resultiert und somit der Einfluss individueller Fahrzeuge deutlich geringer ist als im Gebiet „Altstadt“. Auch die mittlere geladene Energie pro EFZ und Woche bleibt in den Untersuchungsgebieten in Szenarien mit höherer Durchdringung nahezu konstant. Dies ist auf die Fahrprofile zurückzuführen, wonach die EFZ in der Woche für ihre Fahrten im Mittel nur ca. 46 - 50 % der verfügbaren Speicherkapazität (bezogen auf das mittlere EFZ mit 58,22 kWh) benötigen.

Insgesamt resultiert aus der Perspektive der Elektrofahrzeugnutzer auch bei einer starken Begrenzung der Ladeleistung, wie z. B. bei statischem Lastmanagement mit nur 3 kW je EFZ am HA, im statistischen Mittel über alle EFZ und das gesamte Jahr nur marginale Änderungen in der Flottencharakteristik. Dies verdeutlicht einerseits den bereits bestehenden Mehrwert für z. B. Gewerbekunden, welche mittels Lastmanagement bei Elektrifizierung ihrer Flotten potenziell trotzdem keinen höheren Leistungspreise zahlen müssen. Andererseits wird auch das zukünftige Potenzial der verfügbaren Flexibilität von EFZ für Netz- und/oder Systemdienstleistung deutlich, welches durch bidirektional ladende Fahrzeuge noch signifikant gesteigert werden kann.

www.ffe.de
Simulation der zukünftigen Netzbelastung durch gesteuerte Ladevorgänge
7
Auswirkungen auf Stakeholder und Handlungsempfehlungen

Auf Basis der beschriebenen Simulationen wurden Auswirkungen auf Stakeholder analysiert und Handlungsempfehlungen abgeleitet (vgl. Tabelle 4-2), wobei verschiedene Perspektiven berücksichtigt wurden (VNB, Fahrzeugnutzer, Netzanschlussnehmer/Eigentümer/Mieter).

Tabelle 4-2: Handlungsempfehlungen aus verschiedenen Perspektiven

|  |   |   |
| --- | --- | --- |
|  Zentrale Lastmanagement Strategie respektive gezielte Steuerbarkeit aus VNB-Sicht sinnvoller. | Lastmanagement für HA mit Jahresleistungspreissystem oder begrenzter Netzanschlusskapazität empfehlenswert. | Dynamisches Lastmanagement oder eine optimierte Ladestrategie sollten präferiert werden.  |
|  PtH-Systemen tragen signifikant zur Netzbelastung bei. Steuerbarkeit und Einbezug in Lastmanagementsystemen ist sinnvoll. | Einbezug von PtH-Systemen in das Lastmanagement sinnvoll zur Last- und Kostenreduktion. | Einsatz einer Priorisierungslogik ist empfehlenswert. First-Come-First-Served-Logik bereits ausreichend zur Deckung des Mobilitätsbedarfs.  |
|  Fokus des dezentralen Lastmanagements auf Hausanschlüsse mit vielen EFZ (Flotten). | Lastmanagement bei Hausanschlüssen mit EFZ-Flotten und hohen Lastspitzen sinnvoll. |   |

Die Simulationsergebnisse zeigen, dass alle Strategien des gesteuerten Ladens nur eine sehr geringe Absenkung der Leistungsspitze auf Netzebene bewirken und sich somit nur marginal positiv auf die Leitungsauslastung und das Spannungsband auswirken. Gleichzeitig wäre der Aufwand einer tatsächlichen Implementierung sehr hoch, da in der betrachteten Konfiguration alle EFZ in die Ladsteuerungen einbezogen werden und demnach alle HA über ein eigenes Lastmanagementsystem verfügen müssten. Aus Perspektive des VNB wäre eine pauschale Einführung von Lastmanagement am HA auf Basis der Simulationen nicht zu empfehlen und potenziell eine übergeordnete Lastmanagementstruktur zu präferieren. Dies ist allerdings mit regulatorischen Hürden verbunden, da solche Eingriffe gesetzlich legitimiert werden müssten. Die kontroverse Diskussion um die Novelle zu § 14a des Energiewirtschaftsgesetztes zeigt, dass die Umsetzung zentraler Regelung auf deutlichen Widerstand seitens Industrie und Autoherstellern stößt, weshalb zum aktuellen Zeitpunkt noch keine Entscheidung zur genauen Ausgestaltung getroffen wurde. Unabhängig davon, ob das Lastmanagement auf HA-Ebene oder zentral durchgeführt wird, könnte eine vereinfachte Implementierung eines Lastmanagements, mit Fokus auf Hausanschlüsse mit vielen flexiblen Verbrauchern respektive hoher Last, zur Entlastung des Netzes beitragen. Jedoch muss auch hierfür der regulatorische Rahmen und ein potenzieller Anreiz (z. B. verminderte/variable Netzentgelte) geschaffen werden. Speziell für die beiden untersuchten Netzgebiete Altstadt und Pasing in München muss festgehalten werden, dass auch im ungesteuerten Fall nur in einzelnen Verteilungen Netzüberlastungen auftreten. Im Verteilungsmittel sind keine Überlastungen zu erwarten, weshalb die Verpflichtung zum Lastmanagement im Hinblick auf den analysierten Zeithorizont (Jahr 2030) aus Perspektive des Netzes noch nicht notwendig ist. Die Auswertungen zeigen

www.ffe.de

Entwicklung der Netzbelastung im mittleren Verteilnetz der Stadt München
außerdem, dass neben den im Projekt fokussierten Elektrofahrzeugen auch ein großer Anteil der Belastung auf PtH-Systeme zurückzuführen ist. Diese müssen perspektivisch ebenfalls flexibel gesteuert werden und können dadurch zur Entlastung beitragen.

Aus Sicht eines Netzanschlussnehmers kann ein Lastmanagement auf HA-Ebene finanziell durchaus sinnvoll sein. Einige Unternehmen bieten bereits kommerziell Lastmanagement für Elektrofahrzeugflotten an. Hierbei wäre beispielsweise der „ChargePilot“ von The Mobility House, „E-Mobility“ von SAP, „XENON charge“ von gridX und „EisBär SCADA“ von ABB zu nennen. Ein Lastmanagementsystem ist einerseits an HA sinnvoll, an denen die Ladeleistung aller Ladepunkte größer als die verfügbare Netzanschlusskapazität ist. Andererseits ist ein Lastmanagement besonders für Netzanschlussnehmer sinnvoll, deren Netzentgelt über ein Jahresleistungspreissystem abgerechnet wird. In der Regel sind dies kommerzielle Verbraucher mit einem Jahresenergiebedarf größer 100 MWh (RLM-Kunden). Ein Großteil des Netzentgeltes hängt in diesem Fall von der Jahresleistungsspitze des HA ab. Für Kunden mit einer Jahresbenutzungsdauer ≥ 2500 Stunden pro Jahr fielen hierfür in München im Niederspannungsnetz 2020 beispielsweise 111 €/kW pro Jahr an /SWM-07 20/. Das Einsparpotenzial hängt von der Anzahl Ladepunkte am HA und der Anzahl ladender EFZ ab. Abbildung 4-9 zeigt die durchschnittliche erreichte Reduktion der Jahresleistungsspitze abhängig von der Anzahl EFZ am Hausanschlusspunkt. Bereits ab einem gesteuerten EFZ kann eine Reduktion der Jahresleistungsspitze des HA erreicht werden. Aufgrund der zusätzlich notwendigen Ladeleistung durch die EFZ steigt mit der Anzahl an EFZ somit auch das Einsparpotenzial bei Anwendung eines Lastmanagements. In den Simulationen wurde an HA mit mehr als drei Fahrzeugen die Leistungsspitze mittels dynamischem Lastmanagement, gegenüber der ungesteuerten Referenzsimulation im Verteilungsmittel um ca. 20 kW reduziert. Unter Annahme des oben genannten Leistungspreises entspricht dies einer Einsparung von mehr als 2.000 € pro Hausanschluss und Jahr. Auch hier ist die potenzielle Einbindung von PtH-Systemen sinnvoll, da durch den Einbezug die Reduktion der Spitzenlast respektive die Einsparung nochmals deutlich verbessert werden können.

![img-25.jpeg](img-25.jpeg)
Abbildung 4-9: Reduktion der Leistungsspitze für die verschiedenen Ladestrategien bezogen auf ungesteuertes Laden

www.ffe.de

Auswirkungen auf Stakeholder und Handlungsempfehlungen
9
Aus Perspektive des Fahrzeugnutzers hat ein Lastmanagement, sofern dieser nicht auch die Rolle des Hausanschlussnehmers einnimmt, keinen direkten Vorteil sondern schränkt diesen ein. Werden mehrere Fahrzeuge lediglich für kurze Zeit angesteckt und die verfügbare Ladeleistung reicht nicht zum Laden aller Fahrzeuge aus, kann der gewünschte SOC bei Abfahrt möglicherweise nicht erreicht und im Worstcase die Fahrt nicht ohne Zwischenladung absolviert werden. Abbildung 4-10 verdeutlicht die statistische Auswertung der Simulation, wobei das Verteilungsmittel des Anteils der Fahrten pro Jahr dargestellt ist, bei welchen der Abfahrts-SOC kleiner dem für die nächste Fahrt gewünschten SOC ist.

![img-26.jpeg](img-26.jpeg)
Abbildung 4-10: Auswirkungen der Ladestrategien auf den gewünschten SOC bei Abfahrt

Insgesamt zeigt sich, dass die in den Simulationen analysierten Elektromobilisten durch Lastmanagement nur marginal eingeschränkt wurden. In 97,5 % aller Ladevorgänge wird der für die nächste Fahrt notwendige SOCWunsch durch die Ladung erreicht. Die größte Einschränkung wird dabei durch statisches Lastmanagement hervorgerufen, wobei 3 kW je Elektrofahrzeug am HA einer sehr extremen Einschränkung entspricht (auch die nominale Ladeleistung bei Hausanschlüssen mit nur einem Fahrzeug wurde entsprechend dieser Formel auf 3 kW reduziert).

Zur Reduktion der Einschränkungen für die Fahrzeugnutzer, ist die Implementierung einer Priorisierungslogik von Fahrzeugen empfehlenswert. Verschiedene Priorisierungsstrategien wurden in /FFE-118 20/ untersucht. Die Priorisierung kann sich z. B. nach dem Ansteckzeitpunkt, der Ansteckdauer oder dem SOC der EFZ richten. Fahrzeuge, die früher abfahren, können gegenüber Fahrzeugen mit langen Standzeiten bevorzugt geladen werden. Dafür ist es jedoch wichtig, dass das Lastmanagementsystem Informationen über die Ansteckdauer oder den SOC der EFZ erhält. In der Praxis kann dies zum Beispiel über eine App oder über einen Prognosealgorithmus erfolgen.

www.ffe.de

Entwicklung der Netzbelastung im mittleren Verteilnetz der Stadt München
# 5 Energiewirtschaftliche Bewertung der Entwicklung der Elektromobilität

Mit 32 % hatte der Energiesektor den größten Beitrag zu den gesamtdeutschen Emissionen im Jahr 2019 /FME-01 20/. 25 % der Gesamtendenergie im Jahr 2018 wurde in Deutschland vom Verkehrssektor verbraucht /IEA-01 20/ und 84 % der CO₂ Emissionen im Verkehrssektor sind im Jahr 2016 durch den Straßenverkehr entstanden /FFE-104 19/. Um die deutschen Klimaschutzziele zu erreichen, müssen Emissionen durch z. B. Elektrifizierungsmaßnahmen vermindert werden. Es ist vorgesehen, die Emissionen im Verkehr durch Elektromobilität und Ladestrom aus erneuerbaren Energien zu reduzieren /ÜNB-01 19/. Außerdem trägt der Technologiewechsel auf Elektrofahrzeuge dazu bei, lokale Emissionen zu mindern und die Luftqualität zu verbessern /FFE-02 20/. Bestehende Literatur prognostiziert die Zunahme von Spitzenlasten im Stromnetz durch Elektromobilität /ISI-27 15/ /ENGEL-01 18/. Da die meisten Ladevorgänge zu Hause stattfinden, treten Überlastungen besonders in den unteren Spannungsebenen auf /US-02 19/ /PSLU-01 18/ /MUL-01 19/. Vor allem Städte stellen einen Konzentrationspunkt für starke Lastzunahmen durch Ladevorgänge dar, da EFZ primär in urbanen Regionen erwartet werden /US-02 19/. Im Rahmen des Projekts wurden die Ladevorgänge aus energiewirtschaftlicher Perspektive analysiert und dadurch die Lastentwicklung und resultierende Emissionen beurteilt, sowie mögliche Geschäftsmodelle im Kontext der Elektromobilität analysiert.

## 5.1 Bewertung von Ladevorgängen zukünftiger Elektromobilität in München aus energiewirtschaftlicher Perspektive

Zur Analyse des Einflusses zukünftiger Elektromobilität auf den aggregierten Lastgang der Stadt München wurden sowohl auf unterschiedlichem Ladestrategien, als auch auf verschiedene Szenarien der Elektrifizierung (äquivalent zu den Szenarien in Kapitel 3.3.4) untersucht. Das Ziel waren die Bewertung der Lastveränderung auf Ebene Münchens sowie der Auswirkungen hinsichtlich der resultierenden Emissionen.

## 5.1.1 Entwicklung des Summenlastgangs der Stadt München

In einem ersten Schritt wurde der reale Summenlastgang Münchens aus dem Jahr 2019 analysiert, um diesen als Referenz zur Beurteilung der Änderung zu nutzen. Es folgte die synthetische Modellierung des Summenlastgangs für München im Status quo und für das Jahr 2030. Dazu wurden Daten aus dem eXtremOS-Projekt synthetisiert, zur Verifizierung mit den realen Daten verglichen und die Lastgänge der Sektoren Industrie, GHD, Verkehr und Haushalte aggregiert. Abbildung 5-1 verdeutlicht die prognostizierte Änderung der Lastgänge in Form einer sortierten Jahresdauerlinie. Die Analyse der Sektorenlastgänge zeigt, dass Gewerbe- und Haushaltslasten den Summenlastgang Münchens dominieren. Durch Elektrifizierung der Wärmeversorgung privater Haushalte wird ein Anstieg der Last im Haushaltssektor erwartet. Im GHD Sektor werden keine signifikanten Änderungen erwartet. Energieeffizienz- und Suffizienz-Maßnahmen in der Industrie sorgen zukünftig für einen niedrigeren Industrielastgang. Ein Zuwachs an Elektromobilität erhöht die Lastkurve des öffentlichen Verkehrs.

www.ffe.de
Bewertung von Ladevorgängen zukünftiger Elektromobilität in München aus energiewirtschaftlicher Perspektive
![img-27.jpeg](img-27.jpeg)

Abbildung 5-1 Jahresdauerlinien zu prognostizierten Sektor-Lastgängen für die Jahre 2020 und 2030

Zum besseren Vergleich wurden Typlastgänge in Abhängigkeit der Wochentage und der Saison berechnet. Die Analyse der modellierten Typlastgänge verdeutlicht, dass der jährliche Summenlastgang der Stadt München wöchentliche Muster aufweist, mit Lastspitzen an Wochentagen und niedrigeren Lasten an Wochenenden und an Feiertagen. Typisch für Kontinentalklima ist die Netzbelastung im Winter höher als im Sommer, da zu dieser Zeit der Bedarf nach Beleuchtung und elektrischen Heizungen stärker ist /VDEW-01 99/. Vereinzelte Lastspitzen im Sommer können z. B. durch Kühlungsprozesse erklärt werden /IEA-03 20/. Die Analyse der Typtage zeigt, dass Lastspitzen am Vormittag und am Abend auftreten, während in der Nacht nur eine geringe Grundlast vorherrscht. Der synthetische Summenlastgang bildet die Form des realen Lastgangs von 2019 mit ausreichender Genauigkeit nach.

Im nächsten Schritt wurden Lastgänge von Elektrofahrzeugen für verschiedene Sensitivitäten modelliert, um darauf die zusätzlich resultierende Belastung abzuleiten. Dafür wurden mit dem Modell GridSim für jede Sensitivität 1.000 Ladelastgänge erzeugt. Die Ladeszenarien wurden differenziert nach ungesteuertem Laden, eigenverbrauchoptimiertem Laden oder Laden mit Peakshaving am Hausanschluss. Auch wurde die Unterscheidung vorgenommen ob das Fahrzeug einer Wohn- oder Gewerbeeinheit zugeordnet war. Durch Überlagerung und Mittelwertbildung wurden aus den einzelnen Ladelastgängen mittlere Lastprofile je Sensitivität gebildet. Die Hochrechnung der mittleren Lastgänge mit prognostizierten Hochlaufzahlen ergab aggregierte Lastgänge für Status quo, Referenz- und Extremszenario. Bei den Zukunftsszenarien wurde zwischen ungesteuertem und gesteuertem Laden unterschieden.

Im Status quo wurden 502 EFZ auf München verteilt, wodurch sowohl die Maximallast, als auch die durchschnittliche Ladelast im Status quo vernachlässigbar gering ist. Durch das Referenzszenario und die Verteilung von 71.163 EFZ auf München steigt die Belastung durch die Elektromobilität bis zum Jahr 2030 stark an. Die maximale Ladelast der überlagerten Profile beträgt 70,08 MW (im Mittel um 19,23 MW) im ungesteuerten Fall und 58,31 MW (im Mittel um 17,20 MW) für gesteuertes Laden. Im Extremszenario mit 221.725 Elektrofahrzeugen steigt die maximale Ladelast im ungesteuerten Fall um 210,44 MW (im Mittel um 51,57 MW) und im gesteuerten Fall auf 156,64 MW (im Mittel um 47,51 MW). Der Vergleich der zusätzlichen

www.ffe.de

Energiewirtschaftliche Bewertung der Entwicklung der Elektromobilität
Belastung durch Elektromobilität in Relation zur realen Summenlast im Jahr 2019 verdeutlicht, dass die Ladevorgänge den Summenlastgang der Stadt München deutlich verändern. Im Verhältnis zur prognostizierten Lastentwicklung der anderen Sektoren liegt die Änderung der Last durch Ladevorgänge in einer vergleichbaren Größenordnung.

## 5.1.2 Bewertung der resultierenden Emissionen

Zur Bewertung der aus den Ladevorgängen potenziell resultierenden Emissionen wurde im Projekt ein vereinfachter Ansatz gewählt. Auf Basis von zwei CO₂-Emissionszeitreihen wurden die Emissionen je Stützjahr kalkuliert und die Änderungen verglichen. Die Emissionszeitreihen basieren auf einem Referenz/Startszenario (Start) und einem Klimaschutzszenario (fuEL), in welchen der in den Stützjahren erwartete Kraftwerkseinsatz respektive Strommix abgebildet wurde /FFE-104 19/. Abbildung 5-2 verdeutlicht die Änderungen der Emissionen zwischen 2020 und 2030 für die verschiedenen Szenarien und Sektoren. Durch die zusätzlichen Ladevorgänge resultiert ein zusätzlicher Strombedarf, welcher in 2030 voraussichtlich auch noch durch zum Teil fossile Erzeugung gedeckt wird, wodurch die Änderung der Emissionen durch Ladevorgänge positiv in dieser Bilanz erscheinen. Die Kompensation der Emissionen von Fahrten durch Verbrenner-Fahrzeugen, sowie die Bilanzierung der Wertschöpfungskette von Ladeinfrastruktur und Elektrofahrzeugen wurde in dieser Darstellung explizit nicht berücksichtigt.

![img-28.jpeg](img-28.jpeg)
Abbildung 5-2 Berechnete CO₂-Emissionsänderungen für die verschiedenen Sektoren und Szenarien

Die Berücksichtigung der Kompensation von Verbrenner-Fahrten, würde in der Bilanz die Emissionen deutlich reduzieren. So würde der Technologiewechsel z. B. bei Annahme der Kompensation der Fahrten einer Flotte von Mittelklasse Benzinern (mit durchschnittlicher Fahrleistung von 13.600 km/Jahr und Ausstoß von 0,17 kg CO₂ / km) im Extrem-Szenario der Reduktion von ca. 520.000 t CO₂ / Jahr entsprechen.

Insgesamt sinken die Emissionen trotz der Annahme einer steigenden Gesamtlast bis zum Jahr 2030 in München. Grund hierfür ist der deutlich veränderte Strommix, welcher entsprechend der Prognosen zukünftig einen größeren Anteil an erneuerbaren Energien enthalten wird. Im Referenzszenario betrugen die CO₂-Emissionen für das Jahr 2030 70,6 %, nach dem Klimaschutzszenario lediglich 58,1 % der heutigen Emissionen.

www.ffe.de

Bewertung von Ladevorgängen zukünftiger Elektromobilität in München aus energiewirtschaftlicher Perspektive
Bewertung ausgewählter Geschäftsmodelle

Maßnahmen zur Förderung von Elektromobilität, z. B. der sogenannte „Umweltbonus“ /BAFA-04 16/ sowie entsprechende LIS /BR-02 19/, führen nicht nur zu einer steigenden Netzbelastung durch zusätzliche Ladevorgänge, sondern beeinflussen verschiedene energiewirtschaftliche Geschäftsmodelle. Unter der Prämisse „Alternativen zum Netzausbau“ wurden Geschäftsmodelle untersucht, welche die Verteilnetzbelastung durch EFZ verhindern (niedrige Ladeleistung), reduzieren (anderweitige Energieaufnahme) oder zeitlich beeinflussen (zeitliche Verschiebung von Ladevorgängen).

Als Grundlage für die Geschäftsmodellentwicklung und -bewertung wurden, zunächst entsprechend relevante Use Cases (Anwendungsfälle) identifiziert. Unter einem Use Case versteht man eine Spezifikation einer Menge von Aktionen, welche von einem System durchgeführt wird und ein beobachtbares Ergebnis erbringt /DKE-01 17/. Ein Geschäftsmodell dagegen ist eine vereinfachte Abbildung einer auf Gewinn ausgerichteten Unternehmung, bestehend aus den wesentlichen Elementen der Unternehmung und deren Verknüpfungen. Dabei ermöglicht das Geschäftsmodell eine Differenzierung gegenüber Wettbewerbern, die Klarstellung von Zusammenhängen und die Erzielung eines Wettbewerbsvorteils. /SCHAL-01 13/ Ein Use Case beschreibt somit ein Gesamtsystem, während sich ein Geschäftsmodell auf die wirtschaftliche Ausgestaltung aus Sicht einer Unternehmung (eines Akteurs) fokussiert. Um Geschäftsmodelle im Projekt zu entwickeln, wurden zunächst mögliche Anwendungsfälle beschrieben. Anhand der am Use Case beteiligten Akteure wurden anschließend Geschäftsmodelle abgeleitet und bewertet.

Die im Projekt analysierten Use Cases beschränken sich auf unidirektionale Ladevorgänge an privater und gewerblicher Ladeinfrastruktur, wobei die Restriktion einer netzentlastenden Wirkung in den Fokus gestellt wurde. Da bei den Use Cases tarifoptimiertes Laden und zeitliche Arbitrage viele Elektrofahrzeuge gleichzeitig auf ein Marktsignal reagieren, können diese Use Cases zu Netzüberlastungen führen /FFE-08 22/. Auch das Grünstromladen kann sich je nach Definition sowohl netzstützend als auch netzbelastend auswirken. Netzbelastung tritt auf, wenn viele Ladevorgänge anhand eines zentralen Signals ausgerichtet werden, ähnlich zum tarifoptimierten Laden. /FAT-01 21/ Da diese Use Cases nur unter bestimmten Bedingungen zu einer Reduktion der Netzbelastung führen bzw. auch der gegenteilige Fall eintreten kann, wurden diese Use Cases in dieser Untersuchung nicht berücksichtigt. Die Use Cases Regelleistungserbringung und Blindleistungsbereitstellung dienen der Erhaltung der Netzfrequenz bzw. -qualität, dienen jedoch nur bedingt der Reduktion der Netzbelastung, weshalb diese nicht berücksichtigt wurden. Die Use Cases Redispatch, Powerbox und Notstromversorgung erfordern in sinnvoller Anwendung das bidirektionale Laden und wurden im Projekt ebenfalls nicht berücksichtigt. Entsprechend der Use Cases und den beteiligten Akteuren werden die Geschäftsmodelle von zentralen Stakeholdern der Energiewirtschaft untersucht. Da ein Use Case sehr komplex und umfassend werden kann, wurde die Ausgestaltung hinsichtlich Detailtiefe und Umfang durch folgende Aspekte eingeschränkt:

Geschäftsmodelle von Unternehmen, welche Lademanagementsysteme, Ladesäulen oder Messinstrumente herstellen, wurden nicht betrachtet.
Aus Komplexitäts-Gründen wurden weitere energiewirtschaftliche Akteure, wie z. B. Bilanzkreiskoordinatoren, Bilanzkreisverantwortliche oder Messstellenbetreiber, nicht berücksichtigt.

www.ffe.de

Energiewirtschaftliche Bewertung der Entwicklung der Elektromobilität
Die Geschäftsmodelle wurden generisch beschrieben. Eine konkrete, unternehmensspezifische Ausgestaltung von Geschäftsmodellen wurde nicht behandelt.
In der Praxis kann die Umsetzung einer Kombination aus mehreren Use Cases sinnvoll sein (Multi-Use Anwendungsfälle). Dies war ebenfalls kein Fokus der Analysen.

Die Geschäftsmodelle wurden je Use Case aus Sicht der beteiligten Akteure in Form von Business Model Canvas' entwickelt und durch eine SWOT-Analyse bewertet. Das Business Model Canvas nach Osterwalder und Pigneur (2011) beschreibt das Werteversprechen, die Kundensegmente, die Kundenbeziehung, Kanäle, Schlüsselaktivitäten, Schlüsselressourcen und Schlüsselpartner, sowie die Kosten und Erlöse eines Geschäftsmodells /OST-01 11/. Mithilfe einer SWOT-Analyse (Abkürzung engl. für „Strengths“, „Weaknesses“, „Opportunities“ und „Threads“, dt. „Stärken“, „Schwächen“, „Chancen“ und „Risiken“) kann ein Geschäftsmodell analysiert werden, indem unternehmensinterne Stärken und Schwächen sowie externen Chancen und Risiken identifiziert werden /SRG-01 15/.

Die erarbeiteten Geschäftsmodelle werden nachfolgend durch eine kurze Beschreibung des zugrundeliegenden Use Cases, einer grafischen Darstellung des Business Model Canvas, einer SWOT-Analyse inklusive Einordnung und Fazit dargestellt.

## 5.2.1 Spitzenlastbegrenzung

Beim Use Case Spitzenlastbegrenzung wird die Lastspitze an einem Gewerbestandort mit registrierender Leistungsmessung durch gesteuertes Laden von gewerblich genutzten EFZ oder denen von Angestellten begrenzt. Die EFZ laden nur zu Zeiten mit geringer Gewerbelast und insbesondere wird das Laden zu Zeiten hoher Gewerbelast vermieden, um die gesamte Standortlast zu reduzieren. Durch die Begrenzung der Höchstlast innerhalb eines bestimmten Abrechnungszeitraums (Monat/Jahr) bezahlt das Gewerbe einen geringeren Leistungspreis als Bestandteil der Netznutzungsentgelte. Der Preisanreiz für diesen Use Case entstammt somit aus dem Netz. Da die Fahrzeuge unidirektional laden, wird der Use Case als Spitzenlastbegrenzung und nicht als Spitzenlastkappung, bei der EFZ bidirektional Strom zu Zeiten von hoher Gewerbelast einspeisen können, bezeichnet. Durch die Begrenzung der maximalen Standortlast wird zusätzliche Netzbelastung vermieden. Die Steuerung der EFZ erfolgt durch ein Lastmanagement lokal am Gewerbestandort.

Beim Use Case Spitzenlastbegrenzung ergibt sich für das Gewerbe ein Geschäftsmodell, indem es ein Lademanagement für die Ladevorgänge zur Begrenzung der Spitzenlast am Standort verwendet.

## Gewerbe begrenzt Standortlast

Das Gewerbe reduziert die Netzbelastung durch die Begrenzung der maximalen Standortlast. Gleichzeitig kann es seinen Angestellten günstigen Ladestrom am Arbeitsplatz zur Verfügung stellen sowie eigene elektrische Gewerbefahrzeuge günstig laden. Durch die Tatsache, dass es sich um einen behind-the-meter Use Case und eine Eigenoptimierung handelt, gibt es bei diesem Geschäftsmodell neben den Angestellten keine weiteren Kund:innen. Das Gewerbe profitiert selbst von reduzierten Netzentgelten, indem es einen geringeren Leistungspreis bezahlt. Zu den Schlüsselaktivitäten des Gewerbes gehört insbesondere die Optimierung der Ladezeiten in Abhängigkeit der Fahrpläne für Verbrauchsprozesse am Standort /NEXT-04 21/. Eine wichtige Ressource bei diesem Geschäftsmodell ist das Lastmanagement am Standort. Das Business Model Canvas zu diesem Geschäftsmodell wird in Abbildung 5-3 und die SWOT-Analyse in Abbildung 5-4 dargestellt.

www.ffe.de
Bewertung ausgewählter Geschäftsmodelle
5
6
Energiewirtschaftliche Bewertung der Entwicklung der Elektromobilität

Gewerbe begrenzt Standortlast

|  Schlüsselpartner | Schlüsselaktivitäten | Wertversprechen | Kundenbeziehung | Kundensegmente  |
| --- | --- | --- | --- | --- |
|  • VNB
• Angestellte, die die EFZ des Gewerbes dienstlich nutzen (z.B. für die Auslieferung) und am Gewerbestandort laden
• IT-Dienstleister (Hersteller eines Lastmanagements) | • Optimierung von Ladevorgängen in Abhängigkeit der Fahrpläne für Verbrauchssprozesse am Gewerbe
• Bereitstellung von Ladestudien am Gewerbestandort für Angestellte/elektrische Gewerbefahrzeugkosten | • Reduzierte Netzbelastung durch die Begrenzung der Maximaleistung innerhalb eines bestimmten Zeitraums am Standort
• Günstiger Ladestrom für Angestellte am Arbeitsplatz (=Gewerbestandort) | • Regelmäßiger (z.B. monatliches) Abrechnung an die Angestellten | • Angestellte, die mit eigenem Fahrzeug am Gewerbestandort laden
• Darüber hinaus keine Kund:innen, da es sich um einen Behind-the-meter Use Case und eine Eigenoptimierung handelt  |
|  Ersten |   | Erlöse  |   |   |
|  • Investitions- und Betriebskosten für das Lastmanagement
• Investitions-, Installations- und Wartungskosten der Ladesäule(n)
• Personal- und Softwarekosten für die Abstimmung von Fahrplänen für die Verbrauchsprozesse am Standort
• Opportunitätskosten für möglicherweise suboptimale Ladevorgänge | • Reduzierte Netznutzungsentgelte durch geringeren Leistungspreis  |   |   |   |

Abbildung 5-3: Business Model Canvas: Geschäftsmodell eines Gewerbes zum Use Case Spitzenlastbegrenzung

Gewerbe begrenzt Standortlast

|  Stärken | Schwächen  |
| --- | --- |
|  • Kostenersparnisse durch geringere Netznutzungsentgelte (geringerer Leistungspreis) | • EFZ können nicht nach Bedarf schneller geladen oder Verbrauchsprozesse hochgefahren werden, da die Netzanschlussleistung evtl. nicht ausreicht
• Änderungen der Verbrauchsprozesse erfordern ein hohes Maß an Koordination mit EFZ-Ladevorgängen  |
|  Chancen | Risiken  |
|  • Ladestationen für EFZ können installiert werden, ohne dass der Netzanschluss erweitert werden muss
• Durch vermiedenen Netzausbau zukünftig keine steigenden Netzentgelte | • Mithilfe der Angestellten erforderlich: Angestellte könnten fehlerhafte oder gar keine Eingangsinformationen für das Lastmanagement liefern (z.B. gewünschter SOC, Abfahrtszeit, ...)  |

Abbildung 5-4: SWOT-Analyse: Analyse des Geschäftsmodells eines Gewerbes zum Use Case Spitzenlastbegrenzung

Langfristig bietet die Spitzenlastbegrenzung Kostenersparnisse beim Stromverbrauch. Dabei ist es essentiell, dass die Ladevorgänge optimal auf die Verbrauchsprozesse abgestimmt werden und die Netzanschlussleistung ausreichend groß ausgelegt wird, damit keine Verluste durch Lastkonflikte entstehen. Dabei sollte auch berücksichtigt werden, ob die Standortlast zukünftig steigen wird. Das Gewerbe ist zur Optimierung der Erlöse darauf angewiesen, dass eine ausreichende Anzahl an Angestellten beim Lademanagement partizipiert. Eine größtmögliche Automatisierung von Prozessen und Kommunikationsschnittstellen verhindert, dass durch menschliche Fehler Fahrzeuge nicht korrekt geladen werden.

## 5.2.2 Eigenverbrauchsoptimiertes Laden

Beim Eigenverbrauchsoptimierten Laden werden ein oder mehrere EFZ mit dem Strom aus einer am Standort installierten Erzeugungseinheit z. B. PV-Anlage geladen. Bei Bedarf, wenn die Zeiten der Ladevorgänge nicht mit den Zeiten der PV-Stromerzeugung übereinstimmen, kann ein Energiespeicher zu Erhöhung des Eigenstromverbrauchs am Standort verwendet
werden. Die Verwendung von eigenerzeugtem PV-Strom zum Laden des EFZ ist günstiger als der Strombezug aus dem Netz, weshalb das „Preissignal“ einen lokalen Ursprung am Standort hat. Der Use Case wird priorisiert im privaten Haushalt umgesetzt, weshalb dieser primär Eigenheimbesitzer:innen adressiert und das Geschäftsmodell darin liegt, den Ladestrom aus der eigenen PV-Anlage bereitzustellen. Der Mehrwert des Use Cases besteht im günstigeren Ladestrom für Elektrofahrzeugfahrer:innen. Daneben profitiert der Netzbetreiber von einer geringeren Netzbelastung. Da der Netzbetreiber bei der Umsetzung dieses Use Cases nicht aktiv beteiligt ist, hat er kein eigenes Geschäftsmodell.

Grundsätzlich ist das Eigenverbrauchsoptimierte Laden auch an einem Gewerbestandort oder an einer öffentlichen Ladesäule möglich. Jedoch ist der gewerbliche Vertrieb von PV-Strom deutlich komplexer als der private Eigenverbrauch. Das liegt unter anderem an einer anderen Rollenverteilung beim öffentlichen Laden. Nicht der Standortbetreiber der Ladeinfrastruktur (CPO), sondern der Elektromobilitätsdienstleister (EMP) legt die Preise für die Ladevorgänge fest /BDEW-11 20/.

## Eigenheimbesitzer:in lädt eigenverbrauchsoptimiert

Ziel des Use Cases ist der Verbrauch von günstigem und grünem Ladestrom durch die Nutzung von PV-Strom am Standort. Im Haushalt gibt es keine Kund:innen, da es sich um einen behind-the-meter Use Case und eine Eigenoptimierung handelt. Die Erlöse des Geschäftsmodells bestehen also rein aus den Kosteneinsparungen des günstigeren Ladestroms. Neben den Anfangsinvestitionen für die PV-Anlage kann von dem Geschäftsmodell langfristig ohne große Kostenpunkte profitiert werden.

Durch den hohen Autarkiegrad können sich Haushalte mit dem eigenverbrauchsoptimierten Laden gegen hohe Strompreise absichern. Um die Abhängigkeit von der fluktuierenden Stromerzeugung weiter zu reduzieren und sicherzustellen, dass für die erforderlichen Ladevorgänge auch in Zeiten ohne Sonneneinstrahlung PV-Strom vorhanden ist, kann die zusätzliche Installation eines Stromspeichers am Standort sinnvoll sein. Das Business Model Canvas zu diesem Geschäftsmodell wird in Abbildung 5-5 und die SWOT-Analyse in Abbildung 5-6 dargestellt.

![img-29.jpeg](img-29.jpeg)
Eigenheimbesitzer:in lädt eigenverbrauchsoptimiert

Abbildung 5-5: Business Model Canvas: Geschäftsmodell von Eigenheimbesitzer:innen zum Use Case Eigenverbrauchserhöhung

www.ffe.de

Bewertung ausgewählter Geschäftsmodelle
8
www.ffe.de
Energiewirtschaftliche Bewertung der Entwicklung der Elektromobilität

Eigenheimbesitzer:in lädt eigenverbrauchsoptimiert

|  Stärken ⚠️ | Schwächen 🚶  |
| --- | --- |
|  • Unabhängigkeit von Marktpreisen für Strom durch Eigenstromerzeugung
• Ausnutzung von erneuerbaren Energien zum Laden des/ der EFZ
• Hoher Autarkiegrad | • Abhängigkeit von fluktuierender Stromerzeugung
• Notwendige Messung für Steuerung des Ladevorgangs in Abhängigkeit der PV-Leistung  |
|  Chancen | Risiken  |
|  • Absicherung gegen zukünftig möglicherweise steigende oder schwankende Strompreise | • Ladevorgänge sind nicht immer in die Zeiten der PV-Erzeugung verschiebbar  |

Abbildung 5-6: SWOT-Analyse: Analyse des Geschäftsmodells von Eigenheimbesitzer:innen zum Use Case Eigenverbrauchserhöhung

## 5.2.3 Flottenmanagement

Beim Flottenmanagement wird die maximale Netzbezugsleistung am Gewerbe durch eine optimierte Ladung der Fahrzeuge reduziert. Dazu werden die Ladevorgänge der EFZ unter Berücksichtigung der maximalen Netzbezugsleistung unterschiedlich priorisiert. Zur Bestimmung des Priorisierungslevels für einzelne Fahrzeuge müssen vor dem Ladevorgang über eine Schnittstelle, z. B. eine App oder das Fahrzeug-Backend, die Fahrzeugdaten (z. B. SOC) und die Nutzerdaten (z. B. die gewünschte Abfahrtszeit) an das Lademanagement kommuniziert werden. Der Use Case Flottenmanagement ist eine Erweiterung des Use Cases Spitzenlastbegrenzung und schließt diesen ein. Der Use Case bietet Gewerbebetrieben die Möglichkeit zur Reduktion der Netzentgelte durch einen niedrigeren Leistungspreis und zusätzliche Wertschöpfung durch das Sicherstellen der Mobilität. Der Ursprung des Preissignals stammt aus dem Netz. Der Use Case Flottenmanagement ist für Gewerbestandorte, an denen mehrere Fahrzeuge gleichzeitig laden, sinnvoll. Für Haushalte wird kein Leistungspreis berechnet und hier laden selten mehrere Fahrzeuge gleichzeitig, wodurch der Use Case keine Anwendung beim privaten Laden findet. Da an öffentlichen Ladesäulen nicht der CPO den Strom (und die Netzentgelte) bezahlt, sondern die Elektrofahrzeugfahrer:innen über den EMP, findet auch hier der Use Case keine Anwendung /BDEW-11 20/. Aus dem Use Case ergibt sich daher ein Geschäftsmodell für Gewerbebetriebe.

## Gewerbe nutzt Flottenmanagement

Das Gewerbe bezahlt durch die Begrenzung der maximalen Standortlast geringere Netzentgelte und profitiert dadurch von günstigem Ladestrom bei gleichzeitigem Erreichen des erforderlichen SOCs aller Fahrzeuge. Das Gewerbe kann die Kosteneinsparungen für den Ladestrom beliebig an seine Angestellten weitergeben. Zur Anpassung der Ladeleistungen wird ein Lademanagementsystem eingesetzt.

Das Geschäftsmodell ist vergleichsweise einfach umzusetzen, wobei der Algorithmus zur Priorisierung einen zentralen Stellhebel darstellt. Hier sollte das Gewerbe darauf achten, dass der Algorithmus effizient und fair umgesetzt ist. Das Business Model Canvas zu diesem Geschäftsmodell wird in Abbildung 5-7 und die SWOT-Analyse in Abbildung 5-8 dargestellt.
www.ffe.de
Bewertung ausgewählter Geschäftsmodelle
9

![img-30.jpeg](img-30.jpeg)
Gewerbe nutzt Flottenmanagement

Abbildung 5-7: Business Model Canvas: Geschäftsmodell eines Gewerbes zum Use Case Flottenmanagement

![img-31.jpeg](img-31.jpeg)
Gewerbe nutzt Flottenmanagement

Abbildung 5-8: SWOT-Analyse: Analyse des Geschäftsmodells eines Gewerbes zum Use Case Flottenmanagement

## 5.2.4 Lokale Netzdienstleistung

Beim Use Case lokale Netzdienstleistung stellen Anschlussnutzer lokale Flexibilität für den VNB zur Behebung von Netzengpässen durch die Steuerung von Ladevorgängen zur Verfügung. Durch die Möglichkeit der Steuerung von Ladevorgängen wird die Netzbelastung zeitlich verschoben. Die Elektrofahrzeugfahrer:innen laden basierend auf einem Signal des VNBs oder werden durch diesen direkt gesteuert. Die Bereitstellung der Flexibilität wird vergütet, wobei verschiedene Vergütungsmodelle denkbar sind, wie z. B. reduzierte Netznutzungsentgelte. Unabhängig vom Vergütungsmodell stammt das Preissignal im Use Case lokaler Netzdienstleistung aus dem Netz. Neben dem VNB sind Anschlussnehmer und Anschlussnutzer am Use Case beteiligt. Der Anschlussnehmer schließt den Vertrag zur Erbringung von Flexibilität mit dem VNB. Der Anschlussnutzer wiederum ist die betroffene Person der Flexibilitätserbringung. Im Haushalt ist das häufig die gleiche Person. Im Gewerbe sind die Anschlussnutzer die Angestellten, die ihre EFZ laden. Geschäftsmodelle zum Use Case
lokale Netzdienstleistung ergeben sich sowohl für Eigenheimbesitzer:innen und Gewerbe, die Flexibilität verkaufen, als auch für Netzbetreiber, die Flexibilität einkaufen.

## Eigenheimbesitzer:in oder Gewerbe stellt Flexibilität zur Verfügung

Bei diesem Geschäftsmodell verkaufen Eigenheimbesitzer:innen oder Gewerbe ihre Bereitschaft zu flexiblem Lastverbrauch durch die Verschiebung von Ladevorgängen an (Verteil-)Netzbetreiber. Die Schlüsselaktivität bei diesem Geschäftsmodell ist das zeitliche Verschieben von Ladevorgängen auf Anfrage des Netzbetreibers bzw. die Bereiterklärung zur direkten Steuerung der Ladevorgänge durch den Netzbetreiber. Je nach Ausgestaltung der Vergütung bei Haushalten tritt ein Energieversorgungsunternehmen als Schlüsselpartner auf, um z. B. die verminderten Netzentgelte abzurechnen. Das Business Model Canvas zu diesem Geschäftsmodell wird in Abbildung 5-9 und die SWOT-Analyse in Abbildung 5-10 dargestellt.

![img-32.jpeg](img-32.jpeg)
Eigenheimbesitzer:in/Gewerbe stellt Flexibilität zur Verfügung

Abbildung 5-9: Business Model Canvas: Geschäftsmodell von Eigenheimbesitzer:innen bzw. eines Gewerbes zum Use Case lokale Netzdienstleistung

![img-33.jpeg](img-33.jpeg)
Eigenheimbesitzer:in/Gewerbe stellt Flexibilität zur Verfügung

Abbildung 5-10: SWOT-Analyse: Analyse der Geschäftsmodelle von Eigenheimbesitzer:innen bzw. eines Gewerbes zum Use Case lokale Netzdienstleistung

www.ffe.de

Energiewirtschaftliche Bewertung der Entwicklung der Elektromobilität
Durch die Vermeidung von Netzüberlastungen, wird der Netzausbaubedarf reduziert, wodurch Letztverbraucher langfristig von niedrigeren Netzentgelten profitieren. Da der Flexibilitätsabruf die Nutzerbedürfnisse stark einschränken kann, sollte bei der Vertragsgestaltung mit dem VNB stark darauf geachtet werden, unter welchen Bedingungen der Netzbetreiber die Flexibilität abrufen darf. Das ist insbesondere für Unternehmen relevant, falls diese etwaige Lieferverpflichtungen nicht einhalten können.

## Verteilnetzbetreiber kauft lokale Flexibilität

Das entsprechende Geschäftsmodell des VNBs wird so ausgestaltet, dass er Eigenheimbesitzer:innen und Gewerben eine Vergütung im Gegenzug für die Bereitstellung zum flexiblen Lastverbrauch verspricht. Durch den Flexibilitätsabruf kann der Netzbetreiber verhindern, dass seine Netzbetriebsmittel überlastet werden. Außerdem kann er mancherorts den Netzausbaubedarf reduzieren. Ihm entstehen für den zusätzlichen Koordinations- und Abrechnungsaufwand erhöhte Personalkosten.

Durch den Abruf von lokaler Flexibilität und dem dadurch vermiedenen Netzausbau kann der Netzbetreiber die Kompetenz ausbauen, dass er zukünftig stärker auf intelligente Regelung anstatt den physischen Ausbau des Netzes setzt. Dass somit langfristig Netzentgelte eingespart werden können, sollte an mögliche Kund:innen kommuniziert werden, damit für sie der (finanzielle) Anreiz zur Bereitstellung von Flexibilität mit den entsprechenden Einschränkungen im persönlichen Verhalten ausreichend ist. Um die Kosten für den zusätzlichen Personalaufwand und die Flexibilitätskoordination möglichst gering zu halten, sollten diese Prozesse weitestgehend automatisiert und digitalisiert werden. Das Business Model Canvas zu diesem Geschäftsmodell wird in Abbildung 5-11 und die SWOT-Analyse in Abbildung 5-12 dargestellt.

![img-34.jpeg](img-34.jpeg)

Verteilnetzbetreiber kauft lokale Flexibilität

Abbildung 5-11: Business Model Canvas: Geschäftsmodell eines Verteilnetzbetreibers zum Use Case lokale Netzdienstleistung

www.ffe.de

Bewertung ausgewählter Geschäftsmodelle
2
www.ffe.de
Energiewirtschaftliche Bewertung der Entwicklung der Elektromobilität

Verteilnetzbetreiber kauft lokale Flexibilität

|  Stärken ⚡ | Schwächen ⚡  |
| --- | --- |
|  • Vermeidung von Kosten und Ressourcen für den Netzausbau
• Vermeidung der Überlastung von Netzbetriebsmitteln | • Zusätzlicher Personalaufwand und Kosten für die Flexibilitätskoordination  |
|  Chancen | Risiken  |
|  • Wachsendes Geschäftsfeld durch den Ausbau der Elektromobilität in Deutschland | • Falls keine direkte Steuerung durch den Netzbetreiber erfolgt: Netzstabilität ist gefährdet, wenn sich Kund:innen mit lokaler Steuerung nicht an Ladepläne halten
• Höhe der Netzentgeltreduktion ist ggf. nicht hoch genug, um einen Anreiz zur Flexibilitätsbereitstellung zu geben
→ Konflikte mit Kund:innen können auftreten, wenn diese keine externe Steuerung der EFZ zulassen möchten  |

Abbildung 5-12: SWOT-Analyse: Analyse des Geschäftsmodells eines Verteilnetzbetreibers zum Use Case lokale Netzdienstleistung

## 5.2.5 Fazit zu den Geschäftsmodellen

Die hier beschriebenen Geschäftsmodelle beruhen alle auf Formen des flexiblen Ladeverhaltens, um die Netzbelastung zu reduzieren, zu verschieben oder zu vermeiden und erfordern dadurch lange Standzeiten und die Einbindung der Ladpunkte und Fahrzeug in ein Steuerungssystem. Dies kann z. B. ein Lademanagementsystem sein, welches Ladevorgänge der EFZ entweder direkt durch den Netzbetreiber oder durch eine dritte Person (Angestellte, Elektrofahrzeugfahrer:in) steuert. Daraus entstehen ähnliche Geschäftsmodellmuster mit ähnlichen Schlüsselaktivitäten, -ressourcen, -partnern und Kostenstrukturen.

Die Geschäftsmodelle bieten durch die Reduktion des erforderlichen Netzausbaus aufgrund steigender Elektromobilität die Chance, geringere Netzausbaukosten für Netzbetreiber zu ermöglichen und somit auch die Netzentgelte für Letztverbraucher zu senken bzw. zukünftig weniger stark ansteigen zu lassen. Neben den Chancen weisen die Geschäftsmodelle allerdings auch Risiken auf. Für Letztverbraucher entsteht durch die Geschäftsmodelle und die Verschiebung von Ladevorgängen das Risiko, dass ihre Fahrzeuge zum Abfahrtszeitpunkt nicht ausreichend geladen sind. Es ist wichtig zu bestimmen, wie hoch der finanzielle Anreiz sein muss, damit Letztverbraucher einer freiwilligen Bereitstellung von Flexibilität zustimmen.

Letztendlich stehen die „behind-the-meter“ Geschäftsmodelle, also diejenigen, die ausschließlich hinter dem Netzanschlusspunkt stattfinden, geringeren Unsicherheiten gegenüber wie die „in-front-the-meter“ Geschäftsmodelle. Hier gilt es unter anderem, die Rechte von Netzbetreibern zu klären, z. B. bis zu welchem Ausmaß das Abschalten von Letztverbrauchern in Netzengpasssituationen gestattet ist.
# 6 Veröffentlichung von Ladelastgängen

Die Verfügbarkeit von Daten wird im Kontext der Digitalisierung und zunehmenden Flexibilisierung im Verteilnetz immer wichtiger. Im Hinblick auf die Elektromobilität nimmt die Verfügbarkeit von Daten eine besondere Rolle ein. Bei Ladepunkten im öffentlichen Raum bietet die Aufzeichnung der Ladevorgänge ein Maß für die Frequentierung durch Elektrofahrzeuge und damit den wirtschaftlichen Betrieb. Im privaten und gewerblichen Umfeld können aus der Aufzeichnung Indikationen für den effizienten Betrieb z. B. zur Eigenverbrauchoptimierung oder ein Lastmanagement abgeleitet werden.

Auch in der Wissenschaft stellt die Verfügbarkeit einer ausreichend großen Datenbasis eine Grundvoraussetzung für fundierte Analysen dar, welche jedoch häufig nicht zureichend gegeben ist. Das Ziel eines Arbeitspaktes im Rahmen des Projekts war die Aufzeichnung und öffentliche Publikation von detaillierten Ladelastgängen an privater bzw. gewerblicher Ladeinfrastruktur.

Im Zeitraum von 2019 bis 2022 wurden Ladevorgänge in einer zeitlichen Auflösung von fünf Minuten aufgezeichnet, plausibilisiert und unter öffentlicher Lizenz publiziert. Gemessen wurden unter anderem Strom-, Spannungs- und Leistungswerte einzelner Phasen an Schnellladestationen und Wallboxen, so wie die resultierenden Gesamtlastgänge. Im Gegensatz zu den Schnellladestationen, verfügten die Wallboxen nicht über separate Messeinrichtungen, sondern wurden gesammelt über eine Stromschiene gemessen. Aus den gemessenen Zeitreihen wurden über Algorithmen die Zeitfenster extrahiert in welchen Ladevorgänge stattfanden. Im Fall der individuell vermessenen Schnellladestationen konnte dies durch einfache Algorithmen umgesetzt werden, da einerseits Ladevorgänge durch größere Leistungssprünge charakterisiert wurden, andererseits keine Überlagerung mit weiteren Verbrauchern stattfand. Bei der gesammelten Messung von Wallboxen an Stromschienen wurden speziellere Algorithmen genutzt, um die Ladesequenzen aus der aggregierten Leistungszeitreihe zu extrahieren.

Um einen qualitativ hochwertigen Datensatz für die Veröffentlichung auf der FfE OpenData Plattform zu erhalten, mussten die Daten zunächst aufbereitet werden. Hierzu wurden die Rohdaten in einem übergeordneten Datensatz zusammengefügt und Fehler sowie Inkonsistenzen plausibilisiert. Für die Identifizierung der Ladevorgänge diente in erster Linie die Leistungskurve der Wirkleistung (Active Power). Bei einem ausreichend großen Sprung der Ladeleistung aus dem definierten Standby-Leistungsbereich wurde der Start eines Ladevorgangs antizipiert. Dafür wird zunächst der Leistungsgradient berechnet, der den Unterschied der gemessenen Leistung zwischen zwei aufeinanderfolgenden Zeitschritten angibt. Da insbesondere bei Schnellladestationen mit hoher Leistung (bis zu 50 kW), die volle Ladeleistung nicht immer direkt bei Anschluss eines EFZ einsetzt wird und die Messwerte einen Durchschnittswert des gemessenen Zeitintervalls (5 min) darstellen, konnte es vorkommen, dass sich ein Leistungssprung über mehrere Zeitschritte aufteilt. In diesen Fällen mit verzögertem Leistungssprung wurde der erste Zeitschritt mit erhöhter Leistung als Start eines Ladevorgangs definiert.

Das Ende eines Ladevorgangs kann auf zwei Arten eingeleitet und entsprechend klassifiziert werden. Entweder das E-Fahrzeug ist vollgeladen oder das Fahrzeug wird vor Erreichen der Kapazitätsgrenzen getrennt und die Leistung fällt abrupt ab. Abbildung 6-1 zeigt einen Ladevorgang aus dem Datensatz, bei dem das Fahrzeug vollgeladen wurde und es ab einer

www.ffe.de
3
Ladedauer von 01:05 Stunden zu einem deutlichen Spannungsabfall zum Ende der Ladedauer kommt. Dies ist auf das häufig angewandte, batterieschonende IU-Ladeverfahren zurückzuführen, wonach zunächst eine Konstantstromladung erfolgt. Sobald die Ladeschlussspannung der Batterie erreicht ist (bei etwa 70 % bis 80 % der Nennkapazität), wird mit konstanter Spannung weiter geladen, was einen kontinuierlich abnehmenden Stromfluss zur Folge hat, wodurch die Ladeleistung ebenfalls kontinuierlich abnimmt.

![img-35.jpeg](img-35.jpeg)
Abbildung 6-1: Schematische Darstellung eines Ladevorgangs mit Vollladung

In Abbildung 6-2 ist ein Ladevorgang zu sehen, der abgebrochen wurde, bevor die Vollladung erreicht werden konnte, wobei die Ladeleistung abrupt auf 0 kW abfällt. Zudem wurde in selten Fällen beobachtet, dass die Leistung von Fahrzeug oder Wallbox im Verlauf des Ladevorgangs begrenzt wurde und das Leistungsniveau auf ein niedrigeres, konstantes Level begrenzt wurde. Dies war besonders bei Ladevorgängen der Wallboxen mit niedriger Ladeleistung an den Stromschienen zu erkennen.

![img-36.jpeg](img-36.jpeg)
Abbildung 6-2 Schematische Darstellung eines abgebrochenen Ladevorgangs

Während bei der Auswertung der Daten der Schnellladestationen mehrere aufeinander folgende Leistungssprünge ignoriert wurden, deuteten mehrere Leistungssprünge an einer Stromschiene auf mehrere Fahrzeuge hin, die nacheinander an verschiedene Wallboxen angeschlossen wurden. Ohne die entsprechenden Abrechnungsinformationen konnten aus

www.ffe.de
Veröffentlichung von Ladelastgängen
dem an den Stromschienen gemessenen Gesamtlastgang beim Vorliegen mehrerer parallel verlaufender Ladevorgänge keine eindeutige Zuordnung der Start- und Endzeiten zu einem Ladevorgang erfolgen, weshalb explizit nur Ladevorgänge extrahiert wurden, die nicht durch weitere Ladevorgänge überlagert wurden.

Des Weiteren wurden Situationen aufgezeichnet, in welchen Fahrzeuge kurz hintereinander von einer Ladestation abgesteckt und erneut Fahrzeuge angeschlossen wurden. In diesem Fall fiel die Leistungskurve nur sehr kurz (z. B. ein Zeitschritt) auf das Standby-Leistungsniveau. Da nicht unterschieden werden konnte, ob es sich hierbei um einen Messfehler oder einen neuen Ladevorgang handelt, wurden diese („unbestimmten“) Fälle ebenfalls exkludiert. Abbildung 6-3 zeigt ein Beispiel eines solchen unbestimmten Ladevorgangs. Bei diesem Beispiel können innerhalb der ersten beiden Stunden des Ladevorgangs starke Schwankungen in der Ladeleistung beobachtet werden, die als untypisch für einen Ladevorgang einzustufen sind. Ladevorgänge, die abseits dieses Musters eindeutige Leistungsschwankungen aufweisen, wurden entsprechend gekennzeichnet und für die Schnellladestationen mit in den Datensatz aufgenommen.

![img-37.jpeg](img-37.jpeg)
Abbildung 6-3 Schematische Darstellung eines unbestimmten Ladevorgangs

Das Ergebnis der Auswertung waren die aus den gemessenen Leistungskurven extrahierten Ladevorgänge, für die weitere Kennwerte bestimmt wurden. Die aufbereiteten und extrahierten Ladevorgänge wurden schließlich zu einem Datensatz zusammengefügt. Zeilen mit Messwerten, die keinem Ladevorgang zugeordnet wurden, sowie unbestimmte Ladevorgänge und solche mit fehlerhaften Messwerten wurden dabei entfernt. Anschließend wurde jedem Ladevorgang ein UUID (Universally Unique Identifier) zugwiesen, da die numerischen Werte nur für die jeweilige Messstation eindeutig waren. Außerdem wurden Einträge entfernt, die in den relevanten Spalten fehlende Messwerte (NaNs) aufwiesen, sowie Spalten, die in dem gefilterten Datensatz keine Messwerte aufwiesen. Insgesamt konnten so über den gesamten Beobachtungszeitraum 2.054 Ladevorgänge in den Datensatz überführt werden.

www.ffe.de
Der veröffentlichte Datensatz weiß drei verschiedene „internal_id_type Felder“ auf:

Internal_id_type[1] nummeriert den Ladevorgang in kontinuierlicher, aufsteigender Reihenfolge.
Internal_id_type[2] beschreibt, ob es sich um eine individuell vermessene Schnellladestation, oder um kollektiv vermessene Wallbox von einer der Stromschienen handelt.
Internal_id_type[3] hält schließlich die gesamten Messwerte für den Ladevorgang. Ein Überblick dieser Messwerte ist in Tabelle 6-1 zu finden.

Tabelle 6-1 Messwerte des Datensatzes

|  Messwert | Beschreibung  |
| --- | --- |
|  Local timestamp | Zeitstempel (Lokalzeit)  |
|  UTC offset | Sommer / Winterzeit offset um von Lokalzeit auf UTC zu rechnen  |
|  Frequency | Netzfrequenz  |
|  Current A, B, C | Strom gemessen an drei Phasen  |
|  Voltage A-N, B-N, C-N | Spannung gegenüber Neutraleiter gemessen an drei Phasen  |
|  Power factor | Leistungsfaktor  |
|  Active power | Wirkleistung  |
|  Apparent power | Scheinleistung  |
|  Reactive power | Blindleistung  |
|  Active power A, B, C | Wirkleistung gemessen an drei Phasen  |
|  Active energy | Wirkenergie  |
|  Reactive energy | Blindenergie  |
|  Active energy A, B, C | Wirkenergie gemessen an drei Phasen  |
|  Voltage A-B, B-C, C-A | Spannung einer Phase gegenüber einer anderen Phase  |
|  Voltage L-L (Avg.) | Durchschnittliche Leiter-Leiter Spannung  |
|  Voltage L-N (Avg.) | Durchschnittliche Leiter-Neutraleiter Spannung  |

Der aufbereitete Datensatz steht seit dem 16.02.2022 auf der FfE OpenData Plattform unter folgendem Link zur Verfügung: Hyperlink-Ladedatensatz

Die Creative Commons Lizenz CC-BY-4.0 erlaubt das Teilen und Bearbeiten der veröffentlichten Daten unter der Bedingung der Namensnennung.

www.ffe.de
Veröffentlichung von Ladelastgängen
# 7 Förderung und Projektpartner

Die Bearbeitung der hier beschriebenen Inhalte erfolgt im Verbundprojekt München elektrisiert – M⁰ durch die FfE. Die FfE-Aktivitäten im Verbundprojekt M⁰ werden im Rahmen des „Sofortprogramms Saubere Luft 2017-2020“ des Bundesministeriums für Wirtschaft und Klimaschutz (BMWK) gefördert (Förderkennzeichen: 01MZ18010B). Im Rahmen des Verbundprojekts M⁰ werden die Vorhaben der FfE durch das Referat für Klima- und Umweltschutz (ehemals in Referat für Gesundheit und Umwelt) der Landeshauptstadt München, die Lehrstühle für Fahrzeugtechnik sowie Verkehrstechnik der Technischen Universität München begleitet. Als assoziierte Partner beteiligen sich die Handwerkskammer für München und Oberbayern sowie die IHK München und Oberbayern. Des Weiteren werden die FfE-Aktivitäten durch die Bereitstellung von Daten, von den Stadtwerke München in ihrer Funktion als lokaler VNB unterstützt.

www.ffe.de
# 8 Literatur

BAFA-04 16 Elektromobilität (Umweltbonus) in: http://www.bafa.de/bafa/de/wirtschaftsfoerderung/elektromobilitaet/index.html (13.07.2016). Eschborn: Bundesamt für Wirtschaft und Ausfuhrkontrolle (BAFA), 2016

BDEW-11 20 Bundesverband der Energie- und Wasserwirtschaft: Elektromobilität Definition der Ladeinfrastruktur-Marktrollen. Berlin: BDEW Bundesverband der Energie- und Wasserwirtschaft e.V., 2020.

BNETZA-06 19 Marktstammdatenregister - Öffentliche Einheitenübersicht. In: https://www.marktstammdatenregister.de/MaStR/Einheit/Einheiten/OeffentlicheEinheitenuebersicht. (Abruf am 2019-03-07); Bonn: Bundesnetzagentur, 2019.

BR-02 19 Masterplan Ladeinfrastruktur der Bundesregierung (Ziele und Maßnahmen für den Ladeinfrastrukturaufbau bis 2030). Ausgefertigt am 2019-10-09; Berlin: Die Bundesregierung, 2019.

BRD-01 20 Bundesregierung: Klimaschutz - Verkehr. In: https://www.bundesregierung.de/breg-de/themen/klimaschutz/verkehr-1672896. (Abruf am 2020-04-05); Berlin: Bundesregierung, 2020.

DESTATIS-15 13 Zensusdatenbank des Zensus 2011: https://ergebnisse.zensus2011.de/; Wiesbaden: Statistische Ämter des Bundes und der Länder, 2013.

DESTATIS-10 15 Statistisches Bundesamt: Zeitverwendungserhebung - Aktivitäten in Stunden und Minuten für ausgewählte Personengruppen. Wiesbaden: Statistisches Bundesamt, 2015

DESTATIS-05 16 Forschungsdatenzentrum des Statistischen Bundesamtes: Datenangebot, Zeitverwendungserhebung / Zeitbudgeterhebung in: http://www.forschungsdatenzentrum.de/bestand/zve/index.asp (zuletzt abgerufen am: 05.09.2016). (Archived by WebCite® at http://www.webcitation.org/6kHzy2uGO). Wiesbaden: Statistische Ämter des Bundes und der Länder, 2016

DKE-01 17 Generische Anforderungen an Intelligente Elektrizitätsversorgungssysteme (Smart Grids) - Teil 1: Anwendung der Anwendungsfallmethodik speziell auf die Festlegung von generischen Anforderungen an Smart Grids nach dem IEC-Systemansatz (DIN IEC/TS 62913-1 (IEC SyCSmartEnergy/57/CD:2017)). Ausgefertigt am 2016-06, Version vom 2017-12; Berlin: DKE Deutsche Kommission Elektrotechnik Elektronik Informationstechnik in DIN und VDE, 2017.

ENGEL-01 18 Engel, Hauke et al.: The potential impact of electric vehicles on global energy systems. Frankfurt: McKinsey &amp; Company, 2018.

FAT-01 21 Fattler, Steffen: Economic and Environmental Assessment of Electric Vehicle Charging Strategies. Dissertation. Herausgegeben durch die TU München, geprüft von Wagner, Ulrich und Wietschel, Martin: München, 2021.

FERS-01 19 Ferstl, Joachim: Entwicklung und Validierung eines rasterbasierten Modells zur kleinräumigen Prognose der installierten Leistung von Photovoltaikanlagen. Masterarbeit. Herausgegeben durch Universität Augsburg - Institut für Geographie, betreut durch Dr.-Ing. Schmid, Tobias; Prof. Timpf, Sabine: Augsburg, 2019.

FFE-21 17 Corradini, Roger; Konetschny, Claudia; Schmid, Tobias: FREM - Ein regionalisiertes Energiesystemmodell in: et - Energiewirtschaftliche

www.ffe.de
Tagesfragen Heft 1/2 2017. München: Forschungsstelle für Energiewirtschaft, 2017

FFE-23 18
Fattler, Steffen et al.: Charge optimization of privately and commercially used electric vehicles and its influence on operational emissions. Munich: Research Center for Energy Economics, 2018.

FFE-30 18
Konetschny, Claudia et al.: Oberflächennahe Geothermie im außerstädtischen Wohngebäudebestand - Potenzialanalyse zur Nutzung von Erdwärmpumpen im Gebäudebestand. In: BWK Bd. 70 (2018) Nr. 7/8. Düsseldorf: VDI Verlag, 2018.

FFE-104 19
Fattler, Steffen et al.: Dynamis Hauptbericht - Dynamische und intersektorale Maßnahmenbewertung zur kosteneffizienten Dekarbonisierung des Energiesystems. München: Forschungsstelle für Energiewirtschaft e.V., 2019.

FFE-142 19
Ebner, Michael et al.: Kurzstudie Elektromobilität - Modellierung für die Szenarienentwicklung des Netzentwicklungsplans. München: Forschungsstelle für Energiewirtschaft, 2019.

FFE-144 19
Fattler, Steffen; Conrad, Jochen; Regett, Anika et al.: Dynamis Hauptbericht - Dynamis - Dynamische und intersektorale Maßnahmenbewertung zur kosteneffizienten Dekarbonisierung des Energiesystems - Online: https://www.ffe.de/dynamis. München: Forschungsstelle für Energiewirtschaft e.V., 2019. DOI: 10.34805/ffe-144-19

FFE-34 19
Müller, Mathias et al.: Laufendes Projekt: München elektrisiert. In: www.ffe.de/Me. (Abruf am 2019-05-06); (Archived by WebCite® at https://www.webcitation.org/78AO7e0WF); München: Forschungsstelle für Energiewirtschaft e.V. (FfE), 2019.

FFE-97 19
Weiß, Andreas et al.: Ermittlung und Bewertung von Charakteristiken der Netzbelastung im Verteilnetz - Beschreibung einer Methode zur Clustering von Metadaten für die Auswahl individueller Netzbelastungsgebiete in München. In: BWK - Das Energie Fachmagazin Ausgabe 07/08 2019. Düsseldorf: Verein Deutscher Ingenieure (VDI), 2019.

FFE-02 20
Weiß, Andreas et al.: Simulative Analyse der aktuellen und zukünftigen Netzbelastung urbaner Versorgungsgebiete. In: Tagung Zukünftige Stromnetze; Berlin: Forschungsstelle für Energiewirtschaft e.V., 2020.

FFE-118 20
Weiß, Andreas et al.: Spitzenlastkappung durch uni- und bidirektionales Laden von Elektrofahrzeugen und Analyse der resultierenden Netzbelastung in Verteilnetzen. In: Forschung im Ingenieurwesen, Volume 85(2). 469-476. Berlin: Springer Nature, 2021. https://doi.org/10.1007/s10010-020-00424-z.

FFE-61 20
Müller, M.; Biedenbach, F.; Reinhard, J. Development of an Integrated Simulation Model for Load and Mobility Profiles of Private Households. Energies 2020, 13, 3843.

FFE-04 21
Weiß, Andreas et al.: Simulative Analyse der zukünftigen Netzbelastung - Interaktion von PV und Elektromobilität im urbanen Verteilnetz. In: Tagung Zukünftige Stromnetze 2021. Berlin: Conexio, 2021.

FFE-45 21
Weiß, Andreas et al.: Simulation and analysis of future electric mobility load effects in urban distribution grids. Berlin: International ETG Congress 2021. VDE Verlag GmbH, 2021.

FFE-61 21
Weiß, Andreas et al.: Analyse methodischer Modellierungsansätze im Kontext von Verteilnetzsimulationen. In: IEWT 2021 - 12. Internationale Energiewirtschaftstagung; Wien: TU Wien, 2021.

www.ffe.de
FFE-08 22 Schulze, Yannic et al.: Netzbelastungen durch optimal am Spotmarkt vermarktete bidirektionale Elektrofahrzeuge. In: Zukünftige Stromnetze; München: Forschungsstelle für Energiewirtschaft e. V., 2022.

FFE-13 22 Biedenbach, Florian et al.: Simulative Analyse der zukünftigen Netzbelastung – Auswirkungen verschiedener Lastmanagement-Strategien. In: Zukünftige Stromnetze; Berlin: Conexio GmbH, 2022.

FFE-36 22 Weiß, A.; Biedenbach, F.; Müller, M. Probabilistic Load Profile Model for Public Charging Infrastructure to Evaluate the Grid Load. Energies 2022, 15, 4748. https://doi.org/10.3390/en15134748

FGH-01 18 Vennegeerts, Hendrik et al.: Metastudie Forschungsüberblick Netzintegration Elektromobilität. Mannheim: Forschungsgemeinschaft für elektrische Anlagen und Stromwirtschaft, 2018.

FME-01 20 Federal Ministry for the Environment, Nature Conservation, Building and Nuclear Safety: Climate Action in Figures - Facts, Trends and Incentives for German Climate Policy. Berlin: Federal Ministry for the Environment, Nature Conservation and Nuclear Safety, 2020.

IEA-01 20 Data and Statistics - Total final consumption (TFC) by sector, Germany 1990 - 2018: https://www.iea.org/data-and-statistics?country=GERMANY&amp;fuel=Energy%20consumption&amp;indicator=TFCShareBySector; Paris, France: International Energy Agency, 2020 (überarbeitet: 2020).

IEA-03 20 Delmastro, Chiara et al.: Cooling. In: https://www.iea.org/reports/cooling. (Abruf am 2020-11-06); Paris: International Energy Agency, 2020.

INFAS-01 18 Nobis, Claudia et al.: Mobilität in Deutschland – MiD Ergebnisbericht. Bonn: infas, DLR, IVT und infas 360, 2018.

INFAS-01 19 Mobilität in Deutschland 2017 - Datensatz; Bonn: infas Institut für angewandte Sozialwissenschaft GmbH, 2019.

ISI-27 15 Boßmann, T. et al.: The shape of future electricity demand: Exploring load curves in 2050s Germany and Britain. In: Energy 90 (2015) 1317-1333. Karlsruhe, Germany, South Kensington, UK: Fraunhofer Institute for Systems and Innovation Research ISI, Imperial College Business School, Imperial College London, 2015.

ISI-30 15 REM2030 Driving Profiles Database V2015: https://www.rem2030.de/rem2030-de/REM-2030-Fahrprofile.php; Karlsruhe: Fraunhofer Institute of Systems and Innovation Research ISI, 2015.

JOOSS-01 21 Jooß, Niklas: Optimierte Spitzenlastkappung bidirektionaler Elektrofahrzeuge in Gewerbebetrieben und Analyse der resultierenden Verteilnetzbelastung - Optimised peak load shaving of bidirectional electric vehicles in commercial enterprises and analyses of the resulting distribution grid load. Masterarbeit. Herausgegeben durch die Technische Universität München - Fakultät für Maschinenwesen, betreut durch Prof. Dr. Hamacher, Thomas und Dr.-Ing. Kuhn, Philipp: München, 2021.

LHM-01 19 Messergebnisse Stickstoffdioxid (NO2) 2018 - Ergänzende Stickstoffdioxidmessungen: https://www.muenchen.de/rathaus/Stadtverwaltung/Referat-fuer-Gesundheit-und-Umwelt/Luft_und_Strahlung/Stickstoffdioxidmessungen.html#messergebnisse-2018_5; München: Landeshauptstadt München Referat für Gesundheit und Umwelt, 2019.

www.ffe.de
MUL-01 19
Vopava, Julia et al.: Investigating the Impact of E-Mobility on the Electrical Power Grid Using a Simplified Grid Modelling Approach. In: energies 13, 39. Leoben, Austria: Chair of Energy Network Technology, Montanuniversität, Institute of Physics, Montanuniversität, 2019.

NEXT-04 21
Next Kraftwerke: Was ist Peak Shaving?. In: https://www.nextkraftwerke.de/wissen/peak-shaving. (Abruf am 2021-12-23); Köln: Next Kraftwerke GmbH, 2021.

NOW-02 20
Zentrales Datenmonitoring des Förderprogramms Elektromobilität vor Ort; Berlin: NOW GmbH, 2020.

ÖKO-02 17
Kenkmann, Tanja; Hesse, Tilman Dr.; Hülsmann, Friederike Dr.; Timpe, Christof; Hoppe, Klaus: Klimaschutzziel und -strategie München 2050. Freiburg: Öko-Institut e.V., 2017

OST-01 11
Osterwalder, Alexander; Pigneur, Yves: Business Model Generation - Ein Handbuch für Visionäre, Spielveränderer und Herausforderer. Frankfurt / New York: Campus Verlag GmbH, 2011

PSLU-01 18
Gerossier, Alexis et al.: Modeling and Forecasting Electric Vehicle Consumption Profiles. In: 11th Mediterranean Conference on Power Generation, Transmission, Distribution and Energy Conversion (MEDPOWER 2018); Dubrovnik, Croatia: MINES ParisTech, PERSEE-Center for Processes, Renewable Energies and Energy Systems, PSL University, 2018.

RGU-01 09
Kommunaler Klimaschutz in der Landeshauptstadt München. München: Referat für Gesundheit und Umwelt, Landeshauptstadt München, 2009

SCHAL-01 13
Schallmo, Daniel R.A.: Geschäftsmodell-Innovation - Grundlagen, bestehende Ansätze, methodisches Vorgehen und B2B-Geschäftsmodelle. Wiesbaden: Springer Gabler, 2013

SCHM-01 18
Schmid, Tobias: Dynamische und kleinräumige Modellierung der aktuellen und zukünftigen Energienachfrage und Stromerzeugung aus Erneuerbaren Energien. Dissertation. Herausgegeben durch Technische Universität München, geprüft von Prof. Wagner, Ulrich und Prof. Kolbe, Thomas H.: München, 2018.

SRG-01 15
Hornung-Prähauser, Veronika et al.: Methoden zur Geschäftsmodell-Entwicklung für AAL-Lösungen durch Einbeziehung der EndanwenderInnen. Salzburg: Salzburg Research GmbH, 2015.

SWM-02 18
Aktualisierte Umwelterklärung 2018 - Energieerzeugung / Wassergewinnung / Verteilnetze. München: Stadtwerke München, 2018.

SWM-07 20
Preisblatt 5 (Netz 1 - München) - (Konzessionsabgaben). München: SWM Infrastruktur GmbH &amp; Co. KG, 2020.

ÜNB-01 19
Netzentwicklungsplan Strom 2030 (Version 2019), zweiter Entwurf. Berlin, Dortmund, Bayreuth, Stuttgart: 50Hertz Transmission GmbH, Amprion GmbH, TenneT TSO GmbH, TransnetBW GmbH, 2019.

US-02 19
Göhler, Georg et al.: Netzbelastungen und Netzdienstleistungen durch Elektrofahrzeuge: Metastudie. Stuttgart: Universität Stuttgart, 2019.

VDEW-01 99
VDEW et al.: Repräsentative VDEW-Lastprofile. Cottbus: VDEW, 1999.

VIP-01 12
Wermuth, Manfred: Kraftfahrzeugverkehr in Deutschland 2010 (KiD 2010) - Schlussbericht. Braunschweig: Verkehrsforschung und Infrastrukturplanung GmbH, 2012

www.ffe.de
www.ffe.de

# 9 Anhang
München elektrisiert

FTE

# Altstadt-Lehel-Ludwigsvorstadt-Isarvorstadt (80469)

![img-38.jpeg](img-38.jpeg)
Clustergebiete

![img-39.jpeg](img-39.jpeg)
Untersuchungsgebiet

![img-40.jpeg](img-40.jpeg)

## Key Facts zum Netz:

5 Niederspannungs-Stränge mit je 130-180 m Kabellänge
17 Hausanschlüsse pro Transformator
300 Zählpunkte pro Transformator
0,7 % PV-Durchdringung nach MaStR

![img-41.jpeg](img-41.jpeg)
Entwicklung der Komponentendurchdringung je Hausanschluss in %

www.ffe.de
München elektrisiert

FTE

# Pasing-Obermenzing (81247)

![img-42.jpeg](img-42.jpeg)
Clustergebiete

![img-43.jpeg](img-43.jpeg)
Untersuchungsgebiet

![img-44.jpeg](img-44.jpeg)

## Key Facts zum Netz:

6 Niederspannungs-Stränge mit je 350-400 m Kabellänge
93 Hausanschlüsse pro Transformator
220 Zählpunkte pro Transformator
2,6 % PV-Durchdringung nach MaStR

![img-45.jpeg](img-45.jpeg)
Entwicklung der Komponentendurchdringung je Hausanschluss in %

www.ffe.de
München elektrisiert

FTE

# Milbertshofen-Am Hart (80809)

![img-46.jpeg](img-46.jpeg)
Clustergebiete

![img-47.jpeg](img-47.jpeg)
Untersuchungsgebiet

![img-48.jpeg](img-48.jpeg)

## Key Facts zum Netz:

4 Niederspannungs-Stränge mit je 100-130 m Kabellänge
13 Hausanschlüsse pro Transformator
160 Zählpunkte pro Transformator
1,6 % PV-Durchdringung nach MaStR

![img-49.jpeg](img-49.jpeg)
Entwicklung der Komponentendurchdringung je Hausanschluss in %

www.ffe.de
München elektrisiert

H

# Feldmoching-Hasenbergl (80995)

![img-50.jpeg](img-50.jpeg)
Clustergebiete

![img-51.jpeg](img-51.jpeg)
Untersuchungsgebiet

![img-52.jpeg](img-52.jpeg)

## Key Facts zum Netz:

6 Niederspannungs-Stränge mit je 250-300 m Kabellänge
58 Hausanschlüsse pro Transformator
158 Zählpunkte pro Transformator
5,1 % PV-Durchdringung nach MaStR

![img-53.jpeg](img-53.jpeg)
Entwicklung der Komponentendurchdringung je Hausanschluss in %

www.ffe.de
München elektrisiert

FTE

# Schwabing-Freimann (80939)

![img-54.jpeg](img-54.jpeg)
Clustergebiete

![img-55.jpeg](img-55.jpeg)
Untersuchungsgebiet

![img-56.jpeg](img-56.jpeg)
Haasanschlussleitung
Haasanschluss
Obs 1992

# Key Facts zum Netz:

6 Niederspannungs-Stränge mit je 120 m Kabellänge
24 Hausanschlüsse pro Transformator
91 Zählpunkte pro Transformator
3,9 % PV-Durchdringung nach MaStR

![img-57.jpeg](img-57.jpeg)
Entwicklung der Komponentendurchdringung je Hausanschluss in %

www.ffe.de