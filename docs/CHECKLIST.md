# Sjekkliste: More than a Hack 2027

Frist: **17. november 2026**. Roller: P1 = Data og ball, P2 = Modell, P3 = Dødball, infrastruktur og dokument.

## 0. Oppstart (alle)
- [ ] Velg lagnavn (må være likt registreringsskjemaet). Navnet brukes i filnavnene.
- [ ] Bekreft at laget har 3–5 medlemmer og at alle har akseptert vilkårene.
- [ ] Alle leser OVERVIEW.md, CHALLENGE.md, DATA.md, EVALUATION.md, RULES.md, SUBMISSION.md, FAQ.md.
- [ ] Opprett felles offentlig repo. Sjekk at `data/` er i `.gitignore`.
- [ ] Avtal arbeidsflyt: branches, hvem som merger, hvem som kjører endelig fil.
- [ ] Avtal regelen om LLM-er: ingen video, data eller utdrag fra kampdata til skybaserte LLM-er.
- [ ] Alle har Google-konto (trengs for innleveringsskjemaet).
- [ ] Ta vare på skjema-URL-en når den kommer på e-post.

## 1. Data og miljø
- [ ] Last ned alle delene av datasettet (zip «-003» var bare del 3: kun `data/README.md`).
- [ ] Pakk ut til `data/tactical/` og `data/broadcast/` med originale filnavn (eller sett `MTAH_DATA_DIR`).
- [ ] Sjekk at disse filene finnes:
  - [ ] `BAR-ATH_tactical.mp4`, `_atd.xml`, `_atd.txt`, `_homography.csv`, `_smart_tagging.xml`
  - [ ] `BAR-ATH_broadcast.mp4`, `_atd.xml`, `_atd.txt`, `_homography.csv` (valgfritt)
- [ ] `python -m venv .venv`, installer `requirements-pipeline.txt`, og `requirements-starter.txt` ved behov.
- [ ] `python -m unittest discover tests` består.
- [ ] `python evaluation/run_evaluation.py` viser de fire baselinene.
- [ ] Ingen kampdata er committet (`git status` rent for `data/`).

## 2. Forstå dataen (P1, alle leser)
- [ ] Kjør `starter/01_smart_tagging.ipynb`, `02_tracking_atd.ipynb`, `03_first_submission.ipynb`.
- [ ] Forstå tidsbasene: ATD `frame_id - 1`, Smart Tagging `round(sekunder * 25)`, homografi allerede videoframe.
- [ ] Verifiser vurderte frames: 1. omgang 11 185–90 651, 2. omgang 110 808–190 197.
- [ ] Husk at 2. omgang vurderes fra 110 808 men starter 113 742, og at de 2 934 frames er `D`.
- [ ] Mål ballandel og hullengder i sporingen (kun for å forstå, ikke tune mot fasit).
- [ ] Se på noen sekvenser i videoen for å forstå data og feature-idéer (lov, så lenge det ikke blir etiketter eller frame-spesifikke regler).
- [ ] Forstå at 46 sporplasser ikke er 22 spillere, og at plasser kan bytte identitet.

## 3. Baseline og første fil
- [ ] Kjør `python run_pipeline.py --team <navn>` og få en gyldig fil.
- [ ] Sammenlign mot baselinene på de 5 offentlige minuttene (kun sanity-sjekk).
- [ ] Bekreft at filen validerer med `evaluation/validate_submission.py`.
- [ ] Sjekk antall possessions og snittvarighet i valideringsutskriften (ikke hundrevis av tiny possessions).
- [ ] Mål: første ende-til-ende-fil rundt 6. november.

## 4. Ballsignal (P1)
- [ ] Forbedre interpolasjonen av ballhull (eksempelvis fart/retning, ikke bare rett linje).
- [ ] Estimer ballposisjon fra spillerklynger når ballen mangler lenge.
- [ ] Vurder å bruke homografi/video eller egen ball-deteksjon (må være algoritmisk, ikke manuell).
- [ ] Rens sporingen: dupliserte spillere, byttede identiteter, urealistiske hopp.
- [ ] Lag og dokumenter signaler: ballfart, akselerasjon, retningsendring, avstand per lag, antall spillere nær ballen.
- [ ] Håndter `NaN` eksplisitt overalt.

## 5. Possession-modell (P2)
- [ ] Definer «kontrollert possession» (varighet nær ballen, pasninger, retningsbytter) og skriv regelen ned.
- [ ] Beslutt hva som ikke er bytte: touch uten kontroll, avbøyning, luftduell uten kontroll.
- [ ] Bygg kontrollscore per lag per frame.
- [ ] Bruk Smart Tagging team-tagger som prior, og håndter `N/A` (alle innkast) eksplisitt.
- [ ] Behandle taggintervallene som faser, ikke possessions.
- [ ] Juster tidsmodellen (bytte-kostnader, minimumsvarighet) ut fra fotballreasoning, ikke de 5 minuttene.
- [ ] Bestem en `X`-strategi: stille inne i en possession, gjett ved overganger.
- [ ] Hvis dere trener noe: bruk kun data fra *andre* kamper, og dokumenter kilde, merker og omfang.

## 6. Dødball (P3)
- [ ] Bygg dødball-detektor: ball utenfor bane, fart ≈ 0, spillere som går til restart-punkt.
- [ ] Bruk Smart Tagging `SET PIECES` (innkast, corner, målspark, frispark, avspark) som restart-prior.
- [ ] Bekreft at 2. omgang før avspark (110 808–113 741) er `D`.
- [ ] Håndter død ball når ballen mangler i sporingen.
- [ ] Sørg for at restarts avslutter dødballstrekk og starter en ny possession.

## 7. Evaluering og kvalitetssikring (alle)
- [ ] Bruk de 5 offentlige minuttene kun til å sjekke at pipelinen er riktig (frames på riktig sted, format, etikettenes betydning).
- [ ] Ikke velg mellom to versjoner ut fra de 5 minuttene.
- [ ] Bruk `--bootstrap 1000` og sjekk om intervaller overlapper.
- [ ] Sammenlign versjoner på antall possessions, snittvarighet og andel `D`, og at tallene er realistiske.
- [ ] Se på feilene som mønstre (restarts, korte passninger, hull i ballen) og rett opp i metoden, ikke i enkeltframes.
- [ ] Ingen håndrettelser, ingen frame-spesifikke regler eller konstanter.
- [ ] Skriv enhetstester for nye moduler.

## 8. Regelsjekk (P3, alle bekrefter)
- [ ] Alle etiketter kommer fra algoritmer.
- [ ] Ingen håndmerking av kampen (heller ikke for validering).
- [ ] Ingen bruk av privat fasit eller menneskeskapt data om kampen (event-data, korrigert sporing, annoteringer).
- [ ] De 5 offentlige minuttene er ikke brukt til trening eller tuning.
- [ ] Ingen kampdata, video eller avledede sporingseksporter er i det offentlige repoet.
- [ ] Ingen data eller video er sendt til skybaserte LLM-er.
- [ ] Alle eksterne modeller og data er tilgjengelige for arrangørene og beskrevet i dokumentet.
- [ ] Spørsmål om regler tas som issue *før* dere bygger på noe i en gråsone.

## 9. Reproduserbarhet (P3)
- [ ] Ett kommando-skript regenererer nøyaktig samme CSV (`python run_pipeline.py --team <navn>`).
- [ ] Deterministisk: faste seeds, ingen tilfeldighet, pinnede versjoner.
- [ ] Kjør to ganger og sammenlign filene (`diff` eller hash er lik).
- [ ] Fersk clone + ny venv + data gir samme fil.
- [ ] `requirements` er pinnet, og Python-versjon er oppgitt.
- [ ] README i repoet forklarer installasjon, datasti og kjøring.

## 10. Teknisk dokument (P3, alle bidrar)
Mal: `docs/technical_document.md`. Lever som `<lagnavn>.pdf` eller `.md`.
- [ ] Tilnærming og hvorfor denne.
- [ ] Brukte og droppede data (tabell).
- [ ] Treningsdata: kilde, hvem merket, hvor mye (ingen fra denne kampen), eller «ingen».
- [ ] Possession-regler: løse baller, duellers, restarts, `X`-bruk, 2. omgang-bufferen.
- [ ] Håndtering av manglende og støyende sporing.
- [ ] Alle parametre med begrunnelse, og at ingen er tunet på de 5 minuttene.
- [ ] Begrensninger og hva dere ville gjort med mer tid.
- [ ] Reproduksjonsinstruksjoner.
- [ ] Teammedlemmer og LLM-bruk, med bekreftelse på at ingen kampdata er sendt til skybaserte LLM-er.

## 11. Frys og innlevering (rundt 13.–16. november)
- [ ] Frys metoden i en git-tag, og ikke endre kode etter dette uten ny full kjøring.
- [ ] Generer endelig fil fra frossen kode.
- [ ] Valider *nøyaktig* den filen: `python evaluation/validate_submission.py <lagnavn>.csv`.
- [ ] Sjekk: header `frame,label`, 199 319 rader, `frame` 0–199 318 i rekkefølge, etiketter `H/A/D/X`, ingen tomme celler.
- [ ] Sjekk at det ikke er ca. 164k rader (broadcast ble indeksert ved en feil).
- [ ] Filnavn: `<lagnavn>.csv` og `<lagnavn>.pdf` (eller `.md`).
- [ ] Repoet er offentlig, og URL-en åpner uten innlogging.
- [ ] En annen i laget gjenskaper filen fra fersk clone og bekrefter identisk resultat.
- [ ] Fyll ut Google Form: lagnavn, CSV, repo-URL, dokument.
- [ ] Bare siste svar før fristen teller, så send inn med god margin (helst 15.–16. november).
- [ ] Ta skjermbilde eller lagre bekreftelsen.

## 12. Etter innlevering
- [ ] Hold repoet uendret slik at arrangørene kan kjøre det på nytt.
- [ ] Vent på resultater (før 26. november).
