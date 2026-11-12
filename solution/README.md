# Soluzione: possesso palla solo da dati numerici

Input usati: tracking automatico (ATD) e Smart Tagging del video tattico. Niente video, niente homography, niente broadcast.

## Pipeline
1. `possession/data.py`: carica l'ATD in metri (cache in `solution/cache/`, mai committata).
2. `possession/features.py`: distanza dalla palla per squadra, velocità palla, palla fuori campo, velocità collettiva dei giocatori.
3. `possession/emissions.py`: evidenza per frame per H/A/D; Smart Tagging come prior debole (segno dei tag di squadra deciso in automatico).
4. `possession/decode.py`: HMM a durata minima (Viterbi) per tempo; i cambi di stato hanno un costo.
5. `possession/gaps.py`: classificatore dei buchi della palla. Nei frame in cui la palla non è tracciata stima P(palla ferma) dal contesto (velocità dei giocatori, dove e come la palla è sparita e ricomparsa, durata del buco); il suo log-odds entra come evidenza per D con peso `k_gap`. Regressione logistica allenata solo sulle partite Metrica, salvata in `gap_model.json` (accanto a `params.json`); in inferenza serve solo numpy.
6. `possession/deadball.py`: classificatore di palla ferma su TUTTI i frame (palla vista o no), con gli schemi di ripresa di Vidal-Codina et al. 2022: palla ferma sulla linea laterale, alla bandierina, sull'area piccola o al centro con le squadre nelle proprie metà; giocatori che rallentano e poi ripartono (finestre passate/future). Gradient boosting allenato solo su Metrica, salvato in `deadball_model.pkl` accanto a `params.json`; il suo log-odds entra come evidenza per D con peso `k_dead`. Lo Smart Tagging non è una feature (Metrica non lo ha): SET PIECES resta un prior separato. Su Metrica (leave-one-game-out, 100 configurazioni appaiate) porta lo spellF1 della partita peggiore da 0.374 a 0.413 e il macroF1 da ~0.70 a ~0.75.
7. `possession/setpieces.py`: i calci piazzati dello Smart Tagging come ancore. Per ogni `SET PIECES` si trova nel tracking il frame di ripresa (palla ferma che riparte, nel punto tipico del `Type`); prima c'è evidenza di palla ferma per una durata che dipende dal tipo, dopo c'è il possesso di chi batte; `GOALS` → palla ferma fino al calcio d'inizio. I pesi (`k_sp_dead`, `k_sp_team`, `sp_back_scale`) sono calibrati su Metrica con tag sintetici costruiti dai suoi eventi (`metrica.synthetic_tags`). Su Metrica: +0.018 di spellF1 medio, meglio in 100/100 configurazioni.
8. `possession/calibrate.py` + `possession/metrica.py`: parametri calibrati su spellF1 sulle partite open di Metrica (etichette derivate in automatico dagli eventi), mai sulla partita della challenge. Per somigliare al tracking automatico, nei dati Metrica i buchi nativi della palla (quasi tutti a palla ferma) vengono riempiti e poi si tagliano buchi lunghi casuali (41% dei frame): così "palla mancante" non vuol più dire "palla ferma". In calibrazione i classificatori di ogni partita sono allenati sull'altra (leave-one-game-out). Il classificatore dei buchi resta nel codice ma è spento (`k_gap` = 0): il modello di palla ferma vede già il contesto dei buchi.

## Esecuzione
```bash
pip install -r requirements.txt -r solution/requirements.txt   # + `pip install notebook` per i notebook in locale
# notebook, in ordine: solution/notebooks/01 -> 02 (pesante: calibrazione) -> 03 (submission)
python solution/run.py --out evaluation/results/<team_name>.csv   # rigenera la submission da params.json
python evaluation/validate_submission.py evaluation/results/<team_name>.csv
```
`run.py` si ferma se `params.json` manca (`--allow-defaults` solo per prove, `--params <file>` per un altro percorso). Se `params.json` ha `k_dead > 0` serve anche `deadball_model.pkl` **nella stessa cartella** (o `MTAH_DEAD_MODEL`), altrimenti si ferma.

### Riproducibilità della submission finale
- `solution/requirements.txt` fissa le versioni esatte (numpy, pandas, scipy, scikit-learn, matplotlib). `deadball_model.pkl` è un pickle di scikit-learn: va caricato con la **stessa** versione con cui è stato salvato (la versione è scritta nel file, chiave `sklearn_version`).
- Per la consegna, copiare da Drive in `solution/` i file **finali** `params.json` e `deadball_model.pkl` e committarli (non sono in `.gitignore`). Così `python solution/run.py --out ...` li trova da solo (default: `solution/params.json`, il modello accanto) e rigenera lo stesso CSV senza rifare la calibrazione.
- Controllo: rigenerare il CSV con `run.py` da una copia pulita della repo e confrontarlo con quello consegnato (`cmp`): devono essere identici.

### Su Google Colab
Ogni notebook gira in un runtime separato: quello che deve passare da uno all'altro sta su Drive, in `DRIVE_DIR` (default `MyDrive/mtah2027`):
- `data/` (da caricare a mano), `cache/` (ATD parsato, dati Metrica, log della calibrazione), `params.json` e `deadball_model.pkl` (scritti da 02, letti da 03: stessa versione di scikit-learn).
- La calibrazione (02) scrive ogni configurazione in `cache/calibration_v5.jsonl`: se il runtime cade, rieseguendo riparte da dove era.
- Il notebook 03 si ferma con un errore se `params.json` non c'è.
- Colab clona la branch `team-solution` da GitHub: le modifiche al codice devono essere pushate.
- I notebook 02 e 03 installano `solution/requirements.txt` all'avvio. Se pip cambia una libreria già caricata, il runtime si riavvia da solo una volta: basta rieseguire il notebook dall'inizio.

Tempi indicativi (1 core): parsing ATD ~1 min la prima volta, pipeline ~5 s, calibrazione ~3,6 s per configurazione.

## Continuità
`solution/continuity/` contiene il contesto per chi riprende il lavoro: stato e decisioni (`STRATEGIA.md`), diario degli esperimenti con i numeri (`EXPERIMENTS.md`), script e risultati grezzi degli esperimenti su Metrica. Non serve a generare la submission.
