# Soluzione: possesso palla solo da dati numerici

Input usati: tracking automatico (ATD) e Smart Tagging del video tattico. Niente video, niente homography, niente broadcast.

## Pipeline
1. `possession/data.py`: carica l'ATD in metri (cache in `solution/cache/`, mai committata).
2. `possession/features.py`: distanza dalla palla per squadra, velocità palla, palla fuori campo, velocità collettiva dei giocatori.
3. `possession/emissions.py`: evidenza per frame per H/A/D; Smart Tagging come prior debole (segno dei tag di squadra deciso in automatico).
4. `possession/decode.py`: HMM a durata minima (Viterbi) per tempo; i cambi di stato hanno un costo.
5. `possession/gaps.py`: classificatore dei buchi della palla. Nei frame in cui la palla non è tracciata stima P(palla ferma) dal contesto (velocità dei giocatori, dove e come la palla è sparita e ricomparsa, durata del buco); il suo log-odds entra come evidenza per D con peso `k_gap`. Regressione logistica allenata solo sulle partite Metrica, salvata in `gap_model.json` (accanto a `params.json`); in inferenza serve solo numpy.
6. `possession/calibrate.py` + `possession/metrica.py`: parametri calibrati su spellF1 sulle partite open di Metrica (etichette derivate in automatico dagli eventi), mai sulla partita della challenge. Per somigliare al tracking automatico, nei dati Metrica i buchi nativi della palla (quasi tutti a palla ferma) vengono riempiti e poi si tagliano buchi lunghi casuali (41% dei frame): così "palla mancante" non vuol più dire "palla ferma". In calibrazione il classificatore dei buchi di ogni partita è allenato sull'altra (leave-one-game-out).

## Esecuzione
```bash
pip install -r requirements.txt -r solution/requirements.txt
# notebook, in ordine: solution/notebooks/01 -> 02 (pesante: calibrazione) -> 03 (submission)
python solution/run.py --out evaluation/results/<team_name>.csv   # rigenera la submission da params.json
python evaluation/validate_submission.py evaluation/results/<team_name>.csv
```
`run.py` si ferma se `params.json` manca (`--allow-defaults` solo per prove, `--params <file>` per un altro percorso).

### Su Google Colab
Ogni notebook gira in un runtime separato: quello che deve passare da uno all'altro sta su Drive, in `DRIVE_DIR` (default `MyDrive/mtah2027`):
- `data/` (da caricare a mano), `cache/` (ATD parsato, dati Metrica, log della calibrazione), `params.json` e `gap_model.json` (scritti da 02, letti da 03).
- La calibrazione (02) scrive ogni configurazione in `cache/calibration_v3.jsonl`: se il runtime cade, rieseguendo riparte da dove era.
- Il notebook 03 si ferma con un errore se `params.json` non c'è.
- Colab clona la branch `team-solution` da GitHub: le modifiche al codice devono essere pushate.

Tempi indicativi (1 core): parsing ATD ~1 min la prima volta, pipeline ~5 s, calibrazione ~3,6 s per configurazione.
