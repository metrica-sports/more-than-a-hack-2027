# Soluzione: possesso palla solo da dati numerici

Input usati: tracking automatico (ATD) e Smart Tagging del video tattico. Niente video, niente homography, niente broadcast.

## Pipeline
1. `possession/data.py`: carica l'ATD in metri (cache in `solution/cache/`, mai committata).
2. `possession/features.py`: distanza dalla palla per squadra, velocità palla, palla fuori campo, velocità collettiva dei giocatori.
3. `possession/emissions.py`: evidenza per frame per H/A/D; Smart Tagging come prior debole (segno dei tag di squadra deciso in automatico).
4. `possession/decode.py`: HMM a durata minima (Viterbi) per tempo; i cambi di stato hanno un costo.
5. `possession/calibrate.py` + `possession/metrica.py`: parametri calibrati su spellF1 sulle partite open di Metrica (etichette derivate in automatico dagli eventi), mai sulla partita della challenge.

## Esecuzione
```bash
pip install -r requirements.txt -r solution/requirements.txt
# notebook, in ordine: solution/notebooks/01 -> 02 (pesante: calibrazione) -> 03 (submission)
python solution/run.py --out evaluation/results/<team_name>.csv   # rigenera la submission da params.json
python evaluation/validate_submission.py evaluation/results/<team_name>.csv
```
Tempi indicativi (1 core): parsing ATD ~1 min la prima volta, pipeline ~5 s, calibrazione ~3,6 s per configurazione.
