# More Than A Hack 2027 — ricerca e strategia (29/10/2026, aggiornato 12/11)

Copia del documento del progetto Claude "MoreThanAHack27" (`claude/strategia-possesso-ricerca.md`).

Task: etichetta H/A/D/X per ognuno dei 199.319 frame di BAR-ATH. Classifica su spellF1 (possessi come segmenti, IoU temporale), tie-break macroF1. Floor: baseline_smooth50 spellF1 0.2433. Scadenza 17/11/2026.
Vincolo del team: solo dati numerici (tracking ATD + Smart Tagging), niente video; homography e broadcast ignorati. I calcoli pesanti si eseguono sui server Google, non sul portatile.
Vincolo T&C: i dati della partita e i loro estratti (anche grafici o righe della ground truth pubblica) non vanno caricati su LLM hosted → Claude lavora solo sul codice e sui dati Metrica, mai sui file BAR-ATH.

## Come funziona lo spellF1 (chiarito il 12/11)
Ogni possesso (run massimale di H o di A; D = interruzione) conta UNA volta, indipendentemente dalla durata. L'abbinamento è uno-a-uno per IoU temporale, con F1 mediato sulle soglie 0.3-0.7. Contano il numero di possessi e la posizione dei confini (1 s di ritardo ovunque → 0.65). Non si pesa la durata: un possesso corto mancato costa quanto uno lungo. L'HMM a durata minima con costo di cambio, calibrato direttamente su spellF1, è già la versione "furba" della finestra mobile delle baseline.

## Ancore dei calci piazzati dallo Smart Tagging (12/11, commit c76b3d0)
- `possession/setpieces.py`: per ogni `SET PIECES` trova nel tracking il frame di ripresa (palla ferma che riparte, con bonus se è nel punto tipico del `Type`; fallback: i giocatori accelerano, poi il centro della finestra).
  - Prima della ripresa: evidenza D su una rampa, con durata per tipo (mediane su Metrica: rimessa 7 s, punizione 12, rinvio 20, angolo 28, calcio d'inizio dopo gol 55) scalata da `sp_back_scale`.
  - Dopo la ripresa: 3 s di evidenza per chi batte e contro D (`k_sp_team`). Chi batte = squadra del tag; se il tag indichi chi batte o l'avversario è deciso per accordo con il giocatore più vicino dopo la ripresa. Rimesse `N/A` → giocatore più vicino.
  - `GOALS` → D fino al calcio d'inizio successivo.
- `metrica.synthetic_tags`: tag sintetici dagli eventi Metrica (finestre 3-7 s attorno alla ripresa, Type, squadra, N/A sulle rimesse; tassi di mancati per tipo: rimessa 0.55, angolo 0.9, rinvio 0.65, punizione 0.25, calci d'inizio 1.0; 2 falsi positivi per tempo). Così `k_sp_dead`, `k_sp_team` e `sp_back_scale` vengono calibrati con il resto.
- Il vecchio prior SET PIECES è spento (`k_tag_setpiece` = 0); i tag di squadra di SET PIECES/GOALS non sono più usati come prior di fase. `k_tag_team` resta a mano.
- Su Metrica: ripresa trovata con errore mediano 0.6-0.8 s (65% entro 1 s, ~80% entro 2 s), squadra di chi batte giusta ~100%. spellF1 medio +0.018, meglio in 100/100 configurazioni appaiate; migliore partita peggiore 0.407 → 0.412 (dentro il rumore). Su Metrica c'è poco margine (recall di D già 0.84); l'obiettivo è l'ATD (recall di D 0.33).
- Nello stesso commit:
  - versioni esatte delle dipendenze (numpy 2.2.6, pandas 2.2.3, scipy 1.15.3, scikit-learn 1.6.1, matplotlib 3.10.3), installate dai notebook 02/03 su Colab;
  - `solution/TECHNICAL.md` (documento tecnico);
  - log della calibrazione → calibration_v5.jsonl.

## Controllo sulla finestra pubblica con il modello deadball (12/11)
Solo controllo, niente tuning su questi numeri:
- spellF1 0.3355 (era 0.3077), forchetta bootstrap 0.18–0.50: 2° dietro smooth125 0.4242, ma la differenza è dentro il rumore.
- macroF1 0.6741: 1° su tutti (smooth125 0.58).
- Possessi 14 contro 17 (erano 9); durata media 19.9 s contro 14.2 (era 32.3).
- D: recall 0.33 (era 0.08-0.11), precisione 0.89 → il classificatore si trasferisce in parte. Alive/dead F1 0.92.
- Squadra: team acc 0.83; probabilmente 1-2 possessi interi assegnati alla squadra sbagliata.
- Transizioni: F1 0.16, solo 3 su 20 agganciate; spellF1 0.52 @0.3 → 0.13 @0.7 → confini imprecisi.

## Classificatore di palla ferma su tutti i frame (09/11, commit 314327f)
- `possession/deadball.py`: gradient boosting su 25 feature multi-scala (schemi di ripresa di Vidal-Codina, giocatori che rallentano e ripartono, contesto dei buchi). Il log-odds diventa evidenza per D con peso `k_dead`.
- Allenato solo su Metrica (atd_like, etichette dagli eventi); leave-one-game-out in calibrazione. Il notebook 02 salva `deadball_model.pkl` accanto a params.json.
- Su Metrica: AUC 0.925 / 0.914; spellF1 della partita peggiore 0.374 → 0.413, meglio in 100/100; macroF1 ~0.71 → ~0.75; recall di D 0.78/0.68 → 0.84/0.84.

## Classificatore dei buchi della palla (09/11, commit 159f1bb)
- `possession/gaps.py` (spento, k_gap = 0): nessun guadagno oltre la velocità dei giocatori.
- Protocollo Metrica "atd_like": buchi nativi riempiti, poi buchi lunghi casuali sul 41% dei frame. Su Metrica la palla mancava nel 92-100% dei frame D, una scorciatoia che nell'ATD non vale. Col protocollo atd_like il migliore senza classificatori fa ~0.37, non 0.44.

## Stato al 08/11
- Prima submission `team_name`: spellF1 0.3077, 9 possessi contro 17, durata media 32,3 s contro 14,2. Era la v1 con parametri di default per un problema di Drive, poi risolto.
- La finestra pubblica è più facile della partita intera (smooth50: 0.326 qui contro 0.243). Con 17 possessi il rumore è alto; il notebook 03 fa già --bootstrap 1000.

## Implementazione
Codice in `solution/` (branch team-solution):
- `possession/`: data, features, emissions, decode (HMM a durata minima con Viterbi), metrica, calibrate, pipeline, gaps, deadball, setpieces;
- `run.py`, `requirements.txt`, `README.md`, `TECHNICAL.md`;
- notebook: 01 esplorazione, 02 calibrazione (genera params.json + deadball_model.pkl), 03 pipeline e submission.

X solo fuori dai tempi.

## Verdetto su Jev (TypeSafe AI)
Sconsigliato: è vietato caricare i dati su LLM hosted di terzi; è text-only; decide frame per frame (→ flicker); non è riproducibile.

## Verifica sui dati reali (30/10)
- ATD: palla x,y; 46 slot giocatori con x,y,velocità e squadra. Nessun flag palla viva/morta, possessore, altezza palla, ID giocatore.
- Palla presente 59.3% dei frame valutati; buchi pochi e lunghi (149, mediana ~11 s). Palla mancante nel 34% dei frame a palla ferma.
- Giocatore più vicino alla palla: ~76% di concordanza con la GT.
- Smart Tagging (da DATA.md): 638 istanze; SET PIECES 59 con Type (rimessa 26, angolo 12, rinvio 10, punizione 5, calcio d'inizio dopo gol 4, inizio 2); GOALS 4; Team N/A su tutte le rimesse.

## Letteratura chiave
- Vidal-Codina et al. 2022, Automatic event detection in football using tracking data — https://arxiv.org/abs/2202.00804
- Link & Hoernig 2017, Individual ball possession in soccer, PLOS ONE — https://doi.org/10.1371/journal.pone.0179953
- Borghesi et al. 2023, TCN per il possesso, ESWA 223 — https://www.sciencedirect.com/science/article/pii/S0957417423002816
- Kim et al. 2023, Ball Radar — https://arxiv.org/abs/2306.08206 (github.com/pientist/ballradar)
- Capellera et al. 2024, TranSPORTmer — https://arxiv.org/abs/2410.17785
- Dati di altre partite: https://github.com/metrica-sports/sample-data

## Prossimi passi (scadenza 17/11)
1. Push di c76b3d0, poi notebook 02 (log v5) e 03 su Colab, poi controllo sulla finestra pubblica (possessi, recall di D, transizioni).
2. Consegna: copiare params.json + deadball_model.pkl finali in solution/ e committarli; verificare con run.py + cmp che il CSV sia identico; aggiornare TECHNICAL.md con i numeri finali.
3. Se resta tempo: classificatore di squadra (H contro A) sullo stesso schema.
