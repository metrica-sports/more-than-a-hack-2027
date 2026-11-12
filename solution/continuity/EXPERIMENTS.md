# Diario degli esperimenti (solo partite Metrica)

Tutti gli esperimenti sono **appaiati**: per ogni configurazione casuale dei parametri (stesso seed) si misura lo spellF1 ufficiale (`evaluation/possession_metrics.py`) sulle due partite Metrica *con* e *senza* la novità, a parità di tutto il resto. Le partite sono preparate con `calibrate.prepare(seed=0)`: palla "atd_like", etichette ricavate dagli eventi, classificatori allenati sull'altra partita (leave-one-game-out).

"Partita peggiore" = il minimo dei due spellF1, il criterio con cui `calibrate.best_of` sceglie la configurazione. La banda di rumore dello spellF1 dichiarata dal valutatore è circa 0.04.

Gli script sono in `scripts/`, i risultati grezzi in `results/`. Per rilanciarli: `MTAH_EXP_DIR=<cartella>` (default `solution/cache/experiments`), dalla radice della repo, **dopo aver fatto checkout del commit indicato**: la forma delle tuple restituite da `prepare` è cambiata tra un commit e l'altro.

---

## 1. Protocollo "atd_like" (commit 159f1bb)

**Domanda.** Le partite Metrica somigliano al tracking automatico?
**Trovato.** No: in Metrica la palla manca nel 92–100% dei frame di palla ferma e solo nel 9–15% a palla viva. Nell'ATD manca nel 34% dei frame di palla ferma e ~41% in totale, quindi quasi indipendentemente da D. La vecchia calibrazione sfruttava la scorciatoia "palla mancante = palla ferma".
**Fatto.** `metrica.atd_like_ball`: i buchi nativi vengono riempiti per interpolazione, poi si tagliano buchi lunghi casuali (lognormale, mediana 268 frame) fino al 41% dei frame.
**Effetto.** Il migliore spellF1 su Metrica passa da 0.44 (protocollo vecchio, gonfiato) a ~0.37: numero più onesto.

## 2. Classificatore dei buchi della palla (commit 159f1bb) — `scripts/paired.py`, `scripts/hgb.py`, `scripts/mf.py`

**Domanda.** Nei buchi della palla, un classificatore del contesto aiuta a distinguere palla ferma da stesso possesso?
**Modello.** `gaps.py`: regressione logistica su 16 feature (velocità dei giocatori, ingresso e uscita della palla dal buco, durata, cambio di squadra più vicina).
**Numeri.**
- AUC fuori partita 0.877 / 0.787; la sola velocità dei giocatori fa già 0.868 / 0.792. Il gradient boosting arriva a 0.894 / 0.828.
- 116 configurazioni appaiate (`results/gap_paired.jsonl`): spellF1 +0.009 in media, ma sulle 10 configurazioni migliori nessun guadagno (da −0.018 a +0.009). Il gradient boosting sulle 20 migliori fa lo stesso (−0.002 / −0.004).
- D-F1 sale di 0.005–0.01, macroF1 invariato.

**Conclusione.** Spento (`k_gap = 0`, non più cercato): l'informazione nei buchi è già nella velocità dei giocatori.

## 3. Classificatore di palla ferma su tutti i frame (commit 314327f) — `scripts/db_eval.py`, `scripts/db_paired.py`

**Domanda.** Un classificatore di palla ferma su *ogni* frame, con gli schemi di ripresa di Vidal-Codina, aiuta?
**Modello.** `deadball.py`: 25 feature multi-scala (palla ferma nei punti di ripresa, giocatori che rallentano e ripartono, metà campo separate al calcio d'inizio, contesto dei buchi). Il log-odds entra come evidenza per D con peso `k_dead`.
**Numeri.**
- AUC fuori partita: gradient boosting 0.925 / 0.914, logistica 0.922 / 0.905, sola velocità dei giocatori 0.871 / 0.800.
- 100 configurazioni appaiate (`results/deadball_paired.jsonl`):

| | partita peggiore, migliore config | spellF1 medio | configurazioni migliorate |
|---|---|---|---|
| senza | 0.374 | 0.271 | – |
| logistica | 0.397 | 0.320 | 100/100 |
| gradient boosting | **0.413** | 0.332 | 100/100 |

- Il guadagno cresce con `k_dead` fino a 0.4–0.6 (+0.083 in media).
- Sulla configurazione migliore: macroF1 0.719/0.698 → 0.756/0.754; recall di D 0.78/0.68 → 0.84/0.84.

**Conclusione.** Tenuto, con il gradient boosting.

**Controllo sulla finestra pubblica (12/11, solo controllo):** spellF1 0.3355 (era 0.3077); 14 possessi contro 17 (erano 9); recall di D 0.33 (era 0.08–0.11) con precisione 0.89. Si trasferisce, ma solo in parte.

## 4. Ancore dei calci piazzati dallo Smart Tagging (commit c76b3d0) — `scripts/sp_paired.py`

**Domanda.** I `SET PIECES` dello Smart Tagging, usati come ancore (palla ferma prima della ripresa, possesso di chi batte dopo), aiutano?
**Come si calibra senza tag.** `metrica.synthetic_tags` simula lo Smart Tagging dagli eventi Metrica:
- finestre di 3–7 s, che iniziano 1–4 s prima della ripresa;
- Type del calcio piazzato; squadra di chi batte, `N/A` sulle rimesse;
- tassi di mancati per tipo: rimessa 0.55, angolo 0.9, rinvio 0.65, punizione 0.25, calci d'inizio 1.0;
- 2 falsi positivi per tempo.

**Rilevatore della ripresa.** Errore mediano 0.56 s / 0.76 s; entro 1 s il 65–66%, entro 2 s il 77–82%. Squadra di chi batte giusta in 47/47 e 46/46 casi. Il segno del tag Team è risolto in automatico: accordo 19/24 e 23/26.
**Numeri.** 100 configurazioni appaiate (`results/setpieces_paired.jsonl`): spellF1 medio +0.018, migliorato in 100/100; partita peggiore, migliore configurazione 0.407 → 0.412 (dentro il rumore). Il guadagno cresce un po' con `k_sp_dead`.
**Conclusione.** Tenuto. Su Metrica il margine è piccolo perché la palla ferma è già all'84%; l'obiettivo è l'ATD, dove è al 33%. Da verificare sulla finestra pubblica dopo la calibrazione v5.

---

## Idee non ancora provate

- Costi di cambio diversi per coppia di stati in Viterbi (H→A ≠ H→D): il codice in `decode.py` è già predisposto, mancano 2–3 parametri in `calibrate.SEARCH`.
- Durate realistiche (HMM semi-markoviano) invece della sola durata minima.
- Classificatore di squadra (H contro A) sullo stesso schema del classificatore di palla ferma: direzione della palla, forma delle squadre, chi era in possesso prima e dopo un buco.
- Confronto delle distribuzioni delle feature tra Metrica e ATD, senza usare la ground truth, per capire perché la palla ferma passa dall'84% al 33%. Lo deve fare il team: Claude non può vedere i dati della partita.
