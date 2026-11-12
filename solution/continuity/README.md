# Continuity: contesto per chi riprende il lavoro

Questa cartella serve a non perdere il filo tra una sessione e l'altra (persone del team o Claude).
Non serve a generare la submission: `run.py` e i notebook non la usano.

| file | cosa contiene |
|---|---|
| `STRATEGIA.md` | stato del progetto, decisioni prese, risultati, vincoli, prossimi passi (copia del documento del progetto Claude "MoreThanAHack27", aggiornata al 12/11/2026) |
| `EXPERIMENTS.md` | diario degli esperimenti: domanda, protocollo, numeri, conclusione, commit su cui sono stati fatti |
| `scripts/` | gli script usati per gli esperimenti appaiati su Metrica (storici: ognuno va col commit indicato in `EXPERIMENTS.md`) |
| `results/` | i risultati grezzi (JSONL, una configurazione per riga) di quegli esperimenti |

Regole da ricordare sempre:

- **Mai tarare sulla partita della challenge** (né sulla finestra pubblica): calibrazione solo sulle partite open di Metrica. La finestra pubblica serve solo come controllo di correttezza.
- **Mai caricare i dati BAR-ATH o loro estratti su LLM hosted** (T&C), compresi grafici o righe della ground truth pubblica. Le sessioni Claude lavorano solo su codice e dati Metrica.
- Tutto ciò che è in `results/` riguarda **solo** le partite Metrica.
