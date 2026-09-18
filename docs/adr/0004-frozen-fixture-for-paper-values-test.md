# 0004 — Perché il test sui valori pubblicati gira su una fixture congelata e non su `data/`

## Contesto

`data/` viene aggiornato automaticamente (workflow `update_ec50.yml`, giornaliero per EC50,
mensile per Copernicus) e la pipeline viene rieseguita sui dati aggiornati. Un test che
verificasse i numeri pubblicati dal manoscritto contro `data/` live diventerebbe rosso in
modo permanente non appena la serie si allunga oltre il punto su cui quei numeri sono stati
calcolati — un fallimento di deriva dei dati, non una regressione del codice.

## Alternative valutate

- **Test contro `data/` live, con tolleranze larghe**: scartata — le tolleranze dovrebbero
  crescere nel tempo fino a mascherare regressioni vere.
- **Test contro `results/` precalcolato**: scartata — non verifica che il codice *produca*
  quei numeri, solo che una esecuzione passata li abbia prodotti; inoltre anche `results/`
  viene rigenerato sullo stesso calendario e andrebbe incontro alla stessa deriva.
- **Congelare uno snapshot dei dati esatti su cui i numeri della release sono stati calcolati,
  e rieseguire davvero i moduli di analisi contro quello snapshot**: scelta adottata.

## Decisione

`tests/fixtures/paper_mpb_2026/data/` contiene una copia congelata dei CSV su cui la release
v1.5.0 è stata verificata. La fixture `paper_results` in `tests/conftest.py` sostituisce (via
monkeypatch) `ROOT`/`RESULTS` di ciascun modulo interessato per puntare a quella fixture e a
una directory di output temporanea, poi esegue una volta per sessione di test i quattro moduli
di riproduzione dei valori del manoscritto. `tests/test_paper_values.py` confronta il
risultato con i valori pubblicati, con tolleranza dichiarata.

## Conseguenze

Questo test resta verde mentre `data/` continua a crescere; diventa rosso solo se il *codice*
smette di riprodurre i numeri della release congelata, che è l'invariante che vale davvero
proteggere. La fixture stessa è uno snapshot puntuale e va ricongelata (nuova cartella datata)
per qualunque *futuro* manoscritto che questo pacchetto dovrà verificare — non è pensata per
seguire la serie live.
