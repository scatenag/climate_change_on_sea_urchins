# 0006 — Perché l'aggiornamento automatico dei dati non si sospende e il congelamento avviene con un tag

## Contesto

`update_ec50.yml` riscarica EC50/Copernicus a calendario e rilancia la pipeline di analisi,
committando su `main` qualunque cosa sia cambiata. Questo significa che `data/` e `results/`
su `main` continuano ad avanzare nel tempo indefinitamente. Il manoscritto cita uno stato
numerico specifico del pacchetto (v1.5.0) come indipendentemente verificabile.

## Alternative valutate

- **Sospendere l'auto-update durante la revisione o dopo la pubblicazione**: scartata —
  l'auto-update stesso è una capacità rivendicata nelle conclusioni del lavoro; sospenderlo
  contraddirebbe ciò che viene pubblicato, e non esiste un momento naturale per riattivarlo.
- **Un branch permanentemente congelato, separato da `main`**: scartata — frammenta la storia,
  duplica la manutenzione, e non serve dato che i tag git risolvono già il problema.
- **Un tag git che punta al commit esatto che i `results/` di una release riflettono**: scelta
  adottata (già usata per `v0.1.0-sartori-2023`).

## Decisione

Il workflow di auto-update resta attivo su `main`, sempre, senza modifiche. Il congelamento di
uno stato numerico specifico per citazione/verifica avviene esclusivamente via tag git (es.
`v1.5.0`) più il relativo archivio Zenodo, mai sospendendo o isolando il meccanismo di
aggiornamento live.

## Conseguenze

`HEAD` di `main` diverge da qualunque release taggata col passare del tempo — è atteso e
voluto (vedi la sezione "Reproducing the manuscript's published numbers" del README, e la
didascalia del dashboard sulla stessa divergenza). Chi cita un risultato numerico specifico
deve citare il tag/DOI, non "il repository" genericamente.
