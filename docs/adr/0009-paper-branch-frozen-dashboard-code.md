# 0009 — Il dashboard citato nel lavoro gira da un branch a codice congelato, con i dati aggiornati

## Contesto

Il manoscritto (Marine Pollution Bulletin, in revisione) cita l'indirizzo pubblico del
dashboard. Fino al 29/9/2026 quell'app seguiva `main`. Con V2 `main` è diventato il luogo
dello sviluppo dello strumento generico: ogni modifica al codice arrivava subito nell'app
citata. In un solo giorno, per esempio, la #27 ha rinominato chiavi e riscritto testi di
output. Erano correzioni, ma il meccanismo lascia che l'app citata cambi insieme allo
sviluppo generico.

Il proprietario vuole due cose insieme: l'app citata deve mostrare **i dati sempre
aggiornati**, e il suo **codice** non deve essere snaturato dalle modifiche successive. Il
dashboard generico, quando servirà, sarà un'altra app.

## Rapporto con ADR-0006

ADR-0006 resta valido per ciò che decide: i **numeri** citati si congelano con un tag
(`v1.5.0`) e con l'archivio Zenodo, e l'aggiornamento automatico su `main` non si sospende
mai. Fra le alternative scartava però «un branch permanentemente congelato, separato da
`main`», perché frammenta la storia e duplica la manutenzione. Quell'alternativa serviva a
congelare i numeri, e per quello il tag basta. Questo ADR introduce un branch per uno scopo
che il tag non copre: **il codice dell'app citata, con dati che continuano ad arrivare**. Un
tag non riceve commit, quindi non può ricevere dati. I costi indicati da ADR-0006 restano veri
e sono accettati esplicitamente: una seconda storia e un secondo aggiornamento da mantenere.

## Alternative valutate

- **L'app citata segue `main`** (stato fino al 29/9): scartata, perché ogni modifica dello
  sviluppo generico arriva nell'app citata.
- **L'app citata punta al tag `v1.5.0`**: scartata, perché un tag non riceve dati; l'app
  mostrerebbe per sempre i dati del 14/9.
- **Branch `paper/mpb-2026` a codice congelato, con un aggiornamento dei dati che usa il
  codice del branch**: scelta adottata.

## Decisione

- **`paper/mpb-2026`** parte da `v1.5.0` e contiene solo correzioni di errori visibili nel
  dashboard: etichette mg/L, affermazioni false o scritte a mano sostituite da testo generato
  dai valori, titoli delle figure che descrivono invece di concludere. Contiene anche
  `renv.lock`, che congela R e i pacchetti del DLNM. Niente sviluppo.
- **L'aggiornamento dei dati del branch** è `.github/workflows/update_paper_branch.yml`, su
  `main` perché i workflow a schedule partono solo dal branch predefinito. Scarica il branch,
  installa il suo `requirements-lock.txt` e il suo `renv.lock`, ricalcola con il **suo**
  codice, rigenera le figure, esegue il suo `tests/test_data_quality.py` prima del commit e
  committa sul branch. In caso di fallimento apre una issue che menziona il proprietario. Sul
  branch i numeri cambiano solo perché arrivano dati nuovi.
- **Protezione**: il branch è protetto su GitHub da cancellazione e force push, anche per gli
  amministratori.
- **Regole**:
  - sul branch `paper/mpb-2026` non si scrive senza una richiesta esplicita del proprietario
    del repository;
  - il workflow `update_paper_branch.yml` non si modifica senza una richiesta esplicita del
    proprietario, anche se sta su `main`.
- L'app citata viene pubblicata dal branch. L'app che segue `main` diventa quella di
  sviluppo.

## Conseguenze

- I numeri dell'app citata si allontanano da quelli del lavoro man mano che arrivano dati, già
  dalla prima esecuzione (il foglio EC50 è cambiato dopo il 14/9). È voluto, e il dashboard lo
  dichiara nell'intestazione. Chi vuole i numeri del lavoro usa il tag e il DOI (ADR-0006).
- Il giorno in cui un prodotto Copernicus cambia versione e gli script della v1.5.0 non
  scaricano più, il job fallisce e parte l'allarme; la correzione va chiesta esplicitamente,
  perché tocca il branch.
- Un eventuale rilascio `v1.5.1` (per esempio con la fixture corretta per la prova 224) può
  chiudere lo stato del branch con un tag, senza fermare l'aggiornamento.
