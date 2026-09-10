@docs/roadmap/STATO.md

# Istruzioni permanenti per Claude Code

Leggi `STATO.md` (incluso sopra) prima di rispondere a qualunque richiesta. Se non esiste
ancora, dimmelo e fermati.

## Che progetto è

Stiamo trasformando questo repository da analisi di un singolo caso (EC50 di *Paracentrotus
lividus* al largo di Livorno, dati Copernicus Marine) a strumento generico per lo studio di
relazioni ritardate tra serie ambientali e serie di risposta biologica.

Utenti finali attesi: ecotossicologi e ricercatori su impatti dei cambiamenti climatici.

La roadmap è in `docs/roadmap/`, organizzata per versioni: **V1** il caso attuale, **V2** il
caso che diventa configurazione, **V3** casi multipli e confronto, **V4** meta-tool, **V5**
istanza condivisa. Non lavorare mai su una versione diversa da quella indicata in `STATO.md`
senza che io te lo chieda esplicitamente.

## Regole invarianti (non derogabili)

1. **Nessun risultato scientifico cambia senza mia approvazione esplicita.** Prima e dopo
   ogni modifica esegui la verifica dei risultati di riferimento. Se un numero cambia:
   **fermati**, spiegami cosa è cambiato e perché, e aspetta la mia risposta. Non aggiornare
   mai i valori di riferimento di tua iniziativa.
2. **Niente stato globale che descriva il caso studiato.** Sito, serie di risposta, nomi di
   colonna, finestra temporale, elenco delle variabili viaggiano in un oggetto di contesto
   passato esplicitamente, non come costanti di modulo.
3. **Il nucleo scientifico non conosce l'interfaccia.** I moduli di analisi non importano
   Streamlit, non stampano a schermo, non decidono dove scrivere: ricevono dove scrivere.
4. **Gli artefatti hanno identità derivata dalla specifica, mai nomi fissi.**
5. **Un solo confine per leggere e scrivere dati.** Non spargere accessi al filesystem nei
   moduli.
6. **Le scelte scientifiche stanno nella specifica, non nel codice.** Se una trasformazione
   sui dati (imputazione, riempimento, soglia, finestra) non è leggibile nel file di studio,
   è un difetto da segnalare.
7. **La configurazione descrive, non esegue.** Mai eseguire codice arbitrario proveniente da
   un file di configurazione.
8. **Ogni risultato porta la propria provenienza**: input, versioni, parametri.

## Comandi

```bash
pip install -e ".[test]"
pytest tests/                        # richiede che il pipeline sia già stato eseguito una volta
ccsu-run-pipeline                    # rigenera results/ — MHW detection gira per prima, tutto il resto dipende da lei
ccsu-dashboard                       # equivalente a: streamlit run app.py
Rscript scripts/mhw_lag_analysis.R   # opzionale, solo DLNM
```

Aggiornamento dati (serve credenziali Copernicus, mai interattivo — vedi Trappole):
```bash
python scripts/fetch_copernicus_update.py && python scripts/build_dataset.py && ccsu-run-pipeline
```

## Convenzioni di codice

- Python 3.10+. Ogni modulo di analisi espone un `run()` eseguibile da solo e scrive in
  `results/`; `pipeline.py` li orchestra in ordine.
- Il dashboard legge di norma solo CSV precalcolati. Le eccezioni attuali (Pre/Post split, e
  — dietro filtro data — SARIMAX/correlazioni/CCF) sono deliberate e vanno segnalate in UI.
  Granger causality e stazionarietà sono state rese live una volta e poi ripristinate a
  precalcolate dopo un segfault in produzione: non riprovare senza un motivo forte.
- `scripts/` resta fuori dal pacchetto installabile: richiede credenziali, va eseguito a mano.

## Come lavoriamo

- **Un passo = un branch = una sessione = una pull request.** Se il lavoro non entra in una
  sessione, dimmelo e proponimi come spezzarlo.
- **Prima il piano, poi il codice.** Leggi i file rilevanti e proponi un piano; non scrivere
  codice finché non ti dico di procedere.
- **Test prima dell'implementazione** per ogni contratto o interfaccia nuova.
- **Commit piccoli**, messaggi in formato convenzionale (`feat:`, `fix:`, `refactor:`,
  `test:`, `docs:`, `chore:`).
- **Niente "già che ci sono".** Se noti qualcosa da migliorare fuori dal compito corrente,
  **non toccarlo**: elencamelo alla fine e ne apro una issue.
- **A fine passo**: elenca cosa hai cambiato e perché, aggiorna `docs/roadmap/STATO.md` e
  `CHANGELOG.md`, e segnalami se serve un ADR.

## Decisioni architetturali

Ogni decisione non banale va in `docs/adr/NNNN-titolo-breve.md`: contesto, alternative
valutate, decisione presa, conseguenze. Controlla prima se esiste già un ADR sul tema: se c'è,
si segue; se pensi vada rivisto, dimmelo — non aggirarlo. Le decisioni volutamente rimandate
sono in `docs/adr/0000-decisioni-rimandate.md`: non anticiparle.

## Cose da non fare mai

- Non reimplementare funzionalità che esistono in pacchetti mantenuti; se pensi di doverlo
  fare, chiedimelo prima e spiegami perché l'alternativa non basta.
- Non rifattorizzare il dashboard interamente in un colpo solo: si procede per schede, dietro
  interruttori.
- Non introdurre dipendenze nuove senza chiedermelo.
- Non modificare i file in `data/` e `examples/` se non è quello il compito.
- Non generalizzare oltre quanto serve alla fase corrente: un'astrazione costruita su un solo
  caso d'uso è quasi sempre l'astrazione sbagliata.
- Non cancellare o riscrivere codice che non capisci: chiedimi a cosa serve.

## Trappole già note

- `add_vline` rompe nei subplot plotly con x stringa — usa `add_shape` con
  `xref="x", yref="paper"` (vedi `dashboard.py`, linea verticale di `SPLIT_DATE`).
- `fillna(method=...)` è deprecato in pandas — usa `.ffill().bfill()`.
- Il login interattivo Copernicus fallisce in shell non interattive — usa le env var
  `COPERNICUSMARINE_SERVICE_USERNAME` / `COPERNICUSMARINE_SERVICE_PASSWORD`.
- Il dashboard aggira l'import di Python con un lock globale che serializza tutte le sessioni:
  prima di rifattorizzarlo, verifica se la causa originale è ancora presente.

## Contesto scientifico utile

- La serie osservata **non è necessariamente un biomarker**: l'EC50 del caso attuale è un
  proxy dello stato riproduttivo. Il termine generico è *serie di risposta*; cosa sia lo
  dichiara la specifica, non il nome della colonna.
- I dati ambientali vengono da rianalisi su griglia: la cella più vicina non sempre
  rappresenta il sito reale (es. lagune). Quando è così, va segnalato, non nascosto.
- Le serie sono autocorrelate e stagionali: i test standard sovrastimano la significatività.
  Modelli nulli e correzioni per test multipli sono un requisito, non un abbellimento.
- Con più casi e più ritardi il numero di test cresce in fretta: ogni analisi deve dichiarare
  quanti test esegue.
