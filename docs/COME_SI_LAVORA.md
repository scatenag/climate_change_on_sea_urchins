# Come si lavora su questo repository

> Regole per chi sviluppa, compreso un assistente di codice in qualunque sessione. Sono la fonte
> di riferimento: il `CLAUDE.md` locale (escluso da git) le richiama e non le ripete. Le regole
> che portano un numero fra parentesi quadre nascono da un errore descritto in
> [`roadmap/PERCORSO.md`](roadmap/PERCORSO.md) §3.
>
> Prima di qualunque lavoro: leggere [`roadmap/STATO.md`](roadmap/STATO.md). Non si lavora su una
> versione o una milestone diversa da quella indicata lì senza una richiesta esplicita del
> proprietario.

---

## 1. Regole invarianti del codice

1. **Nessun risultato scientifico cambia senza approvazione esplicita del proprietario.** Prima e
   dopo ogni modifica si esegue il golden master. Se un numero cambia ci si ferma, si spiega
   cosa è cambiato e perché, e si aspetta. I valori di riferimento non si aggiornano mai di
   propria iniziativa.
2. **Niente stato globale che descriva il caso studiato.** Sito, serie di risposta, nomi di
   colonna, finestra, variabili viaggiano in un contesto passato esplicitamente.
3. **Il nucleo scientifico non conosce l'interfaccia.** I moduli di analisi non importano
   Streamlit, non stampano a schermo, non decidono dove scrivere: lo ricevono.
4. **Gli artefatti hanno identità derivata dalla specifica**, mai nomi fissi.
5. **Un solo confine per leggere e scrivere dati** (`common.py`); niente accessi al filesystem
   sparsi nei moduli. I percorsi si risolvono al momento della chiamata [4].
6. **Le scelte scientifiche stanno nella specifica, non nel codice.** Una trasformazione dei
   dati (imputazione, riempimento, soglia, finestra) non leggibile nel file di studio è un
   difetto da segnalare.
7. **La configurazione descrive, non esegue.** Mai eseguire codice proveniente da un file di
   configurazione; YAML sempre con `safe_load`.
8. **Ogni risultato porta la propria provenienza**: input, versioni, parametri.
9. **Ogni sorgente di casualità ha un seme fissato ed esposto come parametro.**

## 2. Regole nate dagli errori

- Negli output nessuna identità, nessun conteggio e nessuna data scritti a mano: tutto viene dai
  dati o dalla specifica [8].
- I testi degli output nascono dai valori calcolati e non contengono conclusioni fisse [8].
- Un dato mancante resta mancante, sia dentro la serie sia ai bordi [6, 7]. Nessun riempimento
  con 0, nessuna interpolazione o ripetizione oltre il primo o l'ultimo valore osservato.
- Una sola implementazione per ogni grandezza, condivisa da pipeline e dashboard [9].
- Un errore non si filtra in silenzio: si mostra dove si guarda il risultato [3].
- Unità ed etichette vengono dalla specifica o dal catalogo, con conversioni nominate in un punto
  solo [1, 2].
- Un test nuovo va visto fallire su un sabotaggio prima di considerarlo valido [5].
- Un test verde solo perché i dati coincidono per caso non verifica niente: i test girano su
  fixture congelate, mai sui dati vivi [5].
- Un numero che dipende dalla macchina non si riporta come valore esatto [10]. Si riporta l'esito che
  resta stabile alla perturbazione del dato e fra le scelte equivalenti (per esempio gli ordini ARIMA
  entro 2 punti di AIC dal migliore), come intervallo o come affermazione qualitativa («0 su 39»), con
  le condizioni della verifica e la data; un conteggio che cambia fra scelte equivalenti non si
  riporta. La misura va fatta e conservata: ADR-0000 voce 10 ne è l'esempio.
- Si committa ciò da cui si calcola [11].
- Tutto ciò che è specifico del caso di Livorno (rimandi al lavoro, esiti di indagini passate)
  sta in `examples/livorno_paracentrotus/NOTES.md`, non nel codice né negli output. È pubblico:
  niente nomi di persone, niente decisioni superate.

## 3. Come si procede

- **Un passo = un branch = una sessione = una pull request.** Se il lavoro non entra in una
  sessione, si propone come spezzarlo.
- **Prima il piano, poi il codice.** Si leggono i file rilevanti, si propone un piano, e non si
  scrive codice finché il proprietario non lo approva. Ogni pezzo è una PR con il suo criterio di
  verifica.
- **Test prima dell'implementazione** per ogni contratto o interfaccia nuova.
- **Si corregge solo ciò che blocca il passo in corso.** Ogni altro difetto trovato va in una
  issue e non si tocca; a fine passo si elenca.
- **Commit piccoli**, messaggi convenzionali (`feat:`, `fix:`, `refactor:`, `test:`, `docs:`,
  `chore:`, `ci:`).
- **A fine passo**: elenco di cosa è cambiato e perché, aggiornamento di `roadmap/STATO.md` e
  `CHANGELOG.md`, segnalazione se serve un ADR.
- **Decisioni.** Ogni decisione non banale va in `adr/NNNN-titolo-breve.md` (contesto,
  alternative, decisione, conseguenze). Se esiste già un ADR sul tema si segue; se va rivisto lo
  si dice, non lo si aggira. Le decisioni rimandate sono in `adr/0000-decisioni-rimandate.md`:
  non si anticipano.

### Ci si ferma e si chiede prima di
- aggiornare un riferimento del golden master o cambiare un risultato pubblicato;
- fare una scelta di metodo;
- scrivere sul branch `paper/mpb-2026` o modificare `.github/workflows/update_paper_branch.yml`
  (ADR-0009): solo su richiesta esplicita del proprietario, anche per correzioni che sembrano
  ovvie;
- introdurre una dipendenza nuova;
- modificare i file in `data/` e `examples/` se non è quello il compito.

## 4. Branch, PR e fusioni

- **Su `main` nessun push diretto**, nemmeno per correzioni piccole o urgenti della CI: ogni
  modifica passa da una PR, e dall'8/10 lo impone GitHub (ruleset 24719143: PR obbligatoria,
  controllo `test` verde, nessuna approvazione richiesta, nessuna eccezione per le persone; unica
  eccezione la deploy key del job di aggiornamento) [13]. Se `main` è rotto, la correzione va su
  un branch e una PR.
- **Prima di ogni push si esegue la suite completa** (`pytest tests/`), non solo i test toccati:
  un test lontano dalla modifica può fallire. Vale anche per `paper/mpb-2026` con la sua suite.
- **Fusioni** con `gh pr merge --merge`, solo con tutti i controlli verdi, mai con `--admin`.
  Fonde il proprietario, o l'assistente quando il proprietario lo chiede per quella PR.
- **Chiamate all'API di GitHub.** Mai con la credenziale git salvata. `gh` autenticato con il
  device flow è autorizzato per creare e fondere PR. Le issue le apre l'assistente quando il
  proprietario lo chiede, come è successo il 3/10/2026, con gli stessi strumenti.
- **Operazioni meccaniche e decisioni.** Dall'8/10/2026 le operazioni meccaniche (fusioni a
  controlli verdi, pulizie, rilettura dello stato, misure) le fa l'assistente senza attendere il
  proprietario; le decisioni restano del proprietario.
- Dopo una fusione: i branch delle PR fuse si cancellano da soli (`delete_branch_on_merge`).

## 5. Copie di lavoro e shell

- Le copie di lavoro (`git worktree`) stanno in un percorso persistente, per esempio
  `../ccsu-worktrees/<nome>` accanto alla cartella del repository, mai nella cartella temporanea
  della sessione, che viene svuotata [14].
- Ogni commit sta su un branch con nome; mai lavoro in *detached HEAD*.
- In una copia di lavoro i test vanno lanciati con `PYTHONPATH=<copia>/src:<copia>`: il pacchetto
  è installato in modo editable dalla cartella principale, e senza si testa il codice sbagliato.
- A fine sessione: `git worktree list`, rimuovere le copie non più servite, `git worktree prune`.
- Dopo un'operazione in blocco (cancellazioni, comandi su elenchi di nomi) si rilegge lo stato
  risultante prima di dichiararla riuscita [15]. La shell è zsh: un elenco in una variabile non
  quotata resta un argomento solo (usare un ciclo `while read`), e `$var:r`, `$var:h`, `$var:s`
  sono modificatori (scrivere `${var}`).

## 6. Dashboard e dati

- Il dashboard legge di norma risultati precalcolati. Le eccezioni (pre/post, e dietro il filtro
  per data correlazioni, CCF e forecast) sono deliberate e segnalate nell'interfaccia; usano le
  stesse funzioni del job. Granger e stazionarietà restano precalcolati dopo un segfault in
  produzione: non renderli vivi senza un motivo forte.
- Il dashboard **del lavoro** (`paper/mpb-2026`) non si tocca. Quello di `main` si riscrive,
  guidato da risultati e specifica (`roadmap/MILESTONE-M1.md` §7), invece di togliere il caso
  specifico dalle sue 2827 righe: la riscrittura non mette a rischio l'app citata, perché la
  dashboard di Livorno resta sul branch del lavoro. Il rifacimento si fa comunque per gruppi di
  pannelli, una PR ciascuno, mai mescolato ad altro lavoro.
- `app.py` forza un import nuovo a ogni esecuzione e un lock globale serializza le sessioni:
  prima di toccarlo, verificare se la causa originale è ancora presente.
- Dopo un push su un branch servito da Streamlit Cloud si verifica che l'app serva davvero il
  codice nuovo: la ridistribuzione non è sempre immediata.
- `scripts/` resta fuori dal pacchetto installabile: richiede credenziali e si esegue a mano o dal
  job.
- **Credenziali Copernicus**: mai `copernicusmarine login` (scrive un file di credenziali in
  `$HOME`); nel job e in locale le variabili d'ambiente `COPERNICUSMARINE_SERVICE_USERNAME` e
  `COPERNICUSMARINE_SERVICE_PASSWORD`; nell'app pubblica solo argomenti espliciti delle chiamate
  (vedi `roadmap/MILESTONE-M1.md` §3).
- L'aggiornamento automatico dei dati non si sospende (ADR-0006).

## 7. Trappole note

- `add_vline` rompe nei subplot plotly con asse x di stringhe: usare `add_shape` con `xref="x"`,
  `yref="paper"`.
- `fillna(method=...)` è deprecato in pandas; e `ffill`/`bfill`/`interpolate` ai bordi
  estrapolano (vedi §2).
- I log dei job di GitHub: `gh run view --log` può restituire vuoto; funziona
  `gh api repos/<owner>/<repo>/actions/jobs/<id>/logs`.
- L'argomento `DATA_DIR` di `scripts/mhw_lag_analysis.R` è verificato solo a mano (output
  identico sui dati reali): nessuna CI di PR lo esercita. Se lo si tocca, riverificarlo allo
  stesso modo.
- Una regola di `.gitignore` senza ancoraggio vale a ogni profondità: dopo un `git add` di una
  cartella di riferimento, controllare che tutti i file siano entrati.

## 8. Contesto scientifico

- La serie osservata **non è necessariamente un biomarker**: l'EC50 del caso di Livorno è un
  proxy dello stato riproduttivo di una popolazione. Il termine generico è *serie di risposta*;
  che cosa sia lo dichiara la specifica. Il verso in cui la risposta peggiora è dichiarato, mai
  dedotto dal nome.
- I dati ambientali vengono da rianalisi su griglia: la cella più vicina non sempre rappresenta il
  sito reale (lagune, aree confinate). Quando è così va segnalato, non nascosto.
- Le serie sono autocorrelate e stagionali: i test standard sovrastimano la significatività.
  Modelli nulli e correzioni per test multipli sono un requisito.
- Ogni analisi dichiara quanti test esegue.

## 9. Comandi

```bash
pip install -e ".[test]"
pytest tests/                        # suite completa (golden master compreso)
pytest -m golden                     # solo il golden master
ccsu-run-pipeline                    # rigenera results/<study_id>/; il rilevamento MHW gira per primo
ccsu-dashboard                       # equivale a: streamlit run app.py
Rscript scripts/mhw_lag_analysis.R   # facoltativo, solo DLNM
```
