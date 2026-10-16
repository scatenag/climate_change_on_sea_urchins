# 0016 — Passo 1 dell'app: credenziali, download nel processo dell'app, partenza del download

## Contesto

Il passo 1 dei propri dati (MA.2) fa scaricare all'utente la SST giornaliera con le sue credenziali Copernicus,
dentro l'app pubblica. MILESTONE-M1.md §3 fissa le regole (mai su disco, mai in cache, download nel processo
dell'app); questo ADR dice come le rispetta il codice e che cosa resta aperto.

## Decisione

- **Le credenziali** si digitano in un `st.form` con `clear_on_submit=True`, in un campo `type="password"` **senza chiave di
  stato**; passano a `download.fetch_daily_sst` come argomenti e a nient'altro. Non vanno in un file, nel contesto di
  sessione, in una cache (nessun `st.cache_*` nel modulo), in un registro, in una variabile d'ambiente, né in un messaggio
  d'errore (ogni messaggio passa dalla maschera di `download`; un errore imprevisto è riportato con tipo e testo mascherato).
  Il login del toolbox, che le salva in un file, non è mai usato. Le variabili locali che le tengono sono cancellate a
  chiamata finita. Test: assenza da file, stato, contesto, testo mostrato, ambiente, stdout, stderr e log, anche quando
  l'errore le contiene, visti fallire su sabotaggio; e un controllo statico sul codice del modulo.
- **Limite noto.** Streamlit tiene in memoria, nello stato dei widget della sessione, l'ultimo valore di un campo finché
  l'interazione successiva non lo svuota: le credenziali stanno in memoria per la durata di un'esecuzione e mai altrove.
  `streamlit.testing` non riproduce lo svuotamento del form (è del browser), quindi i test escludono il valore che il test
  stesso ha digitato e verificano tutto il resto.
- **Il download gira nel processo dell'app**, nel thread della sessione, **senza il lock globale** (dura decine di secondi).
  Il lavoro numerico breve della vista (rilevamento delle ondate, 0.4 s su 8644 giorni) prende il lock per la sua durata.
- **Verifica con il toolbox vero** (`scripts/verify_home_clean.py`): dopo `describe` e `subset` con le credenziali come
  argomenti, nessun file nuovo nella cartella home, cache del catalogo compresa (richiesta del proprietario, 16/10).
- **Dove parte il download** (ADR-0014): dal più antico fra l'inizio della baseline e il primo anno delle misure meno
  `max_lag_months` (arrotondato agli anni), entro l'inizio della copertura del prodotto, che si legge dal catalogo.
- **Nessun valore è scelto in silenzio:** latitudine, longitudine e nome del sito sono vuoti; una virgola decimale è
  rifiutata con il motivo; fuori dal dominio, o con una baseline che il prodotto non copre, il rifiuto avviene prima di
  ogni chiamata di download. I campi della baseline sono precompilati con la proposta (visibile e modificabile).

## PROVVISORIA: il campo «primo anno delle misure»

Il passo 1 precede il caricamento della risposta (passo 2), quindi l'app non sa da quando partono le misure. Si chiede un
campo facoltativo «primo anno delle misure»; se è vuoto il download parte dalla baseline, e se poi la risposta comincia
prima dei dati scaricati meno i ritardi, i primi mesi non si appaiano (lo dirà il passo 3). **Alternative:** (a) chiedere la
risposta prima del sito, cambiando l'ordine dei passi; (b) scaricare sempre dall'inizio del prodotto (circa 1,6 volte i
giorni, tempo da misurare). **Scelta:** il campo facoltativo, reversibile, nessun effetto su Livorno.

## Conseguenze

- `requirements.txt` ora installa l'extra `acquisition` (`-e .[acquisition]`): l'app pubblica ha il toolbox (approvato dal
  proprietario il 16/10, D9). L'installazione sul servizio si verifica a mano dopo la fusione.
- Senza l'extra l'app lo dice (e il resto dell'app regge): così in CI i test girano con un client simulato.
- Il tempo del download su Streamlit Cloud si legge nel manifest (`timing.download_seconds`) e nell'app.
