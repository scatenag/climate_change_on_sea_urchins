# 0016 — Passo 1 dell'app: credenziali, download nel processo dell'app, partenza del download

## Contesto

Il passo 1 dei propri dati (MA.2) fa scaricare all'utente la SST giornaliera con le sue credenziali Copernicus,
dentro l'app pubblica. MILESTONE-M1.md §3 fissa le regole (mai su disco, mai in cache, download nel processo
dell'app); questo ADR dice come le rispetta il codice e che cosa resta aperto.

## Decisione

- **Le credenziali** si digitano in un `st.form` con `clear_on_submit=True` (campo password). La callback del pulsante le
  legge, **svuota subito i due campi nello stato di sessione** e le passa come argomenti a `download.fetch_daily_sst` e a
  nient'altro; il download e la vista girano nella callback. Non vanno in un file, nel contesto di sessione, in una cache
  (nessun `st.cache_*`), in un registro, in una variabile d'ambiente, né in un messaggio d'errore. Ogni messaggio passa dalla
  maschera di `download._mask`, rafforzata dopo la rilettura indipendente: prima il segreto più lungo, anche nelle forme
  citate (`repr`, JSON) e codificate (URL); un segreto più corto di 4 caratteri non si può mascherare senza storpiare il
  testo e **fa ritirare l'intero messaggio**. Spazi attorno a nome e password sono tolti. Il login del toolbox, che le salva in
  un file, non è mai usato. Test: assenza da file, stato, contesto, testo mostrato, ambiente, stdout, stderr e log, anche
  quando l'errore le contiene, visti fallire su sabotaggio; e controlli statici sul codice del modulo.
- **Limite noto.** Streamlit tiene il buffer dei valori di widget della richiesta (che il browser ha già svuotato) fino
  all'esecuzione successiva dello script, cioè alla prossima interazione dell'utente: le credenziali stanno in memoria fino
  a lì e mai altrove. Lo stato che il codice controlla è vuoto subito dopo l'invio, e tutto lo stato è pulito un'esecuzione
  dopo; `streamlit.testing` non riproduce lo svuotamento del form (è del browser), quindi il test verifica subito ciò che il
  codice controlla e, dopo un'altra esecuzione, tutto lo stato grezzo. Il download in una callback non mostra un avanzamento
  (Streamlit mostra soltanto «Running»): l'avanzamento è di M1.14.
- **Il download gira nel processo dell'app**, nel thread della sessione, **senza il lock globale** (dura decine di secondi).
  Il lavoro numerico breve della vista (rilevamento delle ondate, 0.4 s su 8644 giorni) prende il lock per la sua durata.
- **Verifica con il toolbox vero** (`scripts/verify_home_clean.py`): dopo `describe` e `subset` con le credenziali come
  argomenti, nessun file nuovo nella cartella home, cache del catalogo compresa (richiesta del proprietario, 16/10).
- **Dove parte il download** (ADR-0014): dal più antico fra l'inizio della baseline e il primo anno delle misure meno
  `max_lag_months` (arrotondato agli anni), entro l'inizio della copertura del prodotto, che si legge dal catalogo.
- **Nessun valore è scelto in silenzio:** latitudine, longitudine e nome del sito sono vuoti; una virgola decimale è
  rifiutata con il motivo; fuori dal dominio, o con una baseline che il prodotto non copre, il rifiuto avviene prima di
  ogni chiamata di download. I campi della baseline sono precompilati con la proposta (visibile e modificabile).

## Il campo «primo anno delle misure» (confermato dal proprietario il 21/10)

Il passo 1 precede il caricamento della risposta (passo 2), quindi l'app non sa da quando partono le misure. Si chiede un
campo facoltativo «primo anno delle misure»; se è vuoto il download parte dalla baseline, e se poi la risposta comincia
prima dei dati scaricati meno i ritardi, i primi mesi non si appaiano (lo dirà il passo 3). **Alternative:** (a) chiedere la
risposta prima del sito, cambiando l'ordine dei passi; (b) scaricare sempre dall'inizio del prodotto (circa 1,6 volte i
giorni, tempo da misurare). **Scelta:** il campo facoltativo, reversibile, nessun effetto su Livorno.

## Baseline minima di dieci anni (confermata dal proprietario il 21/10)

La rilettura ha mostrato che con una baseline di uno o due anni la «soglia» del 90° percentile di ogni giorno è fatta dei
pochi valori stessi (con un solo anno 220 eventi sul 74% dei giorni, su una serie senza tendenza). **Scelta:** meno di 10 anni
è un errore (quanti ne ha la baseline di Livorno), meno di 15 un avviso, 30 raccomandati. **Alternative:** nessun minimo
(rifiutata: risultati insensati senza avviso sufficiente), minimo 30 (rifiutata: Livorno stesso ha 10). Reversibile, nessun
effetto su Livorno. Noto: `mhw_detection` raggruppa per giorno dell'anno esatto, senza la finestra di ±5 giorni
di Hobday, e con 10–15 anni questo si sente di più (codice esistente, golden master). **Risolto per il motore generico con
l'ADR-0019**; resta com'è nella pipeline di Livorno.

## Altre scelte della rilettura

- Il nome del sito ammette lettere, cifre, spazi e `. , ' ( ) -` (fino a 60 caratteri): finisce in messaggi resi come Markdown.
- Coordinate e anno sono decimali e cifre ASCII semplici (niente `1_0.4`, niente cifre Unicode); l'anno è a quattro cifre e non
  oltre l'ultimo anno coperto.
- L'avviso «ingressi cambiati» compare anche quando gli ingressi non sono validi; cambiare scelta azzera i dati del passo e la
  cartella `inputs`; un download fallito non lascia la serie precedente; il file sparito (cartella ripulita) è detto, non è un errore.
- Sono mostrate le scelte fisse: la scatola di 0.1 gradi attorno al sito, la profondità del catalogo, i ritardi fino a cui il
  download parte per tempo (il valore predefinito della specifica). Il conteggio delle ondate è diviso fra anni della baseline e
  anni dopo, con la nota sul trend (ADR-0014; ADR-0000, voce 21), ed è calcolato una volta per download.
- La vista usa `dataset_builder.environment_dataset`, che marca la risposta segnaposto come mancante e non reale.
- `pyproject.toml`: l'extra `acquisition` è fissato a intervalli (copernicusmarine 2.3, xarray, h5netcdf, h5py); il passo di
  verifica col toolbox vero gira ora con Python 3.12 e `requirements.txt`, la configurazione di produzione. Rischio noto: le
  librerie native (HDF5) girano nello stesso processo della dashboard, fuori dal lock; il toolbox si importa solo quando un
  utente scarica.

## Conseguenze

- `requirements.txt` ora installa l'extra `acquisition` (`-e .[acquisition]`): l'app pubblica ha il toolbox (approvato dal
  proprietario il 16/10, D9). L'installazione sul servizio si verifica a mano dopo la fusione.
- Senza l'extra l'app lo dice (e il resto dell'app regge): così in CI i test girano con un client simulato.
- Il tempo del download su Streamlit Cloud si legge nel manifest (`timing.download_seconds`) e nell'app.
