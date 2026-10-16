# 0015 — L'ingresso dell'app generica: funzioni, contesto per sessione, cartelle con scadenza

## Contesto

L'app di sviluppo (`main`) deve offrire due ingressi: il caso dei ricci (la dashboard di oggi) e i propri dati.
`app.py` forza un reimport di `dashboard.py` a ogni esecuzione sotto un lock globale, perché la dashboard ha tutta
l'interfaccia nel corpo del modulo (un semplice `import` da una seconda esecuzione non la riesegue) e perché due
sessioni in contemporanea avevano causato arresti anomali in produzione. In un processo Streamlit i moduli sono
condivisi da tutte le sessioni: lo stato di uno studio non può stare in variabili di modulo (MILESTONE-M1.md §5.1).

## Decisione

- **Moduli nuovi di sole funzioni** (`app_entry.py`, `own_data.py`): importati una volta, come una libreria; l'interfaccia
  gira in `render()`, chiamata a ogni esecuzione. Non soffrono della causa del reimport e non lo richiedono. Un test
  vieta l'interfaccia a livello di modulo.
- **`app.py` instradatore.** Disegna l'ingresso e solo nel ramo del caso dei ricci esegue il blocco di prima (lock,
  rimozione da `sys.modules`, `import dashboard`), identico. Per i propri dati non prende il lock: i passi futuri lo
  prenderanno solo attorno a calcoli numerici brevi, mai per un download o un'attesa (MILESTONE-M1.md, App minima).
  Un test verifica che lock e reimport stiano nel solo ramo.
- **Contesto di sessione** (`SessionContext`: identificativo, cartella di lavoro, etichetta dello studio) in
  `st.session_state`, creato una volta per sessione e passato esplicitamente ai pezzi. Cambiare scelta toglie ogni
  altra chiave di stato (anche quelle che non sono di un widget); il contesto resta.
- **Cartella di sessione con scadenza** (scostamento da §5.1, approvato il 16/10): Streamlit non ha un evento
  affidabile di fine sessione, quindi la cartella (una per sessione, sotto la cartella temporanea di sistema, mai
  condivisa) si rimuove dopo 6 ore senza che la sessione si faccia vedere (un file `.alive` toccato a ogni
  esecuzione), controllato all'avvio di ogni sessione, e con il pulsante *Delete my data*. Si toccano solo i figli
  diretti con nome da identificativo di sessione, mai collegamenti simbolici né altro; la sessione che parte non si
  tocca mai.
- **`st.set_page_config` passa ad `app.py`** (deve essere il primo comando d'interfaccia; l'ingresso ne scrive prima).
  L'icona resta un'emoji (un'icona-immagine bloccava l'app sulla pulizia dei file di Streamlit Cloud).
- Niente `st.cache_*` nei moduli nuovi (non vedranno mai le credenziali: MA.2 aggiunge la guardia sulle chiavi).

## Conseguenze

- Il caso dei ricci si apre dopo un clic in più; la dashboard non cambia (a parte la configurazione di pagina).
- `ccsu-dashboard` non funzionava già prima di MA.1 (lancia `dashboard.py` come script e fallisce con un import relativo);
  non è toccato e va in una issue quando il proprietario lo chiede.
- Su un server con riavvii frequenti le cartelle orfane restano fino al prossimo avvio di una sessione dopo le 6 ore.
