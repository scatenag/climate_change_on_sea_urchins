# Milestone M1 · Lo strumento generico da capo a fondo

> **Stato (aggiornato il 16/10/2026):** definizione fissata dal proprietario; decisioni D1–D9 approvate il
> 2/10 con le precisazioni riportate in §5; piano riordinato il 2/10 come fetta verticale (§6), chiusa
> con M1.6 il 16/10 (ritardo noto ritrovato su coordinate mai usate). Il 16/10 il proprietario ha
> riordinato il seguito: **prima di M1.7 la fetta prosegue fino all'interfaccia** (pezzo «App minima»,
> piano approvato il 16/10), e da lì ogni variabile o analisi compare nel comando e nell'app
> nella stessa PR. Nessuna scadenza: la milestone si chiude quando i criteri di §4 sono soddisfatti.

---

## 1. Obiettivo

Una versione generale dello strumento che funziona da capo a fondo, mostrata da una demo guidata
registrabile in un video per i convegni. Alla presentazione si usa il video: l'app non deve
reggere una dimostrazione dal vivo davanti a un pubblico, ma deve funzionare davvero, su dati
veri scaricati in quel momento.

Il manoscritto citato afferma che la pipeline è un modello adattabile da altri programmi di
monitoraggio; oggi `docs/ADAPTING.md` ammette che cambiare sito è facile e cambiare indicatore
no. M1 è ciò che rende vera quella frase: alla fine `ADAPTING.md` va riscritto.

Nella strada V1–V5 (`METODO_E_FASI.md`) M1 sta dentro **V2**, di cui porta la genericità fino a
chi non conosce il codice, e anticipa una parte di **V3.1** (ogni analisi dichiara i propri
requisiti) e di **V3.3** (uso senza conoscere il codice). Le celle spostate e la tabella di
confronto delle finestre restano in V2.2; il confronto fra casi e la disciplina inferenziale in
V3.2.

## 2. Struttura

- **Una sola app generica**: l'attuale app di sviluppo, da `main`
  (<https://climate-response-explorer.streamlit.app>). L'app del lavoro
  (<https://climate-change-on-sea-urchins.streamlit.app>, branch `paper/mpb-2026`) resta separata
  e invariata (ADR-0009).
- **All'ingresso** l'utente sceglie fra *il caso dei ricci* e *i propri dati*.
- **Il caso dei ricci.** Livorno è uno studio qualunque dello strumento: calcolato in anticipo
  dal job quotidiano, apribile senza credenziali. Mostra la dashboard e, in sola lettura, i passi
  del percorso compilati con i valori di Livorno, così chi guarda vede che cosa dovrà fornire per
  il proprio caso. Porta l'etichetta di caso del lavoro elaborato dallo strumento generico, con il
  rimando all'app citata nel lavoro per la versione pubblicata.
- **I propri dati**, in tre passi:
  1. sito, periodo e credenziali Copernicus dell'utente; coordinate fuori dalla copertura del
     catalogo rifiutate qui, con il motivo; download reale dei dati ambientali e loro
     visualizzazione, compresi gli eventi di ondata di calore rilevati;
  2. serie di risposta guidata: un modello di CSV scaricabile e domande ricavate da quanto
     abbiamo imparato sul foglio dei ricci (`note-dati-sorgente.md`): risoluzione delle date,
     valori per singola prova o già aggregati, intervalli di confidenza come estremi, quale
     colonna è la risposta, unità e verso della risposta quando gli effetti peggiorano; la
     validazione restituisce errori comprensibili;
  3. dashboard con le analisi applicabili: quelle con requisiti non soddisfatti restano visibili
     ma spente, con il motivo; le analisi lente partono su richiesta, con l'avanzamento visibile.

## 3. Vincoli

### Credenziali
- Sono quelle dell'utente: restano nella sessione e passano direttamente alle chiamate di
  download, senza mai finire su disco. L'app spiega come le usa, e nessun messaggio d'errore le
  mostra.
- **Il login del toolbox Copernicus salva le credenziali in un file**: verificato sul codice di
  `copernicusmarine` 2.3.0, `login()` scrive `$HOME/.copernicusmarine/.copernicusmarine-credentials`
  (o nella cartella di `COPERNICUSMARINE_CREDENTIALS_DIRECTORY`). Su un server condiviso le
  lascerebbe all'utente successivo: **non si usa mai**.
- `subset()` accetta `username` e `password` come argomenti e in quel caso non legge né scrive
  file. Se uno dei due manca, il toolbox ripiega nell'ordine su variabili d'ambiente, file di
  credenziali predefiniti (`~/.copernicusmarine/…`, `~/motuclient/motuclient-python.ini`,
  `~/.netrc`) e infine sul prompt interattivo. Quindi: entrambi i valori obbligatori e non vuoti
  prima della chiamata; mai variabili d'ambiente (sono del processo, condivise fra le sessioni);
  mai nella cache di Streamlit (`st.cache_*` usa gli argomenti come chiave); mai negli argomenti
  o nell'ambiente di un sottoprocesso; ogni messaggio d'errore passa da un filtro che le
  maschera.
- **Dove avviene il download.** Nel processo dell'app, che chiama `subset()` con le credenziali
  della sessione. Il processo che esegue lo studio (`ccsu-run-study`) riceve solo file (specifica,
  CSV, dati scaricati) e non vede mai le credenziali, né come argomenti né nell'ambiente.

### Stato e ripresa
- A ogni passo l'utente può scaricare un **pacchetto** con `study.yaml`, CSV della risposta, dati
  ambientali scaricati e un manifest di provenienza (prodotti, versioni, data del download).
- Ricaricare il pacchetto riprende il lavoro con gli stessi dati e gli stessi risultati.
  Ricaricare il solo `study.yaml` e riscaricare l'ambiente aggiorna lo studio: i risultati
  possono cambiare.
- `study.yaml` porta una versione del formato: un file di una versione precedente si carica,
  oppure si rifiuta con un messaggio che ne indica la versione.

### Input non fidati
- Un file caricato su un'app pubblica è un input non fidato: parsing YAML sicuro, e l'app ignora
  ogni campo diretto al filesystem del server o a sorgenti esterne, come `data_dir` o un foglio
  remoto. I dati entrano solo come file caricati. Le credenziali non entrano mai nel pacchetto.

### Video
- Prima Livorno ricostruito da zero con il percorso dei propri dati (coordinate e CSV delle
  EC50) fino alla dashboard; poi un secondo sito con una serie fittizia, generata dalla
  temperatura reale scaricata per quel sito con un ritardo noto, da ritrovare nella dashboard.

## 4. Criteri di accettazione

| # | Criterio | Come si verifica |
|---|---|---|
| 1 | Livorno passa dal motore generico con deriva zero nel golden master, e la dashboard generica lo mostra senza codice specifico per Livorno | golden master sul comando unico; guardia che rende la dashboard con l'etichetta RSPTEST e fallisce su qualunque letterale del caso |
| 2 | Partendo da zero, su coordinate mai usate, si scaricano dati veri, si carica una serie e si arriva alla dashboard; il ritardo noto della serie fittizia compare nei risultati | test automatico del motore su fixture (ritardo ritrovato, e nessun ritardo dove non ce n'è); prova manuale con credenziali vere |
| 3 | Le analisi con requisiti non soddisfatti compaiono spente, con il motivo | test su uno studio sintetico senza pH, senza dati per prova, senza `split_date` |
| 4 | Un pacchetto scaricato e ricaricato riproduce gli stessi risultati | test di andata e ritorno: stessi file di risultato, byte per byte, sulla stessa macchina, esclusi i campi di provenienza che cambiano a ogni esecuzione; i file confrontati solo nella struttura (#9) si confrontano come nel golden master |
| 5 | Il caso dei ricci si apre senza credenziali e senza conoscere lo strumento | test dell'app (`streamlit.testing`) senza credenziali né variabili d'ambiente; prova con una persona esterna al progetto |

## 5. Decisioni (approvate il 2/10/2026)

| # | Domanda | Decisione |
|---|---|---|
| D1 | Ambito geografico del catalogo | Solo Mediterraneo (i prodotti MEDSEA di Livorno). Coordinate fuori copertura rifiutate al passo 1, con il motivo |
| D2 | Stato globale dello studio (`common.py` lo carica all'import da `CCSU_STUDY`) | Ogni esecuzione di uno studio in un **processo separato** (comando unico). Togliere lo stato di modulo dal nucleo resta per dopo. La dashboard non lo usa: vedi §5.1 |
| D3 | Con quale analisi si ritrova il ritardo noto | CCF generica variabile ambientale → risposta, sulle differenze prime, ritardi 0–12, correzione BH-FDR, riusando `mhw_analysis.compute_ccf`. Serie a verità nota con i caratteri della serie reale (pezzo M1.6) |
| D4 | Cosa contiene il pacchetto | Gli input, **compresi i dati ambientali scaricati**, e il manifest con l'impronta dei risultati; al caricamento si ricalcola e l'app dice se i risultati coincidono |
| D5 | `split_date` per uno studio nuovo | Facoltativo: senza, le analisi pre/post sono spente con il motivo |
| D6 | DLNM (R) nell'app generica | Spento con il motivo «R non disponibile su questo server» |
| D7 | Speciazione del rame su uno studio utente | Requisito dichiarato nella specifica (campo per il contaminante dell'endpoint); senza, spenta con il motivo |
| D8 | Imputazione della risposta | Dichiarata nella specifica; **assente per default**. Lo `study.yaml` di Livorno dichiara la sua (media mobile centrata 12 mesi, `min_periods` 3, doppia passata), con deriva zero |
| D9 | Dove sta il download | Un modulo del pacchetto (extra `acquisition`), perché l'app lo chiama; `scripts/` resta per il job di Livorno |

### 5.1 Due utenti, due studi, un solo processo Streamlit

Le costanti di `common.py` (`DATA`, `RESULTS`, `SPLIT_DATE`, `WINDOWS`) sono fissate all'import, e
in un processo Streamlit i moduli sono condivisi da tutte le sessioni: valgono per uno studio solo.
Per questo la dashboard generica non le legge mai.

- **Calcolo.** Ogni studio gira in un processo separato (`ccsu-run-study`), che carica il proprio
  `study.yaml`: lì le costanti sono quelle dello studio, e il processo finisce con il calcolo.
- **Visualizzazione.** Ogni sessione tiene in `st.session_state` il proprio contesto di studio:
  la specifica letta dal suo `study.yaml`, la cartella dei dati e quella dei risultati (una
  cartella di lavoro per sessione, mai condivisa e cancellata a fine sessione; per Livorno i
  risultati precalcolati, in sola lettura). Il contesto si passa esplicitamente a ogni pannello, e
  i pannelli leggono i file con funzioni che ricevono i percorsi come argomenti (parametri nuovi
  dei lettori di `common.py`, con i valori attuali come default per la pipeline).
- **Reimport forzato e lock globale di `app.py`.** Per `app.py` (`docs/COME_SI_LAVORA.md` §6) il
  pacchetto `dashboard` è reimportato a ogni esecuzione e un lock globale serializza le sessioni.
  Conseguenze per la dashboard generica: (1) nessuno stato di studio sta in variabili di modulo, che a
  ogni esecuzione ripartono da zero; sta in `st.session_state`; (2) le cache (`st.cache_*`) hanno
  come chiave il percorso e l'impronta dei file dello studio, mai il solo nome; (3) il lock fa
  attendere le altre sessioni durante un'esecuzione, quindi nessuna analisi lenta gira sotto il
  lock: parte in un processo separato e la sessione ne legge l'avanzamento; (4) il test delle due
  sessioni parte da `app.py`, il punto d'ingresso vero. M1.13 verifica se la causa originale del
  reimport e del lock è ancora presente prima di toglierli.
- **Verifica.** Un test statico vieta al codice della dashboard generica di usare `common.DATA`,
  `common.RESULTS`, `common.SPLIT_DATE`, `common.WINDOWS` e `config`. Un test dell'app apre nello
  stesso processo due sessioni con studi diversi (etichette EC50 e RSPTEST) e controlla che
  ciascuna mostri il proprio.

## 6. Piano

Ogni pezzo è una PR con il proprio criterio di verifica; i pezzi marcati «una PR per…» sono più
PR. Prima una fetta verticale stretta che attraversa la catena da riga di comando (chiusa); poi la
stessa fetta prosegue fino all'interfaccia (**App minima**, sigle MA.1–MA.5, che lasciano invariata la
numerazione M1.7–M1.17 per non spezzare i rimandi dei documenti e degli ADR); poi si allarga una
variabile e un'analisi alla volta, **e ogni variabile o analisi compare nel comando e nell'app nella
stessa PR**, mai l'una senza l'altra.

### Fetta verticale (riga di comando, senza interfaccia)

**Criterio di uscita:** su coordinate mai usate, con dati scaricati davvero, il comando unico
ritrova il ritardo noto della serie a verità nota.

| Pezzo | Contenuto | Criterio di verifica |
|---|---|---|
| **M1.1 Specifica formato 2 e catalogo con la sola SST** | `format_version`; ambiente per id di catalogo; sorgente `csv` della risposta; `split_date` facoltativo (D5); imputazione dichiarata, assente per default (D8); campo del contaminante (D7). Catalogo dichiarativo con la sola SST giornaliera, dominio Mediterraneo (D1) | Lo `study.yaml` di Livorno (formato 1) si carica e dà deriva zero; una versione sconosciuta è rifiutata nominandola; la voce SST riproduce dataset, variabile e profondità di `scripts/fetch_copernicus_daily.py`; nessun campo eseguibile |
| **M1.2 Download della SST per coordinate nuove** | Dal catalogo; fine dalla copertura del prodotto; coordinate fuori dominio rifiutate con il motivo; controllo della cella di mare, **con un messaggio che distingue una cella di terra da un punto di mare fuori dal prodotto** (per esempio il Mar Nero, che sta dentro il riquadro del dominio ma non è coperto dal prodotto del Mediterraneo: il riquadro di `catalog.py` non basta a saperlo); manifest di provenienza; credenziali solo come argomenti (§3) | In CI, con il toolbox simulato: credenziali passate solo come argomenti, `login` mai chiamato, ambiente e `$HOME` intatti, errore con la password mascherato (visto fallire su sabotaggio). A mano, con credenziali: sulle coordinate di Livorno serie uguale a `data/sst_daily.csv` nei giorni comuni; tempo misurato |
| **M1.3 Sorgente CSV della risposta** | Valori per prova o aggregati, risoluzione delle date dichiarata, intervalli come estremi, unità e verso; errori leggibili con il numero di riga. **Da decidere in questo pezzo, dichiarati nella specifica o riconosciuti dal lettore: separatore di campo, separatore decimale e formato della data del CSV** (un CSV europeo con `;` e la virgola decimale, o `31/01/2020`, non deve essere letto male in silenzio) | I dati per prova di Livorno, esportati nel formato e riletti, danno la stessa serie mensile; un test per ogni classe di errore |
| **M1.4 Costruttore del dataset (SST e risposta)** | Aggregazione della risposta, imputazione come dichiarata, rilevamento MHW, temperatura mensile; con le funzioni di `common.py` (una sola implementazione). Per la fetta la temperatura mensile è la media mensile della SST giornaliera (confermato il 2/10), sostituita dalla variabile mensile del catalogo con M1.8. **Mesi incompleti (approvato il 3/10): un mese con anche un solo giorno di SST mancante è mancante**, senza un campo per una soglia di copertura finché non arriva una fonte con buchi veri. Motivo: la SST di rianalisi non ha buchi interni, quindi la regola scatta sui mesi ai bordi, dove una media parziale è distorta dal ciclo stagionale e cadrebbe sui mesi che i ritardi usano di più. I mesi esclusi sono riportati nella copertura dei risultati | Dalla SST e dai dati per prova della fixture: catalogo MHW identico a quello della fixture; Livorno, con la sua imputazione dichiarata, identico a `data_ec50_ci.csv` della fixture. La regola dei mesi incompleti scatta sia su un mese finale incompleto sia su un mese iniziale incompleto (periodo che comincia a metà mese), e i mesi esclusi compaiono nella copertura riportata nei risultati |
| **M1.5 `ccsu-run-study` con la sola CCF** | Comando unico: costruzione, CCF variabile → risposta (D3) con il numero di test dichiarato, risultati in `results/<study_id>/` con la provenienza; un processo per studio; avanzamento leggibile da un altro processo | Uno studio sintetico gira da capo a fondo in CI, senza rete |
| **M1.6 Serie a verità nota (uscita della fetta)** | Generatore con seme esposto (`known_truth.py`, ADR-0013). Risposta generata dall'**anomalia mensile della SST** (decisione del 16/10, ADR-0012), che scende quando la temperatura sale, con ritardo *k* né 0 né multiplo di 12 (3 mesi), segnale al 50% della varianza, da 1 a 6 prove per mese da aggregare e mesi mancanti in sequenze come nel foglio dei ricci (42% dei mesi, sequenze di 2.5 mesi in media, misurate il 16/10), rumore AR(1). La dichiarazione sta in `response.manifest.json` e `run.json` la riporta | Il massimo di \|r\| della CCF sulle differenze prime cade su *k* con il segno atteso e BH<0.05 al ritardo (`known_truth.verify`); gli altri ritardi che superano FDR sono riportati. Con una serie indipendente (quota di segnale 0) questo non succede. Test su fixture congelata (8 semi con segnale, 8 senza) visti fallire su sabotaggio. A mano, con credenziali (`verify_slice.yml`, etichetta `verifica-fetta`): il criterio su due punti mai usati, scaricati dal servizio vero |

### App minima (subito dopo la fetta; il piano è da approvare prima di scrivere codice)

**Obiettivo.** Nell'app di sviluppo (`main`) una scelta d'ingresso fra *il caso dei ricci*, che per ora porta
alla dashboard di Livorno esistente, e *i propri dati*. Per i propri dati, i tre passi sopra ciò che esiste già
dalla riga di comando: (1) sito, periodo e credenziali con il download della SST; (2) caricamento guidato del CSV
(`suggest_format`, le domande della specifica, gli errori di M1.3) oppure una serie di prova con ritardo noto;
(3) esecuzione di `ccsu-run-study` con l'avanzamento e la CCF mostrata con il verdetto e i limiti dichiarati.
Valgono le regole di §3 (credenziali, download nel processo dell'app, calcolo in un processo separato che non vede
le credenziali) e il contesto per sessione di §5.1. Non c'è pacchetto da scaricare né da ricaricare (M1.10), né
analisi oltre la CCF (M1.7–M1.9): chi chiude la scheda perde il lavoro, e l'app lo dice.

**Come convive con `app.py`, il suo reimport forzato e il lock globale.**

- *Perché esistono.* Il reimport a ogni esecuzione esiste perché `dashboard.py` ha tutta l'interfaccia nel
  corpo del modulo: da una seconda esecuzione un semplice `import` non la rieseguirebbe. Il lock serializza il
  reimport e le chiamate numeriche pesanti (arresti anomali in produzione, cause non chiuse del tutto).
- *I moduli nuovi sono funzioni.* L'interfaccia dei propri dati sta in `render()` di moduli senza codice
  d'interfaccia a livello di modulo (nuovi, nel pacchetto): importati una volta, come qualunque libreria, e
  chiamati a ogni esecuzione. Non soffrono della causa del reimport e non lo richiedono; nessuno stato di studio
  sta in variabili di modulo ma in `st.session_state` (§5.1).
- *`app.py` diventa l'instradatore.* A ogni esecuzione chiama `app_entry.render()`: mostra la scelta e, per i
  ricci, esegue **identico il blocco di oggi** (lock, rimozione dal `sys.modules`, `import dashboard`) e solo in
  quel ramo. Per i propri dati chiama le funzioni dei tre passi. L'indirizzo d'ingresso per Streamlit Cloud
  resta `app.py`. (`ccsu-dashboard` non funzionava già prima: lancia `dashboard.py` come script e fallisce con un import
  relativo; difetto noto, non toccato, segnalato in STATO.)
- *Unica modifica a `dashboard.py`:* `st.set_page_config` passa ad `app.py` (deve essere il primo comando
  d'interfaccia e l'ingresso ora ne scrive prima; fatto in MA.1). Il branch del lavoro ha la sua copia e non è toccato.
- *Il lock per i propri dati.* Si prende **solo attorno ai calcoli numerici brevi in processo** (lettura e
  parsing del CSV, rilevamento delle ondate per la visualizzazione) e mai durante il download (decine di
  secondi) né durante l'attesa del calcolo, o ogni altra sessione si fermerebbe. Il calcolo vero gira in un
  processo separato (`ccsu-run-study`), fuori dal lock.
- *Cartella di sessione.* Una per sessione nella cartella temporanea, mai condivisa. Streamlit non offre un
  evento affidabile di fine sessione: **scostamento da §5.1** («cancellata a fine sessione») da approvare: la
  cartella si cancella a ogni avvio di sessione se più vecchia di 6 ore e non in uso, e con un pulsante
  «Cancella i miei dati».
- *Limite di esecuzioni.* Al massimo 2 processi di calcolo contemporanei sul server; gli altri vedono un
  messaggio di attesa (memoria del servizio condiviso).
- *Le credenziali.* Form che si svuota all'invio; non restano in `st.session_state`, mai in `st.cache_*`,
  mai in ambiente o argomenti di un sottoprocesso; ogni errore passa dal filtro che le maschera.
- *Streamlit.* L'avanzamento si legge con `st.fragment(run_every=…)`, disponibile dalla 1.37: il vincolo
  `streamlit>=1.35` di `pyproject.toml` sale in MA.4 se serve (si cita nella PR).

| Pezzo | Contenuto | Criterio di verifica |
|---|---|---|
| **MA.1 Ingresso, contesto di sessione, instradamento** | `app.py` instradatore e `app_entry.py` (solo funzioni); scelta ricci/propri dati; contesto di studio per sessione in `st.session_state` passato esplicitamente; cartella di sessione con pulizia; ramo dei ricci identico a oggi; i tre passi presenti ma vuoti | `AppTest` che parte da `app.py` (l'ingresso vero), senza credenziali né variabili d'ambiente: la scelta compare; «ricci» rende la dashboard di oggi e i test esistenti e il golden master restano verdi; due sessioni nello stesso processo con contesti diversi mostrano ciascuna il proprio (etichette EC50 e RSPTEST); cambiare scelta e tornare non lascia stato dell'altra. Test statico che vieta ai moduli nuovi `common.DATA/RESULTS/SPLIT_DATE/WINDOWS`, `config` e `st.cache_*` su funzioni con credenziali. La pulizia cancella le cartelle vecchie e mai quelle in uso. Tutti visti fallire su sabotaggio. **Rischio:** se `AppTest` non regge la dashboard di oggi in tempi ragionevoli (legge il foglio, ricade sull'istantanea), la resa dei ricci si prova con il ramo simulato e una verifica a mano, dichiarato nella PR |
| **MA.2 Passo 1: sito, periodo, credenziali, download** | Latitudine, longitudine (numeri con anteprima su mappa; la scelta sulla mappa è in M1.14), nome, baseline climatologica (§6.1) e inizio del download derivato: il più antico fra l'inizio della baseline e il primo mese della risposta meno `max_lag_months`; form per le credenziali; `download.fetch_daily_sst` nel processo dell'app con le credenziali come soli argomenti; i rifiuti del catalogo (terra, entroterra o mare non coperto, fuori dominio) mostrati con il loro motivo; SST giornaliera e eventi di ondata di calore (con il lock breve); durata del download mostrata e scritta nel manifest | `AppTest` con il client Copernicus simulato di M1.2: le credenziali sono assenti dai file della cartella di sessione, da `st.session_state` dopo l'invio, dai messaggi anche quando l'errore le contiene, dalle chiavi delle cache, dall'ambiente del processo, dal manifest (ciascuna vista fallire su sabotaggio); `login` mai chiamato; ogni rifiuto mostra il suo messaggio; fuori dominio rifiutato prima di ogni chiamata. **A mano, con le credenziali del proprietario (mai nelle mie sessioni): tempo del download misurato su Streamlit Cloud**, riportato nel manifest e nella PR al posto della stima di §8 |
| **MA.3 Passo 2: la risposta** | CSV caricato (limiti di M1.3) o *serie di prova con ritardo noto* generata da `known_truth` sulla SST appena scaricata (seme, ritardo e quota di segnale mostrati); `suggest_format` dice che cosa il file sembra e l'utente conferma separatore, decimale e formato della data (se la data è ambigua non si procede finché non sceglie); le domande della specifica (risoluzione, per prova o aggregato, colonne di data, valore ed estremi, unità, verso in cui peggiora, etichetta, natura, imputazione assente salvo dichiarata); gli errori di M1.3 con riga e colonna in chiaro; anteprima della serie mensile; `study.yaml` formato 2 prodotto e scaricabile | `AppTest`: una prova per ogni classe di errore di M1.3, con la riga; un CSV europeo (`;`, virgola decimale, `31/01/2020`) letto bene dopo la conferma e rifiutato con le convenzioni sbagliate; la data ambigua blocca; i dati per prova di Livorno esportati danno la stessa serie mensile del caso precalcolato; lo `study.yaml` prodotto passa `load_study` e non ha campi diretti al filesystem; la serie di prova è, byte per byte, quella di `known_truth.make_study` con lo stesso seme |
| **MA.4 Passo 3: esecuzione e risultati** | `ccsu-run-study` in un processo separato con ambiente costruito da zero (nessuna variabile Copernicus, nessuna credenziale fra gli argomenti: riceve solo file); avanzamento da `progress.json` con aggiornamento periodico, fuori dal lock; al più 2 esecuzioni contemporanee; «Interrompi»; fallimento mostrato con passo e motivo. Risultati dalla cartella di sessione: CCF per ritardo (r, p, p BH, coppie), test dichiarati, eseguiti e non eseguiti, copertura, parametri (driver «SST anomaly» e baseline), **limiti dichiarati** (cerca solo l'associazione non stagionale; coppie di mesi consecutivi reali; autocorrelazione delle differenze, ADR-0000 voce 20) e **verdetto**: per una serie di prova `known_truth.verify` (ritrovato o no, massimo di \|r\|, altri ritardi sopra BH), per dati reali una frase dai valori (quali ritardi superano BH e con quale segno rispetto al verso dichiarato; «nessuno» se nessuno), mai una conclusione fissa | `AppTest` con processo vero su una fixture sintetica: serie a verità nota → l'interfaccia dice «ritrovato» al ritardo giusto (uguale a `known_truth.verify` sui file); serie indipendente → «non ritrovato»; una variabile sentinella nell'ambiente dell'app non compare nell'ambiente del processo figlio (vista fallire su sabotaggio); mentre il calcolo gira un'altra sessione risponde; il fallimento di un passo è mostrato con il motivo; la terza esecuzione contemporanea attende |
| **MA.5 Chiusura della tappa: prova a mano e misura** | Il percorso intero sull'app di sviluppo pubblicata (dopo aver verificato che serva il codice nuovo), con le credenziali del proprietario su coordinate mai usate e la serie di prova: scarica, genera, esegue, ritrova il ritardo noto nell'interfaccia. Misure sul servizio (download, calcolo, memoria); ciò che non regge in issue; nessuna funzione nuova. È la tappa con demo | Criterio 2 di §4 come prova a mano registrata nella PR (tempi, verdetto, messaggi); evidenze e difetti riportati |

**Approvato il 16/10:** il piano in 5 PR, la baseline A, la cartella di sessione cancellata dopo 6 ore di
inattività e con il pulsante (scostamento da §5.1), le credenziali in un form che si svuota all'invio, le stime.

**Stima dell'app minima:** 5 PR, **6–8 sessioni** (§8).

**Da decidere dopo MA.5: quali analisi di Livorno entrano nel motore generico.** Se l'app generica mostrasse
Livorno con le sole analisi generiche e rimandasse all'app del lavoro per le altre, M1.9 si ridurrebbe e il
criterio 1 andrebbe riscritto. Quattro dei quindici moduli di analisi dipendono da un caso, non da requisiti che
ogni studio può avere: la speciazione del rame (`cu_speciation`, il contaminante è il rame), l'eredità termica a
24 °C (`thermal_legacy`, una soglia di *P. lividus*), il controllo negativo (`negative_control`, richiede le
colonne dei controlli del foglio) e il forecast con gli scenari (`forecast`: regressori pH, temperatura e ondate; addestrato su una finestra
fissata nel codice dal 2016, ADR-0000). Sono circa un terzo delle righe delle analisi e circa un quinto dei file del golden master (misura
approssimata del 16/10). Differenza stimata:

| | Tutte le analisi nel motore (oggi) | Solo le generiche |
|---|---|---|
| M1.9 (base, sessioni) | circa 10–14 | circa 7–10 |
| Riscrittura del criterio 1 e del job | — | 0,5–1 |
| **Differenza in sessioni** | | 3–4 di base in meno, meno 0,5–1 per riscrivere il criterio e il job; × 1,3: **circa 3–5 in meno** (resto da 36–48 a circa 32–44) |

Non si risparmia tutto: la pipeline di Livorno con lo stato di modulo (`ccsu-run-pipeline`) non si può ritirare
(restano le quattro analisi e la separazione di `common`, ADR-0000 voce 19, non si fa), i due percorsi vanno
mantenuti, e `main` non mostrerebbe più per Livorno la speciazione, l'eredità termica, il controllo negativo
e il forecast, visibili solo nell'app del lavoro. Il criterio 1 diventerebbe: «Livorno passa dal motore generico
per le analisi generiche con deriva zero (salvo la CCF sulle differenze prime, M1.9), i file delle analisi
specifiche restano prodotti dalla pipeline di Livorno e coperti dal golden master, e la dashboard generica mostra
Livorno senza codice specifico per Livorno, con il rimando all'app del lavoro per il resto». La decisione è del
proprietario, dopo MA.5, con le misure dell'app minima.

### 6.1 Baseline climatologica per uno studio nuovo (da decidere prima di MA.2)

Il costruttore (M1.4) vuole la baseline coperta per intero dalla SST, quindi il download parte dal suo inizio
(non dall'inizio della serie di risposta). Se la risposta comincia prima della baseline, il download parte
dal primo mese della risposta meno `max_lag_months` (i ritardi usano la temperatura di prima); se comincia dopo,
la baseline resta e la SST precedente è solo il riferimento. Tre opzioni:

| | Opzione | Per | Contro |
|---|---|---|---|
| **A** | **1991–2020** per tutti (la norma climatologica a 30 anni dell'OMM), modificabile entro gli anni coperti dal prodotto | Trenta anni: la letteratura sulle ondate di calore raccomanda 30 anni o più (Hobday et al. 2016; *da ricontrollare sul testo prima di citarlo in un ADR*). Non dipende dalla serie di risposta, quindi studi diversi hanno la stessa definizione di «normale» (serve al confronto fra casi, V3.2). Indipendente da quanto è corta la serie | Scarica più giorni (circa 14 000 dal 1991 invece di circa 8 700 dal 2003: stima del 1,6 sul tempo dei dati, da misurare in MA.2); le ondate recenti sono più frequenti perché la baseline è più fredda dei decenni successivi (è la definizione, ma va detto) |
| **B** | **I primi 10 anni completi della serie di risposta** (come Livorno: 2003–2012) | Scarico minimo; la stessa regola dello studio di Livorno | Dieci anni sono pochi per una soglia al 90º percentile (rumorosa); la baseline coincide con l'inizio dello studio, quindi non confrontabile fra studi; una serie di risposta di 12–15 anni lascia poco da analizzare dopo la baseline |
| **C** | **I primi 30 anni del prodotto** (dal 1987, da leggere dal catalogo come si fa per la fine) | Trent'anni, indipendente dalla serie, una sola data per tutti | Termina nel 2016: gli anni successivi, quelli della maggior parte delle serie, sono tutti sopra baseline; la data d'inizio del prodotto dipende dal catalogo e può cambiare |

**Decisione del proprietario (16/10): A**, 1991–2020 come valore proposto, modificabile, con due avvisi (baseline
più corta di 15 anni: soglia rumorosa; baseline che non finisce prima dell'ultimo anno coperto: nessun mese
«dopo»). Motivo e citazione di Hobday et al. (2016) in ADR-0014. **L'inizio della copertura del prodotto si
legge dal catalogo come la fine, senza date scritte nel codice:** se non copre il 1991 l'app lo dice e propone il
primo trentennio coperto. Livorno resta con la baseline che dichiara (2003–2012, golden master): una scelta
diversa, scritta nel suo studio. Se la baseline debba essere fissa o mobile per le metriche delle ondate è
aperto (ADR-0000, voce 21), da decidere prima di M1.9.

### Allargamento (una variabile e un'analisi alla volta; ogni pezzo comando e app insieme)

| Pezzo | Contenuto | Criterio di verifica |
|---|---|---|
| **M1.7 Requisiti dichiarati** | Ogni analisi dichiara cosa le serve (variabili, SST giornaliera, dati per prova, controlli, `split_date`, mesi minimi, contaminante, R); `analyses.json` con stato ed eventuale motivo. **Nell'app, nella stessa PR:** la schermata dei risultati di MA.4 legge `analyses.json` e mostra le analisi non applicabili spente, con il motivo | Uno studio senza pH: speciazione e forecast spenti con il motivo, nessun loro file scritto (criterio 3, motore); `AppTest` sullo stesso studio: le stesse analisi spente con lo stesso motivo (criterio 3, interfaccia) |
| **M1.8 Variabili mensili**, una PR per variabile | Temperatura 0–10 m, salinità, O₂, pH, CO₂ con la conversione Pa→µatm nominata nel catalogo. **Nell'app, nella stessa PR:** la variabile si sceglie nel passo 1, si scarica nello stesso invio e compare nei risultati | Voce di catalogo uguale a quella usata oggi per Livorno; a mano, sulle coordinate di Livorno, serie uguale a `data/env_copernicus.csv` nei mesi comuni; `AppTest` con download simulato della variabile |
| **M1.9 Analisi nel comando unico**, una PR per analisi o famiglia | Le analisi della pipeline, ciascuna con i propri requisiti, fino a Livorno completo; alla fine il job di Livorno usa `ccsu-run-study` (PR su `update_ec50.yml`). **Nell'app, nella stessa PR di ogni analisi:** il suo pannello, letto da `analyses.json` e dai file dell'analisi, con i testi dai valori (assorbe M1.12) | Deriva zero sul golden master a ogni PR; il pannello mostra gli stessi valori dei file e la guardia RSPTEST sulla vista generica non trova letterali del caso (criteri 1 e 3); alla fine il golden master intero eseguito dal comando unico (criterio 1, motore). **Eccezione dichiarata, decisa dal proprietario il 16/10:** quando Livorno passa da `ccsu-run-study`, la CCF sulle differenze prime usa le coppie di mesi consecutivi entrambi reali (ADR-0012), quindi `ccf_results_diff.csv` cambia. Il riferimento del golden master di quel file si aggiorna in modo dichiarato in quella PR, sui numeri, con l'approvazione del proprietario; il resto del golden master resta a deriva zero. L'app del lavoro (`paper/mpb-2026`) non cambia: dice che nessun ritardo è significativo, vero con entrambe le costruzioni (misura del 15/10, `drafts/dossier-tabella-S3-revisione/DOSSIER.md` §6) |
| **M1.10 Pacchetto e caricamento non fidato** | Esportazione e caricamento (D4); YAML sicuro; `data_dir` e sorgenti remote ignorati con avviso. **Comando e app nella stessa PR:** scarico e caricamento del pacchetto in ogni passo | Andata e ritorno con risultati identici (criterio 4): il confronto byte per byte esclude i campi di provenienza che cambiano a ogni esecuzione (data e ora, durata, versione del codice), e l'impronta del pacchetto tratta i file che il golden master confronta solo nella struttura (`STRUCTURAL_ONLY_FILES`, #9) allo stesso modo, cioè sulla struttura, così un pacchetto ricaricato su un'altra macchina non risulta diverso senza esserlo; archivi ostili rifiutati (percorsi fuori cartella, file inattesi, dimensioni); nessuna credenziale nel pacchetto (visto fallire su sabotaggio); stesso percorso dall'interfaccia |

### Interfaccia

| Pezzo | Contenuto | Criterio di verifica |
|---|---|---|
| **M1.11 Il caso dei ricci nell'app generica** | **Legato al passaggio di Livorno nel motore (fine di M1.9).** Livorno precalcolato senza credenziali, con l'etichetta e il rimando all'app del lavoro, e il percorso in sola lettura compilato con i suoi valori; sostituisce il ramo «ricci → dashboard esistente» di MA.1 | Test dell'app senza credenziali né variabili d'ambiente (criterio 5); due sessioni con studi diversi nello stesso processo; guardia RSPTEST (criterio 1, dashboard) |
| **M1.12 Pannelli per analisi** | **Assorbito in M1.7–M1.9:** ogni pannello nasce nella PR della sua analisi | — |
| **M1.13 Via la dashboard di Livorno da `main`** | Rimozione di `dashboard.py` e, se non serve più, del meccanismo di import di `app.py`; l'app del lavoro vive sul suo branch | Nessun riferimento rimasto; l'app di sviluppo apre la vista generica |
| **M1.14 Propri dati, passo 1: completamenti** | Quello che MA.2 lascia: sito scelto su mappa, eventi di ondata di calore in dettaglio. La baseline è decisa prima di MA.2 (§6.1) | Test dell'app con download simulato; la scelta sulla mappa riempie le coordinate |
| **M1.15 Propri dati, passo 2: completamenti** | Quello che MA.3 lascia: modello di CSV scaricabile, intervalli di confidenza come estremi e controlli negativi nelle domande | Errori in chiaro per ogni classe; Livorno ricostruito dal CSV uguale al caso precalcolato |
| **M1.16 Propri dati, passo 3: completamenti** | Analisi lente su richiesta (già in un processo separato da MA.4) con avanzamento per ognuna; ricarica del pacchetto riprende il lavoro | Criteri 3 e 4 dall'interfaccia |
| **M1.17 Prova generale e copione** | Percorso del video eseguito per intero con credenziali vere su coordinate mai usate; copione in `docs/`; `docs/ADAPTING.md` riscritto | I cinque criteri spuntati, con le evidenze nella PR |

## 7. La dashboard: riscriverla o togliere il caso dall'attuale

**A. Riscritta, guidata da risultati e specifica.** Un pannello per analisi: legge lo stato da
`analyses.json` e i valori dai file dell'analisi, con il contesto della sessione (§5.1); i testi
nascono dai valori. Costo stimato **6–8 sessioni** (M1.11–M1.13): ingresso e contesto 1–2,
pannelli 4–5 a gruppi, rimozione della dashboard attuale 1; circa 1200–1600 righe nuove. *(Il 16/10 i
pannelli passano nelle PR delle analisi e l'ingresso in MA.1: il costo si ripartisce, §8.)*

**B. Togliere il caso dalle 2827 righe attuali.** Da sciogliere: 211 occorrenze del letterale
EC50; le costanti di modulo dello studio lette all'import; la seconda implementazione del forecast;
i testi scritti a mano; soprattutto la scheda dei ritardi (753 righe), che calcola dal vivo una
ventina di test che la pipeline non ha (`compute_mhw_deep`: confronto a ritardo fisso di 2 mesi,
dose-risposta per terzili, correlazioni stagionali, estate→autunno, serie annuali; accelerazione del
declino), ciascuno da portare nella pipeline, con una scelta di metodo, o da togliere. Stima
**8–11 sessioni**, e il risultato resterebbe organizzato attorno al racconto di Livorno, non
all'elenco delle analisi applicabili a uno studio.

**Decisione (2/10/2026): A.** Su `main` la dashboard di Livorno non sopravvive come codice a sé: si toglie con
M1.13, e l'app del lavoro resta sul suo branch. Le analisi che oggi esistono solo nella dashboard
spariscono da `main`; quelle da tenere diventano analisi della pipeline, ciascuna con una decisione
(ADR-0000, voce 17).

## 8. Stime

- **Download per un sito tipico.** Misurato nei job del 30/9 su GitHub Actions: circa 5–6 s per
  richiesta mensile (quasi tutto metadati e apertura del dataset), circa 4 s per l'intera SST
  giornaliera 2003–2026 su una cella. Uno studio completo richiede 5 variabili mensili e la SST
  giornaliera, ciascuna con il multiyear più la coda analysis-forecast: circa 12 richieste,
  **1–2 minuti** su GitHub Actions, meno di 5 MB. Su Streamlit Cloud non è misurato: si assume
  2–5 minuti finché M1.2 non lo misura.
- **Fetta verticale** (M1.1–M1.6), chiusa il 16/10: stimata 7–8 sessioni, fatta in **9 PR** (6 pezzi più
  3 non pianificate: la CI rotta da `pyarrow`, i documenti di M1.9, il driver sull'anomalia). Il fattore
  osservato è 1,1–1,3; per prudenza si applica **1,3** al resto.
- **Stime rifatte il 16/10** (una sessione = una PR; base × 1,3):

  | Parte | Base | Con 1,3 |
  |---|---|---|
  | App minima (MA.1–MA.5) | 5 | **6–8** |
  | Allargamento M1.7–M1.10, con i pannelli (assorbe M1.12) e il lato app di ogni pezzo | 18–25 | 23–33 |
  | Il caso dei ricci nell'app generica e via la dashboard di Livorno (M1.11, M1.13) | 2–3 | 3–4 |
  | Completamenti dei tre passi e prova generale (M1.14–M1.17) | 3–4 | 4–5 |
  | **Resto da fare** | **28–37** | **36–48** |

  Fatte 9, **in tutto 45–57 PR** (la stima del 2/10 era 32–42). Una correzione: nel messaggio del 16/10 avevo
  scritto «33–45 in tutto» applicando il fattore al solo resto di allora (25–34); quel numero era il resto,
  non il totale. L'aumento di oggi viene per circa 3 dall'app minima (lavoro che prima era in parte nei passi
  M1.14–M1.16) e per il resto dal fattore osservato. Il rischio più grande è M1.9 (16 moduli, un pannello
  ciascuno): ogni modulo con l'app dentro vale più di 1 solo se l'analisi è lenta o con molti requisiti.
- **Download.** Misurato su GitHub Actions: 14–18 s per la SST giornaliera 2003–2026 di un punto
  (8674 giorni). Su Streamlit Cloud non è misurato: si misura in MA.2 con le credenziali del proprietario.
- **Tappe con demo.** `METODO_E_FASI.md` chiede una tappa con la sua demo ogni tre-sei settimane,
  e la milestone ne stima 32–42 sessioni. La chiusura della fetta verticale (M1.6) è la prima
  tappa, con la sua demo da riga di comando: su coordinate mai usate, un comando scarica la SST,
  costruisce il dataset dalla serie fittizia e ritrova il ritardo noto. Le tappe successive: fine
  dell'allargamento (M1.10, Livorno completo dal comando unico e pacchetto), dashboard generica
  (M1.13), prova generale (M1.17).

## 9. Fuori dalla milestone

Le finestre temporali nei moduli non migrati, il prewhitening ARIMA (issue #9), i prodotti fuori
dal Mediterraneo, il confronto fra casi e la contabilità dei test (V3.2), autenticazione e più
utenti (V5), la rinomina del progetto, la rimozione dello stato di modulo dal nucleo (D2). Ogni
difetto trovato lungo la strada che non blocca il percorso della milestone va in una issue e non
si tocca.

### Decisioni rimandate dalla milestone

- **La coda analysis-forecast della SST giornaliera.** La serie giornaliera scaricata (M1.2) finisce dove
  finisce il prodotto multiyear. Il catalogo registra il dataset analysis-forecast che lo segue nel tempo,
  ma non lo si scarica. Aggiungere quella coda è una **scelta di metodo**, non una comodità: i due prodotti
  non sono la stessa serie (reanalisi contro analisi operative), e cambia la SST degli ultimi mesi e quindi
  le ondate di calore rilevate proprio dove i ritardi più recenti pesano di più. Da decidere insieme alla
  regola dei mesi incompleti (M1.4); registrata in `docs/adr/0000-decisioni-rimandate.md`, voce 18.
