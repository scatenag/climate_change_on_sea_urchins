# Milestone M1 · Lo strumento generico da capo a fondo

> **Stato (2/10/2026):** definizione fissata dal proprietario; decisioni aperte e piano dei pezzi
> **proposti, in attesa di approvazione** (§5, §6). Nessun codice della milestone si scrive prima
> dell'approvazione del piano. Il piano approvato sostituisce §6 in questo file.

---

## 1. Obiettivo

Una versione generale dello strumento che funziona da capo a fondo, mostrata da una demo guidata
registrabile in un video per i convegni. Alla presentazione si usa il video: l'app non deve
reggere una dimostrazione dal vivo davanti a un pubblico, ma deve funzionare davvero, su dati
veri scaricati in quel momento.

Il manoscritto citato (in revisione) afferma che la pipeline è un modello adattabile da altri programmi di
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
  1. sito, periodo e credenziali Copernicus dell'utente; download reale dei dati ambientali e
     loro visualizzazione, compresi gli eventi di ondata di calore rilevati;
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
| 4 | Un pacchetto scaricato e ricaricato riproduce gli stessi risultati | test di andata e ritorno: stessi file di risultato, byte per byte, sulla stessa macchina |
| 5 | Il caso dei ricci si apre senza credenziali e senza conoscere lo strumento | test dell'app (`streamlit.testing`) senza credenziali né variabili d'ambiente; prova con una persona esterna al progetto |

## 5. Decisioni aperte (proposte, da approvare)

| # | Domanda | Proposta | Alternativa |
|---|---|---|---|
| D1 | Ambito geografico del catalogo | Solo Mediterraneo (i prodotti MEDSEA di Livorno); coordinate fuori dominio rifiutate con un messaggio | Prodotti globali (risoluzione, profondità e variabili diverse: un secondo catalogo da validare) |
| D2 | Stato globale dello studio: `common.py` carica lo studio all'import (`DATA`, `SPLIT_DATE`, `WINDOWS` costanti di modulo, scelte da `CCSU_STUDY`), incompatibile con più studi nello stesso processo | Ogni esecuzione di uno studio in un **processo separato** (comando unico); la dashboard legge per percorsi espliciti. Togliere lo stato di modulo resta per dopo (ADR) | Rifattorizzare ora `common.py` e i 16 moduli per ricevere il contesto esplicito |
| D3 | Con quale analisi si ritrova il ritardo noto | CCF generica variabile ambientale → risposta, sulle differenze prime, ritardi 0–12, correzione BH-FDR, riusando `mhw_analysis.compute_ccf`; serie fittizia = temperatura ritardata di *k* mesi + rumore AR(1), seme fissato ed esposto | Serie fittizia generata dall'intensità MHW, ritrovata dalla CCF MHW→risposta che esiste già (nessuna analisi nuova, ma non «dalla temperatura») |
| D4 | Cosa contiene il pacchetto | Solo input e manifest, con l'impronta dei risultati; al caricamento si ricalcola e l'app dice se i risultati coincidono | Anche i risultati, mostrati subito senza ricalcolo |
| D5 | `split_date` per uno studio nuovo | Facoltativo: senza, le analisi pre/post sono spente con il motivo | Obbligatorio, chiesto all'utente |
| D6 | DLNM (R) nell'app generica | Spento con il motivo «R non disponibile su questo server» (coerente con ADR-0000 voce 4) | Installare R sull'app |
| D7 | Speciazione del rame su uno studio utente | Requisito dichiarato nella specifica (campo nuovo per il contaminante dell'endpoint); senza, spenta con il motivo | Spenta per tutti gli studi diversi da Livorno |
| D8 | Imputazione della risposta (oggi solo nel codice, ADR-0000 voce 8) | Dichiarata nella specifica con i valori attuali (media mobile centrata 12 mesi, `min_periods` 3, doppia passata), deriva zero | Lasciata nel codice e dichiarata nell'interfaccia come scelta fissa |
| D9 | Download nel pacchetto installabile | Un modulo del pacchetto (extra `acquisition`), perché l'app lo chiama; `scripts/` resta per il job di Livorno | — |

## 6. Piano proposto

Ogni pezzo è una PR con il proprio criterio di verifica. Prima il motore (verificabile senza
interfaccia e senza rete), poi l'interfaccia.

| Pezzo | Contenuto | Criterio di verifica |
|---|---|---|
| **M1.1 Catalogo delle variabili** | File dichiarativo nel pacchetto: per variabile prodotto, dataset multiyear e analysis-forecast, nome, unità nativa, unità di analisi, conversione nominata (Pa→µatm), profondità, cadenza, dominio | Il catalogo riproduce esattamente dataset, variabili, profondità e conversione usati oggi per Livorno; nessun campo eseguibile |
| **M1.2 Specifica, formato 2** | `format_version`; ambiente per id di catalogo; `split_date` facoltativo (D5); campi di D7/D8; caricamento non fidato che ignora `data_dir` e sorgenti remote | Livorno formato 1 si carica e dà deriva zero; una versione sconosciuta è rifiutata nominandola; YAML con tag Python e `data_dir` ostili rifiutati o ignorati con avviso |
| **M1.3 Download guidato dalla specifica** | Mensili e SST giornaliera, multiyear più coda analysis-forecast, fine dalla copertura del prodotto (nessuna data scritta a mano), controllo della cella di mare, manifest di provenienza; credenziali come argomenti | In CI, con il toolbox simulato: credenziali passate solo come argomenti, `login` mai chiamato, ambiente e `$HOME` intatti, errore con la password mascherato (visto fallire su sabotaggio). A mano, con credenziali: sulle coordinate di Livorno serie uguali a `data/env_copernicus.csv` nei mesi comuni; tempi misurati |
| **M1.4 Sorgente CSV della risposta** | Modello di CSV generato dalle risposte alle domande; lettura per prova o aggregata; errori leggibili con il numero di riga | I dati per prova di Livorno, esportati nel modello e riletti, danno la stessa serie mensile; un test per ogni classe di errore |
| **M1.5 Costruttore del dataset generico** | Da ambiente, risposta e SST ai file che legge `load_data`, con le funzioni di `common.py` (una sola implementazione) e il rilevamento MHW | Dagli input della fixture, file identici byte per byte a quelli della fixture; golden master a deriva zero |
| **M1.6 Requisiti dichiarati** | Ogni analisi dichiara cosa le serve (variabili, SST giornaliera, dati per prova, controlli, `split_date`, mesi minimi, R); `analyses.json` con stato ed eventuale motivo | Studio sintetico senza pH: speciazione e forecast spenti con motivo e nessun loro file scritto; Livorno: tutte eseguite, deriva zero |
| **M1.7 Comando unico** | `ccsu-run-study <study.yaml o pacchetto>`: costruzione, analisi, rapporto; avanzamento leggibile da un altro processo; il job di Livorno lo usa | Golden master eseguito dal comando (criterio 1, parte motore); uno studio sintetico gira in CI senza rete |
| **M1.8 Pacchetto** | Esportazione e caricamento (D4); manifest con versione del codice, prodotti, versioni dei dataset, data del download, impronte | Andata e ritorno con risultati identici (criterio 4); archivi ostili rifiutati (percorsi fuori cartella, file inattesi, dimensioni); nessuna credenziale nel pacchetto (visto fallire su sabotaggio) |
| **M1.9 Serie a verità nota** | Generatore della serie fittizia (D3), seme esposto; CCF generica variabile → risposta | Sulla temperatura della fixture con *k* = 3 il ritardo 3 sopravvive a FDR; con una serie indipendente nessun ritardo sopravvive |
| **M1.10 Dashboard: ingresso e contesto** | Scelta iniziale; contesto di studio passato esplicitamente alle schede; Livorno precalcolato senza credenziali, etichetta e rimando all'app del lavoro, percorso in sola lettura; il dashboard attuale resta dietro un interruttore | Test dell'app senza credenziali (criterio 5); guardia RSPTEST sulla vista generica |
| **M1.11 Dashboard: schede generiche, parte 1** | Panoramica, serie temporali, ondate di calore, correlazioni, pre/post; schede spente con il motivo; testi dai valori | Stessi valori del job per Livorno; studio sintetico con analisi spente (criterio 3) |
| **M1.12 Dashboard: schede generiche, parte 2** | Ritardi (CCF), stazionarietà, changepoint e regime shift, forecast (da `forecast.py`: sparisce la seconda implementazione), legacy termica, speciazione | Come M1.11; nessuna funzione di calcolo duplicata nel dashboard |
| **M1.13 Propri dati, passo 1** | Sito su mappa, periodo, credenziali, download con avanzamento, visualizzazione con gli eventi MHW | Test dell'app con download simulato: credenziali assenti da file, messaggi, cache e pacchetto |
| **M1.14 Propri dati, passo 2** | Domande, modello CSV, caricamento, validazione | Errori mostrati in chiaro per ogni classe; Livorno ricostruito dal CSV equivale al caso precalcolato |
| **M1.15 Propri dati, passo 3** | Analisi applicabili, analisi lente su richiesta con avanzamento (processo separato), pacchetto a ogni passo, ricarica | Criteri 3 e 4 dall'interfaccia |
| **M1.16 Prova generale e copione** | Percorso del video eseguito per intero con credenziali vere su coordinate mai usate; copione in `docs/`; `docs/ADAPTING.md` riscritto | I cinque criteri spuntati, con le evidenze nella PR |

**Stime.**
- **Download per un sito tipico.** Misurato nei job del 30/9 su GitHub Actions: circa 5–6 s per
  richiesta mensile (quasi tutto metadati e apertura del dataset), circa 4 s per l'intera SST
  giornaliera 2003–2026 su una cella. Uno studio richiede 5 variabili mensili e la SST
  giornaliera, ciascuna con il multiyear più la coda analysis-forecast: circa 12 richieste,
  **1–2 minuti** su GitHub Actions, meno di 5 MB. Su Streamlit Cloud non è misurato: si assume
  2–5 minuti finché M1.3 non lo misura.
- **Dashboard capace di mostrare uno studio qualunque** (M1.10–M1.12): **5–7 sessioni**. Il
  dashboard attuale ha 2827 righe, 10 schede, 211 occorrenze del letterale EC50, legge le
  costanti di modulo dello studio caricato all'import, e la scheda dei ritardi (circa 750 righe) è
  quasi tutta specifica del caso; contiene ancora conclusioni scritte a mano e una seconda
  implementazione del forecast.
- **Milestone intera:** 20–26 sessioni, una PR ciascuna. Se il tempo non basta, si taglia M1.12
  alla sola scheda dei ritardi e al forecast; le altre schede restano spente con il motivo «non
  ancora disponibile nella vista generica», e il caso completo resta visibile nell'app del lavoro.

## 7. Fuori dalla milestone

Le finestre temporali nei moduli non migrati, il prewhitening ARIMA (issue #9), i prodotti fuori
dal Mediterraneo (se D1 approvata), il confronto fra casi e la contabilità dei test (V3.2),
autenticazione e più utenti (V5), la rinomina del progetto. Ogni difetto trovato lungo la strada
che non blocca il percorso della milestone va in una issue e non si tocca.
