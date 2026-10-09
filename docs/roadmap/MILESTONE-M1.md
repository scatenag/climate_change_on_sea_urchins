# Milestone M1 · Lo strumento generico da capo a fondo

> **Stato (aggiornato l'8/10/2026):** definizione fissata dal proprietario; decisioni D1–D9 approvate il
> 2/10 con le precisazioni riportate in §5; piano riordinato il 2/10 come fetta verticale (§6), che
> si approva con la fusione di questo documento. Nessuna scadenza: la milestone si chiude quando
> i criteri di §4 sono soddisfatti.

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
PR. Prima una fetta verticale stretta che attraversa la catena da riga di comando; poi si allarga
una variabile e un'analisi alla volta; solo dopo vengono la dashboard e i tre passi.

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
| **M1.6 Serie a verità nota (uscita della fetta)** | Generatore con seme esposto. Riproduce i caratteri della serie reale: risposta che scende quando la temperatura sale, ritardo *k* né 0 né multiplo di 12 (proposta: 3 mesi), da 1 a 6 prove per mese da aggregare, mesi mancanti in sequenze come nel foglio dei ricci (circa metà dei mesi), rumore autocorrelato | Il profilo della CCF su una serie che dipende dalla SST grezza ha, sulle differenze prime, un picco negativo a *k*, uno positivo a *k*+6 di ampiezza simile quando domina il ciclo stagionale, e valori alti ai ritardi vicini a *k*. Per questo il criterio non è «*k* è l'unico ritardo che sopravvive»: **il massimo di \|r\| cade su *k* con il segno atteso, e gli altri ritardi che superano FDR sono riportati**. Se la serie dipende dalla SST grezza o dalle sue anomalie si decide all'inizio di M1.6, guardando il profilo della CCF misurato sulla fixture nei due casi. Con una serie indipendente il massimo di \|r\| non cade su *k* o nessun ritardo supera FDR. A mano, con credenziali: il criterio di uscita su coordinate mai usate |

### Allargamento (una variabile e un'analisi alla volta)

| Pezzo | Contenuto | Criterio di verifica |
|---|---|---|
| **M1.7 Requisiti dichiarati** | Ogni analisi dichiara cosa le serve (variabili, SST giornaliera, dati per prova, controlli, `split_date`, mesi minimi, contaminante, R); `analyses.json` con stato ed eventuale motivo | Uno studio senza pH: speciazione e forecast spenti con il motivo, nessun loro file scritto (criterio 3, motore) |
| **M1.8 Variabili mensili**, una PR per variabile | Temperatura 0–10 m, salinità, O₂, pH, CO₂ con la conversione Pa→µatm nominata nel catalogo | Voce di catalogo uguale a quella usata oggi per Livorno; a mano, sulle coordinate di Livorno, serie uguale a `data/env_copernicus.csv` nei mesi comuni |
| **M1.9 Analisi nel comando unico**, una PR per analisi o famiglia | Le analisi della pipeline, ciascuna con i propri requisiti, fino a Livorno completo; alla fine il job di Livorno usa `ccsu-run-study` (PR su `update_ec50.yml`) | Deriva zero sul golden master a ogni PR; alla fine il golden master intero eseguito dal comando unico (criterio 1, motore) |
| **M1.10 Pacchetto e caricamento non fidato** | Esportazione e caricamento (D4); YAML sicuro; `data_dir` e sorgenti remote ignorati con avviso | Andata e ritorno con risultati identici (criterio 4): il confronto byte per byte esclude i campi di provenienza che cambiano a ogni esecuzione (data e ora, durata, versione del codice), e l'impronta del pacchetto tratta i file che il golden master confronta solo nella struttura (`STRUCTURAL_ONLY_FILES`, #9) allo stesso modo, cioè sulla struttura, così un pacchetto ricaricato su un'altra macchina non risulta diverso senza esserlo; archivi ostili rifiutati (percorsi fuori cartella, file inattesi, dimensioni); nessuna credenziale nel pacchetto (visto fallire su sabotaggio) |

### Interfaccia

| Pezzo | Contenuto | Criterio di verifica |
|---|---|---|
| **M1.11 Dashboard generica: ingresso e contesto** | Scelta iniziale; contesto di studio per sessione (§5.1); Livorno precalcolato senza credenziali, con l'etichetta e il rimando all'app del lavoro, e il percorso in sola lettura | Test dell'app senza credenziali né variabili d'ambiente (criterio 5); due sessioni con studi diversi nello stesso processo |
| **M1.12 Pannelli per analisi**, una PR per gruppo | Un pannello per analisi, letto da `analyses.json` e dai file dell'analisi; analisi spente con il motivo; testi dai valori | Stessi valori dei file di risultato; guardia RSPTEST sulla vista generica (criterio 1, dashboard; criterio 3) |
| **M1.13 Via la dashboard di Livorno da `main`** | Rimozione di `dashboard.py` e, se non serve più, del meccanismo di import di `app.py`; l'app del lavoro vive sul suo branch | Nessun riferimento rimasto; l'app di sviluppo apre la vista generica |
| **M1.14 Propri dati, passo 1** | Sito su mappa, periodo, credenziali, download con avanzamento, visualizzazione con gli eventi MHW | Test dell'app con download simulato: credenziali assenti da file, messaggi, cache e pacchetto |
| **M1.15 Propri dati, passo 2** | Domande, modello CSV, caricamento, validazione | Errori mostrati in chiaro per ogni classe; Livorno ricostruito dal CSV uguale al caso precalcolato |
| **M1.16 Propri dati, passo 3** | Analisi applicabili, analisi lente su richiesta con avanzamento, pacchetto a ogni passo, ricarica | Criteri 3 e 4 dall'interfaccia |
| **M1.17 Prova generale e copione** | Percorso del video eseguito per intero con credenziali vere su coordinate mai usate; copione in `docs/`; `docs/ADAPTING.md` riscritto | I cinque criteri spuntati, con le evidenze nella PR |

## 7. La dashboard: riscriverla o togliere il caso dall'attuale

**A. Riscritta, guidata da risultati e specifica.** Un pannello per analisi: legge lo stato da
`analyses.json` e i valori dai file dell'analisi, con il contesto della sessione (§5.1); i testi
nascono dai valori. Costo stimato **6–8 sessioni** (M1.11–M1.13): ingresso e contesto 1–2,
pannelli 4–5 a gruppi, rimozione della dashboard attuale 1; circa 1200–1600 righe nuove.

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
- **Fetta verticale** (M1.1–M1.6): 7–8 sessioni. **Allargamento** (M1.7–M1.10): 14–20 sessioni,
  quasi tutte in M1.9 (16 moduli). **Dashboard** (M1.11–M1.13): 6–8. **Tre passi e prova
  generale** (M1.14–M1.17): 5–6. In tutto **32–42 sessioni**, una PR ciascuna. La stima si
  rivede alla chiusura di M1.6 con i tempi reali della fetta, prima di cominciare M1.7.
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
