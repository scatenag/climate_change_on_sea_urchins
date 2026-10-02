# Strada di sviluppo e metodo di lavoro

Progetto: evoluzione di `climate_change_on_sea_urchins` da case study a strumento generico.
Orizzonte: mesi. Sviluppo: una persona, con un assistente di codice.

Questo documento fissa **due cose sole**: in che ordine procedere, e come lavorare senza
perdersi. I dettagli tecnici di ogni fase si decidono all'inizio della fase, non ora.

---

## Parte 1 — Cosa decidere ora e cosa no

La regola: **decidi ora solo ciò che è costoso cambiare dopo.** Tutto il resto va rimandato,
perché deciderlo adesso significa deciderlo male, con meno informazioni di quante se ne avranno
tra tre mesi.

### Decisioni costose da prendere ora (8 invarianti)

Non sono tecnologie, sono vincoli di forma. Costano poco oggi ed evitano una riscrittura
domani — inclusa quella verso l'istanza condivisa multi-utente, che così resta possibile
senza doverla progettare adesso.

1. **Niente stato globale che descrive il caso.** Sito, serie di risposta, colonne, finestra non
   sono costanti di modulo: viaggiano in un oggetto di contesto passato esplicitamente.
   *È la differenza tra "un caso" e "N casi", e più avanti tra "un utente" e "N utenti".*

2. **Il nucleo scientifico non conosce l'interfaccia.** La logica di analisi non importa
   Streamlit, non stampa a schermo, non decide dove scrivere: riceve dove scrivere.

3. **Ogni artefatto ha un'identità derivata dalla specifica, non un nome fisso.**
   Oggi `results/corr_all.csv` è un nome fisso: due casi si sovrascrivono. Un identificativo
   derivato dal caso risolve il problema per sempre, e senza costi.

4. **Un solo punto di ingresso/uscita per i dati.** Non `open()` sparso ovunque, ma un
   confine unico. Oggi dietro c'è il filesystem; domani può esserci altro. Non serve
   implementare "altro" adesso: serve non spargere.

5. **Il vocabolario è neutro.** La serie osservata non si chiama "biomarker": lo strumento
   ospiterà proxy di popolazione, indici, endpoint tossicologici e biomarker veri, e un nome
   sbagliato in uno strumento rivolto a ecotossicologi è un difetto, non un dettaglio. Il
   nome canonico interno è *serie di risposta*; che cosa sia lo dichiara la specifica.

6. **Le scelte scientifiche stanno nella specifica, non nel codice.** Se leggendo il file
   di configurazione non riesci a ricostruire ogni trasformazione applicata ai dati
   (imputazioni, riempimenti, finestre, soglie), la scelta è nascosta nel codice ed è un bug
   di trasparenza, non una comodità.

7. **Ogni risultato porta con sé la sua provenienza:** quali input, con quali hash, con
   quali versioni, con quali parametri. Serve a chi sviluppa tra sei mesi, prima che a chiunque altro.

8. **La configurazione non esegue codice arbitrario.** Un file di studio descrive, non
   programma. Vincolo gratuito oggi, che evita una riscrittura il giorno in cui altri
   caricano le loro configurazioni su un'istanza condivisa.

### Decisioni da rimandare esplicitamente

Stanno in `docs/adr/0000-decisioni-rimandate.md`, così non si ridiscutono a ogni sessione:

- Quale tecnologia per l'istanza condivisa, autenticazione, isolamento dei dati → V5
- Se e quando sostituire Streamlit → dopo V3.3
- Se portare in Python le analisi R o tenerle come dipendenza opzionale → V3.1
- Nome definitivo del progetto e rinomina del repo → fine V2
- Formato di pubblicazione dei risultati e sede editoriale → dopo V3.2

---

## Parte 2 — Le versioni

L'asse principale è **V1 → V4**, dove ogni versione è un
salto di capacità dello strumento. Dentro ogni versione stanno i passi operativi, ognuno con
un nome che dice cosa fa.

Ogni passo ha tre proprietà obbligatorie:

- **è utile da solo**: fermandosi lì, si ha comunque qualcosa di meglio di prima
- **finisce con una demo**: qualcosa da mostrare a un collega in cinque minuti
- **riduce un rischio preciso**: non è "lavoro", è "un'incertezza in meno"

```
V1 · IL CASO                          stato attuale: una serie di risposta, un sito, analisi fisse
     V1.1  Messa in sicurezza     ~3 sett.  rischio: rompere la scienza rifattorizzando

V2 · IL CASO DIVENTA CONFIGURAZIONE   risposta, coordinate e finestra si dichiarano
     V2.1  Specifica e risposta   ~6 sett.  rischio: astrazione sbagliata
     V2.2  Prova sul campo        ~4 sett.  rischio: genericità solo dichiarata
     M1    Strumento generico da capo a fondo (dentro V2, anticipa parti di V3.1 e V3.3)
                                            rischio: strumento generico solo per chi ha scritto il codice

V3 · MOLTI CASI, UN CONFRONTO         tuple multiple e confronto tra i risultati
     V3.1  Analisi intercambiabili ~6 sett. rischio: ogni analisi nuova tocca il nucleo
     V3.2  Confronto e inferenza  ~8 sett.  rischio: fabbrica di falsi positivi
     V3.3  Installabile da altri  ~6 sett.  rischio: strumento che gira su un PC solo

V4 · META-TOOL                        organismi e risposte di natura diversa, vocabolari, multiverso

V5 · ISTANZA CONDIVISA                rimandata: si valuta quando ci sarà domanda reale
```

Rispetto alla numerazione originale c'è **una sola aggiunta, V1.1**: la messa in
sicurezza di ciò che esiste già. Non è una nuova capacità — V1 resta V1 — ma è il
prerequisito per poter toccare qualunque cosa senza rompere risultati già pubblicati. E
include la **correzione delle unità CO₂**, che è a tutti gli effetti un difetto di V1, non
un pezzo della generalizzazione: va corretto anche se il progetto si fermasse qui.

Le altre due aggiunte non sono versioni ma sotto-passi: V2.2 (validare V2 su casi reali,
altrimenti la genericità resta dichiarata) e V3.3 (rendere lo strumento installabile da
altri, che è ciò che lo trasforma in uno strumento per la comunità invece che in un progetto
personale). V5 è l'istanza multi-utente, spostata dopo perché non deve bloccare l'avvio.

Le stime sono a settimane-calendario part-time, non a giornate-uomo. Servono solo a capire
se si sta sforando di molto.

---

### V1.1 · Messa in sicurezza

**Rischio che elimina.** Non c'è modo di sapere se un refactor ha cambiato un numero
già pubblicato. Finché è così, ogni modifica è una scommessa.

**Cosa contiene.**
- Congelamento dei risultati attuali come riferimento numerico, e un comando unico che
  verifica se qualcosa è cambiato.
- Seed fissi ovunque ci sia casualità (bootstrap, foreste casuali, surrogati): se un modulo
  non è riproducibile, è il primo bug da chiudere.
- Correzione del bug unità CO₂ (`spco2` è in Pascal, non µatm) — fatta **subito**, perché
  cambia i valori di riferimento e va fatta prima di congelarli, non dopo.
- Impianto di memoria del progetto: `CLAUDE.md`, `docs/roadmap/STATO.md`, `docs/adr/`,
  `CHANGELOG.md`, flusso issue → branch → PR.
- Lint, type check leggero, test in CI.

**Demo.** «Cambio una riga a caso in un modulo di analisi, lancio un comando, e il sistema
mi dice esattamente quali risultati sono cambiati e di quanto.»

**Non fare.** Nessun miglioramento di architettura in questo passo. Solo rete.

---

### V2.1 · Specifica e astrazione della risposta

**Rischio che elimina.** Progettare l'astrazione sbagliata. Si mitiga tenendola minima:
in questo passo si generalizza **solo** ciò che serve per V2.2, niente di più.

**Cosa contiene.**
- Un file di specifica dello studio che descrive: sito, serie di risposta,
  variabili ambientali, finestra temporale, regole di allineamento.
- La serie biologica smette di chiamarsi "EC50" ovunque: nome, unità, etichetta e
  **direzione** (valore alto = organismo più o meno sensibile?) vengono dalla specifica.
  La direzione sembra un dettaglio ed è invece ciò che salva V3.2, quando
  si confronteranno serie che rispondono allo stress in versi opposti.
- Le regole oggi implicite (imputazione a media mobile, riempimento della temperatura da
  altra fonte) diventano scelte dichiarate nella specifica.
- Identità del caso e artefatti indirizzati per caso (invariante 3).
- Ingestione dati dietro un confine unico, con almeno due sorgenti: quella attuale e "un
  file fornito dall'utente" — quest'ultima serve anche a far girare i test senza credenziali.

**Demo.** «Scrivo un file di venti righe, lancio un comando, e ottengo la stessa analisi di
prima su una finestra temporale diversa e su una cella di mare diversa.»

**Non fare.** Non toccare il dashboard oltre le etichette. Non introdurre plugin. Non
generalizzare le analisi.

---

### V2.2 · Prova sul campo (il passo che vale doppio)

**Rischio che elimina.** L'astrazione che sembra giusta finché la usi su un caso solo.
**Il secondo caso è l'unico test onesto del design di V2.1**, e per questo va fatto
subito dopo — non alla fine.

**Cosa contiene.** Due casi che si possono fare subito senza dipendere da altri, più almeno una seconda serie
biologica da procurarsi.

1. **Finestre temporali diverse** sulla stessa serie ricci. Costo zero, verifica immediata
   della stabilità dei risultati.
2. **Celle spaziali spostate** — stessa serie di risposta, dati ambientali presi 50, 200, 500 km
   più in là. Non è un esercizio: è un **test di falsificazione**. Se la correlazione regge
   uguale con l'ambiente di un altro bacino, la correlazione non è ambientale. Serve come
   controllo negativo permanente, ed è forse la cosa più utile che questo strumento possa
   offrire a un ecotossicologo.
3. **Serie sintetiche a verità nota.** Serie costruite apposta con un ritardo, un effetto e
   un rumore noti, per verificare che lo strumento li ritrovi — e che *non* trovi
   niente dove non c'è niente. Non sostituiscono un caso reale, ma validano due cose che un
   caso reale non valida: la correttezza del motore e la sua calibrazione. Sono anche la base
   del calcolo di potenza che servirà in V3.2. Costo: basso. Dipendenze da altri: zero.
4. **La serie dei controlli negativi**, che esiste già ed è stata recuperata. Attenzione a
   come la usi: è stazionaria per costruzione, quindi non è un caso di studio. È il
   **controllo negativo dello strumento**: se il tool ci trova un segnale ambientale, il tool
   è rotto. Ed è anche l'origine di una funzione che vale la pena rendere di prima classe —
   una *serie di controllo* affiancata alla serie di risposta, verificata per stazionarietà e
   drift prima di interpretare qualunque risultato. Ogni laboratorio serio ne ha una: è
   esattamente il genere di cosa che rende il tool credibile agli occhi di un altro gruppo.
5. **Una seconda serie di risposta reale**, se e quando arriva. Il criterio non è la
   rilevanza scientifica del caso, ma quanto è *strutturalmente diversa*: altra unità, altro
   endpoint, possibilmente altra direzione o altra cadenza. Oggi non ne esiste nessun'altra
   che copra il ventennio, e quelle nuove partiranno da zero punti: **non tenerla sul
   percorso critico**. Se arriva da una collaborazione, è un guadagno; se non arriva, V2.2 si
   chiude lo stesso con i punti 1-4.

Vincolo di progetto da annotare fin d'ora, senza farne un compito: **prima o poi arriverà un
sito che la griglia ambientale non risolve** — lagune, aree costiere confinate, acque
interne. Non progettare "il sito" assumendo che sia sempre una cella di modello marino:
quando succederà dovrà essere possibile usare una fonte in situ, o una cella surrogata con
avviso esplicito. Costa nulla tenerlo presente ora, costa una riscrittura scoprirlo dopo.

Ogni caso che rompe qualcosa genera una issue e, se serve, un ritocco a V2.1. È
previsto: è il motivo per cui V2.2 esiste.

**Demo.** «Tre o quattro cartelle di esempio nel repo, studi diversi, un comando li esegue
tutti, la guida all'adattamento è passata da tre pagine di scuse a mezza pagina.»

### M1 · Lo strumento generico da capo a fondo (aggiunta il 2/10/2026)

**Rischio che elimina.** Che la genericità resti verificabile solo da chi conosce il codice: uno
`study.yaml` scritto a mano e un comando lanciato in locale. M1 la porta fino a chi ha solo i propri
dati e le proprie credenziali Copernicus.

**Dove sta.** Dentro V2: il secondo caso reale che V2.2 aspettava diventa «qualunque sito, qualunque
serie caricata», e la serie sintetica a verità nota (punto 3 di V2.2) diventa la parte finale della
demo. Anticipa da V3.1 la dichiarazione dei requisiti di ogni analisi (con rifiuto visibile e
motivato) e da V3.3 l'uso senza conoscere il codice (app con percorso guidato, pacchetto
scaricabile e ricaricabile). Non contiene: celle spostate e tabella di confronto delle finestre
(restano in V2.2), confronto fra casi e contabilità dei test (V3.2), istanza multi-utente (V5).

**Demo.** Un video registrato: Livorno ricostruito da zero dal percorso dei propri dati, poi un
secondo sito con una serie fittizia a ritardo noto, ritrovato nella dashboard.

Definizione, vincoli, criteri di accettazione e piano: `MILESTONE-M1.md`.

---

**Nota sull'ordine.** La milestone M1 (sopra) non ha una scadenza: si chiude quando i suoi
criteri di accettazione sono soddisfatti, e si mostra con un video registrato.

---

### V3.1 · Analisi intercambiabili

**Rischio che elimina.** Che ogni analisi nuova richieda di mettere le mani nel nucleo,
e che analisi specifiche di un dominio (speciazione del rame, ondate di calore) girino su
casi dove non hanno senso, producendo numeri privi di significato invece di un rifiuto.

**Cosa contiene.**
- Ogni analisi dichiara di cosa ha bisogno (quali variabili, quale cadenza, quanti punti
  minimi) e viene saltata **rumorosamente** se i requisiti mancano. Mai fallire in silenzio,
  mai produrre un numero su input inadeguati.
- Separazione tra analisi generiche e analisi specifiche del caso ricci-rame-Livorno: queste
  ultime restano, ma come componenti opzionali dichiarate.
- Forma comune dei risultati: ogni analisi restituisce l'effetto stimato nella stessa
  struttura (stima, intervallo, numerosità, scala, quanti test ha eseguito). **Questa è la
  decisione che rende possibile V3.2**: scrivila come ADR e non derogare.

**Demo.** «Aggiungo un'analisi nuova senza toccare una riga del nucleo, e su un caso che non
soddisfa i suoi requisiti il sistema mi dice perché la salta.»

---

### V3.2 · Confronto tra casi e disciplina inferenziale

**Rischio che elimina.** Il rischio vero di uno strumento come questo: diventare una
macchina per falsi positivi. Ogni tupla in più e ogni ritardo in più moltiplica i gradi di
libertà. Si è già visto sui dati di Livorno: il segnale MHW a lag 1 anno, con il 3,7% di test
significativi contro il 4,9% atteso per puro caso.

**Cosa contiene.**
- Confronto degli effetti tra casi, con visualizzazione a foresta e sintesi a effetti
  casuali, preceduta sempre da un test di eterogeneità: se i casi sono troppo diversi, lo
  strumento deve dirlo ad alta voce invece di aggregare.
- **Contabilità della molteplicità**: quanti test sono stati eseguiti in tutto lo studio,
  quanti ci si aspetterebbe significativi per caso, quanti sopravvivono alla correzione.
  Sempre in vista, mai nascosta in un'appendice.
- Modelli nulli calibrati sulla serie (non il nullo teorico, che su serie autocorrelate
  mente).
- Calcolo della potenza: dato quanti punti ci sono e quanto sono correlati, qual è l'effetto
  minimo rilevabile. Da mostrare **accanto a ogni risultato negativo**, così un
  "nessun effetto" diventa informativo invece che muto.

**Demo.** «Confronto quattro casi, e per ognuno vedo l'effetto, l'intervallo, quanto era
rilevabile, e quanti dei risultati significativi restano tali dopo aver contato tutti i
test che ho fatto.»

Questa fase è ciò che distingue lo strumento da un ciclo `for` sulle analisi. È anche ciò
che lo rende utile alla comunità: chiunque può correlare Copernicus con i propri dati, quasi
nessuno lo fa contando i test.

---

### V3.3 · Installabile e usabile da altri

**Rischio che elimina.** Lo strumento che funziona su un portatile solo.

**Cosa contiene.**
- Riorganizzazione dell'interfaccia: il dashboard oggi è un blocco unico di quasi tremila
  righe con un aggiramento del sistema di import e un lock globale che serializza tutte le
  sessioni. Va spezzato per schede, dietro interruttori, una alla volta. È il lavoro più a
  rischio di rotture silenziose di tutto il piano: isolalo, non mescolarlo ad altro.
- Riga di comando come interfaccia di pari dignità (per molti utenti sarà la principale).
- Installazione documentata, guida di avvio, esempi eseguibili, dati di esempio.
- **Prova di verità**: un collega installa e fa girare un proprio studio senza che chi l'ha
  sviluppato tocchi la sua tastiera. Se non ci riesce, il passo non è finito.

**Demo.** La prova di verità stessa.

---

### V4 · Meta-tool

Confronto tra organismi con risposte non commensurabili, analisi multiverso con
dichiarazione preventiva del piano, modelli gerarchici tra casi, vocabolari controllati e
unità di misura verificate, fonti ambientali oltre il marino. Sono le cose più interessanti
scientificamente, ed è per questo che vanno **dopo**: senza V2 e V3 sono castelli.

---

### V5 · Istanza condivisa

Solo dopo V3.3, e solo se c'è domanda reale. Gli invarianti 1, 3, 4 e 8 della Parte 1
sono ciò che rende V5 un'aggiunta e non una riscrittura: il nucleo non sa cosa sia
un utente, gli artefatti hanno già identità propria, i dati passano già da un confine unico
e la configurazione non esegue codice.

Da decidere all'inizio di V5, non ora: autenticazione, isolamento dei dati, gestione
delle credenziali verso le fonti esterne, quote, governo del dato. Nota per allora: l'autenticazione federata
degli enti di ricerca è una strada che apre lo strumento a tutta la comunità senza gestire
utenze a mano.

---

## Parte 3 — Come lavorare con un assistente di codice senza perdersi

Il problema di fondo: tra una sessione e l'altra l'assistente non ricorda nulla. Chi sviluppa, dopo tre
settimane su altro, neanche. Quindi:

> **La memoria del progetto è il repository, non la conversazione.**
> Qualunque decisione che non finisce in un file, è persa.

### I tre dispositivi di memoria

**1. `CLAUDE.md` alla radice** — letto automaticamente a ogni sessione. Contiene invarianti,
convenzioni, comandi di verifica, e le cose da non fare mai. È il file che impedisce la
deriva lenta, quella in cui alla sessione trenta il codice non somiglia più a quello della
sessione dieci. È locale (escluso da git) e rimanda alle regole versionate in
`docs/COME_SI_LAVORA.md`.

**2. `docs/roadmap/STATO.md`** — dove siamo. Fase corrente, ultimo passo chiuso, prossimo
passo, questioni aperte, cose scoperte da non dimenticare. **Si aggiorna alla fine di ogni
sessione, prima del merge.** È il singolo strumento più efficace contro il perdersi: costa
tre minuti e restituisce il contesto in trenta secondi dopo una pausa di un mese.

**3. `docs/adr/`** — una decisione per file: contesto, alternative valutate, scelta,
conseguenze. Serve a non ridiscutere la stessa cosa a ogni sessione, e a ricordare *perché*
si è deciso così quando tra sei mesi sembrerà assurdo.

A questi si aggiungono, gratis, issue e pull request: sono il diario di cosa è successo.

### Un passo = un branch = una sessione = una PR

Se un passo non entra in una sessione, è troppo grande: spezzalo. Segnali che è troppo
grande: tocca più di una decina di file, mescola scienza e infrastruttura, oppure per
descriverlo serve la parola "e anche".

### Il ciclo, ogni volta uguale

1. **Apri l'issue** — due righe: obiettivo e criterio di fatto. Se non riesci a scrivere il
   criterio di fatto, il passo non è chiaro.
2. **Sessione nuova, branch nuovo.** Mai riusare la sessione precedente: si porta dietro il
   contesto sbagliato.
3. **Il piano prima del codice.** «Leggi questi file, propone un piano, non scrivere
   niente finché non approvo.» Questo singolo passaggio intercetta la gran parte delle
   derive, perché è lì che si vede se ha capito o se sta per rifare mezzo repo.
4. **Test prima dell'implementazione** per ogni contratto nuovo.
5. **Commit piccoli**, messaggi in formato convenzionale.
6. **Verifica**: riferimento numerico verde, test verdi, esempi che girano.
7. **Aggiornare `STATO.md` e `CHANGELOG.md`**; per ogni decisione, l'ADR.
8. **PR, merge, chiudi la sessione.**

### Regole anti-deriva

- **Main sempre verde.** Niente branch che vivono settimane: se il lavoro è incompleto,
  entra spento dietro un interruttore.
- **Vietato il "già che ci sono".** Se Claude propone un miglioramento fuori tema — e lo
  farà, spesso sensato — diventa una issue, non un commit. La lista dei "non ora" è
  altrettanto importante della lista dei "sì".
- **Vietato cambiare numeri senza approvazione esplicita.** È la regola più importante di
  tutte ed è per questo che V1.1 viene prima di tutto.
- **Sessione lunga = sessione da chiudere.** Quando il contesto si riempie la qualità cala
  in modo non evidente: si comincia a riscrivere ciò che c'è già invece di modificarlo.
  Ci si ferma, si scrive lo stato, si riparte puliti.
- **A fine passo, sempre: «elenca cosa hai cambiato e perché».** Fa emergere le
  modifiche silenziose non richieste.

### Come scrivere il prompt di un passo

Sei elementi, sempre gli stessi:

1. **Contesto**: quali file leggere prima di rispondere
2. **Obiettivo**: una frase
3. **Vincoli**: cosa non deve cambiare
4. **Criterio di fatto**: come si verifica che è finito
5. **Fuori tema**: cosa non fare, esplicitamente
6. **Comando di verifica**: cosa lanciare alla fine

Meglio sempre «fai X e fermati» a «fai X, poi Y, poi Z». Le catene lunghe funzionano
finché non sbagliano il primo anello, e poi costruiscono tre passi sopra un errore.

### Ritmo e disciplina del tempo

Una pietra miliare ogni tre-sei settimane, ognuna con la sua demo. Se un passo sfora del
doppio, non è un problema di velocità: è che il passo era mal definito. Ci si ferma e
lo si ri-taglia, senza accelerare.

V1.1 e V2 sono circa un terzo del lavoro complessivo e sbloccano tutto il resto.
Se in qualunque momento serve fermarsi per mesi, meglio farlo **dopo** V2.2: lì lo strumento
è già generico e già validato su casi reali, ed è il punto in cui vale di più rispetto a
quanto è costato.

### Le tre trappole di questo progetto in particolare

1. **Rifattorizzare a lungo prima di avere il secondo caso.** Il design sbagliato si vede
   solo col secondo caso. Per questo V2.2 è presto e non alla fine.
2. **Il grande rifacimento del dashboard.** È il pezzo più grosso e più fragile: per schede,
   dietro interruttori, mai tutto insieme, e mai mescolato ad altro lavoro nello stesso
   branch.
3. **Rincorrere la piattaforma condivisa prima che il motore sia stabile.** È la trappola
   più costosa perché sembra progresso. Gli invarianti della Parte 1 sono lì apposta: tengono
   la porta aperta a costo zero, così la si può ignorare per sei mesi senza pentirsene.
