# Consolidamento della versione citata dal paper

Obiettivo: un artefatto archiviato che riproduce i numeri del lavoro, **senza toccare
l'aggiornamento automatico**, che è ciò che rende vivo lo strumento ed è anche quello che il
paper rivendica nelle conclusioni.

---

## Perché non serve fermare niente

Il workflow di auto-update ha trigger `schedule` (cron giornaliero alle 06:00 e mensile il
giorno 5). I trigger a schedule girano **solo sul branch di default**, e il workflow committa
su `main`. Non tocca i tag.

Un tag è un puntatore immutabile a un albero, e Zenodo archivia le *release*, che sono basate
sui tag. Quindi l'artefatto citabile e congelato **esiste già per costruzione**: non va
protetto da niente, perché niente lo minaccia.

L'unico rischio reale è una collisione durante il lavoro: il cron giornaliero che atterra su
`main` mentre sei a metà. Si risolve lavorando su un branch e mettendo il tag sulla testa del
branch **prima** di fondere. Finestra di esposizione zero, automazione mai ferma.

Un branch congelato dedicato è opzionale. Avete già il precedente, `sartori-2023-supplement`
più il tag `v0.1.0-sartori-2023`: se volete un posto navigabile per il lavoro MPB ha senso,
ma come comodità. L'oggetto citabile resta il tag — un branch si può spostare, un tag per
convenzione no.

---

## La conseguenza che cambia il progetto del test

Se l'automazione continua a girare, un test che verifica i numeri del paper rieseguendo la
pipeline sui dati correnti diventa rosso man mano che la serie si allunga. Una CI
permanentemente rossa è peggio di nessuna CI.

La soluzione è separare i due assi. **Congela la vendemmia di dati come fixture** — sono 320
KB in tutto — e fai che il test rieseguisca il codice di analisi *su quella*. L'invariante
diventa:

> il codice di oggi riproduce ancora i numeri pubblicati a partire dai dati pubblicati

che è la cosa che conta davvero. Una modifica al codice che rompe la riproducibilità fa
fallire la build; la crescita dei dati non la sfiora. Il dashboard e `main` continuano a
vivere sui dati nuovi senza interferenze.

---

## La sequenza

### 1. Branch di lavoro

`v1.5.0/consolidamento-paper`. Nessun workflow da disattivare.

### 2. Correggi le unità del CO₂

Va prima del tag: cambia dati committati, e non ha senso archiviare un errore noto. `spco2`
di Copernicus è in Pascal, fattore 9.8692. Vanno sistemati lo script di ingestione e i suoi
commenti, la tabella delle unità nel README e la nota che dichiara l'unità non verificata.

Verifica, non dare per scontato, che nulla di citato nel manoscritto cambi. L'attesa è che
non cambi niente: Spearman e changepoint sono invarianti per trasformazione monotona,
l'indice di stress standardizza prima della decomposizione, il forecast esclude il CO₂ per
collinearità con il pH, la speciazione usa pH, temperatura e salinità. Se qualcosa si muove,
è un bug da capire prima di proseguire.

### 3. Una sola esecuzione pulita

Tutti i seed fissati, tutto `results/` rigenerato in un colpo. Da qui in avanti ogni file
proviene dalla stessa vendemmia.

### 4. Confronta i numeri con il manoscritto

Il pacchetto è la fonte, il testo è il derivato: si aggiorna il paper, non il codice.

Divergenze note, tutte nel blocco della dose termica:

| finestra | ρ detrended | p detrended | p parziale | coefficiente |
|---|---|---|---|---|
| 12m | −0.222 vs −0.22 | 0.0044 vs 0.0047 | 0.0274 vs 0.027 | −0.0282 vs −0.029 |
| 24m | −0.257 vs −0.26 | 0.0009 vs 0.0009 | 0.0044 vs 0.0042 | −0.0505 vs −0.051 |
| 36m | −0.204 vs −0.20 | 0.0091 vs 0.0092 | 0.0754 vs 0.073 | −0.0390 vs −0.040 |
| 48m | −0.099 | 0.2084 vs 0.20 | 0.3038 vs 0.29 | −0.0274 vs −0.028 |
| 60m | −0.051 | 0.5139 vs 0.51 | 0.7882 vs 0.78 | −0.0089 vs −0.009 |

Nessuna conclusione cambia: ogni superamento e ogni fallimento di Bonferroni resta identico.

Già coerenti: speciazione (Δ pH −0.013528, n 94/69, calo 42.93%, quote 3.576% e 0.523%,
residuo p 7.87×10⁻²⁵), changepoint mensile (φ 0.0595, F 318.96, settembre 2016, IC
giugno-ottobre), VIF (1.412, 1.628, 1.805, 2.306, 2.724).

Da controllare, non ancora verificati: §3.1, il controllo negativo di §3.6, la tabella S2
delle soglie, il conteggio dei 4 test su 89 di §3.4.

### 5. Congela dati e corrispondenza

`tests/fixtures/paper_mpb_2026/` con copia dei CSV di `data/`, e `tests/test_paper_values.py`
che riesegue l'analisi su quella fixture e verifica ogni numero citato, con tolleranze
dichiarate e motivate.

In testa al file un commento che dice a quale release e a quale DOI corrisponde, e che il suo
scopo è far fallire la CI se una modifica al codice rompe la riproducibilità di quanto
pubblicato.

Questo passo vale più del tag.

### 6. Tag, release, DOI

`v1.5.0` sulla testa del branch — non v1.4.1, perché la correzione del CO₂ cambia dati.
Release su GitHub con le note di cosa è cambiato rispetto a v1.4.0. Zenodo produce il DOI.

### 7. Fondi su main

L'automazione riprende senza aver mai smesso. Da qui `main` e la release divergeranno, ed è
giusto così.

### 8. Dichiara la divergenza, come caratteristica

Nel README e nel dashboard: l'analisi pubblicata corrisponde alla release v1.5.0 e al suo
DOI, la pagina riflette i dati più recenti.

---

## Cosa cambia nel manoscritto

- I numeri della dose termica in §3.3, allineati alla vendemmia unica.
- §3.5: il pacchetto adesso fissa l'ordinamento per data e poi ID e restituisce F = 329.5 con
  rottura a settembre 2016. L'intervallo 326-380 resta corretto come misura della sensibilità
  all'ordinamento, ma va detto quale valore produce il pacchetto archiviato.
- Data availability: versione, DOI e **data di taglio dei dati**. Formulazione possibile, che
  trasforma la divergenza da imbarazzo in argomento:

  > The analysis reported here corresponds to release v1.5.0 (DOI …), which archives the
  > environmental and bioassay records as retrieved on <data>. The pipeline continues to
  > ingest new data automatically, so the live dashboard reflects a longer series than the
  > one analysed here; this is the mode of use the package is designed for.

  La conclusione del lavoro dice già che la pipeline offre un modello adattabile «as new data
  accumulate». Che continui ad aggiornarsi non è una crepa da spiegare: è la dimostrazione di
  quella frase.

---

## Prompt per Claude Code

```
Leggi CLAUDE.md e docs/roadmap/STATO.md se esistono; se non esistono, dimmelo e fermati.

CONTESTO
Il pacchetto e' citato in un lavoro in sottomissione come mezzo di verifica indipendente, ma
non ha una vendemmia unica: cu_speciation_summary.json e' del 4 settembre,
thermal_legacy_summary.json del 5, changepoint_ec50.json del 7, e i numeri del manoscritto
vengono da un'esecuzione ancora diversa. Le divergenze sono alla terza cifra e non cambiano
nessuna conclusione, ma vanno chiuse.

VINCOLO CHE VIENE PRIMA DI TUTTI: l'aggiornamento automatico dei dati NON si tocca. E' una
funzionalita' voluta ed e' cio' che il lavoro rivendica nelle conclusioni. Lavoriamo su un
branch, il cron gira solo su main, e il tag lo mettiamo sulla testa del branch prima di
fondere.

Procedi nell'ordine, fermandoti a ogni passo per la mia approvazione. Prima di scrivere
codice, leggi i moduli coinvolti e proponimi un piano.

PASSO 1 - correggi le unita' del CO2
spco2 di Copernicus e' in Pascal, non in microatmosfere: fattore 9.8692. Correggi al momento
dell'ingestione, con la costante nominata e commentata, usata in un punto solo. Sistema i
commenti dello script, la tabella delle unita' nel README e la nota che dichiara l'unita' non
verificata: quella nota va sostituita dalla spiegazione della conversione.
Rifai il controllo incrociato con data/data.csv applicando la stessa conversione a entrambe:
il rapporto tornava 0.99 perche' anche quella serie era in Pascal.
Aggiungi un test che verifica che la serie CO2 stia fra 350 e 500 microatmosfere.
ATTESO: nessun risultato citato nel manoscritto cambia. Spearman, changepoint, indice di
stress (standardizzato prima della decomposizione), forecast (che il CO2 lo esclude gia' per
collinearita' con il pH) e speciazione (che usa pH, temperatura e salinita') sono invarianti.
Cambiano solo valori assoluti, medie, deviazioni, trend in unita' e scale dei grafici.
Elencami cosa cambia e cosa no. Se si muove qualcosa che dovrebbe restare fermo, FERMATI.

PASSO 2 - una sola esecuzione pulita
Verifica che ogni sorgente di casualita' abbia un seed fissato ed esposto. Rigenera tutto
results/ in un'unica esecuzione.

PASSO 3 - estrai i numeri e confrontali col manoscritto
Produci un file di confronto con, per ogni quantita' citata: valore nel manoscritto, valore
prodotto ora, differenza. I valori del manoscritto te li do io quando arrivi qui. Non
modificare nulla per farli coincidere: la fonte e' il pacchetto, il testo si adegua.

PASSO 4 - congela dati e corrispondenza
Copia i CSV di data/ in tests/fixtures/paper_mpb_2026/ (circa 320 KB in tutto).
Scrivi tests/test_paper_values.py che RIESEGUE l'analisi su quella fixture, non sui dati
correnti, e verifica ogni numero citato nel lavoro con tolleranze dichiarate e motivate
(strette per i deterministici, piu' larghe dove c'e' bootstrap).
Questo e' il punto centrale: l'invariante da proteggere e' "il codice di oggi riproduce ancora
i numeri pubblicati a partire dai dati pubblicati". I dati correnti possono crescere quanto
vogliono senza far fallire il test. Scritto contro data/, la CI diventerebbe rossa in
permanenza appena arrivano dati nuovi, e sarebbe peggio di non averlo.
In testa al file, un commento con release e DOI di riferimento e lo scopo del test.

PASSO 5 - rilascio
CHANGELOG.md con la voce v1.5.0: correzione unita' CO2, vendemmia unica, fixture e test di
corrispondenza col lavoro pubblicato, ordinamento deterministico della sequenza grezza (fatto
in cb333cd ma non ancora rilasciato). Tag v1.5.0 sulla testa di QUESTO branch, poi fusione su
main.

PASSO 6 - dichiara la divergenza
Nel README e nel dashboard: l'analisi pubblicata corrisponde alla release v1.5.0 e al suo DOI,
la pagina riflette i dati piu' recenti, e le due cose divergeranno man mano che la serie si
allunga.

VINCOLI
- Il PASSO 1 cambia risultati committati ed e' voluto: elencameli file per file PRIMA di
  aggiornarli e aspetta la mia approvazione.
- Non toccare la logica di nessuna analisi. Questo branch consolida, non migliora.
- Non toccare i workflow.
- Nessuna dipendenza nuova, niente refactoring fuori tema.
```

---

## Tempi

Passi 1 e 2: mezza giornata. Il passo 3 richiede te, non Claude Code: qualcuno deve leggere il
manoscritto con l'elenco dei numeri accanto e spuntarli uno per uno. Il passo 4 è quello che
vale di più, e conviene farlo adesso che il lavoro è fresco e sai quali numeri contano.
