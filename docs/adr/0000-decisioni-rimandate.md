# 0000 — Decisioni rimandate

## Stato

Vivo — raccoglie le decisioni volutamente non ancora prese. Non anticiparle: quando una
viene sciolta, va rimossa da qui e, se non banale, diventa un ADR proprio.

## Perché questo file

`CLAUDE.md` vieta di aggirare le decisioni architetturali già prese ma anche di anticipare
quelle rimandate. Questo file è l'elenco di queste ultime, con il contesto minimo per capire
perché sono aperte e cosa serve per chiuderle. Fonte: sezione "Questioni aperte" di
`docs/roadmap/STATO.md`.

---

## 1. Tempistica delle modifiche al manoscritto

**Domanda:** il manoscritto è già stato sottomesso? Le tre modifiche al testo individuate —
i numeri della dose termica, la frase in §3.5, la sezione Data availability — vanno applicate
nelle bozze già in circolazione oppure devono rientrare prima dell'invio definitivo.

**Contesto:** le modifiche nascono da un disallineamento tra quanto il codice produce
attualmente e quanto il testo del manoscritto dichiara. Finché lo stato della sottomissione
non è chiaro, non è possibile stabilire se si tratta di una correzione su un documento ancora
in bozza o di un errata da gestire con la rivista.

---

## 2. Seconda serie di risposta per V2.2

**Domanda:** quale serie di risposta biologica usare per validare il passo V2.2 (prova sul
campo) della roadmap.

**Contesto:** nessuna delle candidate valutate copre l'intero ventennio della serie EC50
attuale — righting response, volume delle uova, contenuto proteico e NRRT partono tutte da
zero punti storici. La scelta condiziona quanto V2.2 potrà davvero validare l'astrazione
introdotta in V2.1.

---

## 3. Nome definitivo del progetto

**Domanda:** quale sarà il nome definitivo del progetto, e quando avverrà la rinomina del
repository.

**Contesto:** il repository porta ancora il nome del caso di studio originale
(`climate_change_on_sea_urchins`), mentre la roadmap prevede che il progetto diventi uno
strumento generico oltre quel singolo caso. Rinominare il repository prima che l'astrazione
sia effettiva rischierebbe di promettere una genericità non ancora raggiunta.

---

## 4. Destino delle analisi in R

**Domanda:** le analisi in R (attualmente solo DLNM, dopo che SEA e mixed-effects sono stati
portati in Python) restano una dipendenza opzionale del progetto, oppure vengono anch'esse
portate in Python.

**Contesto:** il porting di SEA e mixed-effects in Python ha già ridotto la superficie R a un
solo modulo (`scripts/mhw_lag_analysis.R`, DLNM). Non è ancora deciso se valga la pena
completare il porting per eliminare la dipendenza da R, o se mantenerla come opzionale sia
sufficiente per gli obiettivi della versione corrente.

---

## 6. `label` non sanificato per l'uso nei nomi di file

**Domanda:** `ResponseSpec.label` (`v2.1/response-abstraction`) è una stringa di
visualizzazione usata anche per costruire nomi di file e altre identità di output (etichette
di riga/colonna, campi `"variable"`/`"series"`, chiavi di dizionario). Un secondo caso
potrebbe dichiarare un `label` con spazi o caratteri non adatti a un nome di file (es.
"Righting response (s)"). Va sanificato prima dell'uso in un nome di file? Con quale regola?

**Contesto:** per il caso Livorno non si pone: `label == "EC50"`, già un nome di file valido,
quindi non blocca questo passo (`dist_EC50.csv` resta identico). Rimandata a quando arriva un
secondo caso reale (V2.2) che esponga il problema concretamente, invece di indovinare una
regola di sanificazione senza un caso su cui verificarla.

---

## 7. Finestra sull'intero record resa esplicita

**Domanda:** quando uno studio dichiara finestre temporali (`StudySpec.windows`), anche la
finestra sull'intero record dovrebbe diventare una finestra dichiarata, con la propria
directory `results/<study_id>/<window_id>/`, invece di restare implicita.

**Contesto:** V2.2 (PR 5a/5b) fa percorrere a una sola esecuzione della pipeline tutte le
finestre dichiarate, con i risultati in directory sorelle `results/<study_id>/<window_id>/`.
Uno studio senza finestre (Livorno, oggi) resta com'è: risultati in `results/<study_id>/`,
senza sottodirectory, che concettualmente è una finestra implicita sull'intero record.
Formalizzarla adesso sposterebbe i file appena spostati da ADR-0008 e produrrebbe deriva nel
golden master senza alcun guadagno per un caso che non ha finestre da confrontare. Ma in uno
studio che dichiara finestre, l'intero record è il termine di paragone naturale della tabella
di confronto: lasciarlo implicito, in una directory di forma diversa dalle sorelle, è
un'asimmetria da sciogliere quando arriva il primo studio con finestre.
