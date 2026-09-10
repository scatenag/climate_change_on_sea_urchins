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
