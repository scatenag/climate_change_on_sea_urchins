# Anatomia della sorgente dati (foglio Google EC50)

Rilevato il 12/08/2026 ispezionando il foglio che la pipeline già usa
(`sheet_id` in `examples/livorno_paracentrotus/study.yaml`, esportato come CSV → viene letta
**solo la prima scheda**, l'unica descritta qui).
Ultima modifica del foglio al momento del rilevamento: 30/07/2026, coerente con il recupero dei
controlli negativi; anteriore alla correzione della prova 224 del 28/9/2026 (vedi sotto).

Serve come materiale di progetto per V2.1: la specifica della serie di risposta va disegnata
contro questa realtà, non contro un caso ideale.

---

## Scheda 1 — dati grezzi per test

Colonne: `ID, DATE, EC50, UL, LL, pos, neg, SRT DATE, Replica I/II/III CTRL negativo (malformed)`

### Trappola principale: `pos` e `neg` non sono controlli

Verificato su più righe: `pos = UL − EC50` e `neg = EC50 − LL`. Sono le due semi-ampiezze
dell'intervallo di confidenza, **asimmetriche**.

Chiunque legga `neg` capisce "controllo negativo", incluso un assistente che lavora sul
codice. È il caso d'uso che giustifica da solo tutta la scelta di dichiarare la semantica
delle colonne nella specifica invece di dedurla dai nomi.

Conseguenza di progetto: la specifica deve portare `ci_low` e `ci_high` separati, mai un
singolo "± errore". L'incertezza qui è asimmetrica e in alcuni anni molto (riga 30:
`+5.26 / −10.06`).

### I controlli negativi sono campi del test, non una serie parallela

Sono tre conteggi di larve malformate per singolo test, sulla stessa riga della risposta.
Non hanno date proprie: la loro data è quella del test.

Questo semplifica il progetto della funzione "serie di controllo": va modellata come
**campi dell'osservazione**, non come serie temporale indipendente. Più pulito e più
generale — vale per qualunque laboratorio che affianchi un controllo a ogni misura.

I tre replicati danno anche una stima della variabilità intra-test, utile sia per la carta
di controllo sia, eventualmente, per pesare le osservazioni.

Copertura: presenti su circa due terzi delle righe nella porzione ispezionata, assenti
sulle altre. La specifica deve prevedere che il controllo ci sia solo su parte delle
osservazioni.

### Verificato con chi gestisce il foglio (settembre 2026)

- Le due righe di gennaio 2020 (ID 224 e 225) sono due saggi distinti. Nella prova 224 il secondo
  replicato del controllo negativo era registrato come `1`: era un refuso, corretto nel foglio
  sorgente in `14, 11, 14` (verificato il 28/9/2026). La fixture congelata
  `tests/fixtures/paper_mpb_2026/` conserva il valore originale, perché riproduce i dati della
  release v1.5.0. Storia completa in `examples/livorno_paracentrotus/NOTES.md`.
- Le malformazioni sono contate su 100 larve esaminate: i valori sono già percentuali, non
  conteggi nudi.

### Risoluzione temporale ambigua

La maggior parte delle date è il primo del mese, cioè un marcatore di mese, non una data
reale. Alcune sono date vere (16-Jan-08, 26-Mar-10). Dal 2013 compaiono sequenze come
1/1, 1/2, 1/3 dello stesso mese con giorni consecutivi: plausibilmente **indice di
replicato codificato nel giorno**, non tre test in tre giorni.

Oggi è innocuo perché la pipeline aggrega a mese. Diventa un problema il giorno in cui
qualcuno userà la risoluzione giornaliera. La specifica deve **dichiarare** la risoluzione
della sorgente (mese) invece di lasciar credere a una precisione giornaliera che non c'è.

### Osservazioni multiple per mese

Da 1 a 6 test nello stesso mese. La regola di aggregazione (oggi: media, con conteggio in
`EC50_n`) va dichiarata nella specifica. Nota: `EC50_n` è il numero di **test aggregati**,
non la numerosità biologica — altra ambiguità di nome da non ereditare.

---

## Cosa ne discende per la specifica della serie di risposta

Campi che la specifica deve poter dichiarare, tutti giustificati da qualcosa che sta
davvero in questo foglio:

- mappatura esplicita delle colonne (`neg` docet), mai deduzione dai nomi
- incertezza asimmetrica: `ci_low` e `ci_high` distinti
- risoluzione temporale della sorgente, dichiarata
- regola di aggregazione per periodo, dichiarata, con conteggio delle osservazioni aggregate
- campi di controllo allegati all'osservazione, presenti solo su una parte delle righe
- natura della serie: qui *proxy dello stato riproduttivo di una popolazione*, non biomarker
