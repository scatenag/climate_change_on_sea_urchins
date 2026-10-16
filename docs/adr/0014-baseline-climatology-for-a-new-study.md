# 0014 — La baseline climatologica proposta a uno studio nuovo: 1991–2020, modificabile

## Contesto

Il costruttore del dataset (M1.4) calcola la climatologia delle ondate di calore e l'anomalia mensile della SST
(il driver della CCF generica, ADR-0012) rispetto a una baseline di anni dichiarata nello studio, e vuole la
baseline coperta per intero dalla SST. Il download di uno studio nuovo parte quindi dall'inizio della baseline,
non dall'inizio della serie di risposta. Quale baseline suggerire è una scelta di metodo (decisione del
proprietario, 16/10/2026).

## Alternative valutate

- **A. 1991–2020 per tutti** (la norma climatologica a 30 anni), modificabile. Scelta adottata.
- **B. I primi 10 anni completi della serie di risposta** (come lo studio di Livorno, 2003–2012). Scartata: dieci
  anni sono pochi per un 90° percentile, la baseline dipende dalla serie e non è confrontabile fra studi.
- **C. I primi 30 anni del prodotto.** Scartata: termina in anni lontani da quelli della maggior parte delle serie
  e la data d'inizio dipende dal catalogo.

## Decisione

- Il valore proposto è 1991–2020, una costante con nome (`DEFAULT_BASELINE`) che rimanda a questo ADR. Il motivo
  scientifico: Hobday et al. (2016, *Progress in Oceanography* 141: 227–238): *"a period of 30 years is
  recommended to define a baseline temperature climatology, wherever possible"* (frase riportata dal
  proprietario).
- **Nessuna data di copertura nel codice.** L'inizio e la fine della copertura del prodotto si leggono dal
  catalogo a ogni esecuzione, come già si fa per la fine (M1.2). Se la copertura non comprende il 1991, l'app lo
  dice e propone **il primo trentennio coperto**; se è più corta di trent'anni propone tutto il periodo coperto
  con l'avviso della lunghezza.
- Modificabile dall'utente entro gli anni coperti. **Errore** sotto i 10 anni (decisione provvisoria di MA.2, ADR-0016: con
  pochi anni il 90° percentile di un giorno è fatto dei valori stessi); avvisi: baseline più corta di 15 anni (soglia
  rumorosa) e baseline che non finisce prima dell'ultimo anno coperto (nessun anno «dopo»).
- Il download parte dal più antico fra l'inizio della baseline e il primo mese della serie di risposta meno
  `analysis.max_lag_months` (i ritardi usano la temperatura dei mesi precedenti), entro l'inizio della copertura.
- Livorno resta con la baseline che dichiara (2003–2012): è una scelta diversa, scritta nel suo studio, e il
  golden master non cambia.

## Conseguenze

- Il download copre più giorni (circa 14 000 dal 1991 invece di circa 8 700 dal 2003): il tempo si misura in
  MA.2 (non è stimato qui).
- Con una baseline ferma nel passato e un mare che si scalda, i giorni di ondata recenti crescono anche per il solo
  effetto del trend. Per l'anomalia mensile usata dalla CCF (differenze prime) l'effetto è piccolo; per le
  metriche delle ondate di calore che entrano con M1.9 è la questione aperta di ADR-0000, voce 21.
