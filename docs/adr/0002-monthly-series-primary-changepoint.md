# 0002 — Perché la serie mensile è l'analisi primaria del changepoint e la sequenza ordinale è secondaria

## Contesto

`changepoint.py` applica la procedura QLR/AR(1) a due rappresentazioni della serie EC50: la
serie mensile aggregata (163 date uniche) e la sequenza ordinale a piena risoluzione delle 295
determinazioni individuali. Molte prove nella sequenza ordinale condividono la stessa
`Datetime` (per larga parte del record è registrato solo il mese), quindi il loro ordine
relativo entro un mese non è determinato dai dati stessi.

## Alternative valutate

- **Sequenza ordinale come analisi primaria**: scartata — l'ordine entro le date condivise
  non è ben definito. Verificato: phi varia tra 0.246 e 0.315 e il mese di rottura vincente si
  divide 65/35 tra settembre e giugno su 300 permutazioni casuali dell'ordine entro data.
- **Solo la serie mensile**: scartata — perderebbe la granularità a livello di prova usata dal
  contrasto descrittivo di sezione 3.1.
- **Sequenza ordinale con ordinamento fissato come primaria comunque**: scartata — anche con
  un ordinamento deterministico fissato, il mese esatto della rottura non è un risultato
  stabile, solo l'anno lo è.

## Decisione

La serie mensile è l'analisi di changepoint **primaria** (riproducibile, ordine non
ambiguo). La sequenza ordinale, ordinata per (Datetime, ID) per determinismo (vedi ADR-0003),
è un controllo **secondario** il cui unico risultato stabile è l'anno della rottura (2016), non
il mese specifico.

## Conseguenze

Chi cita un numero di changepoint deve specificare da quale rappresentazione proviene. Il
manoscritto stesso segue questa stessa impostazione primaria/secondaria. Non promuovere la
sequenza ordinale a fonte primaria senza rivedere questo ADR.
