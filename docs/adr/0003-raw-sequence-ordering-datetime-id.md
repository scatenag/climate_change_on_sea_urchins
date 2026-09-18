# 0003 — Perché l'ordinamento della sequenza grezza è fissato per (Datetime, ID)

## Contesto

Circa un terzo delle 295 righe di `data/ec50_raw.csv` condivide la `Datetime` con almeno
un'altra riga, perché molte prove sono registrate a livello di mese, non di giorno.
Ordinare per sola `Datetime` lascia le righe con la stessa data in un ordine che dipende
dall'implementazione dell'ordinamento, non garantito stabile tra versioni di pandas.

## Alternative valutate

- **Ordinare per sola `Datetime`**: scartata — l'ordine non è riproducibile. Verificato:
  l'F del test QLR variava tra 326 e 380 e il mese di rottura saltava tra giugno e settembre
  a seconda di come i pareggi venivano risolti.
- **Ordine arbitrario/casuale**: stesso problema, scartata.
- **Ordinare per (Datetime, ID)**, con ID = ordine di riga originale del foglio sorgente —
  l'unico tiebreaker disponibile che non è a sua volta arbitrario. Scelta adottata.

## Decisione

`data/ec50_raw.csv` è ordinato per (Datetime, ID). Ogni modulo che lo legge (`changepoint.py`,
`negative_control.py`, il contrasto grezzo di `period_split.py`) richiede e verifica questo
ordinamento, fallendo rumorosamente se la colonna `ID` manca.

## Conseguenze

Questo rende l'ordine riproducibile (stesso input → stesso output, sempre) ma **non** rende il
mese di rottura risultante scientificamente significativo di per sé — vedi ADR-0002. Qualunque
futura modifica all'ingestione di `ec50_raw.csv` deve preservare (o ri-derivare correttamente)
`ID` come ordine di riga del foglio sorgente.
