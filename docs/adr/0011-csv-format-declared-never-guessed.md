# 0011 — Il formato del CSV della risposta si dichiara nella specifica, non si indovina

## Contesto

La sorgente CSV della risposta (M1.3) riceve file scritti da chi usa lo strumento, non dal foglio di
Livorno. Un CSV europeo usa `;` come separatore e la virgola decimale (`12,5`), e le date `31/01/2020`;
letto con le convenzioni inglesi darebbe valori e date sbagliati senza nessun segno. Ancora peggio per le
date: `01/02/2020` è il primo febbraio o il 2 gennaio, e nessuna analisi dei dati lo dice.

## Alternative valutate

- **Riconoscere il formato dai dati** e procedere. Scartata: sul separatore e sul decimale è quasi sempre
  giusto, ma per le date non c'è modo di essere sicuri, e un errore qui sposta le osservazioni di mesi.
- **Il formato nel file di studio, con l'ISO come valore predefinito, e un lettore che rifiuta ciò che non
  coincide**, aiutando con ciò che il file *sembra* essere. Scelta adottata.

## Decisione

- `ResponseCsvSourceSpec` dichiara `delimiter` (`,` `;` tab `|`), `decimal` (`.` `,`) e `date_format`
  (strptime, solo direttive `%Y %m %d %H %M %S %b %B` e separatori semplici, con l'anno a quattro cifre
  `%Y` e un mese: `%m`, oppure `%b`/`%B` per i nomi dei mesi, **letti in inglese** qualunque sia la lingua del
  sistema, con un lettore proprio perché quello di `strptime` segue il locale). L'anno a due cifre (`%y`) è
  rifiutato, perché il secolo lo deciderebbe il parser: il messaggio chiede di esportare con quattro cifre.
  Valori predefiniti: virgola, punto, ISO `%Y-%m-%d`. I formati che `suggest_format` può proporre sono tutti
  accettati dal validatore (un test li tiene uniti). Delimitatore e decimale non possono coincidere; la
  risoluzione giornaliera richiede `%d`.
- Il lettore (`response_csv.py`) è rigoroso: un file che non coincide con la dichiarazione è rifiutato, con
  riga, colonna, valore e ciò che il file sembra essere. Le date con l'ora non sono troncate; `nan`/`inf` e
  numeri non finiti sono rifiutati; un valore fuori dal proprio intervallo è rifiutato; due righe nello
  stesso periodo in un file già aggregato sono rifiutate. I problemi sono raccolti (fino a 20) perché si
  corregga il file una volta sola.
- `suggest_format()` dice che cosa il file sembra (separatore, decimale, formati di data che lo leggono tutto,
  e se più d'uno va bene: ambiguità dichiarata). Non decide: la guida dell'interfaccia (M1.15) lo usa per
  chiedere all'utente.
- Un valore vuoto è una misura mancante, tenuta mancante e contata; un mese senza righe non compare;
  senza intervallo nel file la serie non ne ha (nessun limite inventato).
- L'aggregazione delle righe per prova è `common.aggregate_period`, la stessa implementazione che sta dietro
  il foglio di Livorno (`aggregate_monthly` ne è ora un involucro, con gli stessi nomi di sempre).

## Conseguenze

- Un file in un formato che non è quello predefinito richiede tre righe in più nello `study.yaml`; in cambio
  nessun file è letto male senza che si sappia.
- Limiti dei file (10 MB, 200 000 righe) pronti per il caricamento non fidato (M1.10).
- L'estrazione del nucleo neutro dell'aggregazione è a deriva zero sul golden master; un test confronta
  l'involucro con una copia congelata della vecchia funzione, bit per bit.
