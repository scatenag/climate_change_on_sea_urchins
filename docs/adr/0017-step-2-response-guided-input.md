# 0017 — Passo 2 dell'app: la risposta si chiede, non si indovina

## Contesto

Il passo 2 dei propri dati (MA.3) porta nell'app la serie di risposta: un CSV che l'utente carica, oppure una serie di
prova a ritardo noto. Il lettore (M1.3, ADR-0011) rifiuta ciò che non coincide con la dichiarazione e `suggest_format`
dice che cosa il file sembra; l'app deve chiedere all'utente, non decidere per lui: `01/02/2020` è il primo febbraio o il
2 gennaio, e un decimale con la virgola letto come punto sposta i valori senza segno.

## Decisione

- **Ogni domanda sul formato parte senza risposta** (separatore, colonna della data, colonna del valore, separatore decimale,
  formato della data) e il pulsante di costruzione le richiede tutte. L'app mostra che cosa il file *sembra* (separatore dalla
  prima riga, decimale dalle cifre, i formati di data che leggono tutto il file) e, se la data è ambigua, lo dice («solo tu sai
  quale»). Un separatore sbagliato si nota subito: l'intestazione risulta di una colonna sola.
- **Le domande della specifica**, tutte senza risposta preimpostata: che cosa contiene una riga (una misura, da mediare nel
  mese, oppure un valore già mensile), nome, unità, natura della grandezza, verso in cui il valore si muove quando gli
  organismi stanno peggio, estremi dell'intervallo (facoltativi). Nome e unità sono limitati a caratteri semplici (finiscono in
  messaggi resi come Markdown).
- **Il risultato è una cartella di studio** nella sessione: `study.yaml` formato 2 scritto con `yaml.safe_dump` e riletto con il
  caricatore dello studio (ogni validatore), `response.csv` (i byte caricati, con nome fisso: il nome del file caricato non
  decide niente), la SST e il suo manifest. `ccsu-run-study` la esegue (passo 3, MA.4). Il file `study.yaml` è scaricabile.
- **Il testo che viene dal file** (le celle citate nei problemi del lettore) si mostra come codice, mai come Markdown.
  Limiti di dimensione e righe sono quelli del lettore, controllati prima di leggere il file.
- **La serie di prova** usa `known_truth` (ADR-0013) sulla SST appena scaricata: seme, ritardo e quota di segnale sono
  mostrati e modificabili; è dichiarata «un controllo dello strumento, non una misura».
- `response_csv.decode_text` è la decodifica e il controllo del lettore, estratti perché l'app legge il file prima di chiedere
  (stesso comportamento, stessi messaggi).

## PROVVISORIA: che cosa il passo 2 non offre ancora

Risoluzione solo **mensile** (le date di un file per misura possono avere il giorno, ma la serie di studio è mensile),
**nessuna imputazione** (assente salvo dichiarata, D8; la CCF non la usa), colonne dei controlli e risoluzione giornaliera
assenti. **Alternative:** offrirle ora (rifiutata: ogni scelta è metodo e va nella specifica con il suo motivo, e il
piano le assegna a M1.15). Reversibile, nessun effetto su Livorno.

## Conseguenze

- Una persona che carica un CSV risponde a circa dieci domande: è il prezzo di non leggere niente in silenzio.
- Il test dell'app verifica che le domande non abbiano risposta preimpostata e che ogni risposta sbagliata sia rifiutata con il
  motivo e prima di scrivere lo studio; `streamlit.testing` non può caricare un file vero, quindi i test sostituiscono
  `st.file_uploader`.
