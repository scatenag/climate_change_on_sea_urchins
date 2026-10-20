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
  Il limite di dimensione è doppio: il widget di caricamento rifiuta oltre 10 MB (`server.maxUploadSize` in
  `.streamlit/config.toml`; il valore predefinito di Streamlit è 200 MB) e l'app lo ricontrolla; le righe sono limitate dal lettore
  alla costruzione. L'intestazione è limitata a 200 colonne di al più 100 caratteri.
- **La serie di prova** usa `known_truth` (ADR-0013) sulla SST appena scaricata: seme, ritardo e quota di segnale sono
  mostrati e modificabili; è dichiarata «un controllo dello strumento, non una misura».
- `response_csv.decode_text` è la decodifica e il controllo del lettore, estratti perché l'app legge il file prima di chiedere
  (stesso comportamento, stessi messaggi).

## PROVVISORIA: che cosa il passo 2 non offre ancora

Risoluzione solo **mensile** (le date di un file per misura possono avere il giorno, ma la serie di studio è mensile),
**nessuna imputazione** (assente salvo dichiarata, D8; la CCF non la usa), colonne dei controlli e risoluzione giornaliera
assenti. **Alternative:** offrirle ora (rifiutata: ogni scelta è metodo e va nella specifica con il suo motivo, e il
piano le assegna a M1.15). Reversibile, nessun effetto su Livorno.

## Dopo la rilettura indipendente

- Un file con una virgoletta mai chiusa non fa più cadere l'app (`csv.Error` è un problema del file con la riga); una data fuori
  dagli anni 1678–2261 che pandas può tenere è un problema con la sua riga (prima era un errore non catturato o un messaggio
  senza riga).
- Lo studio non è detto «pronto» se la serie non ha mesi nel periodo della temperatura o ha meno coppie di mesi consecutivi
  (10) di quelle che il comando richiede: si dice quante ne ha e che nessun ritardo si potrà provare. I conteggi sono dei soli
  mesi con un valore.
- Lo studio costruito resta legato alle risposte e al file che lo hanno prodotto: se cambiano, lo si dice («costruiscilo di
  nuovo»); un nuovo download del passo 1 lo toglie. La serie di prova si fa sotto il lock globale (rilevamento delle ondate) e
  rifiuta un seme negativo; una serie con un buco nel passo 1 ferma il passo 2 con il motivo.
- Test aggiunti per ogni domanda obbligatoria (decimale, riga, formato della data, natura, verso), per le opzioni (intervallo,
  «aumenta», «già mensile», tabulazione), per il secondo file e per la ricostruzione fallita.

## Conseguenze

- Una persona che carica un CSV risponde a circa dieci domande: è il prezzo di non leggere niente in silenzio.
- Il test dell'app verifica che le domande non abbiano risposta preimpostata e che ogni risposta sbagliata sia rifiutata con il
  motivo e prima di scrivere lo studio; `streamlit.testing` non può caricare un file vero, quindi i test sostituiscono
  `st.file_uploader`.
