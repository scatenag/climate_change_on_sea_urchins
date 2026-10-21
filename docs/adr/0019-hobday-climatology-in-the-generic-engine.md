# 0019 — La climatologia delle ondate di calore del motore generico segue Hobday et al. (2016)

## Contesto

Il 21/10/2026 il proprietario ha chiesto di misurare, senza cambiare codice, che cosa fa `mhw_detection.py` rispetto al metodo di
Hobday et al. (2016) che il manoscritto dichiara di seguire, e di far seguire Hobday fedelmente al motore generico. La misura
(fuori da git, `drafts/misura-soglia-hobday-2026-10-21/DOSSIER.md`) sulla fixture congelata di Livorno (SST giornaliera
2003-01-01 → 2026-06-30, baseline 2003–2012), confrontando `mhw_detection` con `legacy/marineHeatWaves.py` a parametri standard:

| | `mhw_detection` (pipeline di Livorno) | Hobday (`marineHeatWaves.py`) |
|---|---|---|
| Valori del percentile e della media | solo quelli di quel giorno dell'anno (10 con 10 anni) | tutti i valori entro ±5 giorni, in ogni anno della baseline (110) |
| Lisciatura | media mobile circolare di 11 giorni | media mobile circolare di 31 giorni |
| Giorno dell'anno | `dayofyear` (dopo il 28/2 slitta di uno negli anni bisestili) | calendario di 366 giorni per (mese, giorno); il 29/2 interpolato |
| Evento | una sequenza che tollera buchi di ≤2 giorni e dura in tutto ≥5 | esaurimenti di ≥5 giorni consecutivi, poi uniti se separati da ≤2 giorni |
| Soglia (media su 365 date) | 0.217 °C **sotto** il riferimento (338 giorni su 365 più bassa) | — |
| Giorni della baseline sopra la propria soglia | 13.0% | 9.7% (≈10% per costruzione) |
| Eventi / giorni di ondata 2003–2026 | 109 / 2835 | 78 / 1837 |
| Moderate / Strong / Severe / Extreme | 70 / 22 / 13 / 4 | 54 / 23 / 1 / 0 |

Il «11-day window» del manoscritto descrive la finestra di raccolta dei valori di Hobday (±5 giorni): il codice di Livorno non la
usa, usa 11 giorni solo per lisciare. Con 10 valori il percentile interpolato è distorto verso il basso e i giorni di ondata
aumentano.

## Decisione

Il motore generico (costruttore del dataset e app) usa `mhw_hobday.py`, che riproduce Hobday con i parametri standard, tutti
argomenti:

- giorno dell'anno del calendario a 366 giorni per (mese, giorno), 29/2 pari alla media del 28/2 e del 1/3;
- 90° percentile e media di ogni giorno sui valori entro ±5 giorni (i giorni vicini della serie; un vicino fuori dalla serie non
  esiste) in ogni anno della baseline; poi media mobile circolare di 31 giorni;
- superamento = temperatura strettamente sopra la soglia; evento = almeno 5 giorni consecutivi; due eventi separati da al più 2
  giorni sono uno (i giorni di mezzo gli appartengono);
- categoria = piano di (T − clima)/(soglia − clima) al picco, al massimo 4.

Le **metriche** (intensità sopra la soglia per giorno, picco, cumulata, anomalia dal clima liscio, tabelle mensile e annuale)
restano quelle della pipeline e usano le sue funzioni `_build_event`, `to_monthly`, `to_annual`: cambia la soglia e la regola
degli eventi, non che cosa si misura. Non si fa il riempimento dei giorni mancanti (`maxPadLength`) né le ondate di freddo: la serie
deve avere un valore al giorno, come il costruttore già pretende. Il manifesto dichiara il metodo nella sezione `mhw` della
copertura.

**Verifica:** `tests/test_mhw_hobday.py` confronta soglia e clima stagionale sui 366 giorni (tolleranza 1e-9) e gli eventi uno
per uno (inizio, fine, categoria) con `legacy/marineHeatWaves.py` sulla fixture di Livorno (78 eventi) e su serie sintetiche con
anni bisestili, inizio a metà anno e baseline di 30 anni.

**La pipeline di Livorno non cambia**: `mhw_detection` e il golden master restano come sono, perché passare a Hobday cambierebbe
109 → 78 eventi e i numeri pubblicati (Tabella S3 compresa: con il catalogo di riferimento i test sono 79 e nessuno supera BH o
Bonferroni in nessuna delle 360 combinazioni di ordini ARIMA; la riga delle ondate severe non è valutabile perché ha un solo
evento severo). La decisione di allineare Livorno, e di come correggere la frase del manoscritto, resta del proprietario, dopo la
misura. Conseguenza: lo stesso dato dà cataloghi diversi nel motore generico e nella pipeline di Livorno; un confronto fra i due
non è un confronto a parità di metodo.

## Limiti noti

- **Residuo stagionale.** La media a 31 giorni appiattisce i massimi e i minimi del ciclo stagionale: l'anomalia mensile conserva
  una piccola componente stagionale (circa l'1% dell'ampiezza; 0.09 °C su una serie sintetica con ampiezza 8 °C e nessuna
  variabilità interannuale). Su una SST vera, con anomalie di decimi o unità di grado, è una frazione piccola; ma una risposta con
  un forte ciclo stagionale proprio può associarsi a quel residuo. Il passo 3 dichiara già che la CCF cerca solo l'associazione
  non stagionale (ADR-0018); la limitazione si aggiunge qui e non è corretta.
- La baseline minima (10 anni, ADR-0014) resta invariata: con la raccolta di ±5 giorni la soglia è meno rumorosa a parità di
  anni, ma i valori entro la finestra non sono indipendenti (giorni vicini della stessa stagione dello stesso anno).
- Una baseline che inizia nel mezzo di un anno o non contiene un giorno dell'anno è un errore, non un riempimento.

## Alternative

- Lasciare `mhw_detection` anche nel motore generico: rifiutata (decisione del proprietario); il motore generico non dichiarerebbe
  Hobday senza seguirlo.
- Cambiare anche la pipeline di Livorno: non adesso. Cambia i numeri pubblicati; decide il proprietario.
- Riscrivere le metriche con le definizioni di Hobday (intensità relative al clima): non fatto, cambierebbe che cosa significa
  ogni metrica rispetto a Livorno e alle analisi già scritte.
