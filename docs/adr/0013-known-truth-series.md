# 0013 — Serie a verità nota: il generatore dichiara ciò che ha fatto, il verdetto sta in un solo posto

## Contesto

L'uscita della fetta verticale (M1.6) è ritrovare un ritardo noto, su coordinate mai usate, senza interfaccia. Serve
una risposta di cui si conosca la dipendenza dalla SST e che abbia i caratteri della serie reale, altrimenti il
criterio verifica una serie facile. Il driver della CCF è l'anomalia mensile della SST (ADR-0012): su di essa si genera.

## Decisione

- **Generatore** (`known_truth.py`, `python -m climate_change_on_sea_urchins.known_truth make`): da una SST giornaliera
  scrive una cartella di studio (formato 2, CSV per prova, SST, `response.manifest.json`).
  `risposta(t) = livello + scala · (√q · z(−anomalia(t−k)) + √(1−q) · z(rumore AR(1) 0.3))`, con `q` la quota di varianza
  del valore mensile (50% di partenza), `k` il ritardo (3: né 0 né multiplo di 12; ammessi da 1 a 12 senza i multipli di
  12) e `z` la standardizzazione sui mesi dove il driver è noto. L'anomalia è quella che calcola il costruttore
  (`anomaly_mean`), non una ricostruzione. Segno negativo: la risposta scende quando la temperatura sale (`adverse_direction:
  decrease`). Una sola fonte di casualità, con il seme come parametro, estratta in ordine fisso: stesso seme, stessi byte.
- **Caratteri della serie reale, misurati sul foglio dei ricci il 16/10** (`ec50_sheets.csv`, 280 mesi): 42% dei mesi senza
  misura, in sequenze di lunghezza media 2.49 (le sequenze di mesi reali 3.4), da 1 a 6 prove per mese (1: 66, 2: 77, 3: 12,
  4: 2, 5: 5, 6: 1). La dispersione fra le prove di un mese (30% della scala) si somma alla quota di segnale, che riguarda il
  valore mensile. L'imputazione è dichiarata come quella di Livorno (finestra 12, 3 punti minimi, 2 passaggi): la CCF non la usa.
- **La dichiarazione accanto alla risposta** (`response.manifest.json`: seme, ritardo, quota di segnale, driver e linea di base,
  rumore, caratteri, intervallo dei mesi) è copiata in `run.json` (`response_manifest`) da `ccsu-run-study`, che non sa nulla di
  serie sintetiche: copia la dichiarazione di qualunque risposta che ne abbia una. Il risultato porta con sé ciò che si
  doveva trovare.
- **Verdetto** (`known_truth.verify`, `…known_truth verify RUN_DIR`, uscita 0 se ritrovato): il massimo di |r| sui ritardi
  eseguiti cade sul ritardo noto, con il segno atteso (`r_toward_adverse` > 0) e p corretto BH < 0.05 a quel ritardo. Gli altri
  ritardi che superano BH sono riportati. Scrive `verification.json`.
- **A mano, con credenziali** (`scripts/verify_slice.py`, workflow `verify_slice.yml`, etichetta `verifica-fetta` o avvio manuale,
  stesse precauzioni di `verify_download.yml`): su due punti mai usati dal progetto (Adriatico centrale 42.5N 15.0E, Ionio
  38.5N 18.5E) scarica la SST vera, genera con tre semi, esegue `ccsu-run-study` e verifica. Provato senza rete con un
  sostituto del download (`tests/test_verify_slice_script.py`).

## Misura (16/10, fixture di Livorno, generatore → `ccsu-run-study` → verdetto, 100 semi per caso)

| Quota di segnale | Ritardo | Ritrovato | Massimo di \|r\| sul ritardo | Altri ritardi BH<0.05 (media) |
|---|---|---|---|---|
| 0 (indipendente) | 3 | 0 | 5 | 0.09 (6 casi su 100 con almeno un ritardo) |
| 20% | 3 | 68 | 94 | 0.18 |
| 50% | 3 | 100 | 100 | 0.20 |
| 80% | 3 | 100 | 100 | 0.45 |
| 50% | 7 | 99 | 99 | 0.30 |
| 50% | 11 | 99 | 100 | 0.19 |

Con una risposta indipendente i 6 casi su 100 con un ritardo significativo sono il limite già registrato (ADR-0000 voce 20).

## Conseguenze

- Il criterio non dipende dalla fortuna di un seme: i test usano otto semi fissi per caso e la misura sopra dice la quota attesa.
- Il generatore chiede una SST con la linea di base per intero; scaricare dalla data d'inizio della linea di base (nota di M1.14).
- Sul segnale al 20% il ritardo si ritrova in due casi su tre: è il limite di potenza con circa 100 coppie, non un difetto.
