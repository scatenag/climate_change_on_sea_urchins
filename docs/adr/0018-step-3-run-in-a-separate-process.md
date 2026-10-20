# 0018 — Passo 3 dell'app: l'analisi gira in un processo separato che riceve solo file

## Contesto

Il passo 3 dei propri dati (MA.4) fa eseguire `ccsu-run-study` sullo studio costruito nel passo 2 e mostra il risultato. Le regole
sono quelle di MILESTONE-M1.md §3 e §5.1: il calcolo in un processo separato che non vede mai le credenziali (né come argomenti
né nell'ambiente), nessuna analisi lenta sotto il lock globale, un contesto per sessione. `common` carica lo studio di Livorno
all'importazione (ADR-0000 voce 19): un processo per studio è anche ciò che lo rende sicuro.

## Decisione

- **Il processo** è `python -m climate_change_on_sea_urchins.run_study STUDIO --inputs CARTELLA --out CARTELLA`, avviato dalla
  cartella della sessione, in un gruppo di processi proprio (per poterlo fermare), senza shell. I suoi argomenti sono percorsi
  assoluti. **Il suo ambiente si costruisce da un elenco di nomi ammessi** (`PATH`, `LANG`, `PYTHONPATH`, `VIRTUAL_ENV` e poco
  altro) più un `HOME` nuovo e vuoto: nessuna variabile Copernicus, nessun segreto del processo dell'app, nessun `CCSU_STUDY`.
  Un test lo verifica con segreti messi nell'ambiente dell'app e li cerca negli argomenti e nell'ambiente del figlio.
- **L'avanzamento** si legge da `progress.json` (scritto in modo atomico dal comando a ogni passo) con un frammento di
  Streamlit che si riesegue ogni due secondi finché il processo gira, **fuori dal lock globale**. Se il processo muore senza
  scrivere l'avanzamento, il motivo è la coda del suo registro.
- **Al più 2 analisi contemporanee sul server** (il servizio ha poca memoria: una analisi usa pandas, statsmodels e il
  rilevamento delle ondate); la terza è invitata ad aspettare. Il conteggio legge i file `run.pid` delle cartelle di sessione e
  controlla che il processo viva.
- **Si può interrompere**; e l'analisi si ferma e si butta quando l'utente cambia scelta, cancella i dati, scarica di nuovo la
  temperatura o costruisce un altro studio (i risultati non corrisponderebbero più).
- **Il risultato** si descrive dai valori: lags provati, eseguiti e non eseguiti; quali superano la correzione di
  Benjamini-Hochberg e con quale segno rispetto al verso dichiarato come peggiorativo; per una serie di prova a ritardo noto, il
  verdetto di `known_truth.verify`. Per dati veri non si dà mai un verdetto fisso. I **limiti** sono scritti ogni volta: cerca
  solo l'associazione non stagionale; coppie di mesi consecutivi reali; p di Spearman corretti su ritardi con differenze
  autocorrelate (6–8% di analisi con un ritardo che passa su risposte indipendenti, ADR-0000 voce 20); un'associazione non è una
  causa e un ritardo che non passa non prova l'assenza di effetto.
- `streamlit>=1.37` (i frammenti che si riesegono da soli).

## PROVVISORIA: il numero massimo di analisi e l'intervallo di aggiornamento

Due analisi contemporanee e un aggiornamento ogni due secondi sono valori scelti senza averli misurati su Streamlit Cloud.
**Alternative:** una sola alla volta (più sicura per la memoria, più attesa), quattro (rischio di esaurire la memoria). Da
rivedere con la misura di MA.5 (tempo e memoria dell'analisi sul servizio). Reversibile, nessun effetto su Livorno.

## Conseguenze

- L'analisi di una persona non rallenta le altre sessioni e non tiene il lock; le credenziali non arrivano mai al processo.
- Su un'installazione non editabile il comando non girerebbe (ADR-0012, ADR-0000 voce 19): su Streamlit Cloud il repository è
  installato in modo editabile, quindi va.
- Il registro del figlio sta nella cartella della sessione e si cancella con essa.
