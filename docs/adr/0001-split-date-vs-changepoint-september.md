# 0001 — Perché SPLIT_DATE è giugno 2016 mentre changepoint.py colloca la rottura a settembre

## Contesto

`common.py::SPLIT_DATE` (stima rank-based, usata come confine pre/post ovunque nel pacchetto)
è fissato al 2016-06-01. `changepoint.py`, applicando la procedura QLR/AR(1) alla serie
mensile EC50, colloca invece la rottura intorno a settembre 2016. Sono due stime dello stesso
regime-shift, ottenute con metodi diversi (uno rank-based più semplice, l'altro parametrico e
consapevole dell'autocorrelazione), e non coincidono esattamente.

## Alternative valutate

- **Adottare la stima QLR come nuovo `SPLIT_DATE`**: avrebbe richiesto ricalcolare ogni numero
  pubblicato che dipende dal taglio pre/post, con relativo cambio di valori di riferimento.
- **Unificare forzando le due stime a coincidere**: non ha senso statistico — sono stime
  indipendenti di metodi diversi, forzarle a coincidere nasconderebbe l'incertezza reale sulla
  data esatta della rottura.
- **Tenere `SPLIT_DATE` come riferimento primario e la stima QLR come conferma indipendente
  secondaria**, senza riconciliarle in un unico numero — scelta adottata.

## Decisione

`SPLIT_DATE = 2016-06-01` resta l'unico riferimento usato da tutti i moduli per il taglio
pre/post. La stima di `changepoint.py` (~settembre 2016) è riportata e interpretata come
conferma indipendente che la rottura cade nella stessa finestra, non come un valore da
riconciliare. I risultati pubblicati sono insensibili alla scelta esatta entro l'intervallo
giugno-settembre (calo EC50 43.0% con giugno, 43.2% con settembre).

## Conseguenze

Chi legge il codice senza il contesto del paper vede un'apparente incoerenza tra `common.py`
e `changepoint.py`. Serve un commento esplicito che rimandi a questo ADR — **fatto** (ADR-0007):
`SPLIT_DATE` non è più una costante di `common.py` ma `ResponseSpec.split_date`, con il
rimando scritto nel commento YAML di `study.yaml` accanto al valore. Non unificare le due
stime senza discuterne prima.
