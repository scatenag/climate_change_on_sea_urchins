# Stato del progetto

> Si aggiorna alla fine di **ogni** sessione di lavoro, prima del merge.
> Tenerlo corto: se supera una pagina, sposta il dettaglio in un ADR o in una issue.

**Ultimo aggiornamento:** fine settembre 2026

---

## Dove siamo

**Versione corrente:** V1 · Il caso — passo **V1.1 Messa in sicurezza**

**Obiettivo del passo:** poter modificare qualunque cosa sapendo subito se un risultato
scientifico è cambiato. Da fine agosto si è aggiunto un obiettivo collegato: il pacchetto è
citato in un lavoro come mezzo di verifica indipendente, quindi deve riprodurre i numeri
pubblicati in modo verificabile.

**Fatto finora** (v1.4.0 e commit successivi)
- `SPLIT_DATE` esplicito al 2016-06-01, propagato a tutti i moduli
- modulo `changepoint.py` con procedura QLR/AR(1), primaria sulla serie mensile
- VIF per finestra in `thermal_legacy`
- ordinamento deterministico della sequenza grezza per (data, ID) — commit `cb333cd`,
  **non ancora incluso in un tag**

**Passo in corso:** consolidamento della versione citata dal paper, branch
`v1.5.0/consolidamento-paper`.

**Prossimo passo:** rete di sicurezza numerica generale. Il test di corrispondenza col lavoro
ne copre una parte, non tutta: resta da congelare il comportamento degli altri moduli.

---

## Questioni aperte

- Il manoscritto è già stato sottomesso? Le tre modifiche al testo — numeri della dose
  termica, frase in §3.5, Data availability — vanno in bozze oppure prima dell'invio.
- Quale seconda serie di risposta per validare V2.2. Nessuna candidata copre il ventennio:
  righting response, volume di uova, contenuto proteico e NRRT partono da zero punti.
- Nome definitivo del progetto e momento della rinomina del repository.
- Se le analisi in R restano una dipendenza opzionale o vengono portate in Python.

---

## Da non dimenticare

- **Le unità del CO₂ sono ancora sbagliate.** `spco2` di Copernicus è in Pascal, fattore
  9.8692. È l'item aperto più vecchio: rilevato ad agosto, mai corretto. Non tocca nessun
  risultato del lavoro (Spearman e changepoint invarianti, stress index standardizzato,
  forecast esclude il CO₂, speciazione usa il pH), ma il README dichiara ancora l'unità come
  non verificata.
- **L'auto-update non si tocca.** È una funzionalità voluta ed è ciò che il lavoro rivendica
  nelle conclusioni. I trigger a schedule girano solo sul branch di default: per congelare
  basta un tag, non serve fermare né isolare niente.
- **Un test sui valori pubblicati va scritto contro una fixture di dati congelata**, non
  contro `data/`, altrimenti la CI diventa rossa in permanenza appena la serie si allunga.
  L'invariante da proteggere è che il codice riproduca i numeri pubblicati a partire dai dati
  pubblicati.
- **La sequenza grezza delle prove non è un oggetto ben definito**: 59 delle 295 condividono
  la data, per via della registrazione a livello di mese. Con ordinamenti diversi F varia fra
  326 e 380 e la rottura salta fra giugno e settembre 2016. Risolto fissando l'ordinamento,
  ma è il motivo per cui la serie mensile è l'analisi primaria.
- `SPLIT_DATE` è giugno 2016 mentre il modulo changepoint riporta settembre: la stima
  rank-based cade a giugno e i risultati sono insensibili alla scelta entro l'intervallo
  (calo 43.0% contro 43.2%). Chi legge il codice senza il paper vede solo un'incoerenza:
  serve un commento in `config.py`.
- Il dashboard aggira il sistema di import di Python e usa un lock globale che serializza
  tutte le sessioni: prima di rifattorizzarlo, capire se la causa è ancora presente.
- Le analisi specifiche del caso (speciazione del rame, ondate di calore, legacy termica)
  hanno senso solo su un endpoint da metalli con pH, temperatura e salinità e serie
  giornaliera: non sono generiche.
- **Il paper promette che la pipeline è un modello adattabile da altri programmi di
  monitoraggio.** Oggi non lo è: `ADAPTING.md` dice che cambiare sito sono quattro costanti,
  cambiare indicatore no. È l'argomento più forte a favore di V2, e la pubblicazione gli dà
  una scadenza implicita.

---

## Versioni

| Versione | Passo | Titolo | Stato |
|---|---|---|---|
| **V1** · Il caso | V1.1 | Messa in sicurezza | in corso |
| **V2** · Il caso diventa configurazione | V2.1 | Specifica e astrazione della risposta | da iniziare |
| | V2.2 | Prova sul campo | da iniziare |
| **V3** · Molti casi, un confronto | V3.1 | Analisi intercambiabili | da iniziare |
| | V3.2 | Confronto e inferenza | da iniziare |
| | V3.3 | Installabile e usabile da altri | da iniziare |
| **V4** · Meta-tool | — | Organismi e risposte di natura diversa, vocabolari, multiverso | rimandata |
| **V5** · Istanza condivisa | — | Multi-utente | rimandata |
