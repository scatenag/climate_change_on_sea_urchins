# 0005 — Perché `requirements.txt` resta permissivo e `requirements-lock.txt` congela l'ambiente della release

## Contesto

`requirements.txt` fissa intervalli di versione permissivi (es. `pandas>=2.0,<3.0`), ciascuno
motivato da un incidente reale (soprattutto installazioni non pinnate su Streamlit Cloud che
risolvevano a major inattese). Questo serve a un'installazione normale, che deve continuare a
ricevere aggiornamenti compatibili. Ma per una release i cui `results/` sono citati come
verificabili da un manoscritto, "compatibile secondo l'intervallo dichiarato" non è la stessa
garanzia di "produce risultati identici byte per byte" — rumore numerico BLAS/LAPACK, o un
cambio di default in un ottimizzatore di statsmodels/scipy entro l'intervallo dichiarato,
possono spostare i risultati anche restando dentro i vincoli.

## Alternative valutate

- **Pinnare `requirements.txt` stesso a versioni esatte**: scartata — congelerebbe per sempre
  le dipendenze del pacchetto installabile, vanificando lo scopo degli intervalli `>=`/`<` e
  bloccando aggiornamenti di sicurezza/bugfix di routine.
- **Nessun blocco d'ambiente**: scartata — rende "verificato riproducibile" una dichiarazione
  non verificabile.
- **Un file di lock esatto separato (`pip freeze`), accanto al `requirements.txt` permissivo**:
  scelta adottata.

## Decisione

`requirements.txt` resta l'insieme permissivo, mantenuto a mano, effettivamente usato per
installare il pacchetto. `requirements-lock.txt` è un `pip freeze` dell'ambiente esatto in cui
i `results/` di una data release sono stati generati e verificati riproducibili (verificato il
2026-09-14 per v1.5.0: rieseguire la pipeline in quell'ambiente esatto riproduce `results/`
byte per byte).

## Conseguenze

`requirements-lock.txt` va rigenerato (e la sua riproducibilità riverificata) a ogni release
che necessita di questa garanzia, non a ogni commit. È materiale di documentazione/verifica,
non il percorso di installazione — l'utente normale continua a fare `pip install -e .` contro
l'insieme permissivo.
