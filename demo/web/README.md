---
title: PulmoEIT
emoji: 🫁
colorFrom: blue
colorTo: indigo
sdk: static
app_file: index.html
pinned: false
license: mit
short_description: Vrais enregistrements EIT, six méthodes de reconstruction
---

# PulmoEIT : voir respirer de vrais poumons

Démo interactive du projet [PulmoEIT](https://github.com/sheene123/pulmoeit).

De vrais enregistrements de tomographie d'impédance électrique (EIT), publiés dans les données
contribuées d'[EIDORS](https://eidors3d.sourceforge.net/data_contrib/) : nouveau-nés, enfant en SDRA
pendant une titration de la PEEP, adulte sain, porc avec air puis liquide dans la plèvre, porcelet
avant et après lavage pulmonaire.

- **Le même cycle respiratoire réel, six méthodes** : NOSER (la famille des moniteurs), le réseau de
  PulmoEIT, et quatre algorithmes d'EIDORS (GREIT, Gauss-Newton une étape et itératif, variation totale).
- **Ce que lirait un moniteur** : ventilation par couche et par côté, centre de ventilation, indice
  d'inhomogénéité, et comment ces mesures changent selon la méthode.
- **Titration de la PEEP** (méthode de Costa) sur les quatre paliers réels de l'enfant : zones
  collabées, zones surdistendues, PEEP proposée.
- **Essai contrôlé** : la méthode suit-elle les volumes d'air et de liquide injectés dans la plèvre ?

Page statique : les reconstructions sont calculées à l'avance (EIDORS v3.12 dans GNU Octave) et
lues par le navigateur. Prototype de recherche, pas un dispositif médical.
