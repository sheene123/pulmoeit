---
title: PulmoEIT
emoji: 🫁
colorFrom: blue
colorTo: indigo
sdk: static
app_file: index.html
pinned: false
license: mit
short_description: Imagerie pulmonaire EIT, NOSER contre réseau appris
---

# PulmoEIT : imagerie pulmonaire par tomographie d'impédance électrique

Démo interactive du projet [PulmoEIT](https://github.com/sheene123/pulmoeit).

- **Visualiser** : un patient simulé (poumons sains, SDRA, intubation sélective, pneumothorax, épanchement
  pleural), la vérité terrain, la reconstruction linéaire NOSER des moniteurs actuels et le réseau PostUNet.
- **Comparer** : corrélation avec la vérité et indices cliniques (GI, centre de ventilation, quadrants).
- **Stresser le modèle** : bruit, ceinture décalée, thorax irrégulier ou aplati, électrode décollée ; le contrôle
  qualité des mesures signale les données inhabituelles et l'électrode suspecte.
- **Tester vos mesures** : fichier CSV `v_ref,v_insp` de 208 tensions (16 électrodes, protocole adjacent).

Tout s'exécute dans le navigateur : la physique et NOSER en Python ([Pyodide](https://pyodide.org)), le réseau
avec [ONNX Runtime Web](https://onnxruntime.ai), à partir du même fichier ONNX que l'API du projet.
Résultats en simulation uniquement ; prototype de recherche, pas un dispositif médical.
