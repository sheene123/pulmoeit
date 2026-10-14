# PulmoEIT — imagerie pulmonaire par tomographie d'impédance électrique, sur de vrais enregistrements

[![ci](https://github.com/sheene123/pulmoeit/actions/workflows/ci.yml/badge.svg)](https://github.com/sheene123/pulmoeit/actions/workflows/ci.yml)
[![Démo sur Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20D%C3%A9mo-Hugging%20Face-yellow)](https://huggingface.co/spaces/sheenee261/pulmoeit)

**Démo en ligne : https://huggingface.co/spaces/sheenee261/pulmoeit**

Le même cycle respiratoire **réel** reconstruit par **six méthodes** : NOSER (la famille des
moniteurs), un réseau de neurones, et quatre algorithmes de référence de la boîte à outils
[EIDORS](https://eidors3d.sourceforge.net) (GREIT, Gauss-Newton en une étape et itératif, variation
totale). Pour chaque méthode : ce que lirait un moniteur (ventilation par couche et par côté, centre
de ventilation, inhomogénéité), la titration de la PEEP par la méthode de Costa (zones collabées et
surdistendues), et un essai contrôlé où l'on connaît la réponse.

> Prototype de recherche. Ce n'est pas un dispositif médical.

## Pourquoi l'EIT, et pourquoi de vraies données

L'EIT injecte de faibles courants par une ceinture de 16 électrodes et mesure les tensions en surface.
On en reconstruit, plusieurs fois par seconde, la carte de ventilation des poumons : sans radiation,
au lit du patient, en continu. C'est un outil de réanimation (SDRA, réglage de la PEEP, pneumothorax).

Le problème inverse est très mal posé : l'image dépend de la méthode de reconstruction et de ses
réglages. En réanimation, le clinicien regarde surtout **les mesures** tirées de l'image (zones
collabées, PEEP proposée, répartition de la ventilation) plutôt que l'image elle-même. La question
de ce projet est donc : **sur de vrais patients, ces mesures changent-elles selon la méthode ?**

Le projet comparait d'abord des méthodes sur des simulations. Ce travail est archivé sous l'étiquette
[`archive-simulation`](https://github.com/sheene123/pulmoeit/tree/archive-simulation) ; il n'en reste que
le réseau, figé, pour le comparer aux méthodes classiques sur de vraies mesures.

## Données

Enregistrements publics des [données contribuées d'EIDORS](https://eidors3d.sourceforge.net/data_contrib/)
(licences Creative Commons avec attribution), appareil Goe-MF II, 16 électrodes, protocole adjacent :

| Enregistrement | Situation | Source |
|---|---|---|
| Nouveau-né, respiration spontanée | 10 jours, sur le ventre | I. Frerichs ; Heinrich et al., Intensive Care Med 2006 |
| Nouveau-né, ventilation mécanique | 58 s | I. Frerichs, `if-neonate-ventilated` |
| Enfant en SDRA | PEEP abaissée de 20 à 14 cmH₂O par paliers, après recrutement | G. K. Wolf, C. Gómez-Laberge et al., IEEE TMI 2012 |
| Adulte sain | au repos et après un effort | C. Gómez-Laberge, `cg_normal_breathing_2006` |
| Porc | 0 à 300 ml d'air puis de liquide injectés dans la plèvre droite | G. Hahn et al., Physiol Meas 2006 |
| Porcelet | sain, après lavage pulmonaire, après surfactant | P. Dargaville et al., Intensive Care Med 2010 |

Pour chaque enregistrement : ordre des 208 mesures vérifié sur le profil des tensions (corrélation
0,80 à 0,93 avec un modèle homogène), sens des électrodes d'après la documentation (vérifié sur
l'essai de la plèvre : l'air injecté à droite apparaît à droite pour les six méthodes), détection des
respirations, puis **cycle moyen** : 16 phases, mesures moyennées phase par phase sur toutes les
respirations. 15 cycles et 12 volumes de la plèvre, soit 237 images par méthode.

## Méthodes et réglages

| Méthode | Réglage (sans simulation) |
|---|---|
| NOSER | reconstruction linéaire en une étape, comme les moniteurs |
| Réseau PulmoEIT | U-Net qui corrige l'image NOSER, entraîné sur des simulations puis **figé** ([modeles/reseau.onnx](modeles/)) |
| EIDORS GREIT | facteur de bruit 0,5, valeur recommandée par EIDORS |
| EIDORS Gauss-Newton une étape | a priori laplacien, hyperparamètre choisi par EIDORS pour le même facteur de bruit |
| EIDORS Gauss-Newton itératif | même hyperparamètre, 5 itérations |
| EIDORS variation totale | principe de l'écart de Morozov : l'écart aux mesures égale le bruit mesuré d'une respiration à l'autre |

EIDORS v3.12 tourne dans GNU Octave (image Docker `gnuoctave/octave:9.2.0`), sur le même maillage,
les mêmes électrodes et le même protocole que les autres méthodes ([scripts/eidors_reel.m](scripts/eidors_reel.m)).

## Résultats

Résultats complets : [reports/reel.json](reports/reel.json).

**Les mesures d'un moniteur bougent peu selon la méthode classique choisie.** Entre les cinq méthodes
classiques, la part de ventilation à droite varie de 2,3 points (médiane sur les 15 cycles, au plus 6,5)
et le centre de ventilation de 2,0 points (au plus 3,8).

**Le réseau entraîné sur des simulations est la méthode la moins d'accord avec les autres.**
Corrélation pixel à pixel de chaque méthode avec les cinq autres, en fin d'inspiration :

| Méthode | Médiane (15 cycles) | Min – max |
|---|---|---|
| EIDORS Gauss-Newton une étape | 0,87 | 0,81 – 0,90 |
| EIDORS Gauss-Newton itératif | 0,86 | 0,80 – 0,90 |
| NOSER | 0,82 | 0,76 – 0,87 |
| EIDORS GREIT | 0,82 | 0,77 – 0,87 |
| EIDORS variation totale | 0,79 | 0,67 – 0,86 |
| Réseau PulmoEIT | **0,67** | 0,56 – 0,82 |

Avec le réseau, l'écart sur la part de ventilation à droite atteint 12 points (nouveau-né).

**Titration de la PEEP chez l'enfant en SDRA (méthode de Costa, 4 paliers réels).** Les six méthodes
proposent la même PEEP ; elles diffèrent sur la part de poumon jugée collabée au palier le plus bas, et
sur l'endroit où elles la placent. La démo montre, pour chaque palier et chaque méthode, la carte des
pixels collabés et surdistendus (plus de la moitié de leur ventilation perdue).

| Méthode | PEEP proposée | Collapsus à 14 cmH₂O | Pixels collabés dans le dos (14) | Surdistension à 20 cmH₂O |
|---|---|---|---|---|
| NOSER | 14,3 | 3 % | 100 % (5 pixels) | 52 % |
| Réseau PulmoEIT | 14,8 | 9 % | 82 % (22 pixels) | 45 % |
| EIDORS GREIT | 14,1 | 1 % | 0 % (2 pixels) | 46 % |
| EIDORS Gauss-Newton une étape | 14,2 | 3 % | 80 % (5 pixels) | 48 % |
| EIDORS Gauss-Newton itératif | 14,2 | 3 % | 60 % (5 pixels) | 48 % |
| EIDORS variation totale | 14,3 | 4 % | 59 % (17 pixels) | 54 % |

Chez un enfant couché sur le dos, on attend le collapsus dans le dos et la surdistension à l'avant :
c'est ce que montrent la plupart des méthodes. GREIT, la plus lisse, ne trouve presque pas de pixel
collabé et ne le localise pas.

**Essai contrôlé (porc, plèvre droite).** Corrélation entre le volume injecté et le changement
d'impédance de l'hémithorax droit :

| Méthode | Air injecté | Air retiré | Liquide injecté |
|---|---|---|---|
| NOSER | 0,999 | 0,995 | 0,986 |
| Réseau PulmoEIT | 0,995 | 0,988 | 0,951 |
| EIDORS GREIT | 0,999 | 0,995 | 0,986 |
| EIDORS Gauss-Newton une étape | 0,999 | 0,995 | 0,985 |
| EIDORS Gauss-Newton itératif | 0,999 | 0,991 | 0,992 |
| EIDORS variation totale | 0,996 | 0,996 | 0,981 |

Toutes les méthodes suivent les volumes. Avec 300 ml d'air, la part de ventilation du côté droit
baisse d'environ 49 % à 40 % pour les cinq méthodes classiques.

**Le plus souvent, c'est le modèle qui limite l'image, pas le bruit.** Pour la variation totale, le
principe de Morozov retient le plus petit hyperparamètre testé sur 14 des 16 groupes. Pour 11 d'entre
eux (adulte, enfant, porc, porcelet sain), même presque sans régularisation, l'écart aux mesures
(10 à 19 %) reste bien au-dessus du bruit mesuré (2 à 7 %) : la forme du thorax et la position des
électrodes, inconnues, pèsent plus que le bruit. Pour les nouveau-nés et le porcelet après lavage,
plus bruités, les deux sont du même ordre.

**Limites.** Pas de vérité terrain chez un être vivant ; un seul patient ou animal par situation ;
sens des électrodes non documenté pour deux enregistrements (nouveau-né ventilé, enfant en SDRA).
Ces comparaisons montrent comment le choix de la méthode change ce qu'un réanimateur lirait, pas
laquelle a raison.

## Reproduire

```bash
pip install -e ".[dev,reel]"
pytest                                      # tests : physique, conventions de l'appareil, indices, Costa
python scripts/preparer_reel.py             # environ 1 h 30 (EIDORS en conteneurs parallèles)
python -m http.server -d demo/web           # la démo en local
python scripts/deployer_space.py --space <utilisateur>/pulmoeit
```

Prérequis : Docker ; EIDORS v3.12 décompressé dans `~/outils_eit/eidors-v3.12` (GPL, hors du dépôt) ;
les archives des enregistrements, téléchargées depuis les pages EIDORS citées ci-dessus, dans
`~/pulmoeit_reel` (chemins dans `JEUX`, en tête de [scripts/preparer_reel.py](scripts/preparer_reel.py)).

## Organisation

```
src/pulmoeit/reel.py     lecture du Goe-MF II, ordre et sens des électrodes, respirations, cycle moyen
src/pulmoeit/peep.py     méthode de Costa (collapsus, surdistension, PEEP proposée), espaces silencieux
src/pulmoeit/fem.py      modèle direct par éléments finis (électrodes à impédance de contact)
src/pulmoeit/recon.py    NOSER
src/pulmoeit/metrics.py  indices de ventilation
scripts/                 préparation des données (Python + EIDORS dans Octave), publication de la démo
modeles/                 NOSER et réseau figés
demo/web/                page statique de la démo
```

## Suite

- [ ] Données de patients supplémentaires (plusieurs patients par situation) auprès des auteurs des enregistrements
- [ ] Mesures cliniques comparées à une référence indépendante (scanner, mécanique ventilatoire)
- [ ] Forme du thorax et position des électrodes estimées à partir des mesures (erreur de modèle)

## Licence

MIT. Les enregistrements restent sous la licence de leurs auteurs (voir les citations ci-dessus).
