# Proposition de sujet de thèse

## Imagerie par impédance électrique portable pilotée par l'IA : reconstruction robuste au transfert simulation → réel et cycle de vie MLOps certifiable pour le monitoring pulmonaire

---

## 1. Contexte

La ventilation mécanique est le principal traitement de l'insuffisance respiratoire
aiguë. Le syndrome de détresse respiratoire aiguë (SDRA) concerne à lui seul environ
10 % des admissions en réanimation (étude LUNG SAFE, Bellani et al., *JAMA* 2016). Régler
un ventilateur (pression expiratoire positive, volume courant) demande de savoir **où va
l'air** dans le poumon : les régions collabées ne sont pas recrutées, d'autres sont
surdistendues.

L'imagerie de référence, le scanner, est irradiante, ponctuelle et impose de transporter
un patient instable. La **tomographie d'impédance électrique (EIT)** apporte la même
information de manière continue, au lit, sans radiation et pour un coût faible. Il suffit
d'une ceinture de 16 à 32 électrodes et d'une électronique de la taille d'un boîtier.
Plusieurs moniteurs sont commercialisés et le consensus TREND (Frerichs et al.,
*Thorax* 2017) en standardise les indices cliniques. Des versions **portables et
portées** (ceintures textiles, monitoring prolongé) apparaissent.

## 2. Problème

L'imagerie EIT reste limitée par quatre verrous :

1. **Problème inverse sévèrement mal posé.** Les moniteurs utilisent des reconstructions
   linéaires régularisées (NOSER, GREIT) qui donnent des images floues, surtout au
   centre du thorax.
2. **Sensibilité à la géométrie.** Une erreur sur la forme du thorax ou sur la position
   de la ceinture crée des artefacts de même ordre de grandeur que le signal utile.
3. **Pas de vérité terrain in vivo.** Les méthodes apprises sont entraînées sur des
   simulations, et le passage *sim-to-real* n'est presque jamais quantifié.
4. **Pas de garde-fou.** Une électrode décollée ou un mouvement produit une image
   fausse mais plausible, et aucun outil ne signale au clinicien qu'il ne doit pas s'y
   fier.

À cela s'ajoute un verrou **d'ingénierie et de réglementation**. Un modèle d'IA embarqué
dans un dispositif médical relève du règlement (UE) 2017/745 (MDR), de l'AI Act
(règlement (UE) 2024/1689, systèmes à haut risque) et de la norme IEC 62304. Il ne peut
être mis à jour qu'avec une traçabilité complète ; aux États-Unis, la FDA encadre ces
mises à jour par un *Predetermined Change Control Plan* (PCCP). Les pratiques MLOps
(Sculley et al., *NeurIPS* 2015) sont rarement pensées pour ces contraintes, encore
moins pour une flotte d'appareils portables.

## 3. Pourquoi c'est une niche

Le sujet est à l'intersection de trois communautés qui se parlent peu :

- la communauté EIT, petite (quelques dizaines d'équipes dans le monde) et centrée sur
  les méthodes mathématiques et le matériel ;
- la communauté de l'apprentissage profond en imagerie médicale, très large mais
  concentrée sur le scanner, l'IRM et l'histologie ;
- la communauté MLOps et réglementaire, qui traite surtout des logiciels d'aide au
  diagnostic sur images statiques, pas des dispositifs embarqués à flux continu.

Les travaux de reconstruction EIT par apprentissage existent (Hamilton & Hauptmann, *IEEE
TMI* 2018, entre autres), mais la robustesse au transfert, la détection des défauts de
mesure et le cycle de vie du modèle dans l'appareil restent très peu étudiés.

## 4. Questions de recherche

**QR1 — Robustesse au transfert.** Comment apprendre une reconstruction qui reste fiable
quand la forme du thorax, la position des électrodes et l'électronique diffèrent de la
simulation ?
*Hypothèse :* conserver l'opérateur physique dans le réseau (post-traitement d'une
reconstruction linéaire, schémas itératifs déroulés) généralise mieux qu'un inverse
appris de bout en bout, à condition d'entraîner avec une *domain randomisation*
réaliste (Tobin et al., *IROS* 2017).

**QR2 — Savoir quand ne pas faire confiance.** Peut-on détecter, directement dans
l'espace des mesures et avant toute reconstruction, les trames hors distribution et
localiser l'électrode défaillante ? Peut-on fournir une incertitude calibrée par pixel
et par indice clinique (ensembles profonds, prédiction conforme) ?

**QR3 — Évaluer ce qui compte au lit.** Les métriques d'image (RMSE, corrélation) sont
mal corrélées à la décision clinique. Il faut évaluer et optimiser directement sur les
indices utilisés au lit : index d'inhomogénéité globale (Zhao et al., *Intensive Care
Med* 2009), centre de ventilation, distribution régionale, recrutement et surdistension.

**QR4 — MLOps certifiable pour dispositif embarqué.** Quelle chaîne garantit la
traçabilité donnée → modèle → appareil, bloque automatiquement un modèle qui régresse,
surveille la dérive sur une flotte d'appareils et autorise des mises à jour dans le
cadre d'un PCCP ?

## 5. Méthodologie et plan de travail (36 mois)

| Lot | Période | Contenu |
|---|---|---|
| **WP1 — Simulation et données** | M1–M9 | Passage du simulateur 2D actuel à la 3D (maillages thoraciques issus de scanners publics) ; modèle d'électrodes complet ; calibration sur données ouvertes de cuve (archive EIT 2D de Kuopio, Hauptmann et al. 2017 ; Kuopio Tomography Challenge 2023) |
| **WP2 — Reconstruction robuste** | M6–M18 | Comparaison inverse appris / post-traitement / schémas déroulés ; conditionnement sur la géométrie mesurée ; banc de robustesse (forme, ceinture, bruit, électronique) |
| **WP3 — Confiance** | M12–M24 | Détection hors distribution dans l'espace des mesures ; localisation des défauts d'électrode ; incertitude calibrée (ensembles, prédiction conforme) évaluée sur des défauts réels provoqués en cuve |
| **WP4 — MLOps embarqué** | M18–M30 | Quantification INT8 et latence sur microcontrôleur ou SoC ; surveillance de dérive sur flotte simulée ; protocole de mise à jour type PCCP ; dossier de traçabilité IEC 62304 et AI Act |
| **WP5 — Validation clinique** | M24–M36 | Étude rétrospective avec un service de réanimation (EIT et scanner simultanés) ; accord entre indices EIT et scanner ; rédaction |

## 6. État actuel (preuve de concept de ce dépôt)

Le dépôt implémente déjà une version 2D complète de la chaîne :

- simulateur éléments finis du *complete electrode model*, validé par la réciprocité et
  par différences finies sur le jacobien ;
- cinq scénarios cliniques (sain, atélectasie dorsale, intubation sélective,
  pneumothorax, épanchement pleural) et une *domain randomisation* de la géométrie, de la
  ceinture, des contacts et du bruit ;
- baseline NOSER et PostUNet, évalués sur un jeu de test de même distribution **et** sur
  un jeu de test décalé (thorax plus variables, ceinture mal placée, bruit plus fort) ;
- contrôle qualité des mesures avec détection et localisation d'électrode décollée,
  test de dérive ;
- pipeline DVC, suivi MLflow, *quality gates*, export ONNX avec test de parité, *model
  card*, API et CI.

Les résultats courants sont dans [reports/report.md](../reports/report.md).

## 7. Valorisation visée

- Revues : *IEEE Transactions on Medical Imaging*, *Physiological Measurement*,
  *Medical Image Analysis*.
- Conférences : MICCAI, IEEE ISBI, IEEE EMBC, conférence internationale annuelle sur
  l'EIT.
- Logiciel et jeux de données simulées ouverts, protocole de benchmark de robustesse
  réutilisable.

## 8. Risques et parades

| Risque | Parade |
|---|---|
| Accès tardif aux données cliniques | Données ouvertes de cuve dès WP1 ; partenariat clinique recherché dès la première année |
| Écart 2D / 3D | Passage 3D planifié en WP1 ; la chaîne MLOps est indépendante de la dimension |
| Méthodes apprises instables sous décalage | Architectures qui gardent la physique, détection hors distribution et repli sur la reconstruction linéaire |
| Complexité réglementaire | Cadrage dès WP4 à partir des textes (MDR, AI Act, IEC 62304, guide PCCP de la FDA) |

## 9. Profil et encadrement envisagés

Co-encadrement entre un laboratoire d'imagerie médicale et de problèmes inverses et un
service de réanimation. Compétences mobilisées : problèmes inverses et éléments finis,
apprentissage profond, ingénierie logicielle et MLOps (conteneurs, CI/CD,
observabilité).

## Références

- Adler A. et al. *GREIT: a unified approach to 2D linear EIT reconstruction of lung images.* Physiological Measurement, 2009.
- Adler A., Boyle A. *Electrical impedance tomography: tissue properties to image measures.* IEEE Transactions on Biomedical Engineering, 2017.
- Bellani G. et al. *Epidemiology, patterns of care, and mortality for patients with acute respiratory distress syndrome in intensive care units in 50 countries* (LUNG SAFE). JAMA, 2016.
- Cheney M. et al. *NOSER: an algorithm for solving the inverse conductivity problem.* International Journal of Imaging Systems and Technology, 1990.
- Frerichs I. et al. *Chest electrical impedance tomography examination, data analysis, terminology, clinical use and recommendations: consensus statement of the TRanslational EIT developmeNt stuDy group.* Thorax, 2017.
- Hamilton S. J., Hauptmann A. *Deep D-bar: real-time electrical impedance tomography imaging with deep neural networks.* IEEE Transactions on Medical Imaging, 2018.
- Hauptmann A. et al. *Open 2D electrical impedance tomography data archive.* arXiv:1704.01178, 2017.
- Sculley D. et al. *Hidden technical debt in machine learning systems.* NeurIPS, 2015.
- Somersalo E., Cheney M., Isaacson D. *Existence and uniqueness for electrode models for electric current computed tomography.* SIAM Journal on Applied Mathematics, 1992.
- Tobin J. et al. *Domain randomization for transferring deep neural networks from simulation to the real world.* IROS, 2017.
- Zhao Z. et al. *Evaluation of an electrical impedance tomography-based global inhomogeneity index for pulmonary ventilation distribution.* Intensive Care Medicine, 2009.
- Règlement (UE) 2017/745 relatif aux dispositifs médicaux ; règlement (UE) 2024/1689 sur l'intelligence artificielle ; IEC 62304 *Medical device software — Software life cycle processes*.
