# Rapport d'évaluation PulmoEIT

## test

| méthode | RMSE | corrélation | erreur GI | erreur CoV (pts %) | erreur régionale (pts %) |
|---|---|---|---|---|---|
| baseline | 0.0710 | 0.847 | 0.158 | 0.64 | 2.12 |
| model | 0.0174 | 0.990 | 0.017 | 0.25 | 0.76 |

| scénario | n | corr. baseline | corr. modèle |
|---|---|---|---|
| healthy | 282 | 0.847 | 0.993 |
| dorsal_atelectasis | 161 | 0.854 | 0.984 |
| one_lung | 90 | 0.874 | 0.994 |
| pneumothorax | 107 | 0.816 | 0.989 |
| pleural_effusion | 160 | 0.844 | 0.989 |

Dérive QC (KS) : D = 0.032, p = 0.79, dérive = False

## test_shift

| méthode | RMSE | corrélation | erreur GI | erreur CoV (pts %) | erreur régionale (pts %) |
|---|---|---|---|---|---|
| baseline | 0.0901 | 0.765 | 0.154 | 1.16 | 3.57 |
| model | 0.0425 | 0.938 | 0.038 | 0.74 | 2.19 |

| scénario | n | corr. baseline | corr. modèle |
|---|---|---|---|
| healthy | 276 | 0.770 | 0.949 |
| dorsal_atelectasis | 157 | 0.748 | 0.919 |
| one_lung | 78 | 0.794 | 0.947 |
| pneumothorax | 116 | 0.754 | 0.936 |
| pleural_effusion | 173 | 0.764 | 0.936 |

Dérive QC (KS) : D = 0.755, p = 5.8e-201, dérive = True

## Contrôle qualité des mesures (électrode décollée)

- AUC détection : 1.000
- Taux de détection au seuil : 100.0% (fausses alarmes : 0.6%)
- Localisation de l'électrode fautive : 99.8%

## Quality gates

- min_corr : OK
- min_corr_gain_over_baseline : OK
- max_cov_mae : OK
- min_fault_auc : OK
- passed : OK
