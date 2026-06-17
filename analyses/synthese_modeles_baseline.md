# Comparaison Initiale Des Modeles

- Jeu de donnees: data/raw/dataset_telemed.csv
- Scenario teste: full - tabulaire complet + texte
- Validation: validation croisee stratifiee a 5 plis sur l'ensemble d'entrainement.
- Jeu de test: 20%, stratifie, random_state=42.
- Meilleur modele initial: xgboost
- Modele sauvegarde: models/triage_model.joblib

Cette premiere comparaison sert a obtenir un point de depart fiable sur le scenario le plus complet. Le choix final ne doit pas encore etre considere comme definitif, car le sujet impose ensuite une analyse plus metier autour des urgences vitales.

## Resultats

| scenario   | model               |   cv_accuracy |   cv_f1_weighted |   cv_recall_class_2 |   cv_critical_errors |   cv_very_critical_errors |   test_accuracy |   test_f1_weighted |   test_recall_class_2 |   test_critical_errors |   test_very_critical_errors |   test_inference_ms_per_row |
|:-----------|:--------------------|--------------:|-----------------:|--------------------:|---------------------:|--------------------------:|----------------:|-------------------:|----------------------:|-----------------------:|----------------------------:|----------------------------:|
| full       | logistic_regression |        0.9459 |           0.9461 |              0.9297 |                   83 |                        13 |          0.9494 |             0.9492 |                0.9322 |                     20 |                           6 |                      0.0002 |
| full       | random_forest       |        0.9339 |           0.9339 |              0.9052 |                  112 |                        22 |          0.9405 |             0.9404 |                0.9085 |                     27 |                           4 |                      0.0198 |
| full       | xgboost             |        0.949  |           0.9489 |              0.884  |                  137 |                         5 |          0.9549 |             0.9547 |                0.8983 |                     30 |                           2 |                      0.0033 |

## Matrices De Confusion Sur Le Jeu De Test

### logistic_regression

Les lignes correspondent aux vraies classes [0, 1, 2]; les colonnes correspondent aux classes predites [0, 1, 2].

[[991, 20, 0], [34, 648, 28], [6, 14, 275]]

### random_forest

Les lignes correspondent aux vraies classes [0, 1, 2]; les colonnes correspondent aux classes predites [0, 1, 2].

[[979, 28, 4], [37, 649, 24], [4, 23, 268]]

### xgboost

Les lignes correspondent aux vraies classes [0, 1, 2]; les colonnes correspondent aux classes predites [0, 1, 2].

[[993, 18, 0], [26, 667, 17], [2, 28, 265]]

## Prochaine Etape

Le modele xgboost obtient le meilleur score global sur ce premier test. Cependant, logistic_regression obtient un meilleur recall sur la classe 2 (0.9322) et produit 20 erreurs critiques. La suite doit donc comparer tous les scenarios, puis optimiser explicitement la detection des urgences vitales.