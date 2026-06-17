# Comparaison Des Scenarios

- Jeu de donnees: data/raw/dataset_telemed.csv
- Validation: validation croisee stratifiee a 5 plis sur l'ensemble d'entrainement.
- Jeu de test: 20%, stratifie, random_state=42.
- Meilleur modele global selon le F1 pondere: without_sensitive / xgboost

Ce rapport compare l'effet des donnees disponibles sur les performances. Il permet de verifier si le modele depend fortement du texte, des variables cliniques ou de variables sensibles comme le sexe et la zone de vie.

## Description Des Scenarios

| scenario | description |
|---|---|
| full | Tabulaire complet + texte |
| without_sensitive | Sans variables sensibles |
| text_only | Texte seul |
| clinical_only | Donnees cliniques pures |

## Resultats Complets

| scenario          | model               |   cv_accuracy |   cv_f1_weighted |   cv_recall_class_2 |   cv_critical_errors |   cv_very_critical_errors |   test_accuracy |   test_f1_weighted |   test_recall_class_2 |   test_critical_errors |   test_very_critical_errors |   test_inference_ms_per_row |
|:------------------|:--------------------|--------------:|-----------------:|--------------------:|---------------------:|--------------------------:|----------------:|-------------------:|----------------------:|-----------------------:|----------------------------:|----------------------------:|
| without_sensitive | xgboost             |        0.9495 |           0.9494 |              0.8865 |                  134 |                         4 |          0.9554 |             0.9552 |                0.9017 |                     29 |                           2 |                      0.0033 |
| full              | xgboost             |        0.949  |           0.9489 |              0.884  |                  137 |                         5 |          0.9549 |             0.9547 |                0.8983 |                     30 |                           2 |                      0.0035 |
| full              | logistic_regression |        0.9459 |           0.9461 |              0.9297 |                   83 |                        13 |          0.9494 |             0.9492 |                0.9322 |                     20 |                           6 |                      0.0002 |
| without_sensitive | logistic_regression |        0.9464 |           0.9466 |              0.9297 |                   83 |                        13 |          0.9479 |             0.9477 |                0.9322 |                     20 |                           6 |                      0.0002 |
| without_sensitive | random_forest       |        0.9356 |           0.9357 |              0.9009 |                  117 |                        19 |          0.9439 |             0.9439 |                0.9085 |                     27 |                           3 |                      0.019  |
| full              | random_forest       |        0.9339 |           0.9339 |              0.9052 |                  112 |                        22 |          0.9405 |             0.9404 |                0.9085 |                     27 |                           4 |                      0.0188 |
| text_only         | random_forest       |        0.8997 |           0.9013 |              0.9026 |                  115 |                        51 |          0.9107 |             0.9117 |                0.9153 |                     25 |                          11 |                      0.0135 |
| text_only         | logistic_regression |        0.8998 |           0.9013 |              0.9001 |                  118 |                        51 |          0.9107 |             0.9116 |                0.9085 |                     27 |                          11 |                      0.0001 |
| text_only         | xgboost             |        0.8994 |           0.9009 |              0.9001 |                  118 |                        53 |          0.9107 |             0.9116 |                0.9085 |                     27 |                          13 |                      0.0031 |
| clinical_only     | random_forest       |        0.9018 |           0.9004 |              0.7146 |                  337 |                         5 |          0.9058 |             0.9044 |                0.7356 |                     78 |                           2 |                      0.0188 |
| clinical_only     | xgboost             |        0.8934 |           0.8923 |              0.7282 |                  321 |                         5 |          0.8978 |             0.8967 |                0.7559 |                     72 |                           3 |                      0.0017 |
| clinical_only     | logistic_regression |        0.8601 |           0.8609 |              0.7832 |                  256 |                        11 |          0.8586 |             0.8585 |                0.7864 |                     63 |                           7 |                      0      |

## Meilleur Candidat Metier Avant Optimisation

Cette ligne ne correspond pas forcement au meilleur score global. Elle met en avant le modele le plus interessant du point de vue metier, c'est-a-dire celui qui detecte le mieux les urgences vitales tout en limitant les erreurs critiques.

| scenario   | model               |   cv_accuracy |   cv_f1_weighted |   cv_recall_class_2 |   cv_critical_errors |   cv_very_critical_errors |   test_accuracy |   test_f1_weighted |   test_recall_class_2 |   test_critical_errors |   test_very_critical_errors |   test_inference_ms_per_row |
|:-----------|:--------------------|--------------:|-----------------:|--------------------:|---------------------:|--------------------------:|----------------:|-------------------:|----------------------:|-----------------------:|----------------------------:|----------------------------:|
| full       | logistic_regression |        0.9459 |           0.9461 |              0.9297 |                   83 |                        13 |          0.9494 |             0.9492 |                0.9322 |                     20 |                           6 |                      0.0002 |

## Meilleur Modele Par Scenario Selon Le F1 Pondere

| scenario          | model         |   cv_accuracy |   cv_f1_weighted |   cv_recall_class_2 |   cv_critical_errors |   cv_very_critical_errors |   test_accuracy |   test_f1_weighted |   test_recall_class_2 |   test_critical_errors |   test_very_critical_errors |   test_inference_ms_per_row |
|:------------------|:--------------|--------------:|-----------------:|--------------------:|---------------------:|--------------------------:|----------------:|-------------------:|----------------------:|-----------------------:|----------------------------:|----------------------------:|
| clinical_only     | random_forest |        0.9018 |           0.9004 |              0.7146 |                  337 |                         5 |          0.9058 |             0.9044 |                0.7356 |                     78 |                           2 |                      0.0188 |
| full              | xgboost       |        0.949  |           0.9489 |              0.884  |                  137 |                         5 |          0.9549 |             0.9547 |                0.8983 |                     30 |                           2 |                      0.0035 |
| text_only         | random_forest |        0.8997 |           0.9013 |              0.9026 |                  115 |                        51 |          0.9107 |             0.9117 |                0.9153 |                     25 |                          11 |                      0.0135 |
| without_sensitive | xgboost       |        0.9495 |           0.9494 |              0.8865 |                  134 |                         4 |          0.9554 |             0.9552 |                0.9017 |                     29 |                           2 |                      0.0033 |

## Matrices De Confusion Sur Le Jeu De Test

### full / logistic_regression

Les lignes correspondent aux vraies classes [0, 1, 2]; les colonnes correspondent aux classes predites [0, 1, 2].

[[991, 20, 0], [34, 648, 28], [6, 14, 275]]

### full / random_forest

Les lignes correspondent aux vraies classes [0, 1, 2]; les colonnes correspondent aux classes predites [0, 1, 2].

[[979, 28, 4], [37, 649, 24], [4, 23, 268]]

### full / xgboost

Les lignes correspondent aux vraies classes [0, 1, 2]; les colonnes correspondent aux classes predites [0, 1, 2].

[[993, 18, 0], [26, 667, 17], [2, 28, 265]]

### without_sensitive / logistic_regression

Les lignes correspondent aux vraies classes [0, 1, 2]; les colonnes correspondent aux classes predites [0, 1, 2].

[[991, 19, 1], [37, 645, 28], [6, 14, 275]]

### without_sensitive / random_forest

Les lignes correspondent aux vraies classes [0, 1, 2]; les colonnes correspondent aux classes predites [0, 1, 2].

[[982, 27, 2], [34, 653, 23], [3, 24, 268]]

### without_sensitive / xgboost

Les lignes correspondent aux vraies classes [0, 1, 2]; les colonnes correspondent aux classes predites [0, 1, 2].

[[993, 18, 0], [27, 667, 16], [2, 27, 266]]

### text_only / logistic_regression

Les lignes correspondent aux vraies classes [0, 1, 2]; les colonnes correspondent aux classes predites [0, 1, 2].

[[926, 40, 45], [42, 642, 26], [11, 16, 268]]

### text_only / random_forest

Les lignes correspondent aux vraies classes [0, 1, 2]; les colonnes correspondent aux classes predites [0, 1, 2].

[[926, 38, 47], [42, 640, 28], [11, 14, 270]]

### text_only / xgboost

Les lignes correspondent aux vraies classes [0, 1, 2]; les colonnes correspondent aux classes predites [0, 1, 2].

[[928, 38, 45], [44, 640, 26], [13, 14, 268]]

### clinical_only / logistic_regression

Les lignes correspondent aux vraies classes [0, 1, 2]; les colonnes correspondent aux classes predites [0, 1, 2].

[[959, 52, 0], [66, 540, 104], [7, 56, 232]]

### clinical_only / random_forest

Les lignes correspondent aux vraies classes [0, 1, 2]; les colonnes correspondent aux classes predites [0, 1, 2].

[[981, 30, 0], [53, 628, 29], [2, 76, 217]]

### clinical_only / xgboost

Les lignes correspondent aux vraies classes [0, 1, 2]; les colonnes correspondent aux classes predites [0, 1, 2].

[[980, 31, 0], [56, 607, 47], [3, 69, 223]]

## Interpretation

Le modele final ne doit pas etre selectionne uniquement selon le F1 pondere. Pour ce cas d'usage de tri medical, la prochaine etape consiste a reduire les faux negatifs de la classe 2. Cela implique de prioriser le recall de la classe 2 et la reduction des erreurs critiques.

Le scenario full atteint un F1 pondere maximal de 0.9547, contre 0.9552 pour le scenario without_sensitive (ecart de -0.0005). Cet ecart faible confirme que les variables sensibles (sexe, zone_vie) n'apportent quasiment aucune information predictive supplementaire, ce qui justifie leur retrait pour des raisons d'equite sans sacrifier la performance.