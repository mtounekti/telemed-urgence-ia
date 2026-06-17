# Ajustement Du Seuil De Decision — Comparaison Multi-Modeles

- Jeu de donnees: data/raw/dataset_telemed.csv
- Scenario optimise: full - Tabulaire complet + texte
- Baisse maximale acceptee du F1 pondere pendant la recherche: 0.03
- Modeles testes: logistic_regression, random_forest, xgboost

## Principe De L'Optimisation

Le modele produit des probabilites pour les trois classes. Par defaut, il choisit la classe avec la probabilite la plus elevee (seuil 0.5). Pour ce projet, une erreur sur une vraie urgence vitale est plus grave qu'une fausse alerte. Le seuil de decision de la classe 2 a donc ete abaisse afin de predire plus facilement urgence vitale lorsque le modele detecte un risque suffisant.

La metrique favorisee est le recall de la classe 2. Le F1 pondere reste surveille via une contrainte de baisse maximale, afin d'eviter une degradation globale trop importante.

## Resultats Comparatifs — Avant Vs Apres Optimisation

| modele | seuil_retenu | accuracy_avant | accuracy_apres | f1_avant | f1_apres | recall_c2_avant | recall_c2_apres | erreurs_critiques_avant | erreurs_critiques_apres |
|---|---|---|---|---|---|---|---|---|---|
| logistic_regression | 0.15 | 0.9484 | 0.9261 | 0.9482 | 0.9266 | 0.9254 | 0.9661 | 22 | 10 |
| random_forest | 0.1 | 0.9385 | 0.9147 | 0.9383 | 0.9169 | 0.8881 | 0.9898 | 33 | 3 |
| xgboost | 0.01 | 0.9554 | 0.938 | 0.9552 | 0.9388 | 0.8983 | 0.9898 | 30 | 3 |

## Meilleur Modele Apres Optimisation

Le modele retenu apres optimisation est **xgboost** avec un seuil de classe 2 fixe a **0.01**. Il minimise les erreurs critiques (3) tout en conservant le meilleur F1 pondere parmi les modeles a egalite sur ce critere (0.9388).

### Detail — logistic_regression

#### Seuils testes

|   threshold |   accuracy |   f1_weighted |   recall_class_2 |   critical_errors |   very_critical_errors |
|------------:|-----------:|--------------:|-----------------:|------------------:|-----------------------:|
|        0.5  |     0.9484 |        0.9482 |           0.9254 |                22 |                      7 |
|        0.4  |     0.9494 |        0.9493 |           0.939  |                18 |                      7 |
|        0.3  |     0.9425 |        0.9424 |           0.9492 |                15 |                      6 |
|        0.2  |     0.934  |        0.9343 |           0.9559 |                13 |                      6 |
|        0.15 |     0.9261 |        0.9266 |           0.9661 |                10 |                      4 |
|        0.1  |     0.9107 |        0.912  |           0.9763 |                 7 |                      3 |
|        0.07 |     0.8919 |        0.8942 |           0.9763 |                 7 |                      3 |
|        0.05 |     0.8681 |        0.8712 |           0.9864 |                 4 |                      1 |
|        0.03 |     0.8125 |        0.8154 |           0.9932 |                 2 |                      1 |
|        0.01 |     0.7088 |        0.6997 |           0.9966 |                 1 |                      1 |

#### Matrice De Confusion Avant Optimisation

[[991, 20, 0], [34, 648, 28], [7, 15, 273]]

#### Matrice De Confusion Apres Optimisation

[[989, 16, 6], [34, 593, 83], [4, 6, 285]]

### Detail — random_forest

#### Seuils testes

|   threshold |   accuracy |   f1_weighted |   recall_class_2 |   critical_errors |   very_critical_errors |
|------------:|-----------:|--------------:|-----------------:|------------------:|-----------------------:|
|        0.5  |     0.9385 |        0.9383 |           0.8881 |                33 |                      8 |
|        0.4  |     0.94   |        0.9399 |           0.9119 |                26 |                      4 |
|        0.3  |     0.9385 |        0.9385 |           0.9254 |                22 |                      3 |
|        0.2  |     0.933  |        0.9335 |           0.9492 |                15 |                      2 |
|        0.15 |     0.9306 |        0.9314 |           0.9661 |                10 |                      2 |
|        0.1  |     0.9147 |        0.9169 |           0.9898 |                 3 |                      1 |
|        0.07 |     0.8988 |        0.9024 |           0.9898 |                 3 |                      1 |
|        0.05 |     0.8795 |        0.8855 |           0.9898 |                 3 |                      1 |
|        0.03 |     0.8433 |        0.8538 |           0.9898 |                 3 |                      1 |
|        0.01 |     0.7029 |        0.7303 |           0.9966 |                 1 |                      1 |

#### Matrice De Confusion Avant Optimisation

[[979, 29, 3], [39, 651, 20], [8, 25, 262]]

#### Matrice De Confusion Apres Optimisation

[[947, 28, 36], [29, 605, 76], [1, 2, 292]]

### Detail — xgboost

#### Seuils testes

|   threshold |   accuracy |   f1_weighted |   recall_class_2 |   critical_errors |   very_critical_errors |
|------------:|-----------:|--------------:|-----------------:|------------------:|-----------------------:|
|        0.5  |     0.9554 |        0.9552 |           0.8983 |                30 |                      2 |
|        0.4  |     0.9549 |        0.9547 |           0.9119 |                26 |                      2 |
|        0.3  |     0.9559 |        0.9558 |           0.9254 |                22 |                      2 |
|        0.2  |     0.9563 |        0.9563 |           0.939  |                18 |                      2 |
|        0.15 |     0.9563 |        0.9563 |           0.9458 |                16 |                      2 |
|        0.1  |     0.9583 |        0.9583 |           0.9729 |                 8 |                      1 |
|        0.07 |     0.9544 |        0.9544 |           0.9729 |                 8 |                      1 |
|        0.05 |     0.9509 |        0.9511 |           0.9797 |                 6 |                      1 |
|        0.03 |     0.9484 |        0.9488 |           0.9864 |                 4 |                      1 |
|        0.01 |     0.938  |        0.9388 |           0.9898 |                 3 |                      1 |

#### Matrice De Confusion Avant Optimisation

[[993, 18, 0], [26, 668, 16], [2, 28, 265]]

#### Matrice De Confusion Apres Optimisation

[[990, 14, 7], [23, 609, 78], [1, 2, 292]]

## Interpretation Generale

Tous les modeles testes voient leur recall de classe 2 augmenter fortement apres ajustement du seuil, au prix d'une baisse controlee du F1 pondere (contrainte de 0.03). Le modele xgboost ressort comme le meilleur compromis global : il atteint un recall de classe 2 de 0.9898 et ne laisse passer que 3 erreurs critiques sur le jeu de test, tout en conservant la meilleure performance globale parmi les modeles obtenant ce niveau de securite. Ce choix est coherent avec le risque metier du tri d'urgence, ou manquer une urgence vitale est plus grave qu'emettre une fausse alerte.