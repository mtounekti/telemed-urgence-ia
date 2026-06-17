# Vocabulaire Discriminant Par Niveau D'Urgence

- Jeu de donnees: data/raw/dataset_telemed.csv
- Colonne analysee: description_symptomes
- Methode: vectorisation TF-IDF avec mots seuls et groupes de deux mots.

Ce rapport aide a comprendre comment le langage naturel du patient peut contribuer a la prediction. Il ne s'agit pas d'une preuve medicale, mais d'une analyse statistique des termes les plus associes a chaque niveau d'urgence dans le jeu de donnees.

## Lecture Des Resultats

Le score indique a quel point un terme est plus present dans une classe que dans les autres. Plus le score est eleve, plus le terme est caracteristique de cette classe dans le dataset.

## Classe 0 - Non urgent

| terme | score_association |
|---|---|
| légère | 0.03476 |
| traitement | 0.02816 |
| suite | 0.02776 |
| médical | 0.02438 |
| activité sportive | 0.02438 |
| médical activité | 0.02438 |
| certificat | 0.02438 |
| certificat médical | 0.02438 |
| sportive | 0.02438 |
| activité | 0.02438 |
| marche prolongée | 0.02267 |
| marche | 0.02267 |
| douleur genou | 0.02267 |
| prolongée | 0.02267 |
| genou | 0.02267 |
| genou marche | 0.02267 |
| information | 0.02198 |
| simple information | 0.02198 |
| simple | 0.02198 |
| information horaires | 0.02198 |

## Classe 1 - Urgence relative

| terme | score_association |
|---|---|
| saignement persistant | 0.02375 |
| persistant | 0.02375 |
| traumatisme direct | 0.02375 |
| nez saignement | 0.02375 |
| direct nez | 0.02375 |
| direct | 0.02375 |
| fièvre 39 | 0.02031 |
| frissons persistants | 0.02031 |
| persistants | 0.02031 |
| 39 frissons | 0.02031 |
| forte fièvre | 0.02031 |
| frissons | 0.02031 |
| 39 | 0.02031 |
| forte | 0.02031 |
| violents accompagnés | 0.01916 |
| lumière | 0.01916 |
| sensibilité lumière | 0.01916 |
| violents | 0.01916 |
| tête violents | 0.01916 |
| accompagnés sensibilité | 0.01916 |

## Classe 2 - Urgence vitale

| terme | score_association |
|---|---|
| respiratoire | 0.0322 |
| sévère | 0.03163 |
| plaie | 0.02782 |
| convulsions généralisées | 0.02733 |
| persistantes minutes | 0.02733 |
| persistantes | 0.02733 |
| généralisées | 0.02733 |
| généralisées persistantes | 0.02733 |
| minutes | 0.02733 |
| convulsions | 0.02733 |
| gonflement visage | 0.02415 |
| anaphylactique gonflement | 0.02415 |
| gorge | 0.02415 |
| anaphylactique | 0.02415 |
| gonflement | 0.02415 |
| visage | 0.02415 |
| réaction anaphylactique | 0.02415 |
| visage gorge | 0.02415 |
| perte | 0.0209 |
| mémoire immédiate | 0.01965 |

## Conclusion

Le texte apporte une information complementaire aux constantes vitales. Les scenarios text_only et full permettent de verifier que le modele exploite bien la description libre des symptomes.

Limite importante : TF-IDF repere des associations de mots, mais ne comprend pas le contexte comme un medecin. Par exemple, il ne gere pas parfaitement la negation, l'ironie, les fautes importantes ou les descriptions tres ambigues.