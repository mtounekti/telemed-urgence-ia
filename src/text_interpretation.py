"""
Script d'interprétation du texte patient
Identifie les termes les plus caractéristiques de chaque niveau d'urgence.
Usage : python -m src.text_interpretation --data data/raw/dataset_telemed.csv
"""

from __future__ import annotations

import argparse
import os
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

ANALYSES_DIR = "analyses"
TOP_N_TERMS = 20

FRENCH_STOPWORDS = [
    "alors", "au", "aucuns", "aussi", "autre", "avant", "avec", "avoir", "bon",
    "car", "ce", "cela", "ces", "ceux", "chaque", "ci", "comme", "comment",
    "dans", "des", "du", "dedans", "dehors", "depuis", "devrait", "doit",
    "donc", "dos", "début", "elle", "elles", "en", "encore", "essai", "est",
    "et", "eu", "fait", "faites", "fois", "font", "hors", "ici", "il", "ils",
    "je", "juste", "la", "le", "les", "leur", "là", "ma", "maintenant", "mais",
    "mes", "mine", "moins", "mon", "mot", "même", "ni", "nommés", "notre",
    "nous", "ou", "où", "par", "parce", "pas", "peut", "peu", "plupart",
    "pour", "pourquoi", "quand", "que", "quel", "quelle", "quelles", "quels",
    "qui", "sa", "sans", "ses", "seulement", "si", "sien", "son", "sont",
    "sous", "soyez", "sujet", "sur", "ta", "tandis", "tellement", "tels",
    "tes", "ton", "tous", "tout", "trop", "très", "tu", "votre", "vous",
    "vu", "ça", "étaient", "état", "étions", "été", "être", "un", "une",
    "de", "ne", "se", "ce", "ces", "cet", "cette", "afin", "ainsi",
    "plus", "patient", "après", "demande", "besoin", "depuis", "deux",
]

def compute_association_scores(df: pd.DataFrame, text_col: str, target_col: str):
    """
    pour chaque classe, calcule un score d'association de chaque terme:
    moyenne du TF-IDF du terme dans la classe, moins la moyenne du TF-IDF
    du même terme dans les autres classes. Un score élevé indique un terme
    surreprésenté dans cette classe par rapport au reste du dataset
    """
    texts = df[text_col].fillna('').tolist()

    vectorizer = TfidfVectorizer(
        max_features=3000,
        ngram_range=(1, 2),
        sublinear_tf=True,
        stop_words=FRENCH_STOPWORDS,
        min_df=3
    )

    tfidf_matrix = vectorizer.fit_transform(texts)
    terms = vectorizer.get_feature_names_out()

    results_by_class = {}
    classes = sorted(df[target_col].unique())

    for cls in classes:
        mask_in = (df[target_col] == cls).values
        mean_in_class = np.asarray(tfidf_matrix[mask_in].mean(axis=0)).flatten()
        mean_out_class = np.asarray(tfidf_matrix[~mask_in].mean(axis=0)).flatten()
        association_score = mean_in_class - mean_out_class

        df_terms = pd.DataFrame({
            'terme': terms,
            'score_association': association_score
        }).sort_values('score_association', ascending=False)

        results_by_class[cls] = df_terms.head(TOP_N_TERMS).reset_index(drop=True)

    return results_by_class


def generate_report(results_by_class: dict, labels: dict):
    os.makedirs(ANALYSES_DIR, exist_ok=True)

    lines = []
    lines.append("# Vocabulaire Discriminant Par Niveau D'Urgence\n")
    lines.append("- Jeu de donnees: data/raw/dataset_telemed.csv")
    lines.append("- Colonne analysee: description_symptomes")
    lines.append("- Methode: vectorisation TF-IDF avec mots seuls et groupes de deux mots.\n")

    lines.append(
        "Ce rapport aide a comprendre comment le langage naturel du patient peut contribuer a "
        "la prediction. Il ne s'agit pas d'une preuve medicale, mais d'une analyse statistique "
        "des termes les plus associes a chaque niveau d'urgence dans le jeu de donnees.\n"
    )

    lines.append("## Lecture Des Resultats\n")
    lines.append(
        "Le score indique a quel point un terme est plus present dans une classe que dans les "
        "autres. Plus le score est eleve, plus le terme est caracteristique de cette classe "
        "dans le dataset.\n"
    )

    for cls, df_terms in results_by_class.items():
        label = labels.get(cls, str(cls))
        lines.append(f"## Classe {cls} - {label}\n")
        lines.append("| terme | score_association |")
        lines.append("|---|---|")
        for _, row in df_terms.iterrows():
            lines.append(f"| {row['terme']} | {round(row['score_association'], 5)} |")
        lines.append("")

    lines.append("## Conclusion\n")
    lines.append(
        "Le texte apporte une information complementaire aux constantes vitales. Les scenarios "
        "text_only et full permettent de verifier que le modele exploite bien la description "
        "libre des symptomes.\n"
    )
    lines.append(
        "Limite importante : TF-IDF repere des associations de mots, mais ne comprend pas le "
        "contexte comme un medecin. Par exemple, il ne gere pas parfaitement la negation, "
        "l'ironie, les fautes importantes ou les descriptions tres ambigues."
    )

    report_path = os.path.join(ANALYSES_DIR, "synthese_vocabulaire_discriminant.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"Rapport généré : {report_path}")


def main():
    parser = argparse.ArgumentParser(description="Interprétation du vocabulaire par classe d'urgence")
    parser.add_argument("--data", required=True, help="Chemin vers le CSV brut")
    args = parser.parse_args()

    print(f"⏳ Chargement des données depuis {args.data}...")
    df = pd.read_csv(args.data)

    labels = {0: "Non urgent", 1: "Urgence relative", 2: "Urgence vitale"}

    print("⏳ Calcul des scores d'association par classe...")
    results_by_class = compute_association_scores(df, 'description_symptomes', 'niveau_urgence')

    for cls, df_terms in results_by_class.items():
        label = labels.get(cls, str(cls))
        top5 = ", ".join(df_terms['terme'].head(5).tolist())
        print(f"   Classe {cls} ({label}) — top 5 : {top5}")

    generate_report(results_by_class, labels)


if __name__ == "__main__":
    main()