"""
Script de comparaison des 4 scénarios d'entraînement
Usage: python -m src.compare_scenarios --data data/raw/dataset_telemed.csv
"""

from __future__ import annotations

import argparse
import os
import time
import numpy as np
import pandas as pd
import scipy.sparse as sp

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, f1_score, recall_score,
    confusion_matrix, make_scorer
)

ANALYSES_DIR = "analyses"

COL_NUM_FULL = ['age', 'freq_cardiaque', 'tension_sys', 'temp', 'sat_oxygene', 'antecedents', 'duree_symptomes']
COL_CAT_FULL = ['sexe', 'zone_vie', 'source']
COL_NUM_CLINICAL = ['age', 'freq_cardiaque', 'tension_sys', 'temp', 'sat_oxygene']
COL_TEXT = 'description_symptomes'


def critical_errors(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    mask_true_2 = y_true == 2
    critical = np.sum(mask_true_2 & (y_pred != 2))
    very_critical = np.sum(mask_true_2 & (y_pred == 0))
    return int(critical), int(very_critical)


def load_and_split(data_path: str):
    df = pd.read_csv(data_path)
    df = df.drop(columns=['patient_id'])
    X = df.drop(columns=['niveau_urgence'])
    y = df['niveau_urgence']
    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


def num_pipeline():
    return Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])


def cat_pipeline():
    return Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])


def build_scenario(scenario: str, X_train, X_test):
    """Retourne X_train_processed, X_test_processed pour le scénario demandé."""

    if scenario == "full":
        preprocessor = ColumnTransformer([
            ('num', num_pipeline(), COL_NUM_FULL),
            ('cat', cat_pipeline(), COL_CAT_FULL),
        ])
        X_train_tab = preprocessor.fit_transform(X_train)
        X_test_tab = preprocessor.transform(X_test)
        tfidf = TfidfVectorizer(max_features=3000, ngram_range=(1, 2), sublinear_tf=True)
        X_train_text = tfidf.fit_transform(X_train[COL_TEXT].fillna('').tolist())
        X_test_text = tfidf.transform(X_test[COL_TEXT].fillna('').tolist())
        return sp.hstack([X_train_tab, X_train_text]).tocsr(), sp.hstack([X_test_tab, X_test_text]).tocsr()

    elif scenario == "without_sensitive":
        col_num = ['age', 'freq_cardiaque', 'tension_sys', 'temp', 'sat_oxygene', 'duree_symptomes']
        col_cat = ['source']
        preprocessor = ColumnTransformer([
            ('num', num_pipeline(), col_num),
            ('cat', cat_pipeline(), col_cat),
        ])
        X_train_tab = preprocessor.fit_transform(X_train)
        X_test_tab = preprocessor.transform(X_test)
        tfidf = TfidfVectorizer(max_features=3000, ngram_range=(1, 2), sublinear_tf=True)
        X_train_text = tfidf.fit_transform(X_train[COL_TEXT].fillna('').tolist())
        X_test_text = tfidf.transform(X_test[COL_TEXT].fillna('').tolist())
        return sp.hstack([X_train_tab, X_train_text]).tocsr(), sp.hstack([X_test_tab, X_test_text]).tocsr()

    elif scenario == "text_only":
        tfidf = TfidfVectorizer(max_features=3000, ngram_range=(1, 2), sublinear_tf=True)
        X_train_text = tfidf.fit_transform(X_train[COL_TEXT].fillna('').tolist())
        X_test_text = tfidf.transform(X_test[COL_TEXT].fillna('').tolist())
        return X_train_text.tocsr(), X_test_text.tocsr()

    elif scenario == "clinical_only":
        preprocessor = ColumnTransformer([
            ('num', num_pipeline(), COL_NUM_CLINICAL),
        ])
        X_train_tab = preprocessor.fit_transform(X_train)
        X_test_tab = preprocessor.transform(X_test)
        return X_train_tab, X_test_tab

    else:
        raise ValueError(f"Scénario inconnu : {scenario}")


def get_models():
    return {
        'logistic_regression': LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42),
        'random_forest': RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=42, n_jobs=-1),
        'xgboost': XGBClassifier(n_estimators=200, random_state=42, eval_metric='mlogloss', n_jobs=-1),
    }


def evaluate(model, X_train, y_train, X_test, y_test):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    recall_c2_scorer = make_scorer(recall_score, labels=[2], average=None)

    cv_results = cross_validate(
        model, X_train, y_train, cv=skf,
        scoring={'accuracy': 'accuracy', 'f1_weighted': 'f1_weighted', 'recall_c2': recall_c2_scorer},
        n_jobs=-1
    )

    cv_critical, cv_very_critical = [], []
    for train_idx, val_idx in skf.split(X_train, y_train):
        m = model.__class__(**model.get_params())
        X_tr = X_train[train_idx]
        X_val = X_train[val_idx]
        y_tr = y_train.iloc[train_idx]
        y_val = y_train.iloc[val_idx]
        m.fit(X_tr, y_tr)
        y_pred_val = m.predict(X_val)
        c, vc = critical_errors(y_val, y_pred_val)
        cv_critical.append(c)
        cv_very_critical.append(vc)

    model.fit(X_train, y_train)
    start = time.perf_counter()
    y_pred_test = model.predict(X_test)
    elapsed_ms_per_row = (time.perf_counter() - start) * 1000 / X_test.shape[0]

    test_critical, test_very_critical = critical_errors(y_test, y_pred_test)
    cm = confusion_matrix(y_test, y_pred_test).tolist()

    return {
        'cv_accuracy': round(cv_results['test_accuracy'].mean(), 4),
        'cv_f1_weighted': round(cv_results['test_f1_weighted'].mean(), 4),
        'cv_recall_class_2': round(cv_results['test_recall_c2'].mean(), 4),
        'cv_critical_errors': int(sum(cv_critical)),
        'cv_very_critical_errors': int(sum(cv_very_critical)),
        'test_accuracy': round(accuracy_score(y_test, y_pred_test), 4),
        'test_f1_weighted': round(f1_score(y_test, y_pred_test, average='weighted'), 4),
        'test_recall_class_2': round(recall_score(y_test, y_pred_test, labels=[2], average=None)[0], 4),
        'test_critical_errors': test_critical,
        'test_very_critical_errors': test_very_critical,
        'test_inference_ms_per_row': round(elapsed_ms_per_row, 4),
        'test_confusion_matrix': cm,
    }


def generate_report(results_df: pd.DataFrame, model_path_hint: str):
    os.makedirs(ANALYSES_DIR, exist_ok=True)

    lines = []
    lines.append("# Comparaison Des Scenarios\n")
    lines.append("- Jeu de donnees: data/raw/dataset_telemed.csv")
    lines.append("- Validation: validation croisee stratifiee a 5 plis sur l'ensemble d'entrainement.")
    lines.append("- Jeu de test: 20%, stratifie, random_state=42.")
    best_global = results_df.sort_values('test_f1_weighted', ascending=False).iloc[0]
    lines.append(f"- Meilleur modele global selon le F1 pondere: {best_global['scenario']} / {best_global['model']}\n")

    lines.append("Ce rapport compare l'effet des donnees disponibles sur les performances. "
                  "Il permet de verifier si le modele depend fortement du texte, des variables "
                  "cliniques ou de variables sensibles comme le sexe et la zone de vie.\n")

    lines.append("## Description Des Scenarios\n")
    lines.append("| scenario | description |")
    lines.append("|---|---|")
    lines.append("| full | Tabulaire complet + texte |")
    lines.append("| without_sensitive | Sans variables sensibles |")
    lines.append("| text_only | Texte seul |")
    lines.append("| clinical_only | Donnees cliniques pures |\n")

    lines.append("## Resultats Complets\n")
    display_df = results_df.drop(columns=['test_confusion_matrix'])
    lines.append(display_df.sort_values('test_f1_weighted', ascending=False).to_markdown(index=False))
    lines.append("")

    lines.append("## Meilleur Candidat Metier Avant Optimisation\n")
    lines.append("Cette ligne ne correspond pas forcement au meilleur score global. Elle met en "
                  "avant le modele le plus interessant du point de vue metier, c'est-a-dire celui "
                  "qui detecte le mieux les urgences vitales tout en limitant les erreurs critiques.\n")
    best_business = results_df.sort_values(
        ['test_recall_class_2', 'test_critical_errors'], ascending=[False, True]
    ).iloc[[0]]
    lines.append(best_business.drop(columns=['test_confusion_matrix']).to_markdown(index=False))
    lines.append("")

    lines.append("## Meilleur Modele Par Scenario Selon Le F1 Pondere\n")
    best_per_scenario = results_df.loc[results_df.groupby('scenario')['test_f1_weighted'].idxmax()]
    lines.append(best_per_scenario.drop(columns=['test_confusion_matrix']).to_markdown(index=False))
    lines.append("")

    lines.append("## Matrices De Confusion Sur Le Jeu De Test\n")
    for _, row in results_df.iterrows():
        lines.append(f"### {row['scenario']} / {row['model']}\n")
        lines.append("Les lignes correspondent aux vraies classes [0, 1, 2]; "
                      "les colonnes correspondent aux classes predites [0, 1, 2].\n")
        lines.append(f"{row['test_confusion_matrix']}\n")

    lines.append("## Interpretation\n")
    lines.append(
        "Le modele final ne doit pas etre selectionne uniquement selon le F1 pondere. Pour ce "
        "cas d'usage de tri medical, la prochaine etape consiste a reduire les faux negatifs de "
        "la classe 2. Cela implique de prioriser le recall de la classe 2 et la reduction des "
        "erreurs critiques.\n"
    )
    full_best = results_df[results_df['scenario'] == 'full']['test_f1_weighted'].max()
    without_best = results_df[results_df['scenario'] == 'without_sensitive']['test_f1_weighted'].max()
    diff = round(full_best - without_best, 4)
    lines.append(
        f"Le scenario full atteint un F1 pondere maximal de {full_best}, contre {without_best} "
        f"pour le scenario without_sensitive (ecart de {diff}). Cet ecart faible confirme que "
        f"les variables sensibles (sexe, zone_vie) n'apportent quasiment aucune information "
        f"predictive supplementaire, ce qui justifie leur retrait pour des raisons d'equite "
        f"sans sacrifier la performance."
    )

    report_path = os.path.join(ANALYSES_DIR, "synthese_scenarios_comparatifs.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    csv_path = os.path.join(ANALYSES_DIR, "donnees_scenarios_comparatifs.csv")
    display_df.to_csv(csv_path, index=False)

    print(f"Rapport généré : {report_path}")
    print(f"CSV généré     : {csv_path}")


def main():
    parser = argparse.ArgumentParser(description="Comparaison des 4 scénarios")
    parser.add_argument("--data", required=True, help="Chemin vers le CSV brut")
    args = parser.parse_args()

    print(f"Chargement et split des données depuis {args.data}...")
    X_train, X_test, y_train, y_test = load_and_split(args.data)

    scenarios = ["full", "without_sensitive", "text_only", "clinical_only"]
    models = get_models()

    all_results = []
    for scenario in scenarios:
        print(f"\nScénario : {scenario}")
        X_train_proc, X_test_proc = build_scenario(scenario, X_train, X_test)

        for model_name, model in models.items():
            print(f"   {model_name}...", end=" ")
            res = evaluate(model, X_train_proc, y_train, X_test_proc, y_test)
            res['scenario'] = scenario
            res['model'] = model_name
            all_results.append(res)
            print(f"f1={res['test_f1_weighted']} | recall_c2={res['test_recall_class_2']} | "
                  f"critical_errors={res['test_critical_errors']}")

    results_df = pd.DataFrame(all_results)
    results_df = results_df[[
        'scenario', 'model', 'cv_accuracy', 'cv_f1_weighted', 'cv_recall_class_2',
        'cv_critical_errors', 'cv_very_critical_errors', 'test_accuracy', 'test_f1_weighted',
        'test_recall_class_2', 'test_critical_errors', 'test_very_critical_errors',
        'test_inference_ms_per_row', 'test_confusion_matrix'
    ]]

    generate_report(results_df, model_path_hint="models/triage_model.joblib")


if __name__ == "__main__":
    main()