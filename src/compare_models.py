"""
Script de comparaison initiale des modèles
Usage: python -m src.compare_models --data data/raw/dataset_telemed.csv --scenario full
"""

from __future__ import annotations

import argparse
import os
import time
import joblib
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
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, f1_score, recall_score,
    confusion_matrix, make_scorer
)

REPORTS_DIR = "analyses"
FIGURES_DIR = os.path.join(REPORTS_DIR, "figures")

COL_NUM = ['age', 'freq_cardiaque', 'tension_sys', 'temp', 'sat_oxygene', 'antecedents', 'duree_symptomes']
COL_CAT = ['sexe', 'zone_vie', 'source']
COL_TEXT = 'description_symptomes'


def critical_errors(y_true, y_pred):
    """
    Erreurs critiques : vraie classe 2 prédite en 0 ou 1.
    Erreurs très critiques : vraie classe 2 prédite en 0.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    mask_true_2 = y_true == 2
    critical = np.sum(mask_true_2 & (y_pred != 2))
    very_critical = np.sum(mask_true_2 & (y_pred == 0))
    return int(critical), int(very_critical)


def build_preprocessor():
    """Construit le ColumnTransformer pour les variables tabulaires (scénario full)."""
    numeric_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    categorical_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    return ColumnTransformer([
        ('num', numeric_pipeline, COL_NUM),
        ('cat', categorical_pipeline, COL_CAT),
    ])


def load_and_split(data_path: str):
    df = pd.read_csv(data_path)
    df = df.drop(columns=['patient_id'])
    X = df.drop(columns=['niveau_urgence'])
    y = df['niveau_urgence']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    return X_train, X_test, y_train, y_test


def preprocess_full(X_train, X_test):
    """preprocessing scénario full: tabulaire + texte TF-IDF"""
    preprocessor = build_preprocessor()
    X_train_tab = preprocessor.fit_transform(X_train)
    X_test_tab = preprocessor.transform(X_test)

    tfidf = TfidfVectorizer(max_features=3000, ngram_range=(1, 2), sublinear_tf=True)
    X_train_text = tfidf.fit_transform(X_train[COL_TEXT].fillna('').tolist())
    X_test_text = tfidf.transform(X_test[COL_TEXT].fillna('').tolist())

    X_train_full = sp.hstack([X_train_tab, X_train_text]).tocsr()
    X_test_full = sp.hstack([X_test_tab, X_test_text]).tocsr()
    return X_train_full, X_test_full


def get_models():
    return {
        'logistic_regression': LogisticRegression(
            max_iter=1000, class_weight='balanced', random_state=42
        ),
        'random_forest': RandomForestClassifier(
            n_estimators=200, class_weight='balanced', random_state=42, n_jobs=-1
        ),
        'xgboost': XGBClassifier(
            n_estimators=200, random_state=42, eval_metric='mlogloss', n_jobs=-1
        ),
    }


def evaluate_with_cv(model, X_train, y_train, X_test, y_test):
    """évalue un modèle avec CV 5-fold sur train + métriques sur test"""
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    recall_c2_scorer = make_scorer(recall_score, labels=[2], average=None)

    cv_results = cross_validate(
        model, X_train, y_train, cv=skf,
        scoring={'accuracy': 'accuracy', 'f1_weighted': 'f1_weighted', 'recall_c2': recall_c2_scorer},
        n_jobs=-1
    )

    # erreirs critiques en CV (approximation via prédictions par fold)
    cv_critical, cv_very_critical = [], []
    for train_idx, val_idx in skf.split(X_train, y_train):
        m = model.__class__(**model.get_params())
        X_tr = X_train[train_idx] if hasattr(X_train, '__getitem__') else X_train.iloc[train_idx]
        X_val = X_train[val_idx] if hasattr(X_train, '__getitem__') else X_train.iloc[val_idx]
        y_tr = y_train.iloc[train_idx]
        y_val = y_train.iloc[val_idx]
        m.fit(X_tr, y_tr)
        y_pred_val = m.predict(X_val)
        c, vc = critical_errors(y_val, y_pred_val)
        cv_critical.append(c)
        cv_very_critical.append(vc)

    # entraînement final sur tout le train + évaluation test
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
        'model_object': model,
    }


def generate_report(results_df: pd.DataFrame, scenario: str, best_model_name: str, model_path: str):
    os.makedirs(REPORTS_DIR, exist_ok=True)

    lines = []
    lines.append("# Comparaison Initiale Des Modeles\n")
    lines.append(f"- Jeu de donnees: data/raw/dataset_telemed.csv")
    lines.append(f"- Scenario teste: {scenario} - tabulaire complet + texte")
    lines.append(f"- Validation: validation croisee stratifiee a 5 plis sur l'ensemble d'entrainement.")
    lines.append(f"- Jeu de test: 20%, stratifie, random_state=42.")
    lines.append(f"- Meilleur modele initial: {best_model_name}")
    lines.append(f"- Modele sauvegarde: {model_path}\n")
    lines.append("Cette premiere comparaison sert a obtenir un point de depart fiable sur le "
                  "scenario le plus complet. Le choix final ne doit pas encore etre considere "
                  "comme definitif, car le sujet impose ensuite une analyse plus metier autour "
                  "des urgences vitales.\n")

    lines.append("## Resultats\n")
    display_df = results_df.drop(columns=['model_object', 'test_confusion_matrix'])
    lines.append(display_df.to_markdown(index=False))
    lines.append("")

    lines.append("## Matrices De Confusion Sur Le Jeu De Test\n")
    for _, row in results_df.iterrows():
        lines.append(f"### {row['model']}\n")
        lines.append("Les lignes correspondent aux vraies classes [0, 1, 2]; "
                      "les colonnes correspondent aux classes predites [0, 1, 2].\n")
        lines.append(f"{row['test_confusion_matrix']}\n")

    sorted_by_recall = results_df.sort_values('test_recall_class_2', ascending=False).iloc[0]
    lines.append("## Prochaine Etape\n")
    lines.append(
        f"Le modele {best_model_name} obtient le meilleur score global sur ce premier test. "
        f"Cependant, {sorted_by_recall['model']} obtient un meilleur recall sur la classe 2 "
        f"({sorted_by_recall['test_recall_class_2']}) et produit "
        f"{sorted_by_recall['test_critical_errors']} erreurs critiques. La suite doit donc "
        f"comparer tous les scenarios, puis optimiser explicitement la detection des urgences vitales."
    )

    report_path = os.path.join(REPORTS_DIR, "synthese_modeles_baseline.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    csv_path = os.path.join(REPORTS_DIR, "donnees_modeles_baseline.csv")
    display_df.to_csv(csv_path, index=False)

    print(f"Rapport généré : {report_path}")
    print(f"CSV généré     : {csv_path}")


def main():
    parser = argparse.ArgumentParser(description="Comparaison initiale des modèles")
    parser.add_argument("--data", required=True, help="Chemin vers le CSV brut")
    parser.add_argument("--scenario", default="full", choices=["full"], help="Scénario (full uniquement pour cette comparaison initiale)")
    args = parser.parse_args()

    print(f"Chargement et split des données depuis {args.data}...")
    X_train, X_test, y_train, y_test = load_and_split(args.data)

    print("Préprocessing (scénario full)...")
    X_train_proc, X_test_proc = preprocess_full(X_train, X_test)

    models = get_models()
    all_results = []

    for name, model in models.items():
        print(f"Entraînement et évaluation : {name}...")
        res = evaluate_with_cv(model, X_train_proc, y_train, X_test_proc, y_test)
        res['scenario'] = args.scenario
        res['model'] = name
        all_results.append(res)
        print(f"   test_f1_weighted={res['test_f1_weighted']} | "
              f"test_recall_class_2={res['test_recall_class_2']} | "
              f"test_critical_errors={res['test_critical_errors']}")

    results_df = pd.DataFrame(all_results)
    results_df = results_df[[
        'scenario', 'model', 'cv_accuracy', 'cv_f1_weighted', 'cv_recall_class_2',
        'cv_critical_errors', 'cv_very_critical_errors', 'test_accuracy', 'test_f1_weighted',
        'test_recall_class_2', 'test_critical_errors', 'test_very_critical_errors',
        'test_inference_ms_per_row', 'test_confusion_matrix', 'model_object'
    ]]

    best_row = results_df.sort_values('test_f1_weighted', ascending=False).iloc[0]
    best_model_name = best_row['model']
    best_model_obj = best_row['model_object']

    os.makedirs("models", exist_ok=True)
    model_path = os.path.join("models", "triage_model.joblib")
    joblib.dump(best_model_obj, model_path)
    print(f"\n Meilleur modèle ({best_model_name}) sauvegardé : {model_path}")

    generate_report(results_df, args.scenario, best_model_name, model_path)


if __name__ == "__main__":
    main()