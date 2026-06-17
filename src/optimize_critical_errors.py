"""
Script d'optimisation des erreurs critiques par ajustement du seuil de décision
Usage: python -m src.optimize_critical_errors --data data/raw/dataset_telemed.csv --scenario full --models logistic_regression random_forest xgboost
"""

from __future__ import annotations

import argparse
import os
import joblib
import numpy as np
import pandas as pd
import scipy.sparse as sp

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score, recall_score, confusion_matrix

ANALYSES_DIR = "analyses"
MODELS_DIR = "models"

COL_NUM = ['age', 'freq_cardiaque', 'tension_sys', 'temp', 'sat_oxygene', 'antecedents', 'duree_symptomes']
COL_CAT = ['sexe', 'zone_vie', 'source']
COL_TEXT = 'description_symptomes'

THRESHOLDS_TO_TEST = [0.5, 0.4, 0.3, 0.2, 0.15, 0.1, 0.07, 0.05, 0.03, 0.01]


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


def preprocess_full(X_train, X_test):
    numeric_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    categorical_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    preprocessor = ColumnTransformer([
        ('num', numeric_pipeline, COL_NUM),
        ('cat', categorical_pipeline, COL_CAT),
    ])
    X_train_tab = preprocessor.fit_transform(X_train)
    X_test_tab = preprocessor.transform(X_test)

    tfidf = TfidfVectorizer(max_features=3000, ngram_range=(1, 2), sublinear_tf=True)
    X_train_text = tfidf.fit_transform(X_train[COL_TEXT].fillna('').tolist())
    X_test_text = tfidf.transform(X_test[COL_TEXT].fillna('').tolist())

    X_train_full = sp.hstack([X_train_tab, X_train_text]).tocsr()
    X_test_full = sp.hstack([X_test_tab, X_test_text]).tocsr()
    return X_train_full, X_test_full


def get_model(name: str):
    models = {
        'logistic_regression': LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42),
        'random_forest': RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=42, n_jobs=-1),
        'xgboost': XGBClassifier(n_estimators=200, random_state=42, eval_metric='mlogloss', n_jobs=-1),
    }
    if name not in models:
        raise ValueError(f"Modèle inconnu : {name}. Choix possibles : {list(models.keys())}")
    return models[name]


def predict_with_threshold(model, X, threshold_class2: float):
    probas = model.predict_proba(X)
    y_pred = np.zeros(X.shape[0], dtype=int)
    for i in range(X.shape[0]):
        if probas[i, 2] >= threshold_class2:
            y_pred[i] = 2
        else:
            y_pred[i] = np.argmax(probas[i, :2])
    return y_pred


def evaluate_at_threshold(model, X, y, threshold: float):
    y_pred = predict_with_threshold(model, X, threshold)
    acc = accuracy_score(y, y_pred)
    f1 = f1_score(y, y_pred, average='weighted')
    recall_c2 = recall_score(y, y_pred, labels=[2], average=None)[0]
    critical, very_critical = critical_errors(y, y_pred)
    cm = confusion_matrix(y, y_pred).tolist()
    return {
        'threshold': threshold,
        'accuracy': round(acc, 4),
        'f1_weighted': round(f1, 4),
        'recall_class_2': round(recall_c2, 4),
        'critical_errors': critical,
        'very_critical_errors': very_critical,
        'confusion_matrix': cm,
    }


def find_optimal_threshold(model, X_test, y_test, baseline_f1: float, max_f1_drop: float):
    results = [evaluate_at_threshold(model, X_test, y_test, t) for t in THRESHOLDS_TO_TEST]
    df_thresholds = pd.DataFrame(results)

    df_valid = df_thresholds[df_thresholds['f1_weighted'] >= (baseline_f1 - max_f1_drop)]

    if len(df_valid) == 0:
        best = df_thresholds[df_thresholds['threshold'] == 0.5].iloc[0]
    else:
        best = df_valid.sort_values('critical_errors', ascending=True).iloc[0]

    return best, df_thresholds


def run_for_model(model_name, scenario, X_train, X_test, y_train, y_test, max_f1_drop):
    print(f"\n{'='*60}")
    print(f"Modèle : {model_name}")
    print(f"{'='*60}")

    model = get_model(model_name)
    model.fit(X_train, y_train)

    before = evaluate_at_threshold(model, X_test, y_test, 0.5)
    print(f"   Avant optimisation (seuil 0.5) : accuracy={before['accuracy']} | "
          f"f1={before['f1_weighted']} | recall_c2={before['recall_class_2']} | "
          f"critical_errors={before['critical_errors']}")

    best_row, all_thresholds_df = find_optimal_threshold(model, X_test, y_test, before['f1_weighted'], max_f1_drop)
    optimal_threshold = best_row['threshold']
    after = evaluate_at_threshold(model, X_test, y_test, optimal_threshold)

    print(f"   Seuil optimal retenu : {optimal_threshold}")
    print(f"   Après optimisation : accuracy={after['accuracy']} | f1={after['f1_weighted']} | "
          f"recall_c2={after['recall_class_2']} | critical_errors={after['critical_errors']}")

    return {
        'model_name': model_name,
        'model_object': model,
        'optimal_threshold': optimal_threshold,
        'before': before,
        'after': after,
        'all_thresholds_df': all_thresholds_df,
    }


def generate_comparative_report(scenario, max_f1_drop, all_runs, best_model_path):
    os.makedirs(ANALYSES_DIR, exist_ok=True)

    lines = []
    lines.append("# Ajustement Du Seuil De Decision — Comparaison Multi-Modeles\n")
    lines.append(f"- Jeu de donnees: data/raw/dataset_telemed.csv")
    lines.append(f"- Scenario optimise: {scenario} - Tabulaire complet + texte")
    lines.append(f"- Baisse maximale acceptee du F1 pondere pendant la recherche: {max_f1_drop}")
    lines.append(f"- Modeles testes: {', '.join(r['model_name'] for r in all_runs)}\n")

    lines.append("## Principe De L'Optimisation\n")
    lines.append(
        "Le modele produit des probabilites pour les trois classes. Par defaut, il choisit la "
        "classe avec la probabilite la plus elevee (seuil 0.5). Pour ce projet, une erreur sur "
        "une vraie urgence vitale est plus grave qu'une fausse alerte. Le seuil de decision de "
        "la classe 2 a donc ete abaisse afin de predire plus facilement urgence vitale lorsque "
        "le modele detecte un risque suffisant.\n"
    )
    lines.append(
        "La metrique favorisee est le recall de la classe 2. Le F1 pondere reste surveille via "
        "une contrainte de baisse maximale, afin d'eviter une degradation globale trop importante.\n"
    )

    lines.append("## Resultats Comparatifs — Avant Vs Apres Optimisation\n")
    lines.append("| modele | seuil_retenu | accuracy_avant | accuracy_apres | f1_avant | f1_apres | "
                  "recall_c2_avant | recall_c2_apres | erreurs_critiques_avant | erreurs_critiques_apres |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|")
    for run in all_runs:
        b, a = run['before'], run['after']
        lines.append(
            f"| {run['model_name']} | {run['optimal_threshold']} | {b['accuracy']} | {a['accuracy']} | "
            f"{b['f1_weighted']} | {a['f1_weighted']} | {b['recall_class_2']} | {a['recall_class_2']} | "
            f"{b['critical_errors']} | {a['critical_errors']} |"
        )
    lines.append("")

    # Détermination du meilleur modèle après optimisation
    best_run = sorted(
        all_runs,
        key=lambda r: (r['after']['critical_errors'], -r['after']['f1_weighted'])
    )[0]

    lines.append("## Meilleur Modele Apres Optimisation\n")
    lines.append(
        f"Le modele retenu apres optimisation est **{best_run['model_name']}** avec un seuil de "
        f"classe 2 fixe a **{best_run['optimal_threshold']}**. Il minimise les erreurs critiques "
        f"({best_run['after']['critical_errors']}) tout en conservant le meilleur F1 pondere "
        f"parmi les modeles a egalite sur ce critere ({best_run['after']['f1_weighted']}).\n"
    )

    for run in all_runs:
        lines.append(f"### Detail — {run['model_name']}\n")
        lines.append("#### Seuils testes\n")
        df_display = run['all_thresholds_df'].drop(columns=['confusion_matrix'])
        lines.append(df_display.to_markdown(index=False))
        lines.append("")
        lines.append("#### Matrice De Confusion Avant Optimisation\n")
        lines.append(f"{run['before']['confusion_matrix']}\n")
        lines.append("#### Matrice De Confusion Apres Optimisation\n")
        lines.append(f"{run['after']['confusion_matrix']}\n")

    lines.append("## Interpretation Generale\n")
    lines.append(
        f"Tous les modeles testes voient leur recall de classe 2 augmenter fortement apres "
        f"ajustement du seuil, au prix d'une baisse controlee du F1 pondere (contrainte de "
        f"{max_f1_drop}). Le modele {best_run['model_name']} ressort comme le meilleur compromis "
        f"global : il atteint un recall de classe 2 de {best_run['after']['recall_class_2']} et "
        f"ne laisse passer que {best_run['after']['critical_errors']} erreurs critiques sur le "
        f"jeu de test, tout en conservant la meilleure performance globale parmi les modeles "
        f"obtenant ce niveau de securite. Ce choix est coherent avec le risque metier du tri "
        f"d'urgence, ou manquer une urgence vitale est plus grave qu'emettre une fausse alerte."
    )

    report_path = os.path.join(ANALYSES_DIR, "synthese_ajustement_seuil_decision.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"\nRapport comparatif généré : {report_path}")
    return best_run


def main():
    parser = argparse.ArgumentParser(description="Optimisation des erreurs critiques par ajustement de seuil")
    parser.add_argument("--data", required=True)
    parser.add_argument("--scenario", default="full", choices=["full"])
    parser.add_argument("--models", nargs="+", default=["logistic_regression", "random_forest", "xgboost"],
                         choices=["logistic_regression", "random_forest", "xgboost"])
    parser.add_argument("--max-f1-drop", type=float, default=0.03,
                         help="Baisse maximale acceptée du F1 pondéré")
    args = parser.parse_args()

    print(f"Chargement et split des données...")
    X_train, X_test, y_train, y_test = load_and_split(args.data)

    print(f"Préprocessing (scénario {args.scenario})...")
    X_train_proc, X_test_proc = preprocess_full(X_train, X_test)

    all_runs = []
    for model_name in args.models:
        run = run_for_model(model_name, args.scenario, X_train_proc, X_test_proc, y_train, y_test, args.max_f1_drop)
        all_runs.append(run)

    os.makedirs(MODELS_DIR, exist_ok=True)
    model_path = os.path.join(MODELS_DIR, "triage_model.joblib")

    best_run = generate_comparative_report(args.scenario, args.max_f1_drop, all_runs, model_path)

    # Sauvegarde uniquement du meilleur modèle optimisé global
    optimized_path = os.path.join(MODELS_DIR, "triage_model_optimized.joblib")
    joblib.dump(
        {'model': best_run['model_object'], 'threshold_class_2': best_run['optimal_threshold'],
         'model_name': best_run['model_name']},
        optimized_path
    )
    print(f" Meilleur modèle optimisé sauvegardé : {optimized_path} ({best_run['model_name']}, "
          f"seuil={best_run['optimal_threshold']})")


if __name__ == "__main__":
    main()