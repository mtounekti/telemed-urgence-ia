# Interface Streamlit—Telemed Urgence IA

import streamlit as st
import requests
import json
import pandas as pd
from datetime import datetime

# config
API_URL = "http://127.0.0.1:8000"

st.set_page_config(
    page_title="Telemed Urgence IA",
    page_icon="🏥",
    layout="wide"
)

# Header
st.title("🏥 Telemed Urgence IA")
st.markdown("**Système de diagnostic assisté et tri d'urgence multimodal**")
st.divider()

# Sidebar: statut API
with st.sidebar:
    st.header("⚙️ Statut API")
    try:
        resp = requests.get(f"{API_URL}/health", timeout=2)
        if resp.status_code == 200:
            st.success("✅ API connectée")
            data = resp.json()
            st.caption(f"Modèle : {data['model']}")
            st.caption(f"Version : {data['version']}")
        else:
            st.error("❌ API non disponible")
    except:
        st.error("❌ API non disponible")
    
    st.divider()
    st.header("📋 Navigation")
    page = st.radio("", ["🔍 Prédiction", "📊 Historique"])

# Page prédiction
if page == "🔍 Prédiction":
    st.subheader("📝 Informations du patient")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**Données administratives**")
        sexe = st.selectbox("Sexe", ["F", "H"])
        age = st.number_input("Âge", min_value=0, max_value=120, value=45)
        zone_vie = st.selectbox("Zone de vie", ["U", "R"],
                                 help="U = Urbain / R = Rural")
        source = st.selectbox("Source", ["appel", "chat"])
        antecedents = st.selectbox("Antécédents médicaux", [0, 1],
                                    format_func=lambda x: "Oui" if x == 1 else "Non")

    with col2:
        st.markdown("**Constantes vitales**")
        freq_cardiaque = st.number_input("Fréquence cardiaque (bpm)",
                                          min_value=30, max_value=250, value=80)
        tension_sys = st.number_input("Tension systolique (mmHg)",
                                       min_value=50, max_value=300, value=120)
        temp = st.number_input("Température (°C)",
                                min_value=34.0, max_value=43.0,
                                value=37.0, step=0.1)
        sat_oxygene = st.number_input("Saturation O2 (%)",
                                       min_value=50.0, max_value=100.0,
                                       value=98.0, step=0.1)
        duree_symptomes = st.number_input("Durée des symptômes (heures)",
                                           min_value=0.0, value=24.0, step=0.5)

    with col3:
        st.markdown("**Description des symptômes**")
        description = st.text_area(
            "Décrivez les symptômes",
            height=200,
            placeholder="Ex: Douleur thoracique intense avec essoufflement..."
        )

    st.divider()

    # bouton de prédiction
    if st.button("🔍 Analyser l'urgence", type="primary", use_container_width=True):
        if not description.strip():
            st.warning("⚠️ Veuillez renseigner la description des symptômes.")
        else:
            payload = {
                "sexe": sexe,
                "age": float(age),
                "zone_vie": zone_vie,
                "source": source,
                "freq_cardiaque": float(freq_cardiaque),
                "tension_sys": float(tension_sys),
                "temp": float(temp),
                "sat_oxygene": float(sat_oxygene),
                "antecedents": float(antecedents),
                "duree_symptomes": float(duree_symptomes),
                "description_symptomes": description
            }

            with st.spinner("Analyse en cours..."):
                try:
                    resp = requests.post(f"{API_URL}/predict", json=payload)
                    result = resp.json()

                    niveau = result['niveau_urgence']
                    label  = result['label']
                    probas = result['probabilites']

                    # affichage résultat
                    st.divider()
                    st.subheader("📊 Résultat de l'analyse")

                    if niveau == 0:
                        st.success(f"## ✅ {label}")
                    elif niveau == 1:
                        st.warning(f"## ⚠️ {label}")
                    else:
                        st.error(f"## 🚨 {label}")

                    # Probabilités
                    col_a, col_b, col_c = st.columns(3)
                    col_a.metric("Non urgent",
                                 f"{probas['non_urgent']*100:.1f}%")
                    col_b.metric("Urgence relative",
                                 f"{probas['urgence_relative']*100:.1f}%")
                    col_c.metric("Urgence vitale",
                                 f"{probas['urgence_vitale']*100:.1f}%")

                    if niveau == 2:
                        st.error("🚨 **Action immédiate requise** — "
                                 "Contacter les secours d'urgence (15 / 112)")

                except Exception as e:
                    st.error(f"Erreur API : {e}")

# page: Historique
elif page == "📊 Historique":
    st.subheader("📊 Historique des inférences")

    limit = st.slider("Nombre d'entrées à afficher", 5, 50, 10)

    try:
        resp = requests.get(f"{API_URL}/history?limit={limit}")
        data = resp.json()

        if data['count'] == 0:
            st.info("Aucune inférence enregistrée pour le moment.")
        else:
            st.caption(f"{data['count']} inférence(s) affichée(s)")
            df = pd.DataFrame(data['history'])
            df['niveau_urgence'] = df['niveau_urgence'].map(
                {0: '✅ Non urgent', 1: '⚠️ Urgence relative', 2: '🚨 Urgence vitale'}
            )
            st.dataframe(df[['timestamp', 'age', 'sexe',
                              'freq_cardiaque', 'sat_oxygene',
                              'niveau_urgence', 'label']],
                         use_container_width=True)
    except Exception as e:
        st.error(f"Erreur : {e}")