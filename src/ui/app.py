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
    page = st.radio("", ["🔍 Prédiction", "📊 Historique", "📝 Feedbacks"])

# Page prédiction
if page == "🔍 Prédiction":
    st.subheader("📝 Informations du patient")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**Données administratives**")
        sexe = st.selectbox("Sexe", ["F", "H"])
        age = st.number_input("Âge", min_value=0, max_value=120, value=45)
        zone_vie = st.selectbox("Zone de vie", ["U", "R"], help="U = Urbain / R = Rural")
        source = st.selectbox("Source", ["appel", "chat"])
        antecedents = st.selectbox("Antécédents médicaux", [0, 1],
                                    format_func=lambda x: "Oui" if x == 1 else "Non")

    with col2:
        st.markdown("**Constantes vitales**")
        freq_cardiaque = st.number_input("Fréquence cardiaque (bpm)", min_value=30, max_value=250, value=80)
        tension_sys = st.number_input("Tension systolique (mmHg)", min_value=50, max_value=300, value=120)
        temp = st.number_input("Température (°C)", min_value=34.0, max_value=43.0, value=37.0, step=0.1)
        sat_oxygene = st.number_input("Saturation O2 (%)", min_value=50.0, max_value=100.0, value=98.0, step=0.1)
        duree_symptomes = st.number_input("Durée des symptômes (heures)", min_value=0.0, value=24.0, step=0.5)

    with col3:
        st.markdown("**Description des symptômes**")
        description = st.text_area(
            "Décrivez les symptômes", height=200,
            placeholder="Ex: Douleur thoracique intense avec essoufflement..."
        )

    st.divider()

    if st.button("🔍 Analyser l'urgence", type="primary", use_container_width=True):
        if not description.strip():
            st.warning("⚠️ Veuillez renseigner la description des symptômes.")
        else:
            payload = {
                "sexe": sexe, "age": float(age), "zone_vie": zone_vie,
                "source": source, "freq_cardiaque": float(freq_cardiaque),
                "tension_sys": float(tension_sys), "temp": float(temp),
                "sat_oxygene": float(sat_oxygene), "antecedents": float(antecedents),
                "duree_symptomes": float(duree_symptomes),
                "description_symptomes": description
            }
            with st.spinner("Analyse en cours..."):
                try:
                    resp = requests.post(f"{API_URL}/predict", json=payload)
                    result = resp.json()
                    # Stocker le résultat dans session_state
                    st.session_state['last_result'] = result
                    st.session_state['last_description'] = description
                    st.session_state['feedback_sent'] = False
                except Exception as e:
                    st.error(f"Erreur API : {e}")

    # Affichage du résultat depuis session_state
    if 'last_result' in st.session_state:
        result = st.session_state['last_result']
        niveau = result['niveau_urgence']
        label  = result['label']
        probas = result['probabilites']

        st.divider()
        st.subheader("📊 Résultat de l'analyse")
        st.info(f"📋 Symptômes analysés : *{st.session_state['last_description']}*")

        if niveau == 0:
            st.success(f"## ✅ {label}")
        elif niveau == 1:
            st.warning(f"## ⚠️ {label}")
        else:
            st.error(f"## 🚨 {label}")

        col_a, col_b, col_c = st.columns(3)
        col_a.metric("Non urgent", f"{probas['non_urgent']*100:.1f}%")
        col_b.metric("Urgence relative", f"{probas['urgence_relative']*100:.1f}%")
        col_c.metric("Urgence vitale", f"{probas['urgence_vitale']*100:.1f}%")

        if niveau == 2:
            st.error("🚨 **Action immédiate requise** — Contacter les secours d'urgence (15 / 112)")

        # Feedback
        if not st.session_state.get('feedback_sent', False):
            st.divider()
            st.subheader("💬 Votre retour")
            st.caption("Ce feedback aide à améliorer le modèle.")

            col_f1, col_f2 = st.columns(2)
            with col_f1:
                utile = st.radio(
                    "Cette prédiction était-elle correcte ?",
                    ["✅ Oui, correcte", "❌ Non, incorrecte"],
                    horizontal=True
                )
            with col_f2:
                niveau_reel = st.selectbox(
                    "Niveau réel (optionnel)", [None, 0, 1, 2],
                    format_func=lambda x: "— Non renseigné —" if x is None else
                                         ["Non urgent", "Urgence relative", "Urgence vitale ⚠️"][x]
                )

            commentaire = st.text_input(
                "Commentaire (optionnel)",
                placeholder="Ex: Le patient était plus grave que prévu..."
            )

            if st.button("📤 Envoyer le feedback"):
                try:
                    feedback_payload = {
                        "niveau_predit": niveau,
                        "utile": utile == "✅ Oui, correcte",
                        "niveau_reel": niveau_reel,
                        "commentaire": commentaire if commentaire else None,
                    }
                    resp_fb = requests.post(
                        f"{API_URL}/feedback",
                        params=feedback_payload,
                        timeout=5
                    )
                    if resp_fb.status_code == 200:
                        st.session_state['feedback_sent'] = True
                        st.success("✅ Feedback enregistré, merci !")
                    else:
                        st.error("Erreur lors de l'envoi du feedback.")
                except Exception as e:
                    st.error(f"Erreur : {e}")
        else:
            st.success("✅ Feedback déjà envoyé pour cette prédiction.")

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

# page: Feedbacks
elif page == "📝 Feedbacks":
    st.subheader("📝 Historique des feedbacks")

    limit_fb = st.slider("Nombre de feedbacks à afficher", 5, 50, 10)

    try:
        resp = requests.get(f"{API_URL}/feedbacks?limit={limit_fb}")
        data = resp.json()

        if data['count'] == 0:
            st.info("Aucun feedback enregistré pour le moment.")
        else:
            st.caption(f"{data['count']} feedback(s) affiché(s)")
            df_fb = pd.DataFrame(data['feedbacks'])
            df_fb['utile'] = df_fb['utile'].map({True: '✅ Correcte', False: '❌ Incorrecte'})
            df_fb['niveau_predit'] = df_fb['niveau_predit'].map(
                {0: 'Non urgent', 1: 'Urgence relative', 2: 'Urgence vitale ⚠️'}
            )
            df_fb['niveau_reel'] = df_fb['niveau_reel'].map(
                {0: 'Non urgent', 1: 'Urgence relative', 2: 'Urgence vitale ⚠️', None: '—'}
            )
            st.dataframe(df_fb, use_container_width=True)

            # Stats rapides
            st.divider()
            col1, col2, col3 = st.columns(3)
            total = len(df_fb)
            correctes = (df_fb['utile'] == '✅ Correcte').sum()
            col1.metric("Total feedbacks", total)
            col2.metric("Prédictions correctes", correctes)
            col3.metric("Taux de satisfaction", f"{correctes/total*100:.1f}%" if total > 0 else "—")

    except Exception as e:
        st.error(f"Erreur : {e}")