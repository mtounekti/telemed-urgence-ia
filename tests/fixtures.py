"""
Fixtures de données pour les tests — suit strictement la structure
du dataset_telemed.csv (colonnes, bornes physiologiques, types).
"""

# Cas critique: urgence vitale claire (constantes dégradées + texte alarmant)
PATIENT_CRITIQUE = {
    "sexe": "F",
    "age": 65,
    "zone_vie": "U",
    "source": "appel",
    "freq_cardiaque": 130,
    "tension_sys": 190,
    "temp": 39.8,
    "sat_oxygene": 85.0,
    "antecedents": 1,
    "duree_symptomes": 1.0,
    "description_symptomes": "Douleur thoracique intense avec essoufflement sévère",
}

# Cas non urgent: constantes normales, motif bénin
PATIENT_NON_URGENT = {
    "sexe": "H",
    "age": 30,
    "zone_vie": "U",
    "source": "chat",
    "freq_cardiaque": 70,
    "tension_sys": 115,
    "temp": 37.0,
    "sat_oxygene": 99.0,
    "antecedents": 0,
    "duree_symptomes": 48.0,
    "description_symptomes": "Renouvellement ordonnance traitement habituel",
}

# Cas limite: valeurs proches des bornes physiologiques mais valides
PATIENT_LIMITE_VALIDE = {
    "sexe": "F",
    "age": 0,
    "zone_vie": "R",
    "source": "appel",
    "freq_cardiaque": 30,
    "tension_sys": 50,
    "temp": 34.0,
    "sat_oxygene": 50.0,
    "antecedents": 0,
    "duree_symptomes": 0,
    "description_symptomes": "abc",
}

# Cas urgence relative: entre les deux extrêmes
PATIENT_RELATIF = {
    "sexe": "H",
    "age": 45,
    "zone_vie": "R",
    "source": "chat",
    "freq_cardiaque": 100,
    "tension_sys": 145,
    "temp": 38.2,
    "sat_oxygene": 94.0,
    "antecedents": 1,
    "duree_symptomes": 24.0,
    "description_symptomes": "Traumatisme direct avec saignement persistant",
}


def invalid_payload(field: str, value):
    """Retourne PATIENT_CRITIQUE avec un seul champ modifié pour tester une violation de borne"""
    payload = dict(PATIENT_CRITIQUE)
    payload[field] = value
    return payload


# Violations de bornes physiologiques (une par variable numérique/catégorielle contrainte)
INVALID_CASES = {
    "sexe_invalide": invalid_payload("sexe", "X"),
    "zone_vie_invalide": invalid_payload("zone_vie", "Z"),
    "source_invalide": invalid_payload("source", "email"),
    "age_negatif": invalid_payload("age", -1),
    "age_trop_eleve": invalid_payload("age", 150),
    "freq_cardiaque_trop_basse": invalid_payload("freq_cardiaque", 10),
    "freq_cardiaque_trop_haute": invalid_payload("freq_cardiaque", 300),
    "tension_sys_trop_basse": invalid_payload("tension_sys", 10),
    "temp_trop_basse": invalid_payload("temp", 20.0),
    "temp_trop_haute": invalid_payload("temp", 50.0),
    "sat_oxygene_negative": invalid_payload("sat_oxygene", -5.0),
    "sat_oxygene_trop_haute": invalid_payload("sat_oxygene", 150.0),
    "antecedents_invalide": invalid_payload("antecedents", 2),
    "duree_symptomes_negative": invalid_payload("duree_symptomes", -1.0),
    "description_vide": invalid_payload("description_symptomes", ""),
}