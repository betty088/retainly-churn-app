import streamlit as st
import pandas as pd
import numpy as np
import time
from datetime import datetime
from itertools import combinations
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns

APP_NAME = "Retainly"

st.set_page_config(page_title=f"{APP_NAME} — Prévention du Churn", layout="wide", page_icon="◆")

# ============================================================
# STYLE — palette adoucie (mauve/nude/bleu-gris), navbar épurée
# ============================================================
st.markdown(f"""
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap" rel="stylesheet">
<style>
    html, body, [class*="css"]  {{ font-family: 'Inter', sans-serif; }}
    .stApp {{ background-color: #FAF8F7; }}

    section[data-testid="stSidebar"] {{
        background-color: #FFFFFF;
        border-right: 1px solid #ECE6E5;
    }}

    .navbar {{
        display: flex; justify-content: space-between; align-items: center;
        padding: 16px 6px; margin-bottom: 26px;
        border-bottom: 1px solid #ECE6E5;
        flex-wrap: wrap; gap: 18px;
    }}
    .navbar-name {{ font-size: 19px; font-weight: 800; color: #3D2B35; letter-spacing: -.01em; }}
    .navbar-sub {{ font-size: 12.5px; color: #9C8E92; margin-top: 1px; }}

    .nav-steps {{ display: flex; align-items: center; gap: 4px; }}

    .hero {{
        padding: 42px 42px;
        border-radius: 20px;
        background: linear-gradient(120deg, #F7F2F1 0%, #F1EEF5 50%, #EEF1F5 100%);
        border: 1px solid #ECE6E5;
        margin-bottom: 30px;
    }}
    .hero-eyebrow {{
        display: inline-block; color: #8B5A6B; font-size: 12.5px; font-weight: 700;
        letter-spacing: .05em; margin-bottom: 12px; text-transform: uppercase;
    }}
    .hero h1 {{ font-size: 30px; margin: 0 0 12px 0; color: #3D2B35; font-weight: 800; letter-spacing: -.02em; max-width: 620px; }}
    .hero p {{ color: #7A6E72; font-size: 15px; margin: 0; line-height: 1.65; max-width: 580px; }}

    h2, h3 {{ color: #3D2B35 !important; font-weight: 700 !important; letter-spacing: -.01em; }}
    p, label, .stMarkdown, .stCaption {{ color: #7A6E72; }}

    .card {{
        background-color: #FFFFFF;
        border: 1px solid #ECE6E5;
        border-radius: 16px;
        padding: 22px 24px;
        margin-bottom: 14px;
        box-shadow: 0 2px 10px rgba(61,43,53,.04);
    }}

    .empty-state {{
        background-color: #FFFFFF; border: 1px dashed #D9CFCF; border-radius: 16px;
        padding: 30px 26px; text-align: center; margin: 10px 0;
    }}
    .empty-state-icon {{
        width: 42px; height: 42px; border-radius: 12px; background-color: #F1EEF5;
        display: flex; align-items: center; justify-content: center; margin: 0 auto 14px auto;
        color: #8B5A6B; font-size: 19px; font-weight: 700;
    }}
    .empty-state-title {{ color: #3D2B35; font-weight: 700; font-size: 15px; margin-bottom: 5px; }}
    .empty-state-desc {{ color: #9C8E92; font-size: 13.5px; }}

    .overview-label {{
        font-size: 11.5px; color: #AFA0A4; font-weight: 700; letter-spacing: .06em;
        text-transform: uppercase; margin-bottom: 12px;
    }}
    .flow {{ display: flex; flex-direction: column; gap: 0; }}
    .flow-item {{
        display: flex; gap: 12px; align-items: flex-start;
        padding: 11px 0; border-bottom: 1px solid #F3EFEE;
    }}
    .flow-item:last-child {{ border-bottom: none; }}
    .flow-num {{
        width: 22px; height: 22px; border-radius: 7px;
        background-color: #F1EEF5; color: #6B5B73; font-weight: 700; font-size: 11.5px;
        display: flex; align-items: center; justify-content: center;
        flex-shrink: 0;
    }}
    .flow-title {{ color: #3D2B35; font-weight: 600; font-size: 13.5px; margin: 0; }}
    .flow-desc {{ color: #AFA0A4; font-size: 12px; margin: 2px 0 0 0; }}

    div[data-testid="stMetric"] {{
        background-color: #FFFFFF;
        border: 1px solid #ECE6E5;
        border-radius: 14px;
        padding: 16px 18px;
        box-shadow: 0 2px 10px rgba(61,43,53,.04);
    }}
    div[data-testid="stMetricLabel"] {{ color: #AFA0A4; font-weight: 500; }}
    div[data-testid="stMetricValue"] {{ color: #3D2B35; font-weight: 700; }}

    .risk-bar-track {{
        width: 100%; height: 7px; background: #F3EFEE; border-radius: 5px;
        overflow: hidden; margin-top: 9px;
    }}
    .risk-bar-fill {{ height: 100%; border-radius: 5px; }}

    .pipeline-step {{
        background-color: #FBFAFA;
        border-left: 2px solid #E4DCE0;
        border-radius: 8px;
        padding: 11px 15px;
        margin-bottom: 7px;
        color: #6E6266;
        font-size: 13.5px;
        animation: fadeIn .3s ease-in;
    }}
    .pipeline-step-done {{ border-left: 2px solid #8B5A6B; }}
    @keyframes fadeIn {{ from {{opacity:0; transform: translateY(3px);}} to {{opacity:1; transform: translateY(0);}} }}

    .badge-alerte {{
        background-color: #F7ECEE; color: #7A2E3D; border: 1px solid #EAD6DA;
        padding: 6px 16px; border-radius: 20px; font-weight: 800; font-size: 13px;
        letter-spacing: .03em;
    }}
    .badge-ok {{
        background-color: #EDF1F5; color: #2F4A66; border: 1px solid #D8E1EA;
        padding: 6px 16px; border-radius: 20px; font-weight: 800; font-size: 13px;
        letter-spacing: .03em;
    }}
    .badge-already {{
        background-color: #FAF5EA; color: #7A5B22; border: 1px solid #EADFC3;
        padding: 4px 12px; border-radius: 14px; font-weight: 700; font-size: 12px;
    }}
    .why-box {{
        background-color: #FBFAFA; border: 1px solid #ECE6E5; border-left: 3px solid #8B5A6B;
        border-radius: 12px; padding: 16px 18px; margin-top: 12px; color: #4A3E42; font-size: 14px;
        line-height: 1.65; font-weight: 500;
    }}

    .stButton>button {{
        background-color: #6B5B73;
        color: #FFFFFF; border: none; border-radius: 9px; font-weight: 600;
        padding: 10px 4px;
    }}
    .stButton>button:hover {{ background-color: #7C6B85; }}

    .stDownloadButton>button {{
        background-color: #FFFFFF; color: #6B5B73; border: 1px solid #D9CFD3;
        border-radius: 9px; font-weight: 600;
    }}
    .stDownloadButton>button:hover {{ background-color: #F7F2F1; }}

    div[data-testid="stDataFrame"] {{ border-radius: 12px; overflow: hidden; }}
    hr {{ border-color: #ECE6E5 !important; }}

    .nav-arrow {{ text-align: center; color: #B5A8AC; font-size: 18px; font-weight: 700; padding-top: 6px; }}
    button[kind="secondary"] {{
        background-color: #FFFFFF !important;
        color: #1A1418 !important;
        font-weight: 800 !important;
        border: 1.5px solid #D9CFD3 !important;
    }}
    button[kind="secondary"]:hover {{
        background-color: #F7F2F1 !important;
        border-color: #8B5A6B !important;
        color: #1A1418 !important;
    }}
    button[kind="secondary"] p {{ color: #1A1418 !important; font-weight: 800 !important; }}
</style>

<div class="navbar">
    <div>
        <div class="navbar-name">{APP_NAME}</div>
        <div class="navbar-sub">Plateforme de prévention du churn</div>
    </div>
</div>
""", unsafe_allow_html=True)

if 'page' not in st.session_state:
    st.session_state.page = "Données & Modèle"

step_targets = {
    "1 · Import": "Données & Modèle",
    "2 · Dashboard": "Données & Modèle",
    "3 · Modèle": "Données & Modèle",
    "4 · Client": "Analyser un client",
    "5 · Agent": "Analyser un client",
    "6 · Digital Twin": "Analyser un client",
    "7 · Recommandation": "Analyser un client",
}
items = list(step_targets.items())
widths = []
for i in range(len(items)):
    widths.append(4)
    if i != len(items) - 1:
        widths.append(1)
nav_cols = st.columns(widths)
col_idx = 0
for i, (label, target) in enumerate(items):
    with nav_cols[col_idx]:
        if st.button(label, key=f"navstep_{label}", use_container_width=True):
            st.session_state.page = target
            st.rerun()
    col_idx += 1
    if i != len(items) - 1:
        with nav_cols[col_idx]:
            st.markdown('<div class="nav-arrow">→</div>', unsafe_allow_html=True)
        col_idx += 1

# ============================================================
# ETAT DE SESSION
# ============================================================
defaults = {
    'model': None, 'scaler': None, 'X': None, 'y': None,
    'feature_columns': None, 'numeric_features': None,
    'X_test': None, 'y_test': None, 'y_proba': None, 'y_pred': None,
    'retention_levers': None,
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.markdown(f"""
    <div style="font-size:16px; font-weight:800; color:#3D2B35;">{APP_NAME}</div>
    <div style="font-size:12.5px; color:#AFA0A4; margin:3px 0 0 0;">Prévention du churn client</div>
    """, unsafe_allow_html=True)
    st.divider()
    page = st.radio("Navigation", ["Données & Modèle", "Analyser un client"],
                     label_visibility="collapsed", key="page")
    st.divider()

    if st.session_state.model is not None:
        if st.button("Réinitialiser", use_container_width=True):
            for k in list(defaults.keys()):
                st.session_state[k] = defaults[k]
            st.rerun()
        st.divider()
        auc = roc_auc_score(st.session_state.y_test, st.session_state.y_proba)
        st.markdown('<div class="overview-label">Modèle actif</div>', unsafe_allow_html=True)
        st.markdown(f"**ROC-AUC** {auc:.3f}")
        st.caption(f"{len(st.session_state.feature_columns)} variables · {len(st.session_state.retention_levers or {})} leviers")

st.markdown("""
<div class="hero">
    <span class="hero-eyebrow">Plateforme de rétention client</span>
    <h1>Anticipez le départ de vos clients avant qu'il ne soit trop tard</h1>
    <p>Chargez vos données clients, entraînez le modèle de prédiction, puis laissez l'agent analyser chaque client individuellement et recommander l'action de rétention la plus efficace pour son profil.</p>
</div>
""", unsafe_allow_html=True)

# ============================================================
# PAGE 1 — DONNEES & MODELE
# ============================================================
if page == "Données & Modèle":
    st.subheader("1 · Charger vos données")
    uploaded_file = st.file_uploader("Fichier CSV (données clients)", type=["csv"])

    if uploaded_file is not None:
        df_raw = pd.read_csv(uploaded_file)
        st.success(f"Fichier chargé — {df_raw.shape[0]} lignes, {df_raw.shape[1]} colonnes")
        with st.expander("Aperçu des données"):
            st.dataframe(df_raw.head())

        st.subheader("Dashboard — aperçu des données")
        d1, d2, d3 = st.columns(3)
        d1.metric("Nombre de clients", f"{df_raw.shape[0]}")
        d2.metric("Nombre de variables", f"{df_raw.shape[1]}")
        d3.metric("Valeurs manquantes", f"{int(df_raw.isnull().sum().sum())}")
        num_cols_preview = df_raw.select_dtypes(include=[np.number]).columns.tolist()
        if num_cols_preview:
            st.caption("Distribution d'une variable numérique (aperçu rapide)")
            preview_col = st.selectbox("Variable à visualiser", num_cols_preview, key="dash_preview_col")
            fig_d, ax_d = plt.subplots(figsize=(7, 2.8))
            fig_d.patch.set_alpha(0)
            ax_d.set_facecolor("none")
            ax_d.hist(df_raw[preview_col].dropna(), bins=25, color='#8B5A6B')
            ax_d.tick_params(colors='#7A6E72')
            for spine in ax_d.spines.values(): spine.set_color('#ECE6E5')
            st.pyplot(fig_d)

        st.subheader("2 · Configuration")
        col1, col2 = st.columns(2)
        all_columns = df_raw.columns.tolist()

        with col1:
            target_col = st.selectbox("Colonne cible (churn / a quitté ?)", all_columns)
            id_col = st.selectbox("Colonne identifiant (optionnel)", ["Aucune"] + all_columns)

        with col2:
            numeric_features = st.multiselect(
                "Colonnes numériques (âge, montant, durée...)",
                [c for c in all_columns if c != target_col],
            )

        remaining_cols = [c for c in all_columns if c not in numeric_features and c != target_col and c != id_col]
        categorical_features = st.multiselect(
            "Colonnes catégorielles à encoder", remaining_cols, default=remaining_cols,
        )

        st.subheader("3 · Leviers de rétention")
        st.caption("Actions que l'agent pourra tester, seules ou combinées, pour réduire le risque")
        lever_cols = st.multiselect("Colonnes-leviers disponibles", categorical_features)

        if st.button("Lancer l'analyse et entraîner le modèle", type="primary"):
            status = st.status("Traitement en cours...", expanded=True)

            status.write("Nettoyage des variables numériques...")
            df = df_raw.copy()
            for col in numeric_features:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                median_val = df[col].median()
                df[col] = df[col].fillna(median_val if pd.notna(median_val) else 0)

            if id_col != "Aucune" and id_col in df.columns:
                df = df.drop(id_col, axis=1)

            def is_textual(series):
                return not pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series)

            status.write("Encodage de la variable cible et des variables catégorielles...")
            if is_textual(df[target_col]):
                uniques = df[target_col].dropna().unique()
                if len(uniques) == 2:
                    df[target_col] = df[target_col].map({uniques[0]: 0, uniques[1]: 1})

            for col in categorical_features:
                if is_textual(df[col]):
                    uniques = df[col].dropna().unique()
                    if len(uniques) == 2:
                        df[col] = df[col].map({uniques[0]: 0, uniques[1]: 1})

            remaining_multi = [c for c in categorical_features if is_textual(df[c])]
            df = pd.get_dummies(df, columns=remaining_multi, drop_first=True)
            df = df.dropna(subset=[target_col])

            X = df.drop(target_col, axis=1)
            y = df[target_col].astype(int)
            X = X.select_dtypes(include=[np.number, 'bool']).astype(float)

            status.write("Séparation entraînement / test et standardisation...")
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )

            scaler = StandardScaler()
            valid_numeric = [c for c in numeric_features if c in X.columns]
            X_train_s, X_test_s = X_train.copy(), X_test.copy()
            if valid_numeric:
                X_train_s[valid_numeric] = scaler.fit_transform(X_train[valid_numeric])
                X_test_s[valid_numeric] = scaler.transform(X_test[valid_numeric])

            status.write("Entraînement du modèle (Random Forest)...")
            model = RandomForestClassifier(n_estimators=200, max_depth=10, class_weight='balanced', random_state=42)
            model.fit(X_train_s, y_train)

            y_pred = model.predict(X_test_s)
            y_proba = model.predict_proba(X_test_s)[:, 1]

            status.write("Construction des leviers de rétention et de leurs combinaisons...")
            simple_levers = {}
            for col in lever_cols:
                if col in X.columns:
                    simple_levers[f"{col}"] = {col: 1}
                else:
                    matching = [c for c in X.columns if c.startswith(f"{col}_")]
                    for m in matching:
                        others = [c for c in matching if c != m]
                        changes = {m: 1}
                        for o in others:
                            changes[o] = 0
                        label = m.replace(f"{col}_", f"{col} : ")
                        simple_levers[label] = changes

            retention_levers = {f"Activer : {name}": changes for name, changes in simple_levers.items()}
            lever_names = list(simple_levers.keys())
            for a, b in combinations(lever_names, 2):
                combined = {**simple_levers[a], **simple_levers[b]}
                retention_levers[f"Package : {a} + {b}"] = combined

            st.session_state.update({
                'model': model, 'scaler': scaler, 'X': X, 'y': y,
                'feature_columns': X.columns.tolist(), 'numeric_features': valid_numeric,
                'X_test': X_test_s, 'y_test': y_test, 'y_proba': y_proba, 'y_pred': y_pred,
                'retention_levers': retention_levers,
            })

            status.update(label="Modèle entraîné avec succès", state="complete", expanded=False)

        if st.session_state.model is not None:
            st.info("Modèle prêt. Ouvrez « Analyser un client » dans le menu de gauche pour lancer l'agent.")

    else:
        st.markdown("""
        <div class="empty-state">
            <div class="empty-state-icon">＋</div>
            <div class="empty-state-title">Aucune donnée chargée</div>
            <div class="empty-state-desc">Importez un fichier CSV de clients ci-dessus pour commencer l'analyse.</div>
        </div>
        """, unsafe_allow_html=True)

    if st.session_state.model is not None:
        st.divider()
        st.subheader("Performance du modèle")
        c1, c2 = st.columns([1, 1.4])
        with c1:
            auc = roc_auc_score(st.session_state.y_test, st.session_state.y_proba)
            report = classification_report(st.session_state.y_test, st.session_state.y_pred, output_dict=True)
            m1, m2 = st.columns(2)
            m1.metric("ROC-AUC", f"{auc:.3f}")
            m2.metric("Accuracy", f"{report['accuracy']:.3f}")
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.caption("Facteurs les plus déterminants")
            importances = pd.DataFrame({
                'feature': st.session_state.feature_columns,
                'importance': st.session_state.model.feature_importances_
            }).sort_values('importance', ascending=False).head(8)
            fig2, ax2 = plt.subplots(figsize=(5, 3.4))
            fig2.patch.set_alpha(0)
            ax2.set_facecolor("none")
            ax2.barh(importances['feature'][::-1], importances['importance'][::-1], color='#8B5A6B')
            ax2.tick_params(colors='#7A6E72')
            for spine in ax2.spines.values(): spine.set_color('#ECE6E5')
            st.pyplot(fig2)
            st.markdown('</div>', unsafe_allow_html=True)
        with c2:
            cm = confusion_matrix(st.session_state.y_test, st.session_state.y_pred)
            fig, ax = plt.subplots(figsize=(4.6, 3.6))
            fig.patch.set_alpha(0)
            mauve_cmap = sns.light_palette("#8B5A6B", as_cmap=True)
            sns.heatmap(cm, annot=True, fmt='d', cmap=mauve_cmap, ax=ax, cbar=False)
            ax.set_xlabel("Prédit", color='#7A6E72'); ax.set_ylabel("Réel", color='#7A6E72')
            ax.tick_params(colors='#7A6E72')
            st.pyplot(fig)

# ============================================================
# CLASSES — Agent d'abord, Digital Twin ensuite (piloté par l'Agent)
# ============================================================
class ClientDigitalTwin:
    def __init__(self, features_dict, model, scaler, numeric_features, feature_columns):
        self.state = features_dict.copy()
        self.model = model
        self.scaler = scaler
        self.numeric_features = numeric_features
        self.feature_columns = feature_columns

    def _prepare(self, state_dict):
        row = pd.DataFrame([state_dict])[self.feature_columns]
        row_scaled = row.copy()
        if self.numeric_features:
            row_scaled[self.numeric_features] = self.scaler.transform(row[self.numeric_features])
        return row_scaled

    def get_risk(self, state_dict=None):
        state_dict = state_dict or self.state
        proba = self.model.predict_proba(self._prepare(state_dict))[0][1]
        return round(proba * 100, 2)


class RetentionAgent:
    def __init__(self, model, scaler, numeric_features, feature_columns, retention_levers, risk_threshold=50):
        self.model = model
        self.scaler = scaler
        self.numeric_features = numeric_features
        self.feature_columns = feature_columns
        self.retention_levers = retention_levers
        self.risk_threshold = risk_threshold

    def analyze_client(self, client_raw_dict, progress_callback=None):
        twin = ClientDigitalTwin(client_raw_dict, self.model, self.scaler,
                                  self.numeric_features, self.feature_columns)

        if progress_callback: progress_callback("risk_computed")
        base_risk = twin.get_risk()

        if base_risk < self.risk_threshold:
            if progress_callback: progress_callback("decision_ok")
            return {'statut': 'OK', 'risque_initial': base_risk,
                    'message': "Client à faible risque — aucune action requise."}

        if progress_callback: progress_callback("decision_alerte")

        results, skipped = [], []
        for action_name, changes in self.retention_levers.items():
            already_active = all(client_raw_dict.get(k) == v for k, v in changes.items())
            if already_active:
                skipped.append(action_name)
                continue
            simulated_state = client_raw_dict.copy()
            simulated_state.update(changes)
            new_risk = twin.get_risk(simulated_state)
            results.append({'action': action_name, 'nouveau_risque': new_risk,
                             'reduction': round(base_risk - new_risk, 2)})
            if progress_callback: progress_callback("simulation", action_name, new_risk)

        if not results:
            if progress_callback: progress_callback("done")
            return {'statut': 'ALERTE', 'risque_initial': base_risk,
                    'message': "Toutes les actions possibles sont déjà actives chez ce client.",
                    'deja_actifs': skipped}

        ranked = sorted(results, key=lambda x: -x['reduction'])
        best = ranked[0]
        if progress_callback: progress_callback("done")

        return {'statut': 'ALERTE', 'risque_initial': base_risk,
                'recommandation': best['action'], 'nouveau_risque': best['nouveau_risque'],
                'reduction': best['reduction'], 'toutes_options': ranked[:10],
                'nb_testees': len(results), 'deja_actifs': skipped}


def risk_color(risk):
    if risk < 40: return "#4F6B8A"
    if risk < 70: return "#A5813F"
    return "#9B4D5D"


def build_text_report(client_idx, result):
    lines = [
        f"{APP_NAME} — Rapport d'analyse client",
        f"Généré le {datetime.now().strftime('%d/%m/%Y à %H:%M')}",
        "=" * 50,
        f"Client (index) : {client_idx}",
        f"Statut : {result['statut']}",
        f"Risque initial estimé : {result['risque_initial']}%",
    ]
    if result['statut'] == 'OK':
        lines.append(result['message'])
    else:
        if 'recommandation' in result:
            lines += [
                f"Risque après action recommandée : {result['nouveau_risque']}%",
                f"Réduction obtenue : -{result['reduction']} points",
                f"Action recommandée : {result['recommandation']}",
                "",
                f"Cette recommandation a été retenue parmi {result['nb_testees']} actions et",
                "combinaisons testées par le Digital Twin, comme celle produisant la plus",
                "forte baisse du risque de départ pour ce client.",
                "",
                "Détail des simulations testées (classées par efficacité) :",
            ]
            for opt in result['toutes_options']:
                lines.append(f"  - {opt['action']} → {opt['nouveau_risque']}% (réduction {opt['reduction']} pts)")
        else:
            lines.append(result.get('message', ''))
        if result.get('deja_actifs'):
            lines.append("")
            lines.append("Leviers déjà actifs chez ce client (non testés) :")
            for a in result['deja_actifs']:
                lines.append(f"  - {a}")
    return "\n".join(lines)


def render_result(result, client_idx):
    st.divider()
    st.caption(f"Client analysé : {client_idx}")
    if result['statut'] == 'OK':
        st.markdown('<span class="badge-ok">Risque faible</span>', unsafe_allow_html=True)
        st.markdown(f"""
        <div class="card">
            <div style="font-size:34px; font-weight:800; color:{risk_color(result['risque_initial'])};">
                {result['risque_initial']}%
            </div>
            <div class="risk-bar-track">
                <div class="risk-bar-fill" style="width:{result['risque_initial']}%; background:{risk_color(result['risque_initial'])};"></div>
            </div>
            <p style="margin-top:10px;">{result['message']}</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown('<span class="badge-alerte">Alerte churn</span>', unsafe_allow_html=True)

        if 'recommandation' in result:
            c1, c2, c3 = st.columns(3)
            with c1:
                st.markdown(f"""
                <div class="card">
                    <div style="color:#AFA0A4; font-size:13px;">Risque initial</div>
                    <div style="font-size:28px; font-weight:800; color:{risk_color(result['risque_initial'])};">{result['risque_initial']}%</div>
                    <div class="risk-bar-track"><div class="risk-bar-fill" style="width:{result['risque_initial']}%; background:{risk_color(result['risque_initial'])};"></div></div>
                </div>""", unsafe_allow_html=True)
            with c2:
                st.markdown(f"""
                <div class="card">
                    <div style="color:#AFA0A4; font-size:13px;">Risque après action</div>
                    <div style="font-size:28px; font-weight:800; color:{risk_color(result['nouveau_risque'])};">{result['nouveau_risque']}%</div>
                    <div class="risk-bar-track"><div class="risk-bar-fill" style="width:{result['nouveau_risque']}%; background:{risk_color(result['nouveau_risque'])};"></div></div>
                </div>""", unsafe_allow_html=True)
            with c3:
                st.markdown(f"""
                <div class="card">
                    <div style="color:#AFA0A4; font-size:13px;">Réduction obtenue</div>
                    <div style="font-size:28px; font-weight:800; color:#8B5A6B;">-{result['reduction']} pts</div>
                    <div style="margin-top:6px; color:#3D2B35; font-size:13px; font-weight:600;">{result['recommandation']}</div>
                </div>""", unsafe_allow_html=True)

            n_alt = result['nb_testees'] - 1
            st.markdown(f"""
            <div class="why-box">
                <strong style="color:#3D2B35;">Pourquoi cette recommandation ?</strong><br>
                Ce client a été analysé individuellement : parmi les {result['nb_testees']} actions et combinaisons
                testées par le Digital Twin pour son profil précis, « {result['recommandation']} » est celle qui
                produit la plus forte baisse de son risque de départ (-{result['reduction']} points),
                devançant les {n_alt} autres options simulées pour ce même client.
            </div>
            """, unsafe_allow_html=True)

            st.caption("Top simulations testées pour ce client, classées par efficacité :")
            df_opts = pd.DataFrame(result['toutes_options'])
            st.dataframe(
                df_opts,
                use_container_width=True, hide_index=True,
                column_config={
                    "action": "Action testée",
                    "nouveau_risque": st.column_config.ProgressColumn(
                        "Nouveau risque", format="%.1f%%", min_value=0, max_value=100),
                    "reduction": st.column_config.NumberColumn("Réduction (pts)", format="%.2f"),
                }
            )

            dcol1, dcol2 = st.columns(2)
            with dcol1:
                st.download_button(
                    "Exporter le rapport (.txt)",
                    data=build_text_report(client_idx, result),
                    file_name=f"rapport_client_{client_idx}.txt",
                    mime="text/plain",
                    key=f"dl_txt_{client_idx}",
                )
            with dcol2:
                st.download_button(
                    "Exporter les simulations (.csv)",
                    data=df_opts.to_csv(index=False).encode('utf-8'),
                    file_name=f"simulations_client_{client_idx}.csv",
                    mime="text/csv",
                    key=f"dl_csv_{client_idx}",
                )
        else:
            st.markdown(f'<div class="card">{result.get("message","")}</div>', unsafe_allow_html=True)

        if result.get('deja_actifs'):
            st.caption("Leviers déjà actifs chez ce client (non testés) :")
            st.markdown(" ".join(f'<span class="badge-already">{a}</span>' for a in result['deja_actifs']),
                        unsafe_allow_html=True)


# ============================================================
# PAGE 2 — ANALYSE D'UN CLIENT
# ============================================================
if page == "Analyser un client":
    if st.session_state.model is None:
        st.markdown("""
        <div class="empty-state">
            <div class="empty-state-icon">◆</div>
            <div class="empty-state-title">Aucun modèle entraîné</div>
            <div class="empty-state-desc">Rendez-vous dans « Données & Modèle » pour importer vos données et entraîner le modèle avant d'analyser un client.</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        if st.session_state.get('last_result') is not None:
            st.subheader("Dernière analyse effectuée")
            render_result(st.session_state['last_result'], st.session_state['last_client_idx'])
            st.divider()

        st.subheader("Sélectionner un nouveau client")
        client_idx = st.selectbox("Client (index dans l'échantillon de test)",
                                   st.session_state.X_test.index.tolist())
        client_state = st.session_state.X_test.loc[client_idx].to_dict()

        if not st.session_state.retention_levers:
            st.markdown("""
            <div class="empty-state">
                <div class="empty-state-icon">!</div>
                <div class="empty-state-title">Aucun levier de rétention défini</div>
                <div class="empty-state-desc">Retournez dans « Données & Modèle » pour sélectionner au moins un levier avant de lancer l'agent.</div>
            </div>
            """, unsafe_allow_html=True)

        launch = st.button("Lancer l'agent sur ce client", type="primary",
                            disabled=not st.session_state.retention_levers)

        if launch:
            agent = RetentionAgent(
                model=st.session_state.model, scaler=st.session_state.scaler,
                numeric_features=st.session_state.numeric_features,
                feature_columns=st.session_state.feature_columns,
                retention_levers=st.session_state.retention_levers
            )

            steps_placeholder = st.empty()
            log = []

            def render_log():
                html = "".join(f'<div class="pipeline-step pipeline-step-done">{s}</div>' for s in log)
                steps_placeholder.markdown(html, unsafe_allow_html=True)

            def cb(event, *args):
                if event == "risk_computed":
                    log.append("Étape 1 — Chargement du client dans le Digital Twin et calcul du risque initial.")
                elif event == "decision_ok":
                    log.append("Étape 2 — Décision de l'Agent : risque faible, aucune simulation nécessaire.")
                elif event == "decision_alerte":
                    log.append("Étape 2 — Décision de l'Agent : risque élevé, activation du Digital Twin.")
                elif event == "simulation":
                    log.append(f"Étape 3 — Simulation : {args[0]} → risque estimé {args[1]}%")
                elif event == "done":
                    log.append("Étape 4 — Comparaison de toutes les simulations et sélection de la meilleure.")
                render_log()
                time.sleep(0.12)

            result = agent.analyze_client(client_state, progress_callback=cb)
            st.session_state['last_result'] = result
            st.session_state['last_client_idx'] = client_idx
            st.rerun()

st.divider()
foot1, foot2 = st.columns([3, 1])
with foot1:
    st.caption(f"{APP_NAME} — Plateforme de prévention du churn basée sur un Digital Twin et un Agent de recommandation")
    st.caption("Les données importées restent locales à votre session et ne sont pas conservées après la fermeture de l'application.")
with foot2:
    st.caption("Version 1.0")
