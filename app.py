"""
app.py — Dashboard NER sur Documents Administratifs
spaCy custom · Entités nommées · F1 par type · Visualisation
"""

import json
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import spacy

sys.path.insert(0, str(Path(__file__).parent / "src"))
from generate_training_data import generate_dataset, LABELS as ENTITY_LABELS
from train_ner import split_dataset, train_ner_model
from evaluate_ner import evaluate_on_test

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="NER — Documents Administratifs",
    page_icon="🏷️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ───────────────────────────────────────────────────────────────────────

st.markdown("""
<style>
.stApp { background-color: #0f1117; }
[data-testid="stMetric"] {
    background: linear-gradient(135deg, #1e2130, #252a3d);
    border: 1px solid #2d3550; border-radius: 12px; padding: 16px 20px;
}
[data-testid="stMetricValue"] { color: #e2e8f0; font-size: 1.8rem; font-weight: 700; }
[data-testid="stMetricLabel"] { color: #94a3b8; font-size: 0.78rem; text-transform: uppercase; letter-spacing: 0.08em; }
h1 { color: #e2e8f0 !important; font-weight: 800 !important; }
h2, h3 { color: #cbd5e1 !important; font-weight: 600 !important; }
[data-testid="stSidebar"] { background-color: #141624 !important; border-right: 1px solid #2d3550; }
[data-testid="stSidebar"] * { color: #cbd5e1 !important; }
button[data-baseweb="tab"] { color: #94a3b8 !important; }
button[data-baseweb="tab"][aria-selected="true"] { color: #60a5fa !important; border-bottom-color: #60a5fa !important; }
.ent-tag {
    display: inline-block; padding: 2px 8px; border-radius: 6px;
    font-size: 0.78rem; font-weight: 700; margin: 0 2px;
}
.card { background: linear-gradient(135deg,#1e2130,#252a3d); border:1px solid #2d3550; border-radius:12px; padding:20px; margin-bottom:12px; }
</style>
""", unsafe_allow_html=True)

# ── Couleurs par entité ───────────────────────────────────────────────────────

ENT_COLORS = {
    "MONTANT":    ("#1d4ed8", "#dbeafe"),
    "ORGANISME":  ("#065f46", "#d1fae5"),
    "DISPOSITIF": ("#7c3aed", "#ede9fe"),
    "DUREE":      ("#92400e", "#fef3c7"),
    "ZONE_GEO":   ("#9f1239", "#ffe4e6"),
}

# ── Chargement / entraînement ─────────────────────────────────────────────────

@st.cache_data
def load_dataset_cached():
    path = Path(__file__).parent / "data" / "ner_dataset.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return generate_dataset(600)


@st.cache_resource
def train_model_cached(n_iter: int):
    dataset = load_dataset_cached()
    train_data, _ = split_dataset(dataset, train_ratio=0.8)
    nlp, history = train_ner_model(train_data, n_iter=n_iter)
    return nlp, history


@st.cache_data
def evaluate_cached(n_iter: int):
    dataset = load_dataset_cached()
    _, test_data = split_dataset(dataset, train_ratio=0.8)
    nlp, _ = train_model_cached(n_iter)
    return evaluate_on_test(nlp, test_data)


def highlight_entities(text: str, entities: list) -> str:
    """Génère du HTML avec les entités surlignées."""
    if not entities:
        return f'<span style="color:#94a3b8">{text}</span>'
    result = []
    prev = 0
    for start, end, label in sorted(entities, key=lambda x: x[0]):
        bg, fg = ENT_COLORS.get(label, ("#1e2130", "#e2e8f0"))
        result.append(f'<span style="color:#cbd5e1">{text[prev:start]}</span>')
        result.append(
            f'<span class="ent-tag" style="background:{bg};color:{fg}">'
            f'{text[start:end]}<sup style="font-size:0.65rem;margin-left:3px">{label}</sup>'
            f'</span>'
        )
        prev = end
    result.append(f'<span style="color:#cbd5e1">{text[prev:]}</span>')
    return "".join(result)


# ── Header ────────────────────────────────────────────────────────────────────

st.markdown("""
<div style="padding:24px 0 8px 0">
  <h1 style="margin:0;font-size:2rem">🏷️ NER — Documents Administratifs</h1>
  <p style="color:#64748b;margin:4px 0 0 0;font-size:0.95rem">
    Reconnaissance d'entités nommées · spaCy custom · 5 types · 600 exemples annotés
  </p>
</div>
""", unsafe_allow_html=True)

st.divider()

# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("### ⚙️ Paramètres d'entraînement")
    n_iter = st.slider("Itérations d'entraînement", 10, 50, 30, step=5)
    st.divider()
    st.markdown("### 🏷️ Entités détectées")
    for label, (bg, fg) in ENT_COLORS.items():
        st.markdown(
            f'<span class="ent-tag" style="background:{bg};color:{fg}">{label}</span>',
            unsafe_allow_html=True
        )
    st.divider()
    st.markdown("### 📖 À propos")
    st.caption("Modèle NER entraîné from scratch sur spaCy blank('fr'). Aucun modèle pré-entraîné requis.")

# ── Chargement ────────────────────────────────────────────────────────────────

with st.spinner("Entraînement du modèle NER..."):
    nlp, history = train_model_cached(n_iter)
    metrics = evaluate_cached(n_iter)
    dataset = load_dataset_cached()
    _, test_data = split_dataset(dataset, train_ratio=0.8)

# ── KPIs ──────────────────────────────────────────────────────────────────────

macro = metrics.get("macro_avg", {})
c1, c2, c3, c4 = st.columns(4)
c1.metric("📋 Exemples train", int(len(dataset) * 0.8))
c2.metric("🧪 Exemples test", int(len(dataset) * 0.2))
c3.metric("🎯 Macro F1", f"{macro.get('f1', 0):.3f}")
c4.metric("📊 Précision macro", f"{macro.get('precision', 0):.3f}")

st.divider()

# ── Onglets ───────────────────────────────────────────────────────────────────

tab1, tab2, tab3 = st.tabs(["📈 Performances", "🔬 Inférence interactive", "📉 Courbe d'entraînement"])

# ── Tab 1 : Métriques F1 ─────────────────────────────────────────────────────

with tab1:
    st.subheader("F1 · Précision · Rappel par type d'entité")

    rows = []
    for label in ENTITY_LABELS:
        m = metrics.get(label, {})
        rows.append({
            "Entité": label,
            "Précision": m.get("precision", 0),
            "Rappel": m.get("recall", 0),
            "F1": m.get("f1", 0),
            "Support": m.get("support", 0),
        })
    df_metrics = pd.DataFrame(rows)

    col_left, col_right = st.columns(2)

    with col_left:
        fig = go.Figure()
        for metric_name, color in [("F1", "#60a5fa"), ("Précision", "#34d399"), ("Rappel", "#f59e0b")]:
            fig.add_trace(go.Bar(
                name=metric_name,
                x=df_metrics["Entité"],
                y=df_metrics[metric_name],
                marker_color=color,
            ))
        fig.update_layout(
            barmode="group",
            paper_bgcolor="#0f1117", plot_bgcolor="#141624",
            font_color="#cbd5e1", legend_font_color="#cbd5e1",
            title="Métriques par type d'entité",
            yaxis=dict(range=[0, 1.1]),
            margin=dict(t=40, b=10, l=10, r=10),
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_right:
        fig2 = px.scatter(
            df_metrics, x="Précision", y="Rappel",
            size="Support", color="Entité", text="Entité",
            color_discrete_sequence=list(c[0] for c in ENT_COLORS.values()),
            title="Précision vs Rappel (taille = support)",
            size_max=40,
        )
        fig2.update_traces(textposition="top center", textfont_color="#cbd5e1")
        fig2.update_layout(
            paper_bgcolor="#0f1117", plot_bgcolor="#141624",
            font_color="#cbd5e1", xaxis=dict(range=[0, 1.1]), yaxis=dict(range=[0, 1.1]),
            margin=dict(t=40, b=10, l=10, r=10),
        )
        st.plotly_chart(fig2, use_container_width=True)

    st.dataframe(
        df_metrics.style
            .format({"Précision": "{:.3f}", "Rappel": "{:.3f}", "F1": "{:.3f}"})
            .background_gradient(subset=["F1"], cmap="Blues"),
        use_container_width=True, hide_index=True
    )

# ── Tab 2 : Inférence interactive ────────────────────────────────────────────

with tab2:
    st.subheader("Testez le modèle sur vos propres textes")

    examples = [
        "La CAF verse 500 euros aux bénéficiaires du RSA en Ile-de-France pendant 6 mois.",
        "Le ministère du Travail alloue 2 millions d euros au contrat d engagement jeune sur 24 mois.",
        "Pôle emploi accompagne les jeunes dans les zones rurales grâce à la prime d activité.",
        "L ANAH finance MaPrimeRénov à hauteur de 750 000 euros sur 18 mois dans les QPV.",
    ]

    user_input = st.text_area(
        "Entrez un texte administratif :",
        value=examples[0], height=100,
    )

    col_ex, _ = st.columns([2, 3])
    with col_ex:
        selected = st.selectbox("Ou choisissez un exemple :", examples)
        if st.button("Utiliser cet exemple"):
            user_input = selected

    if user_input:
        doc = nlp(user_input)
        ents = [(e.start_char, e.end_char, e.label_) for e in doc.ents]

        st.markdown("**Résultat :**")
        st.markdown(
            f'<div class="card" style="font-size:1rem;line-height:1.8">{highlight_entities(user_input, ents)}</div>',
            unsafe_allow_html=True
        )

        if ents:
            cols = st.columns(len(ents))
            for i, (s, e, label) in enumerate(ents):
                bg, fg = ENT_COLORS.get(label, ("#1e2130", "#e2e8f0"))
                cols[i].markdown(
                    f'<div class="card" style="text-align:center">'
                    f'<p style="color:#94a3b8;font-size:0.7rem;margin:0">{label}</p>'
                    f'<p style="color:{fg};font-size:1rem;font-weight:700;margin:4px 0 0 0">{user_input[s:e]}</p>'
                    f'</div>',
                    unsafe_allow_html=True
                )
        else:
            st.info("Aucune entité détectée dans ce texte.")

# ── Tab 3 : Courbe de loss ────────────────────────────────────────────────────

with tab3:
    st.subheader("Convergence de l'entraînement")
    df_history = pd.DataFrame(history)
    fig_loss = px.line(
        df_history, x="iter", y="loss", markers=True,
        title="Loss NER au fil des itérations",
        color_discrete_sequence=["#f59e0b"],
        labels={"iter": "Itération", "loss": "Loss NER"},
    )
    fig_loss.update_layout(
        paper_bgcolor="#0f1117", plot_bgcolor="#141624",
        font_color="#cbd5e1", margin=dict(t=40, b=10, l=10, r=10),
    )
    st.plotly_chart(fig_loss, use_container_width=True)
    st.caption(f"Loss finale : **{df_history['loss'].iloc[-1]:.4f}** après {n_iter} itérations.")
