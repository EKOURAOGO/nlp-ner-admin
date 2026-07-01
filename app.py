"""
app.py — NER Documents Administratifs
Dashboard pro · Design inspiré Linear/Vercel
"""

import json
import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import spacy

sys.path.insert(0, str(Path(__file__).parent / "src"))
from generate_training_data import generate_dataset, LABELS as ENTITY_LABELS
from train_ner import split_dataset, train_ner_model
from evaluate_ner import evaluate_on_test

st.set_page_config(page_title="NER · Documents Administratifs", layout="wide",
                   initial_sidebar_state="collapsed")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
*, *::before, *::after { box-sizing: border-box; }
html, body, [data-testid="stAppViewContainer"], .stApp {
    background-color: #0A0B0E !important;
    font-family: 'Inter', sans-serif !important; color: #E2E8F0 !important;
}
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 2rem 2.5rem !important; max-width: 1400px; }
[data-testid="stSidebar"] { background-color: #111318 !important; border-right: 1px solid #1C1F28 !important; }
[data-testid="stSidebar"] * { color: #94A3B8 !important; font-family: 'Inter', sans-serif !important; }
.app-header { padding: 0 0 2rem 0; border-bottom: 1px solid #1C1F28; margin-bottom: 2rem; }
.app-eyebrow { font-size: 0.7rem; font-weight: 600; letter-spacing: 0.14em;
    text-transform: uppercase; color: #10B981; margin-bottom: 0.5rem; }
.app-title { font-size: 2rem; font-weight: 700; color: #F1F5F9; line-height: 1.15; letter-spacing: -0.02em; }
.app-subtitle { font-size: 0.88rem; color: #64748B; margin-top: 0.35rem; }
.kpi-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 0.9rem; margin-bottom: 2rem; }
.kpi-card { background: #111318; border: 1px solid #1C1F28; border-top: 2px solid #10B981;
    border-radius: 10px; padding: 1.1rem 1.4rem; }
.kpi-label { font-size: 0.68rem; font-weight: 600; color: #475569;
    text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 0.4rem; }
.kpi-value { font-size: 1.9rem; font-weight: 700; color: #F1F5F9;
    font-variant-numeric: tabular-nums; line-height: 1; }
.kpi-sub { font-size: 0.72rem; color: #475569; margin-top: 0.3rem; }
.section-title { font-size: 0.72rem; font-weight: 600; color: #475569;
    text-transform: uppercase; letter-spacing: 0.12em;
    margin: 1.5rem 0 0.9rem 0; padding-bottom: 0.5rem; border-bottom: 1px solid #1C1F28; }
.ent-inline { display:inline; line-height: 2.2; }
.ent-token { padding: 3px 7px; border-radius: 5px; font-size: 0.9rem;
    font-weight: 500; margin: 0 1px; }
.ent-label { font-size: 0.58rem; font-weight: 700; vertical-align: super;
    margin-left: 3px; letter-spacing: 0.06em; }
.inference-box { background: #111318; border: 1px solid #1C1F28;
    border-radius: 10px; padding: 1.4rem 1.6rem; margin: 1rem 0;
    line-height: 2.2; font-size: 0.95rem; color: #CBD5E1; }
.ent-chip { display: inline-flex; align-items: center; gap: 8px;
    background: #111318; border: 1px solid #1C1F28; border-radius: 8px;
    padding: 0.65rem 1rem; margin: 0.3rem; }
.ent-chip-label { font-size: 0.65rem; font-weight: 700; text-transform: uppercase;
    letter-spacing: 0.1em; }
.ent-chip-value { font-size: 0.88rem; font-weight: 600; color: #F1F5F9; }
[data-baseweb="tab-list"] { background: transparent !important; border-bottom: 1px solid #1C1F28 !important; }
[data-baseweb="tab"] { background: transparent !important; color: #475569 !important;
    font-size: 0.82rem !important; font-weight: 500 !important;
    padding: 0.55rem 1.1rem !important; border-radius: 0 !important;
    border-bottom: 2px solid transparent !important; }
[aria-selected="true"][data-baseweb="tab"] {
    color: #E2E8F0 !important; border-bottom: 2px solid #10B981 !important;
    background: transparent !important; }
.stTextArea textarea { background: #111318 !important; color: #CBD5E1 !important;
    border: 1px solid #2D3142 !important; border-radius: 8px !important;
    font-family: 'Inter', sans-serif !important; font-size: 0.9rem !important; }
</style>
""", unsafe_allow_html=True)

ENT_STYLES = {
    "MONTANT":    {"bg":"#1e3a8a","fg":"#93c5fd","border":"#3b82f6"},
    "ORGANISME":  {"bg":"#064e3b","fg":"#6ee7b7","border":"#10b981"},
    "DISPOSITIF": {"bg":"#4c1d95","fg":"#c4b5fd","border":"#8b5cf6"},
    "DUREE":      {"bg":"#78350f","fg":"#fcd34d","border":"#f59e0b"},
    "ZONE_GEO":   {"bg":"#881337","fg":"#fda4af","border":"#f43f5e"},
}
LAYOUT = dict(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#111318",
              font=dict(family="Inter", color="#94A3B8", size=11),
              margin=dict(t=32, b=12, l=12, r=12),
              xaxis=dict(gridcolor="#1C1F28", linecolor="#1C1F28"),
              yaxis=dict(gridcolor="#1C1F28", linecolor="#1C1F28"))

@st.cache_data
def load_dataset_cached():
    p = Path(__file__).parent / "data" / "ner_dataset.json"
    if p.exists():
        with open(p) as f: return json.load(f)
    return generate_dataset(600)

@st.cache_resource
def train_model_cached(n_iter):
    dataset = load_dataset_cached()
    train_data, _ = split_dataset(dataset, train_ratio=0.8)
    nlp, history = train_ner_model(train_data, n_iter=n_iter)
    return nlp, history

@st.cache_data
def evaluate_cached(n_iter):
    dataset = load_dataset_cached()
    _, test_data = split_dataset(dataset, train_ratio=0.8)
    nlp, _ = train_model_cached(n_iter)
    return evaluate_on_test(nlp, test_data)

def highlight(text, entities):
    if not entities:
        return f'<span style="color:#94A3B8">{text}</span>'
    result, prev = [], 0
    for s, e, label in sorted(entities, key=lambda x: x[0]):
        st_ = ENT_STYLES.get(label, {"bg":"#1C1F28","fg":"#E2E8F0","border":"#2D3142"})
        result.append(f'<span style="color:#CBD5E1">{text[prev:s]}</span>')
        result.append(
            f'<span class="ent-token" style="background:{st_["bg"]};color:{st_["fg"]};'
            f'border:1px solid {st_["border"]}">'
            f'{text[s:e]}<span class="ent-label" style="color:{st_["fg"]}">{label}</span>'
            f'</span>')
        prev = e
    result.append(f'<span style="color:#CBD5E1">{text[prev:]}</span>')
    return "".join(result)

with st.sidebar:
    st.markdown("### Entraînement")
    n_iter = st.slider("Itérations", 10, 50, 30, step=5)
    st.markdown("---")
    st.markdown("### Entités")
    for label, st_ in ENT_STYLES.items():
        st.markdown(f'<span style="background:{st_["bg"]};color:{st_["fg"]};border:1px solid {st_["border"]};padding:2px 8px;border-radius:5px;font-size:0.75rem;font-weight:600">{label}</span>', unsafe_allow_html=True)

with st.spinner("Entraînement NER…"):
    nlp, history = train_model_cached(n_iter)
    metrics = evaluate_cached(n_iter)
    dataset = load_dataset_cached()
    _, test_data = split_dataset(dataset, train_ratio=0.8)

macro = metrics.get("macro_avg", {})

st.markdown(f"""
<div class="app-header">
  <div class="app-eyebrow">NLP · Reconnaissance d'entités nommées</div>
  <div class="app-title">NER sur Documents<br>Administratifs</div>
  <div class="app-subtitle">spaCy blank("fr") · 5 types d'entités · 480 exemples d'entraînement</div>
</div>
<div class="kpi-grid">
  <div class="kpi-card"><div class="kpi-label">Exemples train</div><div class="kpi-value">{int(len(dataset)*0.8)}</div><div class="kpi-sub">80 % du dataset</div></div>
  <div class="kpi-card"><div class="kpi-label">Exemples test</div><div class="kpi-value">{int(len(dataset)*0.2)}</div><div class="kpi-sub">20 % du dataset</div></div>
  <div class="kpi-card"><div class="kpi-label">Macro F1</div><div class="kpi-value">{macro.get("f1",0):.3f}</div><div class="kpi-sub">Sur 5 entités</div></div>
  <div class="kpi-card"><div class="kpi-label">Précision macro</div><div class="kpi-value">{macro.get("precision",0):.3f}</div><div class="kpi-sub">Moyenne pondérée</div></div>
</div>
""", unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["Performances F1","Inférence interactive","Courbe d'entraînement"])

with tab1:
    st.markdown('<div class="section-title">Métriques par type d\'entité</div>', unsafe_allow_html=True)
    ENTITY_LABELS = ["MONTANT","ORGANISME","DISPOSITIF","DUREE","ZONE_GEO"]
    rows = [{"Entité":l, "Précision":metrics.get(l,{}).get("precision",0),
             "Rappel":metrics.get(l,{}).get("recall",0),
             "F1":metrics.get(l,{}).get("f1",0),
             "Support":metrics.get(l,{}).get("support",0)} for l in ENTITY_LABELS]
    df_m = pd.DataFrame(rows)

    cl, cr = st.columns(2, gap="large")
    with cl:
        fig = go.Figure()
        colors = {"F1":"#10B981","Précision":"#6366F1","Rappel":"#F59E0B"}
        for metric, color in colors.items():
            fig.add_trace(go.Bar(name=metric, x=df_m["Entité"], y=df_m[metric],
                                 marker=dict(color=color, opacity=0.85, cornerradius=3)))
        fig.update_layout(**LAYOUT, barmode="group", height=300,
                          yaxis=dict(range=[0,1.15], gridcolor="#1C1F28"),
                          title="F1 · Précision · Rappel", title_font_size=12,
                          title_font_color="#94A3B8",
                          legend=dict(font=dict(color="#94A3B8"),
                                      bgcolor="rgba(0,0,0,0)"))
        st.plotly_chart(fig, use_container_width=True)
    with cr:
        # Radar chart
        cats = ENTITY_LABELS + [ENTITY_LABELS[0]]
        fig_r = go.Figure(go.Scatterpolar(
            r=[metrics.get(l,{}).get("f1",0) for l in ENTITY_LABELS] + [metrics.get(ENTITY_LABELS[0],{}).get("f1",0)],
            theta=cats, fill="toself",
            line=dict(color="#10B981", width=2),
            fillcolor="rgba(16,185,129,0.12)",
        ))
        fig_r.update_layout(paper_bgcolor="rgba(0,0,0,0)", height=300,
                            font=dict(family="Inter", color="#94A3B8", size=10),
                            margin=dict(t=20,b=20,l=20,r=20),
                            polar=dict(
                                bgcolor="#111318",
                                radialaxis=dict(visible=True, range=[0,1], gridcolor="#1C1F28",
                                                tickfont=dict(size=8,color="#475569")),
                                angularaxis=dict(gridcolor="#1C1F28", tickfont=dict(size=9,color="#94A3B8")),
                            ),
                            title="Vue radar F1", title_font_size=12, title_font_color="#94A3B8")
        st.plotly_chart(fig_r, use_container_width=True)

    st.dataframe(df_m.style.format({"Précision":"{:.3f}","Rappel":"{:.3f}","F1":"{:.3f}"}),
                 use_container_width=True, hide_index=True)

with tab2:
    st.markdown('<div class="section-title">Testez le modèle</div>', unsafe_allow_html=True)
    EXAMPLES = [
        "La CAF verse 500 euros aux bénéficiaires du RSA en Ile-de-France pendant 6 mois.",
        "Le ministère du Travail alloue 2 millions d euros au CEJ sur 24 mois.",
        "Pôle emploi accompagne les jeunes dans les zones rurales grâce à la prime d activité.",
        "L ANAH finance MaPrimeRénov à hauteur de 750 000 euros sur 18 mois dans les QPV.",
    ]
    col_inp, col_ex = st.columns([3,1], gap="large")
    with col_ex:
        st.markdown('<div class="section-title">Exemples</div>', unsafe_allow_html=True)
        chosen = None
        for ex in EXAMPLES:
            if st.button(ex[:40]+"…", key=f"ex_{ex[:10]}", use_container_width=True):
                chosen = ex

    with col_inp:
        default = chosen if chosen else EXAMPLES[0]
        user_input = st.text_area("Texte administratif :", value=default, height=110,
                                   label_visibility="collapsed")

    if user_input:
        doc = nlp(user_input)
        ents = [(e.start_char, e.end_char, e.label_) for e in doc.ents]

        st.markdown(f'<div class="inference-box">{highlight(user_input, ents)}</div>',
                    unsafe_allow_html=True)

        if ents:
            st.markdown('<div class="section-title">Entités extraites</div>', unsafe_allow_html=True)
            chips = ""
            for s, e, label in ents:
                st_ = ENT_STYLES.get(label, {"bg":"#1C1F28","fg":"#E2E8F0","border":"#2D3142"})
                chips += (f'<span class="ent-chip" style="border-color:{st_["border"]}">'
                          f'<span class="ent-chip-label" style="color:{st_["fg"]}">{label}</span>'
                          f'<span class="ent-chip-value">{user_input[s:e]}</span></span>')
            st.markdown(f'<div style="display:flex;flex-wrap:wrap">{chips}</div>',
                        unsafe_allow_html=True)
        else:
            st.markdown('<p style="color:#475569;font-size:0.85rem;margin-top:0.5rem">Aucune entité détectée.</p>', unsafe_allow_html=True)

with tab3:
    st.markdown('<div class="section-title">Convergence de l\'entraînement</div>', unsafe_allow_html=True)
    df_h = pd.DataFrame(history)
    fig_l = go.Figure()
    fig_l.add_trace(go.Scatter(x=df_h["iter"], y=df_h["loss"], mode="lines",
                               line=dict(color="#F59E0B", width=2),
                               fill="tozeroy", fillcolor="rgba(245,158,11,0.07)"))
    fig_l.update_layout(**LAYOUT, height=280,
                        xaxis_title="Itération", yaxis_title="Loss NER",
                        title="Convergence NER", title_font_size=12,
                        title_font_color="#94A3B8")
    st.plotly_chart(fig_l, use_container_width=True)
    final_loss = df_h["loss"].iloc[-1]
    st.markdown(f'<p style="font-size:0.8rem;color:#475569">Loss finale : <strong style="color:#F59E0B">{final_loss:.4f}</strong> après {n_iter} itérations.</p>', unsafe_allow_html=True)
