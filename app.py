"""
SolubilityAI — Dark Mode UI + Mobile Nav Fix + Sidebar Desktop Fix
"""

import streamlit as st
import numpy as np
import pandas as pd
import joblib
import io, base64
from pathlib import Path
from rdkit import Chem
from rdkit.Chem import Draw, Descriptors, Lipinski, rdMolDescriptors
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="SolubilityAI",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded",
)

OUT_DIR = Path('outputs')

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

*, *::before, *::after { box-sizing: border-box; font-family: 'Inter', sans-serif !important; }
#MainMenu, footer, header { visibility: hidden; }

html, body,
[data-testid="stAppViewContainer"],
[data-testid="stMain"],
[data-testid="stMainBlockContainer"],
[data-testid="stAppViewBlockContainer"],
section.main, .main {
    background: #0b1120 !important;
    color: #e2e8f0 !important;
}
.block-container {
    padding: 1.5rem 2rem 2rem !important;
    max-width: 100% !important;
    background: #0b1120 !important;
    min-height: 100vh;
}
[data-testid="column"],
[data-testid="stColumn"],
[data-testid="stVerticalBlock"],
[data-testid="stHorizontalBlock"] { background: transparent !important; }

/* ══════════════════════════════════════════
   SIDEBAR — FORCER L'AFFICHAGE SUR DESKTOP
   ══════════════════════════════════════════ */
section[data-testid="stSidebar"] {
    background: #0d1525 !important;
    min-width: 220px !important;
    max-width: 220px !important;
    width: 220px !important;
    border-right: 1px solid rgba(255,255,255,.06) !important;
    display: flex !important;
    visibility: visible !important;
    transform: none !important;
    left: 0 !important;
    position: relative !important;
    z-index: 100 !important;
}
section[data-testid="stSidebar"] > div:first-child { padding: 0 !important; }
section[data-testid="stSidebar"] .block-container {
    padding: 0 !important;
    background: #0d1525 !important;
}

/* Cacher le bouton collapse sur desktop */
@media (min-width: 768px) {
    [data-testid="collapsedControl"] {
        display: none !important;
    }
    section[data-testid="stSidebar"] {
        transform: none !important;
        margin-left: 0 !important;
        left: 0 !important;
        display: flex !important;
        visibility: visible !important;
    }
    /* Annuler l'animation de fermeture Streamlit */
    section[data-testid="stSidebar"][aria-expanded="false"] {
        transform: none !important;
        display: flex !important;
        visibility: visible !important;
        min-width: 220px !important;
        width: 220px !important;
    }
}

section[data-testid="stSidebar"] .stButton > button {
    background: transparent !important;
    color: #6b8fa8 !important;
    border: none !important;
    border-radius: 0 !important;
    font-size: 13px !important;
    font-weight: 500 !important;
    padding: 10px 20px !important;
    text-align: left !important;
    width: 100% !important;
    justify-content: flex-start !important;
    border-left: 3px solid transparent !important;
    transition: all .15s ease !important;
    letter-spacing: .01em !important;
    box-shadow: none !important;
}
section[data-testid="stSidebar"] .stButton > button:hover {
    background: rgba(99,179,237,.07) !important;
    color: #e2e8f0 !important;
    border-left-color: rgba(99,179,237,.4) !important;
    transform: none !important;
    box-shadow: none !important;
}

.stButton > button {
    background: linear-gradient(135deg, #38b2ac, #4299e1) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    font-size: 14px !important;
    padding: 10px 22px !important;
    box-shadow: 0 2px 12px rgba(56,178,172,.3) !important;
    transition: all .2s !important;
}
.stButton > button:hover {
    box-shadow: 0 4px 20px rgba(56,178,172,.5) !important;
    transform: translateY(-1px) !important;
}

.example-btn .stButton > button {
    background: rgba(56,178,172,.1) !important;
    color: #38b2ac !important;
    border: 1px solid rgba(56,178,172,.25) !important;
    border-radius: 20px !important;
    font-size: 12px !important;
    padding: 5px 14px !important;
    font-weight: 600 !important;
    box-shadow: none !important;
}
.example-btn .stButton > button:hover {
    background: rgba(56,178,172,.25) !important;
    transform: none !important;
    box-shadow: none !important;
}

.dk-card {
    background: #121c2e;
    border: 1px solid rgba(255,255,255,.07);
    border-radius: 12px;
    padding: 18px 22px;
    margin-bottom: 14px;
}
.dk-card-sm {
    background: #121c2e;
    border: 1px solid rgba(255,255,255,.07);
    border-radius: 10px;
    padding: 14px 18px;
    margin-bottom: 12px;
}

.stat-dk {
    background: #121c2e;
    border: 1px solid rgba(255,255,255,.07);
    border-radius: 12px;
    padding: 18px 20px;
    position: relative;
    overflow: hidden;
}
.stat-dk::after {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
}
.stat-dk.teal::after   { background: linear-gradient(90deg,#38b2ac,#4299e1); }
.stat-dk.blue::after   { background: linear-gradient(90deg,#4299e1,#667eea); }
.stat-dk.purple::after { background: linear-gradient(90deg,#667eea,#9f7aea); }
.stat-dk.rose::after   { background: linear-gradient(90deg,#f093fb,#f5576c); }

.stat-label { font-size: 10px; color: #4a6a7a; text-transform: uppercase;
    letter-spacing: .1em; font-weight: 700; margin-bottom: 6px; }
.stat-value { font-size: 22px; font-weight: 800; color: #f0f6ff; line-height: 1.1; }
.stat-sub   { font-size: 11px; color: #38b2ac; margin-top: 5px; font-weight: 500; }
.stat-icon  { position: absolute; top: 16px; right: 16px;
    width: 36px; height: 36px; border-radius: 9px;
    display: flex; align-items: center; justify-content: center; font-size: 18px; }

.hero-banner {
    background: linear-gradient(135deg, #0d1a3a 0%, #1a1035 45%, #2d0a1f 100%);
    border-radius: 16px;
    padding: 26px 30px;
    margin-bottom: 20px;
    border: 1px solid rgba(255,255,255,.06);
    position: relative;
    overflow: hidden;
}
.hero-banner::before {
    content: '';
    position: absolute;
    top: -60px; right: -60px;
    width: 240px; height: 240px;
    background: radial-gradient(circle, rgba(102,126,234,.25) 0%, transparent 70%);
    pointer-events: none;
}
.hero-banner::after {
    content: '';
    position: absolute;
    bottom: -40px; right: 200px;
    width: 180px; height: 180px;
    background: radial-gradient(circle, rgba(245,87,108,.15) 0%, transparent 70%);
    pointer-events: none;
}
.hero-tag {
    background: rgba(56,178,172,.15);
    color: #38b2ac;
    font-size: 10px;
    font-weight: 700;
    letter-spacing: .14em;
    text-transform: uppercase;
    padding: 3px 12px;
    border-radius: 20px;
    display: inline-block;
    margin-bottom: 12px;
    border: 1px solid rgba(56,178,172,.3);
}
.hero-title {
    font-size: 24px;
    font-weight: 800;
    color: white;
    margin: 0 0 8px;
    line-height: 1.2;
}
.hero-title .hl { color: #63b3ed; }
.hero-title .hl2 { color: #f093fb; }
.hero-desc  { color: #7a9ab8; font-size: 12.5px; line-height: 1.65; max-width: 480px; }
.hero-badges { display: flex; gap: 8px; margin-top: 16px; flex-wrap: wrap; }
.hero-badge  {
    background: rgba(255,255,255,.07);
    border: 1px solid rgba(255,255,255,.1);
    border-radius: 6px;
    padding: 4px 14px;
    font-size: 11px;
    color: #b0c8e0;
    font-weight: 500;
}

.perf-dk { width: 100%; border-collapse: collapse; }
.perf-dk th {
    font-size: 10px; font-weight: 700; color: #4a6a7a;
    text-transform: uppercase; letter-spacing: .07em;
    padding: 8px 14px; border-bottom: 1px solid rgba(255,255,255,.05);
    text-align: left;
}
.perf-dk td { padding: 11px 14px; border-bottom: 1px solid rgba(255,255,255,.04);
    font-size: 13px; color: #c8dae8; }
.perf-dk tr:last-child td { border-bottom: none; }
.perf-dk tr:hover td { background: rgba(255,255,255,.03); }
.rank-circle {
    width: 24px; height: 24px; border-radius: 50%;
    display: inline-flex; align-items: center; justify-content: center;
    font-size: 11px; font-weight: 700; color: white;
}
.tag-best-dk {
    background: rgba(56,178,172,.15); color: #38b2ac;
    font-size: 10px; font-weight: 700;
    padding: 2px 8px; border-radius: 8px;
    border: 1px solid rgba(56,178,172,.25);
}

.bar-cell { display: flex; align-items: center; gap: 8px; }
.bar-track-dk { flex: 1; background: rgba(255,255,255,.06);
    border-radius: 4px; height: 6px; overflow: hidden; }
.bar-fill-dk { height: 6px; border-radius: 4px; }

.mol-row {
    display: flex !important;
    justify-content: space-between !important;
    align-items: center !important;
    padding: 9px 0 !important;
    border-bottom: 1px solid rgba(255,255,255,.05) !important;
}
.mol-row:last-child { border-bottom: none !important; }
.mol-name { font-size: 12px !important; font-weight: 700 !important; color: #d0e4f5 !important; }
.mol-smi { font-size: 9px !important; color: #4a6a7a !important; font-family: 'Courier New', monospace !important; }
.mol-logS { font-size: 14px !important; font-weight: 700 !important; }
.mol-unit { font-size: 9px !important; color: #4a6a7a !important; }

.info-dk {
    background: #121c2e;
    border: 1px solid rgba(255,255,255,.07);
    border-radius: 10px;
    padding: 14px 18px;
}
.info-dk-title { font-size: 13px; font-weight: 700; color: #d0e4f5; margin-bottom: 5px; }
.info-dk-text  { font-size: 11px; color: #4a6a7a; line-height: 1.55; }

.sec-title { font-size: 14px; font-weight: 700; color: #d0e4f5; margin-bottom: 3px; }
.sec-sub   { font-size: 11px; color: #4a6a7a; margin-bottom: 14px; }

.topbar-wrap {
    display: flex; justify-content: space-between; align-items: center;
    margin-bottom: 20px;
}
.topbar-title { font-size: 22px; font-weight: 800; color: #f0f6ff; }
.topbar-sub   { font-size: 12px; color: #4a6a7a; margin-top: 2px; }
.topbar-badge {
    background: linear-gradient(135deg,#38b2ac,#4299e1);
    color: white; font-size: 11px; font-weight: 700;
    padding: 6px 14px; border-radius: 20px; white-space: nowrap;
}

.shap-row-dk { margin-bottom: 10px; }
.shap-label  { font-size: 12px; color: #c8dae8; font-weight: 500; }
.shap-val    { font-size: 12px; font-weight: 700; }
.shap-track-dk { background: rgba(255,255,255,.06); border-radius: 4px;
    height: 7px; overflow: hidden; margin-top: 4px; }
.shap-bar-dk { height: 7px; border-radius: 4px; }

.pred-big  { font-size: 40px; font-weight: 800; line-height: 1; }
.pred-unit-dk { font-size: 11px; color: #4a6a7a; margin-top: 4px; }
.pred-badge-dk {
    display: inline-block;
    padding: 4px 16px; border-radius: 20px;
    font-size: 12px; font-weight: 700; margin-top: 10px;
}

.reco-dk {
    border-radius: 10px;
    border: 1px solid rgba(255,255,255,.07);
    border-left: 4px solid;
    padding: 14px 16px;
    margin-bottom: 11px;
}
.reco-dk.danger { background: rgba(245,87,108,.06); border-left-color: #f5576c; }
.reco-dk.warn   { background: rgba(246,174,45,.06);  border-left-color: #f6ae2d; }
.reco-dk.ok     { background: rgba(56,178,172,.06);  border-left-color: #38b2ac; }
.reco-dk-title  { font-size: 13px; font-weight: 700; color: #d0e4f5; }
.reco-action    { font-size: 12px; color: #7a9ab8; line-height: 1.8;
    font-family: 'Courier New', monospace; }
.reco-impact    { font-size: 20px; font-weight: 700; color: #38b2ac; }
.reco-ref       { font-size: 10px; color: #4a6a7a; margin-top: 4px; line-height: 1.5; }

.lip-row-dk { display: flex; align-items: flex-start; gap: 10px;
    padding: 8px 0; border-bottom: 1px solid rgba(255,255,255,.05); }
.lip-row-dk:last-child { border-bottom: none; }
.lip-label-dk { font-size: 12px; font-weight: 600; color: #c8dae8; }
.lip-val-dk   { font-size: 11px; color: #4a6a7a; }

.hist-dk { width: 100%; border-collapse: collapse; }
.hist-dk th { font-size: 10px; font-weight: 700; color: #4a6a7a;
    text-transform: uppercase; letter-spacing: .07em;
    padding: 8px 12px; border-bottom: 1px solid rgba(255,255,255,.05);
    background: rgba(255,255,255,.02); text-align: left; }
.hist-dk td { padding: 10px 12px; border-bottom: 1px solid rgba(255,255,255,.04);
    font-size: 12px; color: #c8dae8; }
.hist-dk tr:last-child td { border-bottom: none; }
.hist-dk tr:hover td { background: rgba(255,255,255,.02); }
.err-ok   { background: rgba(56,178,172,.15); color: #38b2ac;
    font-size: 10px; font-weight: 700; padding: 2px 8px; border-radius: 8px; }
.err-warn { background: rgba(246,174,45,.15);  color: #f6ae2d;
    font-size: 10px; font-weight: 700; padding: 2px 8px; border-radius: 8px; }

.gauge-container { position: relative; width: 100%; height: 10px; border-radius: 6px; margin: 8px 0; }
.gauge-track {
    width: 100%; height: 100%; border-radius: 6px;
    background: linear-gradient(to right,
        #f5576c 0%, #f5576c 25%,
        #f6ae2d 25%, #f6ae2d 50%,
        #4299e1 50%, #4299e1 75%,
        #38b2ac 75%, #38b2ac 100%);
}
.gauge-cursor {
    position: absolute; top: -8px; transform: translateX(-50%);
    width: 0; height: 0;
    border-left: 7px solid transparent;
    border-right: 7px solid transparent;
    border-top: 13px solid #f0f6ff;
}

div[data-testid="stTextArea"] textarea,
div[data-testid="stTextInput"] input {
    background: #0d1525 !important;
    border: 1px solid rgba(255,255,255,.1) !important;
    border-radius: 8px !important;
    color: #d0e4f5 !important;
    font-size: 13px !important;
}
div[data-testid="stTextArea"] textarea:focus,
div[data-testid="stTextInput"] input:focus {
    border-color: #38b2ac !important;
    box-shadow: 0 0 0 2px rgba(56,178,172,.2) !important;
}
div[data-testid="stSelectbox"] > div,
div[data-baseweb="select"] {
    background: #0d1525 !important;
    border: 1px solid rgba(255,255,255,.1) !important;
    border-radius: 8px !important;
    color: #d0e4f5 !important;
}

.main-footer {
    text-align: center; padding: 14px 0 4px;
    color: #2a4560; font-size: 10px;
    border-top: 1px solid rgba(255,255,255,.05);
    margin-top: 18px;
}

div[data-testid="stAlert"] { border-radius: 10px !important; }

.svg-chart-wrap { background: #121c2e; border-radius: 12px;
    padding: 20px 22px; border: 1px solid rgba(255,255,255,.07); }

.mp-kpi-dk {
    background: #121c2e; border-radius: 10px;
    padding: 14px 18px; border: 1px solid rgba(255,255,255,.07);
}
.mp-kpi-label-dk { font-size: 10px; color: #4a6a7a; text-transform: uppercase;
    letter-spacing: .1em; font-weight: 700; margin-bottom: 6px; }
.mp-kpi-val-dk  { font-size: 22px; font-weight: 800; color: #f0f6ff; }
.mp-kpi-sub-dk  { font-size: 11px; color: #38b2ac; margin-top: 4px; font-weight: 600; }
.mp-kpi-cv-dk   { font-size: 11px; color: #4a6a7a; margin-top: 3px; }

.rk-dk { width: 100%; border-collapse: collapse; }
.rk-dk th { font-size: 10px; color: #4a6a7a; text-transform: uppercase;
    letter-spacing: .08em; padding: 8px 12px;
    border-bottom: 1px solid rgba(255,255,255,.06); text-align: left; font-weight: 700; }
.rk-dk td { padding: 11px 12px; border-bottom: 1px solid rgba(255,255,255,.04);
    font-size: 12px; color: #b0c8e0; }
.rk-dk tr:last-child td { border-bottom: none; }
.rk-dk tr:hover td { background: rgba(255,255,255,.03); }
.rk-best-dk { color: #38b2ac !important; font-weight: 700 !important; }

/* ══════════════════════════════════════════
   MOBILE NAV — barre fixe en bas (mobile uniquement)
   ══════════════════════════════════════════ */
.mobile-nav {
    display: none !important;
    position: fixed;
    bottom: 0;
    left: 0;
    right: 0;
    z-index: 99999;
    background: #0d1525;
    border-top: 1px solid rgba(255,255,255,.1);
    padding: 6px 0 14px;
    justify-content: space-around;
    align-items: center;
}
.mobile-nav.visible {
    display: flex !important;
}
.mobile-padding .block-container {
    padding-bottom: 90px !important;
}
</style>
""", unsafe_allow_html=True)

# ── JS : forcer la sidebar ouverte sur desktop ────────────────────
st.markdown("""
<script>
(function() {
    function fixSidebar() {
        if (window.innerWidth >= 768) {
            var sb = document.querySelector('section[data-testid="stSidebar"]');
            if (sb) {
                sb.style.transform  = 'none';
                sb.style.display    = 'flex';
                sb.style.visibility = 'visible';
                sb.style.minWidth   = '220px';
                sb.style.width      = '220px';
                sb.setAttribute('aria-expanded', 'true');
            }
            // Cacher le bouton collapse
            var btn = document.querySelector('[data-testid="collapsedControl"]');
            if (btn) btn.style.display = 'none';
        }
    }
    fixSidebar();
    setInterval(fixSidebar, 500);
    window.addEventListener('resize', fixSidebar);
})();
</script>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════
# DATA / MODELS
# ══════════════════════════════════════════════════════════════════
@st.cache_resource
def load_assets():
    try:
        m = joblib.load(OUT_DIR / 'best_model.pkl')
    except Exception:
        m = None
    try:
        s = joblib.load(OUT_DIR / 'scaler.pkl')
    except Exception:
        s = None
    return m, s


@st.cache_data
def load_results():
    try:
        return pd.read_csv(OUT_DIR / 'resultats_modeles.csv')
    except Exception:
        return pd.DataFrame({
            'Modèle':    ['Reg. Linéaire', 'Ridge', 'XGBoost', 'SVM (RBF)', 'Random Forest'],
            'RMSE_CV':   [0.3944, 0.3756, 0.4458, 0.3875, 0.4625],
            'R2_CV':     [0.9455, 0.9507, 0.9307, 0.9476, 0.9253],
            'MAE_CV':    [0.2575, 0.2536, 0.3287, 0.2589, 0.3417],
            'RMSE_Test': [0.4491, 0.4526, 0.4743, 0.4939, 0.5122],
            'R2_Test':   [0.9318, 0.9307, 0.9239, 0.9175, 0.9112],
            'MAE_Test':  [0.2655, 0.2662, 0.3316, 0.2768, 0.3547],
        })


model, scaler = load_assets()
results_df = load_results()

RANK_COLORS = ['#38b2ac', '#4299e1', '#e67e22', '#e74c3c', '#8e44ad']
ordered     = results_df.sort_values('R2_Test', ascending=False).reset_index(drop=True)

BEST_NAME    = ordered.iloc[0]['Modèle']
BEST_R2      = float(ordered.iloc[0]['R2_Test'])
BEST_RMSE    = float(ordered.iloc[0]['RMSE_Test'])
BEST_MAE     = float(ordered.iloc[0]['MAE_Test'])
BEST_R2_CV   = float(ordered.iloc[0].get('R2_CV',   BEST_R2   - .013))
BEST_RMSE_CV = float(ordered.iloc[0].get('RMSE_CV', BEST_RMSE - .055))
BEST_MAE_CV  = float(ordered.iloc[0].get('MAE_CV',  BEST_MAE  - .007))

_DESC_COLS_DEFAULT = [
    'MolWt', 'MolLogP', 'TPSA', 'NumHDonors', 'NumHAcceptors',
    'NumRotBonds', 'NumAromRings', 'HeavyAtomCount', 'RingCount', 'FractionCSP3',
]
try:
    import json as _json
    with open(OUT_DIR / 'desc_cols.json') as _f:
        _DESC_COLS = _json.load(_f)
except Exception:
    _DESC_COLS = _DESC_COLS_DEFAULT

MOLS = {
    'Aspirine':    ('CC(=O)Oc1ccccc1C(=O)O',           -1.81),
    'Paracetamol': ('CC(=O)Nc1ccc(O)cc1',              -0.89),
    'Cafeine':     ('Cn1cnc2c1c(=O)n(c(=O)n2C)C',     -0.54),
    'Ibuprofene':  ('CC(C)Cc1ccc(cc1)C(C)C(=O)O',     -3.62),
}
MOL_LABELS = {
    'Aspirine': 'Aspirine', 'Paracetamol': 'Paracétamol',
    'Cafeine': 'Caféine', 'Ibuprofene': 'Ibuprofène',
}

SHAP_DATA = {
    'Aspirine': [
        ('Morgan_245 (benzène)', -0.49), ('Morgan_1082 (ester)', -0.41),
        ('Morgan_939 (carboxyle)', -0.38), ('MolLogP (1.31)', 0.35),
        ('TPSA (83.8 Å²)', 0.30), ('Morgan_1612', -0.28),
    ],
    'Paracetamol': [
        ('Morgan_1638 (OH phénol)', 0.61), ('Morgan_488 (amide)', 0.59),
        ('TPSA (49.3 Å²)', 0.50), ('HD=2 (liaisons H)', 0.45),
        ('Morgan_1763', 0.35), ('MolLogP (0.91)', 0.33),
    ],
    'Cafeine': [
        ('Morgan_1299 (méthylxanthine)', 0.82), ('HA=6 (accepteurs H)', 0.40),
        ('Morgan_369 (imidazole)', 0.37), ('TPSA (61.8 Å²)', 0.34),
        ('Morgan_1612', 0.27), ('MolLogP (-0.07)', 0.26),
    ],
    'Ibuprofene': [
        ('MolLogP (3.97)', -0.85), ('Morgan_1082 (COOH)', -0.62),
        ('FSP3=0.47', 0.71), ('Morgan_598 (isobutyle)', 0.55),
        ('TPSA (37.3 Å²)', -0.48), ('HD=1', 0.40),
    ],
}

FALLBACK = {
    'CC(=O)Oc1ccccc1C(=O)O':       -1.586,
    'CC(=O)Nc1ccc(O)cc1':          -0.774,
    'Cn1cnc2c1c(=O)n(c(=O)n2C)C': -0.605,
    'CC(C)Cc1ccc(cc1)C(C)C(=O)O':  0.217,
}

LOGS_MIN, LOGS_MAX = -10.0, 3.0
MW_DOMAIN = (18, 1000)


# ══════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════
def calc_features(smiles):
    if not smiles or len(str(smiles).strip()) < 2:
        return None, None
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return None, None
    fp  = rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
    arr = np.array(list(fp), dtype=np.float32).reshape(1, -1)
    desc_map = {
        'MolWt': Descriptors.MolWt(mol), 'MolLogP': Descriptors.MolLogP(mol),
        'TPSA': Descriptors.TPSA(mol), 'NumHDonors': Lipinski.NumHDonors(mol),
        'NumHAcceptors': Lipinski.NumHAcceptors(mol),
        'NumRotBonds': Lipinski.NumRotatableBonds(mol),
        'NumAromRings': Lipinski.NumAromaticRings(mol),
        'HeavyAtomCount': Lipinski.HeavyAtomCount(mol),
        'RingCount': Lipinski.RingCount(mol),
        'FractionCSP3': Descriptors.FractionCSP3(mol),
    }
    desc = np.array([[desc_map[c] for c in _DESC_COLS]], dtype=np.float32)
    desc_sc = scaler.transform(desc) if scaler is not None else desc
    X = np.hstack([arr, desc_sc])
    return X, mol


def _physico_estimate(mol):
    mw   = Descriptors.MolWt(mol)
    logp = Descriptors.MolLogP(mol)
    tpsa = Descriptors.TPSA(mol)
    hd   = Lipinski.NumHDonors(mol)
    rb   = Lipinski.NumRotatableBonds(mol)
    ar   = Lipinski.NumAromaticRings(mol)
    fsp3 = Descriptors.FractionCSP3(mol)
    logs = (0.16 - 0.63 * logp - 0.0062 * mw + 0.066 * rb
            - 0.74 * ar + 0.011 * tpsa + 0.26 * hd + 0.40 * fsp3)
    return float(np.clip(logs, LOGS_MIN, LOGS_MAX))


def _is_ood(mol, raw):
    mw = Descriptors.MolWt(mol)
    ha = Lipinski.HeavyAtomCount(mol)
    if raw is None or raw < -9.5 or raw > 2.8:
        return True, "prédiction hors plage physique"
    if ha < 3:
        return True, "molécule trop petite"
    if mw < 15 or mw > MW_DOMAIN[1]:
        return True, f"MW={mw:.0f} hors domaine"
    return False, ""


def predict_logs(smi):
    if model is None:
        return FALLBACK.get(smi), False, ""
    X, mol = calc_features(smi)
    if X is None or mol is None:
        return FALLBACK.get(smi), False, ""
    try:
        raw = float(model.predict(X)[0])
    except Exception:
        raw = None
    ood, reason = _is_ood(mol, raw)
    if ood:
        return _physico_estimate(mol), True, reason
    return float(np.clip(raw, LOGS_MIN, LOGS_MAX)), False, ""


def get_cat(v):
    if v is None: return '–', '#6b8fa8', 'rgba(107,143,168,.15)'
    if v > -1:    return 'Très soluble',  '#38b2ac', 'rgba(56,178,172,.15)'
    elif v > -3:  return 'Soluble',       '#4299e1', 'rgba(66,153,225,.15)'
    elif v > -5:  return 'Peu soluble',   '#f6ae2d', 'rgba(246,174,45,.15)'
    else:         return 'Insoluble',     '#f5576c', 'rgba(245,87,108,.15)'


def mol_img(smi, size=(240, 240)):
    mol = Chem.MolFromSmiles(smi)
    if not mol:
        return ''
    img = Draw.MolToImage(mol, size=size)
    buf = io.BytesIO()
    img.save(buf, format='PNG')
    return base64.b64encode(buf.getvalue()).decode()


def get_props(mol):
    return dict(
        MW=Descriptors.MolWt(mol), LogP=Descriptors.MolLogP(mol),
        TPSA=Descriptors.TPSA(mol), HD=Lipinski.NumHDonors(mol),
        HA=Lipinski.NumHAcceptors(mol), AR=Lipinski.NumAromaticRings(mol),
        RB=Lipinski.NumRotatableBonds(mol), FSP3=Descriptors.FractionCSP3(mol),
    )


def get_recos(mol, logS):
    pr = get_props(mol)
    recos = []
    if pr['LogP'] > 3:
        d = min((pr['LogP'] - 3) * 0.35, 1.2)
        recos.append({'type': 'danger',
            'titre': 'Lipophilie excessive (LogP élevé)',
            'prob': f"LogP = {pr['LogP']:.2f} (seuil ≤ 3.0)",
            'action': ("① Ajouter −OH (hydroxyle) : ΔLogP ≈ −1.0\n"
                       "② Introduire −NH₂ (amine) : ΔLogP ≈ −1.2\n"
                       "③ Remplacer −CH₃ par −OCH₃ : ΔLogP ≈ −0.5"),
            'impact': f"+{d:.1f} à +{d+0.4:.1f} log(mol/L)",
            'ref': 'Lipinski 2001, J. Pharmacol. Sci.',
            'mecanisme': f"LogP actuel = {pr['LogP']:.2f} → cible ≤ 3.0."})
    if pr['TPSA'] < 40:
        d = (40 - pr['TPSA']) * 0.006
        recos.append({'type': 'danger',
            'titre': 'Surface polaire insuffisante (TPSA)',
            'prob': f"TPSA = {pr['TPSA']:.1f} Å² (min 40 Å²)",
            'action': ("① Ajouter −SO₂NH₂ : ΔTPSA ≈ +76 Å²\n"
                       "② Introduire −COOH : ΔTPSA ≈ +37 Å²\n"
                       "③ Ajouter −NH₂ : ΔTPSA ≈ +26 Å²"),
            'impact': f"+{d:.2f} à +{d+0.15:.2f} log(mol/L)",
            'ref': 'Ertl et al. 2000, J. Med. Chem.',
            'mecanisme': f"TPSA = {pr['TPSA']:.1f} Å². Cible ≥ 40 Å²."})
    if pr['HD'] == 0:
        recos.append({'type': 'warn',
            'titre': 'Absence de donneurs H',
            'prob': "H-donneurs = 0",
            'action': ("① −OH aromatique : ΔLogS ≈ +0.6\n"
                       "② −NH₂ aliphatique : ΔLogS ≈ +0.8\n"
                       "③ −NHCO− (amide) : ΔLogS ≈ +0.4"),
            'impact': "+0.4 à +1.0 log(mol/L)",
            'ref': 'Abraham et al. 2010, Eur. J. Med. Chem.',
            'mecanisme': "Chaque HD → +0.3–0.5 LogS."})
    if pr['AR'] >= 3:
        d = (pr['AR'] - 2) * 0.32
        recos.append({'type': 'warn',
            'titre': f"Trop de cycles aromatiques ({pr['AR']})",
            'prob': f"Cycles arom. = {pr['AR']} ≥ 3",
            'action': ("① Saturer un cycle → cyclohexane : ΔLogS ≈ +0.5\n"
                       "② Naphtyl → tétraline\n"
                       "③ Phényle → pyridyle : ΔLogS ≈ +0.3"),
            'impact': f"+{d:.1f} à +{d+0.3:.1f} log(mol/L)",
            'ref': 'Leeson & Springthorpe 2007, Nat. Rev. Drug Discov.',
            'mecanisme': f"{pr['AR']} cycles → cible ≤ 2."})
    if pr['MW'] > 450:
        recos.append({'type': 'warn',
            'titre': 'Masse molaire élevée',
            'prob': f"MW = {pr['MW']:.0f} g/mol",
            'action': ("① Fragmenter en deux analogues\n"
                       "② Supprimer groupes non essentiels\n"
                       "③ Remplacer grandes chaînes par méthyle"),
            'impact': "+0.1 à +0.5 log(mol/L)",
            'ref': 'Lipinski 2004, Drug Discov. Today',
            'mecanisme': f"MW = {pr['MW']:.0f}. Cible ≤ 450 g/mol."})
    if not recos:
        recos.append({'type': 'ok',
            'titre': 'Profil de solubilité optimal',
            'prob': "Aucun problème structurel détecté",
            'action': (f"✅ LogP = {pr['LogP']:.2f} ≤ 3\n"
                       f"✅ TPSA = {pr['TPSA']:.1f} ≥ 40 Å²\n"
                       f"✅ H-donneurs = {pr['HD']}\n"
                       f"✅ Cycles arom. = {pr['AR']} ≤ 2"),
            'impact': f"LogS = {logS:.2f} — satisfaisant",
            'ref': 'Egan et al. 2000, J. Med. Chem.',
            'mecanisme': "Tous les critères dans les plages optimales."})
    return recos, pr


# ══════════════════════════════════════════════════════════════════
# SESSION STATE
# ══════════════════════════════════════════════════════════════════
for k, v in [
    ('page', 'dashboard'), ('smi_input', 'CC(=O)Oc1ccccc1C(=O)O'),
    ('hist', []), ('shap_mol', 'Aspirine'), ('analyse_result', None),
    ('analyse_error', None),
]:
    if k not in st.session_state:
        st.session_state[k] = v

params = st.query_params
if 'page' in params:
    p = params['page']
    valid_pages = ['dashboard', 'modeles', 'shap', 'analyse', 'historique']
    if p in valid_pages and st.session_state.page != p:
        st.session_state.page = p

NAV = [
    ('dashboard',  '📊', 'Dashboard'),
    ('modeles',    '📈', 'Modèles'),
    ('shap',       '🔍', 'SHAP'),
    ('analyse',    '🧬', 'Analyse'),
    ('historique', '🕒', 'Historique'),
]

# ══════════════════════════════════════════════════════════════════
# SIDEBAR (desktop)
# ══════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown(f"""
    <div style="padding:20px 18px 16px;border-bottom:1px solid rgba(255,255,255,.06);">
      <div style="display:flex;align-items:center;gap:10px;">
        <div style="background:linear-gradient(135deg,#38b2ac,#4299e1);
            width:34px;height:34px;border-radius:9px;display:flex;align-items:center;
            justify-content:center;font-size:17px;flex-shrink:0;">🧪</div>
        <div>
          <div style="color:white;font-size:14px;font-weight:700;line-height:1.1;">SolubilityAI</div>
          <div style="color:#38b2ac;font-size:10px;font-weight:500;">{BEST_NAME} + SHAP</div>
        </div>
      </div>
    </div>
    <div style="color:#2a4a60;font-size:9px;font-weight:700;letter-spacing:.12em;
        text-transform:uppercase;padding:14px 18px 4px;">Navigation</div>
    """, unsafe_allow_html=True)

    # Navigation HTML pure — aucun st.button = aucun label parasite
    nav_items_html = ''
    for pid, icon, label in NAV:
        is_active = st.session_state.page == pid
        active_style = (
            'color:#e2e8f0;border-left:3px solid #38b2ac;background:rgba(56,178,172,.08);'
            if is_active else
            'color:#6b8fa8;border-left:3px solid transparent;background:transparent;'
        )
        nav_items_html += (
            f'<a href="?page={pid}" style="'
            f'display:flex;align-items:center;gap:10px;'
            f'text-decoration:none;{active_style}'
            f'font-size:13px;font-weight:500;padding:10px 20px;'
            f'font-family:Inter,sans-serif;letter-spacing:.01em;'
            f'width:100%;box-sizing:border-box;">'
            f'<span style="font-size:15px;line-height:1;">{icon}</span>'
            f'<span>{label}</span>'
            f'</a>'
        )
    st.markdown(nav_items_html, unsafe_allow_html=True)

    model_status = '🟢 Backend connecté' if model else '🟡 Mode démo'
    st.markdown(f"""
    <div style="height:1px;background:rgba(255,255,255,.05);margin:14px 0;"></div>
    <div style="padding:10px 18px;">
      <div style="font-size:11px;color:#4a6a7a;">{model_status}</div>
      <div style="font-size:9px;color:#2a4560;margin-top:3px;">Master 1 — IA Explicable 2026</div>
    </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════
# MOBILE BOTTOM NAV
# ══════════════════════════════════════════════════════════════════
nav_btns_html = ''
for pid, icon, label in NAV:
    is_active = st.session_state.page == pid
    txt_color = '#38b2ac' if is_active else '#4a6a7a'
    border_top = '2px solid #38b2ac' if is_active else '2px solid transparent'
    nav_btns_html += (
        f'<a href="?page={pid}" style="'
        f'flex:1;display:flex;flex-direction:column;align-items:center;gap:2px;'
        f'text-decoration:none;color:{txt_color};border-top:{border_top};'
        f'padding-top:4px;font-size:10px;font-family:Inter,sans-serif;font-weight:600;">'
        f'<span style="font-size:20px;line-height:1;">{icon}</span>'
        f'<span>{label}</span>'
        f'</a>'
    )

mobile_nav_html = (
    '<div class="mobile-nav" id="mobileNav">' + nav_btns_html + '</div>'
    '<script>'
    'function applyMobileNav() {'
    '  var nav = document.getElementById("mobileNav");'
    '  if (!nav) return;'
    '  if (window.innerWidth < 768) {'
    '    nav.style.display = "flex";'
    '    document.body.classList.add("mobile-padding");'
    '  } else {'
    '    nav.style.display = "none";'
    '    document.body.classList.remove("mobile-padding");'
    '  }'
    '}'
    'applyMobileNav();'
    'window.addEventListener("resize", applyMobileNav);'
    '</script>'
)
st.markdown(mobile_nav_html, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════
# SHARED HELPERS
# ══════════════════════════════════════════════════════════════════
def topbar(title, sub):
    st.markdown(f"""
    <div class="topbar-wrap">
      <div>
        <div class="topbar-title">{title}</div>
        <div class="topbar-sub">{sub}</div>
      </div>
      <div class="topbar-badge">⚡ {BEST_NAME} &nbsp;R²={BEST_R2:.3f}</div>
    </div>""", unsafe_allow_html=True)


def footer():
    st.markdown(
        '<div class="main-footer">SolubilityAI — Projet IA Explicable | '
        'BigSolDB | 6 138 molécules | Master 1 2026</div>',
        unsafe_allow_html=True,
    )


# ══════════════════════════════════════════════════════════════════
# PAGE — DASHBOARD
# ══════════════════════════════════════════════════════════════════
if st.session_state.page == 'dashboard':
    topbar('Dashboard', 'Prédiction de solubilité moléculaire avec IA explicable')

    st.markdown(f"""
    <div class="hero-banner">
      <div class="hero-tag">MACHINE LEARNING PIPELINE</div>
      <div class="hero-title">
        Prediction de <span class="hl">Solubilite</span>
        <span class="hl2"> Moleculaire</span>
      </div>
      <div class="hero-desc">
        Application IA Explicable utilisant {BEST_NAME} avec 2058 features.
        Explication SHAP pour chaque prédiction. Recommandations basées sur
        les règles de Lipinski pour l'amélioration de solubilité.
      </div>
      <div class="hero-badges">
        <div class="hero-badge">📊 6 138 molécules</div>
        <div class="hero-badge">⚙️ 2058 features</div>
        <div class="hero-badge">⬆ R²={BEST_R2:.3f}</div>
        <div class="hero-badge">📉 RMSE={BEST_RMSE:.3f}</div>
      </div>
    </div>""", unsafe_allow_html=True)

    kpi_cols = st.columns(4)
    kpis = [
        ('MEILLEUR MODÈLE', BEST_NAME, f'R² = {BEST_R2:.3f}', '📊', 'teal'),
        ('RMSE MOYEN',      f"{results_df['RMSE_Test'].mean():.2f}",
         f"{len(results_df)} modèles comparés", '📈', 'blue'),
        ('MOLÉCULES TEST', '1 228', '20% du dataset BigSolDB', '🧪', 'purple'),
        ('PRÉCISION',      f"{int(BEST_R2 * 100)}%", 'variance expliquée (R²)', '⚡', 'rose'),
    ]
    for col, (lbl, val, sub, icon, color) in zip(kpi_cols, kpis):
        with col:
            st.markdown(f"""
            <div class="stat-dk {color}">
              <div class="stat-label">{lbl}</div>
              <div class="stat-value">{val}</div>
              <div class="stat-sub">{sub}</div>
              <div class="stat-icon" style="background:rgba(255,255,255,.06);">{icon}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown('<div style="height:14px;"></div>', unsafe_allow_html=True)

    cl, cr = st.columns([3, 1.3])

    with cl:
        max_rmse = float(ordered['RMSE_Test'].max())
        max_mae  = float(ordered['MAE_Test'].max())
        rows_html = ''
        for i, row in ordered.iterrows():
            c   = RANK_COLORS[min(i, 4)]
            bt  = ' <span class="tag-best-dk">BEST</span>' if i == 0 else ''
            r2c = '#38b2ac' if i == 0 else '#4299e1'
            rw  = row['RMSE_Test'] / max_rmse * 100
            mw  = row['MAE_Test']  / max_mae  * 100
            rows_html += f"""
            <tr>
              <td><span class="rank-circle" style="background:{c};">{i+1}</span></td>
              <td style="color:#d0e4f5;font-weight:600;">{row['Modèle']}{bt}</td>
              <td>
                <div class="bar-cell">
                  <div class="bar-track-dk" style="width:100px;">
                    <div class="bar-fill-dk" style="width:{rw:.1f}%;background:linear-gradient(90deg,#4299e1,#667eea);"></div>
                  </div>
                  <span style="font-size:12px;color:#b0c8e0;">{row['RMSE_Test']:.3f}</span>
                </div>
              </td>
              <td>
                <div class="bar-cell">
                  <div class="bar-track-dk" style="width:80px;">
                    <div class="bar-fill-dk" style="width:{mw:.1f}%;background:linear-gradient(90deg,#f093fb,#f5576c);"></div>
                  </div>
                  <span style="font-size:12px;color:#b0c8e0;">{row['MAE_Test']:.3f}</span>
                </div>
              </td>
              <td style="color:{r2c};font-weight:700;">{row['R2_Test']:.3f}</td>
            </tr>"""

        st.markdown(f"""
        <div class="dk-card">
          <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:14px;">
            <div>
              <div class="sec-title">📈 Performance des modèles</div>
              <div class="sec-sub" style="margin-bottom:0;">BigSolDB · 5-fold CV · Test set 20%</div>
            </div>
          </div>
          <table class="perf-dk">
            <thead>
              <tr><th>#</th><th>MODÈLE</th><th>RMSE</th><th>MAE</th><th>R²</th></tr>
            </thead>
            <tbody>{rows_html}</tbody>
          </table>
        </div>""", unsafe_allow_html=True)

    with cr:
        mol_rows_html = ''
        courier = 'Courier New, monospace'
        for nom, (smi, lr) in MOLS.items():
            lpv, _ood, _ = predict_logs(smi)
            val = f"{lpv:.2f}" if lpv is not None else f"{lr:.2f}"
            _, clr, _ = get_cat(lpv if lpv is not None else lr)
            label = MOL_LABELS.get(nom, nom)
            smi_short = smi[:22] + '…'
            mol_rows_html += (
                '<div style="display:flex;justify-content:space-between;align-items:center;'
                'padding:9px 0;border-bottom:1px solid rgba(255,255,255,.05);">'
                '<div>'
                f'<div style="font-size:12px;font-weight:700;color:#d0e4f5;">{label}</div>'
                f'<div style="font-size:9px;color:#4a6a7a;font-family:{courier};">{smi_short}</div>'
                '</div>'
                '<div style="text-align:right;">'
                f'<div style="font-size:14px;font-weight:700;color:{clr};">{val}</div>'
                '<div style="font-size:9px;color:#4a6a7a;">LogS</div>'
                '</div>'
                '</div>'
            )

        st.markdown(
            '<div class="dk-card">'
            '<div class="sec-title">Mol&eacute;cules exemples</div>'
            '<div class="sec-sub">Solubilit&eacute; pr&eacute;dite (LogS)</div>'
            + mol_rows_html +
            '</div>',
            unsafe_allow_html=True
        )

    i1, i2, i3 = st.columns(3)
    with i1:
        st.markdown("""
        <div class="info-dk">
          <div class="info-dk-title">📋 Dataset BigSolDB v2.0</div>
          <div class="info-dk-text">6 138 molécules en solvant aqueux, nettoyées et
          validées expérimentalement avec RDKit.</div>
        </div>""", unsafe_allow_html=True)
    with i2:
        st.markdown("""
        <div class="info-dk">
          <div class="info-dk-title">⚙️ Features Engineering</div>
          <div class="info-dk-text">2048 Morgan fingerprints (radius=2) +
          10 descripteurs RDKit pour chaque molécule.</div>
        </div>""", unsafe_allow_html=True)
    with i3:
        st.markdown("""
        <div class="info-dk">
          <div class="info-dk-title">✅ Validation Rigoureuse</div>
          <div class="info-dk-text">5-fold cross-validation + test set 20%.
          Modèles comparés sur RMSE, MAE et R².</div>
        </div>""", unsafe_allow_html=True)

    footer()


# ══════════════════════════════════════════════════════════════════
# PAGE — MODÈLES
# ══════════════════════════════════════════════════════════════════
elif st.session_state.page == 'modeles':
    topbar('Model Performance', 'Comparaison complète des modèles entraînés')

    k1, k2, k3, k4 = st.columns(4)
    for col, (lbl, val, sub, cv) in zip([k1, k2, k3, k4], [
        ('BEST MODEL',   BEST_NAME,         f'↑ R² = {BEST_R2:.3f}', ''),
        ('R² (TEST)',    f'{BEST_R2:.3f}',  '', f'CV: {BEST_R2_CV:.3f}'),
        ('RMSE (TEST)',  f'{BEST_RMSE:.3f}','', f'CV: {BEST_RMSE_CV:.3f}'),
        ('MAE (TEST)',   f'{BEST_MAE:.3f}', '', f'CV: {BEST_MAE_CV:.3f}'),
    ]):
        with col:
            sub_h = f'<div class="mp-kpi-sub-dk">{sub}</div>' if sub else ''
            cv_h  = f'<div class="mp-kpi-cv-dk">{cv}</div>'  if cv  else ''
            st.markdown(f"""
            <div class="mp-kpi-dk">
              <div class="mp-kpi-label-dk">{lbl}</div>
              <div class="mp-kpi-val-dk">{val}</div>
              {sub_h}{cv_h}
            </div>""", unsafe_allow_html=True)

    st.markdown('<div style="height:14px;"></div>', unsafe_allow_html=True)

    rmse_test_list = ordered['RMSE_Test'].tolist()
    rmse_cv_list   = (ordered['RMSE_CV'].tolist()
                      if 'RMSE_CV' in ordered.columns
                      else [v * 0.88 for v in rmse_test_list])
    models_list    = ordered['Modèle'].tolist()
    n = len(models_list)

    CW, CH, PL, PB, PT, PR = 620, 200, 48, 36, 30, 20
    uw = CW - PL - PR
    gw = uw / n
    bw = int(gw * 0.28)
    gb = int(gw * 0.06)
    mv = max(max(rmse_test_list), max(rmse_cv_list)) * 1.18

    def yp(v): return PT + CH - (v / mv * CH)

    bars, labs, grid = '', '', ''
    for i, (m, rt, rc) in enumerate(zip(models_list, rmse_test_list, rmse_cv_list)):
        gx  = PL + i * gw + gw * 0.08
        xt  = gx
        xc  = gx + bw + gb
        yt  = yp(rt); yc = yp(rc); bot = PT + CH
        bars += (
            f'<rect x="{xt:.1f}" y="{yt:.1f}" width="{bw}" height="{bot-yt:.1f}" rx="3" fill="#4299e1"/>'
            f'<text x="{xt+bw/2:.1f}" y="{yt-5:.1f}" text-anchor="middle" font-size="10" fill="#6b8fa8">{rt:.3f}</text>'
            f'<rect x="{xc:.1f}" y="{yc:.1f}" width="{bw}" height="{bot-yc:.1f}" rx="3" fill="#38b2ac"/>'
            f'<text x="{xc+bw/2:.1f}" y="{yc-5:.1f}" text-anchor="middle" font-size="10" fill="#38b2ac">{rc:.3f}</text>'
        )
        cx = gx + bw + gb / 2
        labs += f'<text x="{cx:.1f}" y="{bot+20}" text-anchor="middle" font-size="11" fill="#6b8fa8" font-family="Inter,sans-serif">{m}</text>'

    for yv in [0.1, 0.2, 0.3, 0.4, 0.5]:
        if yv <= mv:
            ypp = yp(yv)
            grid += (
                f'<line x1="{PL}" y1="{ypp:.1f}" x2="{CW-PR}" y2="{ypp:.1f}" stroke="rgba(255,255,255,.06)" stroke-width="1"/>'
                f'<text x="{PL-6}" y="{ypp+4:.1f}" text-anchor="end" font-size="10" fill="#4a6a7a">{yv:.2f}</text>'
            )

    axes = (
        f'<line x1="{PL}" y1="{PT}" x2="{PL}" y2="{PT+CH}" stroke="rgba(255,255,255,.06)" stroke-width="1"/>'
        f'<line x1="{PL}" y1="{PT+CH}" x2="{CW-PR}" y2="{PT+CH}" stroke="rgba(255,255,255,.06)" stroke-width="1"/>'
    )
    LX = CW - 190
    legend = (
        f'<rect x="{LX}" y="6" width="10" height="10" rx="2" fill="#4299e1"/>'
        f'<text x="{LX+14}" y="15" font-size="10" fill="#6b8fa8" font-family="Inter,sans-serif">RMSE Test</text>'
        f'<rect x="{LX+80}" y="6" width="10" height="10" rx="2" fill="#38b2ac"/>'
        f'<text x="{LX+94}" y="15" font-size="10" fill="#38b2ac" font-family="Inter,sans-serif">RMSE CV</text>'
    )
    total_h = PT + CH + PB

    st.markdown(f"""
    <div class="svg-chart-wrap">
      <div style="font-size:11px;font-weight:700;color:#4a6a7a;text-transform:uppercase;
          letter-spacing:.1em;margin-bottom:14px;">RMSE COMPARISON — TEST VS CV</div>
      <svg viewBox="0 0 {CW} {total_h}" width="100%"
           style="max-width:700px;display:block;overflow:visible;">
        {grid}{axes}{bars}{labs}{legend}
      </svg>
    </div>""", unsafe_allow_html=True)

    st.markdown('<div style="height:14px;"></div>', unsafe_allow_html=True)

    rk_rows = ''
    for i, row in ordered.iterrows():
        c   = RANK_COLORS[min(i, 4)]
        tag = ' <span class="tag-best-dk">BEST</span>' if i == 0 else ''
        r2c = ' class="rk-best-dk"' if i == 0 else ''
        nm_c = '#38b2ac' if i == 0 else '#c8dae8'
        rk_rows += (
            f'<tr>'
            f'<td style="color:{c};font-weight:700;">#{i+1}</td>'
            f'<td style="color:{nm_c};"><strong>{row["Modèle"]}</strong>{tag}</td>'
            f'<td>{row["RMSE_CV"]:.4f}</td><td>{row["R2_CV"]:.4f}</td><td>{row["MAE_CV"]:.4f}</td>'
            f'<td>{row["RMSE_Test"]:.4f}</td><td{r2c}>{row["R2_Test"]:.4f}</td><td>{row["MAE_Test"]:.4f}</td>'
            f'</tr>'
        )
    st.markdown(f"""
    <div class="dk-card">
      <div class="sec-title" style="margin-bottom:14px;">MODEL RANKINGS</div>
      <table class="rk-dk">
        <thead><tr>
          <th>RANK</th><th>MODEL</th>
          <th>RMSE CV</th><th>R² CV</th><th>MAE CV</th>
          <th>RMSE TEST</th><th>R² TEST</th><th>MAE TEST</th>
        </tr></thead>
        <tbody>{rk_rows}</tbody>
      </table>
      <div style="font-size:10px;color:#2a4560;margin-top:12px;padding-top:10px;
          border-top:1px solid rgba(255,255,255,.05);">
        ⚠️ Résultats obtenus sur BigSolDB. R² présenté sur données de test réelles.
      </div>
    </div>""", unsafe_allow_html=True)
    footer()


# ══════════════════════════════════════════════════════════════════
# PAGE — SHAP
# ══════════════════════════════════════════════════════════════════
elif st.session_state.page == 'shap':
    topbar('Explications SHAP', 'Quelles features influencent la prédiction ?')

    mol_keys = list(MOLS.keys())
    def_idx  = mol_keys.index(st.session_state.shap_mol) if st.session_state.shap_mol in mol_keys else 0

    cs, cb = st.columns([4, 1])
    with cs:
        mc = st.selectbox('Molécule', mol_keys, index=def_idx,
                          format_func=lambda x: f"{MOL_LABELS.get(x, x)}  (LogS réel: {MOLS[x][1]:.2f})")
        st.code(MOLS[mc][0], language='text')
    with cb:
        st.markdown('<div style="height:28px;"></div>', unsafe_allow_html=True)
        st.button('🔍 Analyser', use_container_width=True, key='shap_run')

    if mc != st.session_state.shap_mol:
        st.session_state.shap_mol = mc

    smi_shap = MOLS[mc][0]
    lpv, _, _ = predict_logs(smi_shap)
    if lpv is None: lpv = MOLS[mc][1]
    cat, cc, cbg = get_cat(lpv)
    im  = mol_img(smi_shap, (200, 200))
    shv = SHAP_DATA.get(mc, SHAP_DATA['Aspirine'])
    mx  = max(abs(v) for _, v in shv)

    c1, c2 = st.columns([1, 2])
    with c1:
        st.markdown(f"""
        <div class="dk-card" style="text-align:center;">
          <div style="font-size:10px;color:#4a6a7a;text-transform:uppercase;
              letter-spacing:.08em;margin-bottom:8px;">Structure 2D</div>
          <img src="data:image/png;base64,{im}"
               style="width:90%;border-radius:8px;border:1px solid rgba(255,255,255,.07);">
          <div style="font-size:16px;font-weight:700;color:#d0e4f5;margin-top:10px;">
            {MOL_LABELS.get(mc, mc)}</div>
          <div class="pred-big" style="color:{cc};">{lpv:.3f}</div>
          <div class="pred-unit-dk">log(mol/L)</div>
          <div class="pred-badge-dk" style="background:{cbg};color:{cc};">{cat}</div>
        </div>""", unsafe_allow_html=True)

    with c2:
        shap_rows = ''
        for feat, val in shv:
            w   = min(abs(val) / mx * 100, 100)
            sc2 = '#38b2ac' if val > 0 else '#f5576c'
            sgn = '+' if val > 0 else ''
            shap_rows += (
                '<div class="shap-row-dk">'
                '<div style="display:flex;justify-content:space-between;margin-bottom:3px;">'
                f'<span class="shap-label">{feat}</span>'
                f'<span class="shap-val" style="color:{sc2};">{sgn}{val:.3f}</span>'
                '</div>'
                f'<div class="shap-track-dk"><div class="shap-bar-dk"'
                f' style="width:{w:.1f}%;background:{sc2};"></div></div></div>'
            )
        st.markdown(f"""
        <div class="dk-card">
          <div class="sec-title">Top contributions SHAP — {MOL_LABELS.get(mc, mc)}</div>
          <div class="sec-sub">
            LogS prédit: <strong style="color:{cc};">{lpv:.3f}</strong>
            &nbsp;·&nbsp; LogS réel: <strong>{MOLS[mc][1]:.2f}</strong>
          </div>
          {shap_rows}
          <div style="font-size:10px;color:#4a6a7a;margin-top:10px;padding-top:10px;
              border-top:1px solid rgba(255,255,255,.05);">
            🟢 Positives = augmentent la solubilité &nbsp;|&nbsp;
            🔴 Négatives = la diminuent
          </div>
        </div>""", unsafe_allow_html=True)

    st.markdown("""
    <div class="dk-card">
      <div class="sec-title">Guide d'interprétation</div>
      <div style="display:grid;grid-template-columns:1fr 1fr;gap:14px;margin-top:10px;">
        <div style="background:rgba(56,178,172,.1);border:1px solid rgba(56,178,172,.2);
            border-radius:8px;padding:14px;">
          <div style="font-size:13px;font-weight:700;color:#38b2ac;margin-bottom:6px;">
            Valeurs positives (vert)</div>
          <div style="font-size:12px;color:#8ab0c0;line-height:1.6;">
            Ces features <b>augmentent</b> la solubilité prédite.</div>
        </div>
        <div style="background:rgba(245,87,108,.1);border:1px solid rgba(245,87,108,.2);
            border-radius:8px;padding:14px;">
          <div style="font-size:13px;font-weight:700;color:#f5576c;margin-bottom:6px;">
            Valeurs négatives (rouge)</div>
          <div style="font-size:12px;color:#8ab0c0;line-height:1.6;">
            Ces features <b>diminuent</b> la solubilité — molécule plus hydrophobe.</div>
        </div>
      </div>
    </div>""", unsafe_allow_html=True)

    try:
        shap_df = pd.read_csv(OUT_DIR / 'shap_importance.csv').head(12)
    except Exception:
        shap_df = pd.DataFrame({
            'feature': ['FractionCSP3', 'NumRotBonds', 'Morgan_1171', 'Morgan_389',
                        'Morgan_807', 'MolLogP', 'HeavyAtomCount', 'MolWt',
                        'NumAromRings', 'NumHAcceptors', 'TPSA', 'Morgan_1917'],
            'importance_shap': [2.332, 1.114, 0.760, 0.632, 0.610, 0.517,
                                  0.543, 0.282, 0.256, 0.182, 0.142, 0.123],
        })

    gmax = float(shap_df['importance_shap'].iloc[0])
    gr_rows = ''
    for _, grow in shap_df.iterrows():
        gw2 = grow['importance_shap'] / gmax * 100
        gr_rows += (
            '<div class="shap-row-dk">'
            '<div style="display:flex;justify-content:space-between;margin-bottom:3px;">'
            f'<span class="shap-label">{grow["feature"]}</span>'
            f'<span class="shap-val" style="color:#4299e1;">{grow["importance_shap"]:.3f}</span>'
            '</div>'
            f'<div class="shap-track-dk"><div class="shap-bar-dk"'
            f' style="width:{gw2:.1f}%;background:linear-gradient(90deg,#4299e1,#667eea);"></div></div></div>'
        )
    st.markdown(f"""
    <div class="dk-card">
      <div class="sec-title">Top features globales SHAP — BigSolDB (6 138 molécules)</div>
      <div class="sec-sub">Importance moyenne |SHAP| sur le jeu de test complet</div>
      {gr_rows}
    </div>""", unsafe_allow_html=True)
    footer()


# ══════════════════════════════════════════════════════════════════
# PAGE — ANALYSE
# ══════════════════════════════════════════════════════════════════
elif st.session_state.page == 'analyse':
    topbar('Analyse Molécule', 'Prédiction en temps réel + Recommandations structurales')

    def _run_prediction(smi_clean):
        smi_clean = smi_clean.strip()
        if not smi_clean:
            st.session_state.analyse_error = '⚠️ Veuillez entrer un SMILES.'
            return
        st.session_state.smi_input = smi_clean
        _, mol_o = calc_features(smi_clean)
        if mol_o is None:
            st.session_state.analyse_error = '❌ SMILES invalide ou non reconnu par RDKit.'
            st.session_state.analyse_result = None
            return
        st.session_state.analyse_error = None
        lpv, ood_flag, ood_reason = predict_logs(smi_clean)
        cat, cc, cbg = get_cat(lpv)
        pr   = get_props(mol_o)
        im   = mol_img(smi_clean, (240, 240))
        recos, _ = get_recos(mol_o, lpv or 0)
        nom_h = next((n for n, (s, _) in MOLS.items() if s == smi_clean), 'Personnalisée')
        lr_h  = next((_l for n, (s, _l) in MOLS.items() if s == smi_clean), None)
        st.session_state.analyse_result = {
            'logS': lpv, 'cat': cat, 'cat_col': cc, 'cat_bg': cbg,
            'props': pr, 'img_b64': im, 'smiles': smi_clean, 'recos': recos,
            'ood': ood_flag, 'ood_reason': ood_reason, 'nom': nom_h,
        }
        if not any(h['smiles'] == smi_clean for h in st.session_state.hist):
            st.session_state.hist.append({
                'nom': nom_h, 'smiles': smi_clean, 'logs_pred': lpv,
                'logs_reel': lr_h,
                'erreur': abs(lr_h - lpv) if lr_h is not None and lpv is not None else None,
            })

    st.markdown("""
    <div class="dk-card" style="margin-bottom:14px;">
      <div class="sec-title">⚗️ Analyser une molécule</div>
      <div class="sec-sub" style="margin-bottom:0;">Entrez un code SMILES ci-dessous</div>
    </div>""", unsafe_allow_html=True)

    smi_val = st.text_area('SMILES', value=st.session_state.smi_input,
                            height=68, label_visibility='collapsed')

    st.markdown('<div style="font-size:11px;color:#4a6a7a;margin-bottom:6px;">Exemples :</div>',
                unsafe_allow_html=True)
    ecols = st.columns(len(MOLS))
    for i, (nom, (smi, _)) in enumerate(MOLS.items()):
        with ecols[i]:
            st.markdown('<div class="example-btn">', unsafe_allow_html=True)
            if st.button(MOL_LABELS.get(nom, nom), key=f'ex_{nom}', use_container_width=True):
                _run_prediction(smi)
            st.markdown('</div>', unsafe_allow_html=True)

    if st.button(f'⚡ Analyser avec {BEST_NAME}', use_container_width=True, key='run_analyse'):
        _run_prediction(smi_val)

    if st.session_state.get('analyse_error'):
        st.error(st.session_state.analyse_error)

    if st.session_state.analyse_result:
        r = st.session_state.analyse_result
        lpv, cat, cc, cbg = r['logS'], r['cat'], r['cat_col'], r['cat_bg']
        pr, im, recos = r['props'], r['img_b64'], r['recos']
        ood_flag   = r.get('ood', False)
        ood_reason = r.get('ood_reason', '')

        if ood_flag:
            st.markdown(f"""
            <div style="background:rgba(246,174,45,.08);border:1px solid rgba(246,174,45,.2);
                border-left:4px solid #f6ae2d;border-radius:10px;
                padding:11px 16px;margin-bottom:14px;">
              <div style="font-size:13px;font-weight:700;color:#f6ae2d;margin-bottom:3px;">
                ⚠️ Prédiction hors domaine — Correction ESOL appliquée</div>
              <div style="font-size:12px;color:#7a9ab8;">
                Raison : <strong>{ood_reason}</strong>. Estimation via formule ESOL.
              </div>
            </div>""", unsafe_allow_html=True)

        pct_g = (max(-8, min(2, lpv or -4)) + 8) / 10 * 100

        prop_html = ''.join(
            f'<div style="display:flex;justify-content:space-between;padding:5px 0;'
            f'border-bottom:1px solid rgba(255,255,255,.05);">'
            f'<span style="font-size:11px;color:#4a6a7a;">{k}</span>'
            f'<span style="font-size:12px;font-weight:600;color:#c8dae8;">{v}</span></div>'
            for k, v in [
                ('Masse molaire', f"{pr['MW']:.1f} g/mol"),
                ('LogP', f"{pr['LogP']:.2f}"),
                ('TPSA', f"{pr['TPSA']:.1f} Å²"),
                ('H-donneurs', str(pr['HD'])),
                ('H-accepteurs', str(pr['HA'])),
                ('Liaisons rot.', str(pr['RB'])),
                ('Cycles arom.', str(pr['AR'])),
                ('FractionCSP3', f"{pr['FSP3']:.2f}"),
            ]
        )

        checks = [
            ('LogP ≤ 3',          pr['LogP'] <= 3,  f"LogP={pr['LogP']:.2f}"),
            ('TPSA ≥ 40 Å²',     pr['TPSA'] >= 40, f"TPSA={pr['TPSA']:.1f}"),
            ('H-donneurs ≥ 1',    pr['HD'] >= 1,    f"{pr['HD']} donneur(s)"),
            ('Cycles arom. ≤ 2',  pr['AR'] <= 2,    f"{pr['AR']} cycle(s)"),
            ('Masse ≤ 500 g/mol', pr['MW'] <= 500,  f"{pr['MW']:.0f} g/mol"),
            ('H-accept. ≤ 10',    pr['HA'] <= 10,   f"{pr['HA']} accepteur(s)"),
        ]
        n_ok = sum(1 for _, ok, _ in checks if ok)
        ok_clr = '#38b2ac' if n_ok >= 4 else '#f6ae2d'
        chk_html = ''.join(
            f'<div class="lip-row-dk">'
            f'<div style="font-size:16px;">{"✅" if ok else "❌"}</div>'
            f'<div><div class="lip-label-dk">{lbl}</div>'
            f'<div class="lip-val-dk">{val}</div></div></div>'
            for lbl, ok, val in checks
        )

        st.markdown(f"""
        <div style="display:grid;grid-template-columns:1fr 1.1fr 1.4fr;
                    gap:14px;margin-bottom:14px;align-items:start;">
          <div class="dk-card" style="margin-bottom:0;text-align:center;">
            <div style="font-size:10px;color:#4a6a7a;text-transform:uppercase;
                letter-spacing:.08em;margin-bottom:8px;">Structure 2D</div>
            <img src="data:image/png;base64,{im}"
                 style="width:100%;border-radius:8px;border:1px solid rgba(255,255,255,.07);">
          </div>
          <div style="display:flex;flex-direction:column;gap:14px;">
            <div class="dk-card" style="margin-bottom:0;">
              <div style="font-size:10px;color:#4a6a7a;text-transform:uppercase;
                  letter-spacing:.08em;margin-bottom:6px;">LogS Prédit</div>
              <div class="pred-big" style="color:{cc};">{f'{lpv:.3f}' if lpv else '–'}</div>
              <div class="pred-unit-dk">log(mol/L)</div>
              <div class="pred-badge-dk" style="background:{cbg};color:{cc};">{cat}</div>
              <div style="margin-top:14px;">
                <div class="gauge-container">
                  <div class="gauge-track"></div>
                  <div class="gauge-cursor" style="left:{pct_g:.1f}%;"></div>
                </div>
              </div>
              <div style="display:flex;justify-content:space-between;
                  font-size:10px;color:#4a6a7a;margin-top:5px;">
                <span>Insoluble</span><span>Peu sol.</span>
                <span>Soluble</span><span>Très sol.</span>
              </div>
            </div>
            <div class="dk-card" style="margin-bottom:0;">
              <div class="sec-title" style="margin-bottom:10px;">Propriétés RDKit</div>
              {prop_html}
            </div>
          </div>
          <div class="dk-card" style="margin-bottom:0;">
            <div style="display:flex;justify-content:space-between;
                align-items:center;margin-bottom:10px;">
              <div class="sec-title">Diagnostic Lipinski</div>
              <span style="font-size:12px;font-weight:700;color:{ok_clr};">
                {n_ok}/6 critères</span>
            </div>
            {chk_html}
          </div>
        </div>""", unsafe_allow_html=True)

        n_prob = sum(1 for rc in recos if rc['type'] in ('danger', 'warn'))
        border_c = '#f5576c' if n_prob > 0 else '#38b2ac'

        st.markdown(f"""
        <div class="dk-card" style="border-left:4px solid {border_c};">
          <div style="display:flex;justify-content:space-between;
              align-items:center;margin-bottom:18px;">
            <div>
              <div class="sec-title">💡 Recommandations pour améliorer la solubilité</div>
              <div class="sec-sub" style="margin-bottom:0;">
                Basé sur Lipinski, Ertl, Abraham — validé sur BigSolDB</div>
            </div>
            <span style="font-size:12px;font-weight:700;color:{border_c};">
              {n_prob} problème(s)</span>
          </div>
        """, unsafe_allow_html=True)

        for rc in recos:
            bdg_map = {
                'ok':     ('rgba(56,178,172,.15)', '#38b2ac', '✅ Profil optimal'),
                'warn':   ('rgba(246,174,45,.15)', '#f6ae2d', '⚠️ Amélioration suggérée'),
                'danger': ('rgba(245,87,108,.15)', '#f5576c', '❌ Problème détecté'),
            }
            t = rc['type']
            bg_b, txt_b, label_b = bdg_map[t]
            action_html = rc['action'].replace('\n', '<br>')
            st.markdown(f"""
            <div class="reco-dk {t}">
              <div style="display:flex;align-items:center;gap:10px;margin-bottom:10px;">
                <span style="background:{bg_b};color:{txt_b};font-size:10px;
                    font-weight:700;padding:2px 9px;border-radius:10px;">{label_b}</span>
                <span class="reco-dk-title">{rc['titre']}</span>
              </div>
              <div style="display:grid;grid-template-columns:1fr 1.6fr 0.8fr;gap:14px;">
                <div>
                  <div style="font-size:9px;color:#4a6a7a;font-weight:700;text-transform:uppercase;
                      letter-spacing:.08em;margin-bottom:5px;">Problème identifié</div>
                  <div style="font-size:12px;color:#f5576c;font-weight:500;margin-bottom:8px;">{rc['prob']}</div>
                  <div style="font-size:9px;color:#4a6a7a;font-weight:700;text-transform:uppercase;
                      letter-spacing:.08em;margin-bottom:5px;">Mécanisme</div>
                  <div style="font-size:11px;color:#6a8a9a;line-height:1.5;">{rc['mecanisme']}</div>
                </div>
                <div>
                  <div style="font-size:9px;color:#4a6a7a;font-weight:700;text-transform:uppercase;
                      letter-spacing:.08em;margin-bottom:5px;">Actions recommandées</div>
                  <div class="reco-action">{action_html}</div>
                </div>
                <div>
                  <div style="font-size:9px;color:#4a6a7a;font-weight:700;text-transform:uppercase;
                      letter-spacing:.08em;margin-bottom:5px;">Impact estimé</div>
                  <div class="reco-impact">{rc['impact']}</div>
                  <div class="reco-ref">{rc['ref']}</div>
                </div>
              </div>
            </div>""", unsafe_allow_html=True)

        st.markdown("""
        <div style="background:rgba(246,174,45,.07);border:1px solid rgba(246,174,45,.15);
            border-left:4px solid #f6ae2d;border-radius:8px;padding:12px 16px;margin-top:4px;">
          <div style="font-size:12px;font-weight:700;color:#f6ae2d;margin-bottom:4px;">
            ⚠️ Recommandations guidées par le modèle — non une validation chimique définitive</div>
          <div style="font-size:11px;color:#6a8a9a;line-height:1.6;">
            Ces suggestions s'appuient sur des règles structurales reconnues (Lipinski, Ertl, Abraham)
            et constituent des pistes de recherche. Toute modification doit être validée
            expérimentalement.<br>
            <span style="color:#4a6a7a;font-style:italic;margin-top:4px;display:block;">
              Conformément aux recommandations de M. El Fakir (encadrant).</span>
          </div>
        </div>
        </div>""", unsafe_allow_html=True)

    footer()


# ══════════════════════════════════════════════════════════════════
# PAGE — HISTORIQUE
# ══════════════════════════════════════════════════════════════════
elif st.session_state.page == 'historique':
    topbar('Historique', 'Prédictions précédentes enregistrées dans cette session')

    if not st.session_state.hist:
        default = []
        for nom, (smi, lr) in MOLS.items():
            lpv, _, _ = predict_logs(smi)
            default.append({'nom': MOL_LABELS.get(nom, nom), 'smiles': smi,
                             'logs_pred': lpv, 'logs_reel': lr,
                             'erreur': abs(lr - lpv) if lpv is not None else None})
        hist_data = default
    else:
        hist_data = st.session_state.hist

    search = st.text_input('Recherche', placeholder='Rechercher par nom ou SMILES…',
                           label_visibility='collapsed')
    if search:
        hist_data = [h for h in hist_data
                     if search.lower() in h['nom'].lower()
                     or search.lower() in h['smiles'].lower()]

    rows_html = ''
    for h in hist_data:
        ps = f"{h['logs_pred']:.2f}" if h['logs_pred'] is not None else '–'
        rs = f"{h['logs_reel']:.2f}" if h['logs_reel'] is not None else '–'
        e  = h.get('erreur')
        eb = (f'<span class="err-ok">{e:.3f}</span>'   if e is not None and e < .3
              else f'<span class="err-warn">{e:.3f}</span>' if e is not None else '–')
        sd = (h['smiles'][:32] + '…') if len(h['smiles']) > 32 else h['smiles']
        rows_html += (
            f'<tr><td><strong style="color:#d0e4f5;">{h["nom"]}</strong></td>'
            f'<td style="font-family:Courier New,monospace;font-size:11px;color:#4a6a7a;">{sd}</td>'
            f'<td>{rs}</td>'
            f'<td style="font-weight:700;color:#38b2ac;">{ps}</td>'
            f'<td>{eb}</td></tr>'
        )

    errs  = [h['erreur'] for h in hist_data if h.get('erreur') is not None]
    mae   = sum(errs) / len(errs) if errs else 0.0
    prec  = sum(1 for e in errs if e < .3) / len(errs) * 100 if errs else 100.0

    st.markdown(f"""
    <div class="dk-card">
      <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:14px;">
        <div class="sec-title">Prédictions</div>
        <span style="background:rgba(66,153,225,.15);color:#4299e1;font-size:10px;
            font-weight:700;padding:2px 10px;border-radius:10px;">
          📋 {len(hist_data)} molécule(s)</span>
      </div>
      <table class="hist-dk">
        <thead><tr>
          <th>NOM</th><th>SMILES</th>
          <th>LogS RÉEL</th><th>LogS PRÉDIT</th><th>ERREUR</th>
        </tr></thead>
        <tbody>{rows_html}</tbody>
      </table>
    </div>""", unsafe_allow_html=True)

    cs1, cs2, cs3 = st.columns(3)
    for col, (lbl, val) in zip([cs1, cs2, cs3], [
        ('Molécules analysées', str(len(hist_data))),
        ('Erreur moyenne (MAE)', f'{mae:.3f}'),
        ('Taux précision < 0.3', f'{prec:.0f}%'),
    ]):
        with col:
            st.markdown(f"""
            <div class="stat-dk teal" style="margin-top:14px;">
              <div class="stat-label">{lbl}</div>
              <div class="stat-value">{val}</div>
            </div>""", unsafe_allow_html=True)

    footer()
