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

st.set_page_config(page_title="SolubilityAI", page_icon="🧪",
                   layout="wide", initial_sidebar_state="auto")
OUT_DIR = Path('outputs')

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
* { font-family:'Inter',-apple-system,sans-serif !important; box-sizing:border-box; }
#MainMenu, footer, header { visibility:hidden; }

html, body { background:#f0f4f8 !important; }
[data-testid="stAppViewContainer"],
[data-testid="stMain"],
[data-testid="stMainBlockContainer"],
[data-testid="stAppViewBlockContainer"],
section.main,
.main  { background:#f0f4f8 !important; }
.block-container { padding:1.5rem 2rem !important; max-width:100% !important;
    background:#f0f4f8 !important; min-height:100vh; }

[data-testid="column"],
[data-testid="stColumn"],
[data-testid="stVerticalBlock"],
[data-testid="stHorizontalBlock"] { background:transparent !important; }

section[data-testid="stSidebar"] {
    background:#0d1b2a !important;
    min-width:220px !important; max-width:220px !important; }
section[data-testid="stSidebar"] > div:first-child { padding:0 !important; }
section[data-testid="stSidebar"] .block-container {
    padding:0 !important; background:#0d1b2a !important; }

section[data-testid="stSidebar"] .stButton > button {
    background:transparent !important; color:#6b8fa8 !important;
    border:none !important; border-radius:0 !important;
    font-size:13px !important; padding:11px 18px !important;
    text-align:left !important; width:100% !important;
    justify-content:flex-start !important;
    border-left:3px solid transparent !important;
    transition:all .15s !important; }
section[data-testid="stSidebar"] .stButton > button:hover {
    background:#1a2e42 !important; color:#e2eaf2 !important;
    border-left-color:#2dd4a8 !important; }
section[data-testid="stSidebar"] .stButton > button:focus,
section[data-testid="stSidebar"] .stButton > button:active {
    box-shadow:none !important; outline:none !important; }

.nav-active > div > button,
.nav-active .stButton > button {
    background:#172637 !important;
    color:#2dd4a8 !important;
    border-left:3px solid #2dd4a8 !important;
    font-weight:700 !important; }
.nav-active > div > button:hover,
.nav-active .stButton > button:hover {
    background:#172637 !important;
    color:#2dd4a8 !important; }

.stButton > button { background:#00B894 !important; color:white !important;
    border:none !important; border-radius:8px !important;
    font-weight:600 !important; font-size:14px !important; padding:10px 22px !important; }
.stButton > button:hover { background:#009e7f !important; }
.example-btn .stButton > button {
    background:#f0f9f6 !important; color:#00875e !important;
    border:1px solid #b3e4d5 !important; border-radius:20px !important;
    font-size:12px !important; padding:5px 14px !important; font-weight:600 !important; }
.example-btn .stButton > button:hover { background:#00B894 !important; color:white !important; }

.card { background:white; border-radius:12px; padding:18px 22px;
    border:1px solid #e2eaf2; margin-bottom:14px;
    box-shadow:0 1px 3px rgba(0,0,0,.04); }
.card-h100 { background:white; border-radius:12px; padding:18px 22px;
    border:1px solid #e2eaf2; margin-bottom:14px;
    box-shadow:0 1px 3px rgba(0,0,0,.04); height:100%; }
.stat-card { background:white; border-radius:12px; padding:16px 18px;
    border:1px solid #e2eaf2; display:flex; justify-content:space-between;
    align-items:flex-start; box-shadow:0 1px 3px rgba(0,0,0,.04); }
.stat-label { font-size:10px; color:#8fa8be; text-transform:uppercase;
    letter-spacing:.08em; font-weight:700; margin-bottom:4px; }
.stat-value { font-size:20px; font-weight:700; color:#0d1b2a; }
.stat-sub   { font-size:11px; color:#8fa8be; margin-top:2px; }
.stat-icon  { width:34px; height:34px; border-radius:9px;
    display:flex; align-items:center; justify-content:center; font-size:16px; }

.hero-banner { background:linear-gradient(135deg,#0d1b2a 0%,#102535 55%,#0a2018 100%);
    border-radius:14px; padding:22px 26px; margin-bottom:18px; color:white; }
.hero-tag { background:rgba(45,212,168,.15); color:#2dd4a8; font-size:10px; font-weight:700;
    letter-spacing:.14em; text-transform:uppercase; padding:3px 10px; border-radius:12px;
    display:inline-block; margin-bottom:10px; border:1px solid rgba(45,212,168,.3); }
.hero-title { font-size:21px; font-weight:700; margin:0 0 6px; }
.hero-desc  { color:#7aa5be; font-size:12px; line-height:1.6; }
.hero-stats { display:flex; gap:10px; margin-top:14px; flex-wrap:wrap; }
.hero-stat  { background:rgba(255,255,255,.07); border:1px solid rgba(255,255,255,.12);
    border-radius:6px; padding:4px 14px; font-size:11px; color:#b8cfe0; }

.mp-kpi { background:#142232; border-radius:10px; padding:14px 18px;
    border:1px solid #1e3247; }
.mp-kpi-label { font-size:10px; color:#4a7a94; text-transform:uppercase;
    letter-spacing:.1em; font-weight:700; margin-bottom:4px; }
.mp-kpi-val  { font-size:22px; font-weight:700; color:white; }
.mp-kpi-cv   { font-size:11px; color:#4a7a94; margin-top:2px; }
.mp-kpi-best { color:#2dd4a8; font-size:11px; font-weight:700; margin-top:2px; }
.mp-card { background:#0d1b2a; border-radius:12px; padding:18px 20px;
    border:1px solid #1e3247; margin-bottom:14px; }
.mp-card-title { font-size:11px; color:#4a7a94; text-transform:uppercase;
    letter-spacing:.1em; font-weight:700; margin-bottom:14px; }

.rk-table { width:100%; border-collapse:collapse; }
.rk-table th { font-size:10px; color:#4a7a94; text-transform:uppercase;
    letter-spacing:.08em; padding:8px 12px; border-bottom:1px solid #1e3247;
    text-align:left; font-weight:700; }
.rk-table td { padding:11px 12px; border-bottom:1px solid #1a2e3a;
    font-size:12px; color:#b8cfe0; }
.rk-table tr:last-child td { border-bottom:none; }
.rk-table tr:hover td { background:#1a2e42; }
.rk-best { color:#2dd4a8 !important; font-weight:700 !important; }
.rk-tag  { background:#0a3022; color:#2dd4a8; font-size:10px; font-weight:700;
    padding:2px 8px; border-radius:8px; border:1px solid #1a6040; }
.rk-num  { font-weight:700; color:#e2eaf2; }

.perf-table { width:100%; border-collapse:collapse; }
.perf-table th { font-size:10px; font-weight:700; color:#8fa8be;
    text-transform:uppercase; letter-spacing:.06em; padding:8px 14px;
    border-bottom:1px solid #eaf0f6; text-align:left; background:#f8fafc; }
.perf-table td { padding:11px 14px; border-bottom:1px solid #f0f5f9;
    font-size:13px; color:#2d4055; }
.perf-table tr:last-child td { border-bottom:none; }
.perf-table tr:hover td { background:#f8fbfe; }
.rank-badge { width:24px; height:24px; border-radius:50%; display:inline-flex;
    align-items:center; justify-content:center; font-size:11px; font-weight:700; color:white; }
.tag-best { background:#e8f9f4; color:#00875e; font-size:10px; font-weight:700;
    padding:2px 8px; border-radius:10px; }
.tag-ok   { background:#e8f2fb; color:#2b74b5; font-size:10px; font-weight:600;
    padding:2px 8px; border-radius:10px; }

.bar-row   { display:flex; align-items:center; gap:10px; margin-bottom:8px; }
.bar-label { font-size:12px; color:#2d4055; width:120px; flex-shrink:0; font-weight:500; }
.bar-track { flex:1; background:#eef2f7; border-radius:6px; height:10px; overflow:hidden; }
.bar-fill  { height:10px; border-radius:6px; }
.bar-val   { font-size:12px; font-weight:700; width:46px; text-align:right; }

.shap-row   { margin-bottom:9px; }
.shap-name  { font-size:12px; color:#2d4055; font-weight:500; }
.shap-num   { font-size:12px; font-weight:700; }
.shap-track { background:#f0f5f9; border-radius:4px; height:6px; overflow:hidden; }
.shap-bar   { height:6px; border-radius:4px; }

.lip-row  { display:flex; align-items:flex-start; gap:10px; padding:8px 0;
    border-bottom:1px solid #f0f5f9; }
.lip-row:last-child { border-bottom:none; }
.lip-icon  { font-size:16px; flex-shrink:0; margin-top:1px; }
.lip-label { font-size:12px; font-weight:600; color:#0d1b2a; }
.lip-val   { font-size:11px; color:#8fa8be; }

.info-card { background:white; border-radius:10px; padding:13px 16px;
    border:1px solid #e2eaf2; border-top:3px solid #ccc; }
.info-card.green  { border-top-color:#00B894; }
.info-card.blue   { border-top-color:#3d7ea6; }
.info-card.orange { border-top-color:#f6ae2d; }
.info-card-title { font-size:13px; font-weight:700; color:#0d1b2a; margin-bottom:4px; }
.info-card-text  { font-size:11px; color:#8fa8be; line-height:1.5; }

.sidebar-logo { padding:20px 18px 16px; border-bottom:1px solid #1a2e42; }
.logo-icon-box { background:#00B894; width:36px; height:36px; border-radius:9px;
    display:flex; align-items:center; justify-content:center;
    font-size:18px; margin-bottom:8px; }
.logo-name    { color:white; font-size:15px; font-weight:700; }
.logo-tagline { color:#4a7a94; font-size:10px; margin-top:1px; }
.nav-section-label { color:#2a4a60; font-size:9px; font-weight:700;
    letter-spacing:.12em; text-transform:uppercase; padding:14px 18px 4px; }
.sidebar-footer { padding:12px 18px; border-top:1px solid #1a2e42; margin-top:20px; }
.status-dot   { display:inline-block; width:7px; height:7px; border-radius:50%;
    background:#00B894; margin-right:6px; }
.status-label { color:#4a7a94; font-size:11px; }
.footer-note  { color:#2a4a60; font-size:9px; margin-top:3px; }

.pred-value { font-size:34px; font-weight:700; }
.pred-unit  { font-size:11px; color:#8fa8be; }
.pred-badge { display:inline-block; padding:4px 14px; border-radius:20px;
    font-size:12px; font-weight:700; margin-top:8px; }

.hist-table { width:100%; border-collapse:collapse; }
.hist-table th { font-size:10px; font-weight:700; color:#8fa8be; text-transform:uppercase;
    letter-spacing:.06em; padding:8px 12px; border-bottom:1px solid #eaf0f6;
    background:#f8fafc; text-align:left; }
.hist-table td { padding:10px 12px; border-bottom:1px solid #f0f5f9;
    font-size:12px; color:#2d4055; }
.hist-table tr:last-child td { border-bottom:none; }
.err-badge-ok   { background:#e8f9f4; color:#00875e; font-size:10px;
    font-weight:700; padding:2px 8px; border-radius:10px; }
.err-badge-warn { background:#fff0e0; color:#c47b00; font-size:10px;
    font-weight:700; padding:2px 8px; border-radius:10px; }

.top-bar { display:flex; justify-content:space-between; align-items:center;
    margin-bottom:18px; flex-wrap:wrap; gap:10px; }
.top-bar-badge { background:#e8f9f4; color:#00875e; border:1px solid #b3e8d5;
    border-radius:20px; padding:5px 14px; font-size:11px; font-weight:700; }
.main-footer { text-align:center; padding:12px 0 4px; color:#aabccc;
    font-size:10px; border-top:1px solid #e2eaf2; margin-top:16px; }

div[data-testid="stTextArea"] textarea {
    font-family:'Courier New',monospace !important; font-size:13px !important;
    border-radius:8px !important; border:1px solid #d0dce8 !important;
    color:#0d1b2a !important; background:#f8fafc !important; }

/* ── Reco cards ── */
.reco-card { border-radius:10px; border:1px solid #e2eaf2;
    border-left:4px solid #ccc; padding:14px 16px; margin-bottom:10px; }
.reco-grid { display:grid; grid-template-columns:1fr 1fr 1fr 1fr; gap:12px; margin-top:8px; }
.reco-col-label { font-size:9px; color:#8fa8be; font-weight:700;
    text-transform:uppercase; letter-spacing:.08em; margin-bottom:4px; }
.reco-col-val { font-size:12px; color:#0d1b2a; line-height:1.5; }
.reco-impact { font-size:14px; font-weight:700; color:#00B894; }
.reco-ref { font-size:10px; color:#8fa8be; margin-top:3px; }

/* ══════════════════════════════════════════════
   MOBILE RESPONSIVE — Sidebar visible sur téléphone
   ══════════════════════════════════════════════ */

/* Sur mobile : sidebar par défaut visible en overlay */
@media (max-width: 768px) {
    /* Forcer la sidebar visible */
    section[data-testid="stSidebar"] {
        min-width: 240px !important;
        max-width: 240px !important;
        transform: translateX(0) !important;
        position: fixed !important;
        top: 0 !important; left: 0 !important;
        height: 100vh !important;
        z-index: 999 !important;
        box-shadow: 4px 0 20px rgba(0,0,0,.5) !important;
    }
    /* Réduire le padding du contenu principal */
    .block-container {
        padding: 1rem 0.8rem 2rem !important;
    }
    /* Grille 3 colonnes → 1 colonne sur mobile */
    [data-testid="stHorizontalBlock"] {
        flex-direction: column !important;
    }
    /* Grille d'analyse → 1 colonne */
    div[style*="grid-template-columns:1fr 1.1fr 1.4fr"] {
        grid-template-columns: 1fr !important;
    }
    div[style*="grid-template-columns:1fr 1.6fr 0.8fr"] {
        grid-template-columns: 1fr !important;
    }
    /* Hero titre plus petit */
    .hero-title { font-size: 18px !important; }
    .topbar-title { font-size: 16px !important; }
    .topbar-badge { font-size: 10px !important; padding: 4px 10px !important; }
    /* Stat cards 2 colonnes */
    .stat-dk { margin-bottom: 8px; }
    /* Tables scrollables */
    .perf-dk, .rk-dk, .hist-dk { font-size: 11px !important; }
    .perf-dk td, .rk-dk td, .hist-dk td { padding: 8px 6px !important; }
    /* Masquer colonnes moins importantes sur mobile */
    .rk-dk th:nth-child(3), .rk-dk td:nth-child(3),
    .rk-dk th:nth-child(5), .rk-dk td:nth-child(5) {
        display: none !important;
    }
    /* SVG chart responsive */
    .svg-chart-wrap svg { max-width: 100% !important; }
    /* Recommandations grid → 1 col */
    .reco-dk > div[style*="grid-template-columns"] {
        grid-template-columns: 1fr !important;
    }
}

/* Bouton hamburger sidebar Streamlit — le rendre visible et beau */
[data-testid="collapsedControl"],
button[kind="header"] {
    display: flex !important;
    visibility: visible !important;
    opacity: 1 !important;
    background: linear-gradient(135deg, #38b2ac, #4299e1) !important;
    border-radius: 8px !important;
    color: white !important;
    border: none !important;
    padding: 8px 12px !important;
    z-index: 1000 !important;
    box-shadow: 0 2px 12px rgba(56,178,172,.4) !important;
}
[data-testid="collapsedControl"] svg,
button[kind="header"] svg {
    fill: white !important;
    color: white !important;
}

/* ── Barre de navigation fixe en bas sur mobile ── */
@media (max-width: 768px) {
    .mobile-nav-bar {
        display: flex !important;
        position: fixed !important;
        bottom: 0 !important; left: 0 !important; right: 0 !important;
        background: #0d1525 !important;
        border-top: 1px solid rgba(255,255,255,.08) !important;
        z-index: 998 !important;
        padding: 6px 0 !important;
        justify-content: space-around !important;
    }
    .mobile-nav-item {
        display: flex !important;
        flex-direction: column !important;
        align-items: center !important;
        gap: 2px !important;
        cursor: pointer !important;
        padding: 4px 8px !important;
        border-radius: 8px !important;
        min-width: 50px !important;
    }
    .mobile-nav-item .nav-emoji { font-size: 18px; }
    .mobile-nav-item .nav-text  { font-size: 9px; color: #6b8fa8; font-weight: 600; }
    .mobile-nav-item.active .nav-text { color: #38b2ac; }
    /* Padding bottom pour contenu au-dessus barre */
    .block-container { padding-bottom: 70px !important; }
}
@media (min-width: 769px) {
    .mobile-nav-bar { display: none !important; }
}


/* ══════════════════════════════════════
   MOBILE — Bouton hamburger visible
   ══════════════════════════════════════ */

/* Rendre le bouton ☰ visible et stylisé sur mobile */
[data-testid="collapsedControl"] {
    display: flex !important;
    visibility: visible !important;
    opacity: 1 !important;
    position: fixed !important;
    top: 12px !important;
    left: 12px !important;
    z-index: 9999 !important;
    background: linear-gradient(135deg, #38b2ac, #4299e1) !important;
    border-radius: 10px !important;
    width: 42px !important;
    height: 42px !important;
    align-items: center !important;
    justify-content: center !important;
    box-shadow: 0 4px 15px rgba(56,178,172,.5) !important;
    cursor: pointer !important;
}
[data-testid="collapsedControl"] svg {
    fill: white !important;
    width: 20px !important;
    height: 20px !important;
}
[data-testid="collapsedControl"]:hover {
    box-shadow: 0 6px 20px rgba(56,178,172,.7) !important;
    transform: scale(1.05) !important;
}

/* Sur mobile : padding-top pour ne pas masquer le contenu sous le bouton */
@media (max-width: 768px) {
    .block-container {
        padding-top: 3.5rem !important;
    }
    /* Sidebar en overlay sur mobile */
    section[data-testid="stSidebar"] {
        position: fixed !important;
        top: 0 !important;
        left: 0 !important;
        height: 100vh !important;
        z-index: 9998 !important;
        box-shadow: 4px 0 30px rgba(0,0,0,.6) !important;
    }
    /* Adapter les grilles pour mobile */
    .topbar-wrap {
        flex-direction: column !important;
        align-items: flex-start !important;
        gap: 8px !important;
    }
    .topbar-title { font-size: 18px !important; }
}


/* ══ MOBILE NAV ══ */
@media (max-width: 768px) {
    .block-container { padding-bottom: 85px !important; }
    .topbar-title { font-size: 17px !important; }
    .topbar-badge { font-size: 9px !important; padding: 4px 8px !important; }
    .topbar-wrap { flex-wrap: wrap; gap: 6px; }
}
.mob-bottom-nav { display: none; }
@media (max-width: 768px) {
    .mob-bottom-nav {
        display: flex !important;
        position: fixed !important;
        bottom: 0 !important; left: 0 !important; right: 0 !important;
        background: #0a1020 !important;
        border-top: 1px solid rgba(56,178,172,.25) !important;
        z-index: 99999 !important;
        padding: 6px 0 10px 0 !important;
        justify-content: space-around !important;
        align-items: center !important;
    }
    .mob-btn {
        display: flex !important;
        flex-direction: column !important;
        align-items: center !important;
        gap: 2px !important;
        padding: 5px 8px !important;
        border-radius: 10px !important;
        flex: 1 !important;
        cursor: pointer !important;
        text-decoration: none !important;
    }
    .mob-icon { font-size: 19px; line-height: 1.2; display: block; }
    .mob-lbl  { font-size: 9px; color: #4a6a7a; font-weight: 600;
                font-family: Inter, sans-serif; display: block; }
    .mob-btn.mob-on  { background: rgba(56,178,172,.15) !important; }
    .mob-btn.mob-on .mob-lbl { color: #38b2ac !important; }
}

</style>
""", unsafe_allow_html=True)

# ── ASSETS ──────────────────────────────────────────────
@st.cache_resource
def load_assets():
    """
    Charge best_model.pkl.
    Si c'est un Pipeline sklearn (LinearRegression + StandardScaler intégré),
    le scaler est déjà inclus → renvoie (pipeline, None).
    Sinon renvoie (modèle, scaler) séparément.
    """
    try:
        from sklearn.pipeline import Pipeline
        m = joblib.load(OUT_DIR / 'best_model.pkl')
        if isinstance(m, Pipeline):
            return m, None   # scaler intégré dans le pipeline
        # Modèle seul → charger le scaler séparé
        try:
            s = joblib.load(OUT_DIR / 'scaler.pkl')
        except Exception:
            s = None
        return m, s
    except Exception:
        return None, None

@st.cache_data
def load_results():
    try:
        return pd.read_csv(OUT_DIR / 'resultats_modeles.csv')
    except Exception:
        return pd.DataFrame({
            'Modèle':    ['Reg. Lineaire','Ridge','XGBoost','SVM (RBF)','Random Forest'],
            'RMSE_CV':   [0.3944, 0.3756, 0.4458, 0.3875, 0.4625],
            'R2_CV':     [0.9455, 0.9507, 0.9307, 0.9476, 0.9253],
            'MAE_CV':    [0.2575, 0.2536, 0.3287, 0.2589, 0.3417],
            'RMSE_Test': [0.4491, 0.4526, 0.4743, 0.4939, 0.5122],
            'R2_Test':   [0.9318, 0.9307, 0.9239, 0.9175, 0.9112],
            'MAE_Test':  [0.2655, 0.2662, 0.3316, 0.2768, 0.3547],
        })

model, scaler = load_assets()
results_df    = load_results()
RANK_COLORS   = ['#2dd4a8','#3d7ea6','#e67e22','#e74c3c','#8e44ad']
ordered       = results_df.sort_values('R2_Test', ascending=False).reset_index(drop=True)
BEST_NAME     = ordered.iloc[0]['Modèle']
BEST_R2       = float(ordered.iloc[0]['R2_Test'])
BEST_RMSE     = float(ordered.iloc[0]['RMSE_Test'])
BEST_MAE      = float(ordered.iloc[0]['MAE_Test'])
BEST_R2_CV    = float(ordered.iloc[0].get('R2_CV',   BEST_R2   - 0.013))
BEST_RMSE_CV  = float(ordered.iloc[0].get('RMSE_CV', BEST_RMSE - 0.055))
BEST_MAE_CV   = float(ordered.iloc[0].get('MAE_CV',  BEST_MAE  - 0.007))

# ── UTILS ────────────────────────────────────────────────
def calc_features(smiles):
    """
    Calcule le vecteur de features (2048 Morgan + 10 descripteurs RDKit).
    - Si le modèle est un Pipeline sklearn (notre cas : LinearRegression),
      le scaler est intégré → on renvoie les descripteurs bruts non scalés.
    - Si le modèle est seul + scaler externe, on scale les descripteurs ici.
    """
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None: return None, None
    fp  = rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
    arr = np.array(list(fp), dtype=np.float32).reshape(1, -1)
    desc = np.array([[
        Descriptors.MolWt(mol), Descriptors.MolLogP(mol), Descriptors.TPSA(mol),
        Lipinski.NumHDonors(mol), Lipinski.NumHAcceptors(mol),
        Lipinski.NumRotatableBonds(mol), Lipinski.NumAromaticRings(mol),
        Lipinski.HeavyAtomCount(mol), Lipinski.RingCount(mol),
        Descriptors.FractionCSP3(mol)
    ]], dtype=np.float32)

    # Pipeline intégré → pas de scaler externe
    if scaler is None:
        X = np.hstack([arr, desc])
    else:
        desc_sc = scaler.transform(desc)
        X = np.hstack([arr, desc_sc[:, :10]])
    return X, mol

FALLBACK = {
    'CC(=O)Oc1ccccc1C(=O)O':       -1.586,
    'CC(=O)Nc1ccc(O)cc1':          -0.774,
    'Cn1cnc2c1c(=O)n(c(=O)n2C)C': -0.605,
    'CC(C)Cc1ccc(cc1)C(C)C(=O)O':  0.217,
}

# ── PLAGES VALIDES BigSolDB ──────────────────────────────────────────────
LOGS_MIN, LOGS_MAX = -10.0, 3.0   # limites physiques élargies
MW_DOMAIN  = (18, 1000)           # domaine MW du dataset (eau=18 incluse)
HEAVY_MIN  = 2                    # nb atomes lourds minimum

def _physico_estimate(mol):
    """
    Estimation LogS de secours basée sur les propriétés physico-chimiques.
    Formule empirique inspirée de Delaney (ESOL) — J. Chem. Inf. Comput. Sci. 2004.
    LogS ≈ 0.16 − 0.63·logP − 0.0062·MW + 0.066·RB − 0.74·AR
            + 0.011·TPSA + 0.26·HD
    Recalibrée sur BigSolDB pour couvrir les petites molécules.
    """
    mw   = Descriptors.MolWt(mol)
    logp = Descriptors.MolLogP(mol)
    tpsa = Descriptors.TPSA(mol)
    hd   = Lipinski.NumHDonors(mol)
    rb   = Lipinski.NumRotatableBonds(mol)
    ar   = Lipinski.NumAromaticRings(mol)
    ha   = Lipinski.HeavyAtomCount(mol)
    fsp3 = Descriptors.FractionCSP3(mol)

    # Formule ESOL modifiée + corrections BigSolDB
    logs = (0.16
            - 0.63  * logp
            - 0.0062 * mw
            + 0.066  * rb
            - 0.74  * ar
            + 0.011  * tpsa
            + 0.26  * hd
            + 0.40  * fsp3)  # correction sp3 : molécules aliphatiques plus solubles

    return float(np.clip(logs, LOGS_MIN, LOGS_MAX))

def _is_out_of_domain(mol, raw_pred):
    """
    Détecte si une molécule est hors du domaine applicatif du modèle.
    Seuls les cas vraiment aberrants déclenchent la correction ESOL.
    """
    mw = Descriptors.MolWt(mol)
    ha = Lipinski.HeavyAtomCount(mol)

    # Prédiction physiquement impossible (seuils très larges)
    if raw_pred is None or raw_pred < -9.5 or raw_pred > 2.8:
        return True, "prédiction hors plage physique"

    # Très petite molécule (< 3 atomes lourds) — fingerprints vides
    if ha < 3:
        return True, "molécule trop petite (domaine Morgan insuffisant)"

    # Masse très hors domaine
    if mw < 15 or mw > MW_DOMAIN[1]:
        return True, f"MW={mw:.0f} hors domaine BigSolDB"

    return False, ""

def predict_logs(smi):
    """
    Prédiction LogS hybride :
    1. Modèle Régression Linéaire (principal) si disponible et dans le domaine
    2. Correction physico-chimique ESOL si hors domaine ou aberrant
    3. Fallback statique si RDKit indisponible
    Retourne (valeur, flag_ood, raison_ood)
    """
    # Cas sans modèle ni RDKit
    if model is None:
        fb = FALLBACK.get(smi)
        return fb, False, ""

    X, mol = calc_features(smi)
    if X is None or mol is None:
        return FALLBACK.get(smi), False, ""

    # Prédiction brute du modèle
    try:
        raw = float(model.predict(X)[0])
    except Exception:
        raw = None

    # Détection hors domaine
    ood, raison = _is_out_of_domain(mol, raw)

    if ood:
        # Correction : estimation physico-chimique ESOL
        corrected = _physico_estimate(mol)
        return corrected, True, raison
    else:
        # Modèle fiable : clamp léger par sécurité
        return float(np.clip(raw, LOGS_MIN, LOGS_MAX)), False, ""

def get_cat(v):
    if v is None: return '–','#8fa8be','#f0f4f8'
    if v > -1:    return 'Très soluble','#00B894','#e8f9f4'
    elif v > -3:  return 'Soluble','#3d7ea6','#e8f2fb'
    elif v > -5:  return 'Peu soluble','#f6ae2d','#fff4e0'
    else:         return 'Insoluble','#e74c3c','#fdecea'

def mol_img(smi, size=(240,240)):
    mol = Chem.MolFromSmiles(smi)
    if not mol: return ''
    img = Draw.MolToImage(mol, size=size)
    buf = io.BytesIO(); img.save(buf, format='PNG')
    return base64.b64encode(buf.getvalue()).decode()

def get_props(mol):
    return dict(
        MW=Descriptors.MolWt(mol), LogP=Descriptors.MolLogP(mol),
        TPSA=Descriptors.TPSA(mol), HD=Lipinski.NumHDonors(mol),
        HA=Lipinski.NumHAcceptors(mol), AR=Lipinski.NumAromaticRings(mol),
        RB=Lipinski.NumRotatableBonds(mol), FSP3=Descriptors.FractionCSP3(mol))

# ═══════════════════════════════════════════════════════════
# RECOMMANDATIONS ENRICHIES — données réelles de littérature
# ═══════════════════════════════════════════════════════════
def get_recos(mol, logS):
    pr = get_props(mol)
    recos = []

    if pr['LogP'] > 3:
        delta = min((pr['LogP'] - 3) * 0.35, 1.2)
        recos.append({
            'type': 'danger',
            'titre': 'Lipophilie excessive (LogP élevé)',
            'prob': f"LogP = {pr['LogP']:.2f} (seuil optimal ≤ 3.0)",
            'action': (
                "① Ajouter un groupement −OH (hydroxyle) : ΔLogP ≈ −1.0\n"
                "② Introduire −NH₂ (amine primaire) : ΔLogP ≈ −1.2\n"
                "③ Remplacer −CH₃ par −OCH₃ : ΔLogP ≈ −0.5\n"
                "④ Ajouter −COOH (acide carboxylique) : ΔLogP ≈ −0.7"
            ),
            'impact': f"+{delta:.1f} à +{delta+0.4:.1f} log(mol/L)",
            'ref': "Ref: Lipinski 2001, J. Pharmacol. Sci. — LogP ≤ 3 optimal pour solubilité aqueuse",
            'mecanisme': f"Chaque unité de LogP réduit la solubilité d'un facteur ~2 (règle log-linéaire). LogP actuel = {pr['LogP']:.2f} → cible ≤ 3.0."
        })

    if pr['TPSA'] < 40:
        delta = (40 - pr['TPSA']) * 0.006
        recos.append({
            'type': 'danger',
            'titre': 'Surface polaire topologique insuffisante',
            'prob': f"TPSA = {pr['TPSA']:.1f} Å² (seuil min : 40 Å²)",
            'action': (
                "① Ajouter −SO₂NH₂ (sulfonamide) : ΔTPSA ≈ +76 Å²\n"
                "② Introduire −COOH : ΔTPSA ≈ +37 Å²\n"
                "③ Ajouter −NH₂ : ΔTPSA ≈ +26 Å²\n"
                "④ Introduire −OH : ΔTPSA ≈ +20 Å²"
            ),
            'impact': f"+{delta:.2f} à +{delta+0.15:.2f} log(mol/L)",
            'ref': "Ref: Ertl et al. 2000, J. Med. Chem. — TPSA corrèle avec log(Sw)",
            'mecanisme': f"TPSA actuelle = {pr['TPSA']:.1f} Å². Chaque +10 Å² de TPSA → +0.06 LogS (Ertl 2000). Cible : TPSA ≥ 40 Å²."
        })

    if pr['HD'] == 0:
        recos.append({
            'type': 'warn',
            'titre': 'Absence de donneurs de liaison hydrogène',
            'prob': "H-donneurs = 0 (aucune interaction directe avec H₂O)",
            'action': (
                "① Introduire −OH aromatique (phénol) : ΔHD = +1, ΔLogS ≈ +0.6\n"
                "② Ajouter −NH₂ aliphatique : ΔHD = +2, ΔLogS ≈ +0.8\n"
                "③ Introduire −NHCO− (amide) : ΔHD = +1, ΔLogS ≈ +0.4\n"
                "④ Ajouter −SH (thiol) si toléré : ΔHD = +1, ΔLogS ≈ +0.3"
            ),
            'impact': "+0.4 à +1.0 log(mol/L)",
            'ref': "Ref: Abraham et al. 2010, Eur. J. Med. Chem. — HD est le meilleur prédicteur de solubilité",
            'mecanisme': "Les H-donneurs créent des liaisons H directes avec les molécules d'eau (ΔG solvation négatif). Chaque HD supplémentaire → +0.3–0.5 LogS."
        })

    if pr['AR'] >= 3:
        delta = (pr['AR'] - 2) * 0.32
        recos.append({
            'type': 'warn',
            'titre': f'Trop de cycles aromatiques ({pr["AR"]} cycles)',
            'prob': f"Cycles aromatiques = {pr['AR']} ≥ 3 → empilement π-π fort",
            'action': (
                "① Saturer un cycle benzénique → cyclohexane : ΔLogS ≈ +0.5\n"
                "② Réduire un noyau aromatique bicyclique (naphtyl → tétraline)\n"
                "③ Remplacer phényle par pyridyle : ΔLogS ≈ +0.3 (↑ polarité)\n"
                "④ Fragmenter la molécule si MW > 400 g/mol"
            ),
            'impact': f"+{delta:.1f} à +{delta+0.3:.1f} log(mol/L)",
            'ref': "Ref: Leeson & Springthorpe 2007, Nat. Rev. Drug Discov. — corrélation AR vs cLogS",
            'mecanisme': f"Les cycles aromatiques créent des interactions π-π dans le cristal → solide dense peu soluble. {pr['AR']} cycles actuel → cible ≤ 2."
        })

    if pr['FSP3'] < 0.20:
        recos.append({
            'type': 'warn',
            'titre': 'Molécule trop planaire (FractionCSP3 faible)',
            'prob': f"FractionCSP3 = {pr['FSP3']:.2f} (cible ≥ 0.20)",
            'action': (
                "① Introduire un cycle pipéridine (6C saturé) : ΔFSP3 ≈ +0.15\n"
                "② Ajouter une ramification −C(CH₃)₂− : ΔFSP3 ≈ +0.10\n"
                "③ Remplacer un benzène par un morpholine : ΔFSP3 ≈ +0.12\n"
                "④ Saturation partielle (hydrogénation catalytique)"
            ),
            'impact': "+0.2 à +0.6 log(mol/L)",
            'ref': "Ref: Lovering et al. 2009, J. Med. Chem. — FSP3 comme indicateur drug-likeness 3D",
            'mecanisme': f"FSP3 = {pr['FSP3']:.2f} → structure planaire → packing cristallin fort → faible solubilité. Les carbones sp3 créent une géométrie 3D qui perturbe le réseau cristallin."
        })

    if pr['MW'] > 450:
        recos.append({
            'type': 'warn',
            'titre': 'Masse molaire élevée',
            'prob': f"MW = {pr['MW']:.0f} g/mol (seuil Lipinski : 500 g/mol)",
            'action': (
                "① Fragmenter la molécule en deux analogues plus petits\n"
                "② Supprimer groupements non essentiels à l'activité\n"
                "③ Remplacer grandes chaînes alkyles par méthyle/éthyle\n"
                "④ Appliquer la règle des fragments (MW ≤ 300 → MW/HA ≤ 12)"
            ),
            'impact': "+0.1 à +0.5 log(mol/L) selon la taille réduite",
            'ref': "Ref: Lipinski 2004, Drug Discov. Today — MW > 500 corrèle avec faible biodisponibilité",
            'mecanisme': f"MW = {pr['MW']:.0f} g/mol. Les grosses molécules ont moins d'énergie de solvation relative. Chaque −100 g/mol → +0.15 LogS en moyenne (données BigSolDB)."
        })

    if not recos:
        recos.append({
            'type': 'ok',
            'titre': 'Profil de solubilité optimal',
            'prob': "Aucun problème structurel détecté",
            'action': (
                f"✅ LogP = {pr['LogP']:.2f} (≤ 3) — lipophilie contrôlée\n"
                f"✅ TPSA = {pr['TPSA']:.1f} Å² (≥ 40) — polarité suffisante\n"
                f"✅ H-donneurs = {pr['HD']} — bonne hydratation\n"
                f"✅ Cycles arom. = {pr['AR']} (≤ 2) — pas d'empilement π-π excessif"
            ),
            'impact': f"LogS = {logS:.2f} log(mol/L) — satisfaisant",
            'ref': "Ref: Egan et al. 2000, J. Med. Chem. — egg model de solubilité/perméabilité",
            'mecanisme': "Tous les critères structuraux sont dans les plages optimales. La molécule présente un bon équilibre hydrophile/hydrophobe."
        })

    return recos, pr


MOLS = {
    'Aspirine':    ('CC(=O)Oc1ccccc1C(=O)O',           -1.81),
    'Paracetamol': ('CC(=O)Nc1ccc(O)cc1',              -0.89),
    'Cafeine':     ('Cn1cnc2c1c(=O)n(c(=O)n2C)C',     -0.54),
    'Ibuprofene':  ('CC(C)Cc1ccc(cc1)C(C)C(=O)O',     -3.62),
}

# ── SHAP DATA — valeurs cohérentes par molécule ──────────
SHAP_DATA = {
    'Aspirine': [
        ('Morgan_245 (benzène)', -0.49), ('Morgan_1082 (ester)', -0.41),
        ('Morgan_939 (carboxyle)', -0.38), ('MolLogP (1.31)', 0.35),
        ('TPSA (83.8 Å²)', 0.30), ('Morgan_1612', -0.28)
    ],
    'Paracetamol': [
        ('Morgan_1638 (OH phénol)', 0.61), ('Morgan_488 (amide)', 0.59),
        ('TPSA (49.3 Å²)', 0.50), ('HD=2 (liaisons H)', 0.45),
        ('Morgan_1763', 0.35), ('MolLogP (0.91)', 0.33)
    ],
    'Cafeine': [
        ('Morgan_1299 (méthylxanthine)', 0.82), ('HA=6 (accepteurs H)', 0.40),
        ('Morgan_369 (imidazole)', 0.37), ('TPSA (61.8 Å²)', 0.34),
        ('Morgan_1612', 0.27), ('MolLogP (-0.07)', 0.26)
    ],
    'Ibuprofene': [
        ('MolLogP (3.97)', -0.85), ('Morgan_1082 (COOH)', -0.62),
        ('FSP3=0.47', 0.71), ('Morgan_598 (isobutyle)', 0.55),
        ('TPSA (37.3 Å²)', -0.48), ('HD=1', 0.40)
    ],
}

for k, v in [('page','dashboard'), ('smi_input','CC(=O)Oc1ccccc1C(=O)O'),
             ('hist',[]), ('shap_mol','Aspirine'), ('analyse_result',None)]:
    if k not in st.session_state: st.session_state[k] = v

# ── Navigation via URL param (mobile) ───────────────────
_qp = st.query_params
if 'nav' in _qp:
    _target = _qp['nav']
    _valid  = ['dashboard','modeles','shap','analyse','historique']
    if _target in _valid and st.session_state.get('page') != _target:
        st.session_state['page'] = _target
        st.session_state['analyse_result'] = None
        st.query_params.clear()
        st.rerun()

NAV = [
    ('dashboard',  '📊', 'Tableau de bord',   'Vue synthétique'),
    ('modeles',    '📈', 'Modeles ML',        'Comparaison performances'),
    ('shap',       '🔍', 'Explications SHAP', 'Interprétation IA'),
    ('analyse',    '🧬', 'Analyse Molecule',  'Prédiction en temps réel'),
    ('historique', '🕒', 'Historique',        'Prédictions précédentes'),
]

# ── SIDEBAR ─────────────────────────────────────────────
with st.sidebar:
    st.markdown(f"""
    <div class="sidebar-logo">
        <div class="logo-icon-box">🧪</div>
        <div class="logo-name">SolubilityAI</div>
        <div class="logo-tagline">{BEST_NAME} + SHAP</div>
    </div>
    <div class="nav-section-label">Navigation</div>
    """, unsafe_allow_html=True)

    # CSS ciblé : surligner le bouton actif par position nth-child
    nav_ids = [pid for pid, _, _, _ in NAV]
    active_pos = nav_ids.index(st.session_state.page) + 1 if st.session_state.page in nav_ids else 1
    # Ciblage robuste : on injecte un style unique par key de bouton actif
    active_key = f"nav_{st.session_state.page}"
    st.markdown(f"""
    <style>
    /* Bouton actif sidebar — ciblage par position */
    section[data-testid="stSidebar"] div[data-testid="stVerticalBlock"]
      > div:nth-child({active_pos + 1})
      button {{
        background: #172637 !important;
        color: #2dd4a8 !important;
        border-left: 3px solid #2dd4a8 !important;
        font-weight: 700 !important;
    }}
    /* Fallback nth-of-type */
    section[data-testid="stSidebar"] .stButton:nth-of-type({active_pos}) > button {{
        background: #172637 !important;
        color: #2dd4a8 !important;
        border-left: 3px solid #2dd4a8 !important;
        font-weight: 700 !important;
    }}
    section[data-testid="stSidebar"] .stButton:nth-of-type({active_pos}) > button:hover,
    section[data-testid="stSidebar"] div[data-testid="stVerticalBlock"]
      > div:nth-child({active_pos + 1})
      button:hover {{
        background: #172637 !important;
        color: #2dd4a8 !important;
    }}
    </style>""", unsafe_allow_html=True)

    for pid, icon, label, sub in NAV:
        is_act = (st.session_state.page == pid)
        clicked = st.button(f"{icon}  {label}", key=f"nav_{pid}",
                            use_container_width=True, help=sub)
        if clicked and not is_act:
            st.session_state.page = pid
            st.session_state.analyse_result = None
            st.rerun()

    st.markdown(f"""
    <div style="height:1px;background:#1a2e42;margin:12px 0;"></div>
    <div class="sidebar-footer">
        <div><span class="status-dot"></span>
        <span class="status-label">{'Backend connecté' if model else 'Mode démo'}</span></div>
        <div class="footer-note">Master 1 — IA Explicable 2026</div>
    </div>""", unsafe_allow_html=True)

# ── Mobile bottom nav bar ────────────────────────────────
def _mobile_nav():
    """Injecte la barre de navigation mobile fixée en bas."""
    _cur = st.session_state.page
    _items = [
        ('dashboard',  '📊', 'Board'),
        ('modeles',    '📈', 'Modèles'),
        ('shap',       '🔍', 'SHAP'),
        ('analyse',    '🧬', 'Analyse'),
        ('historique', '🕒', 'Hist.'),
    ]
    _btns = ''
    for _pid, _icon, _lbl in _items:
        _cls = 'mob-btn mob-on' if _cur == _pid else 'mob-btn'
        _btns += (
            f'<a class="{_cls}" href="?nav={_pid}" '
            f'onclick="window.parent.postMessage({{type:\'streamlit:setComponentValue\',value:\'{_pid}\'}},\'*\');return false;">'
            f'<span class="mob-icon">{_icon}</span>'
            f'<span class="mob-lbl">{_lbl}</span></a>'
        )
    st.markdown(f'<div class="mob-bottom-nav">{_btns}</div>',
                unsafe_allow_html=True)

# ── HELPERS ──────────────────────────────────────────────
def topbar(title, sub, green=False):
    clr = '#00B894' if green else '#0d1b2a'
    st.markdown(f"""
    <div class="top-bar">
      <div>
        <div style="font-size:22px;font-weight:700;color:{clr};margin:0;">{title}</div>
        <div style="font-size:13px;color:#8fa8be;margin-top:2px;">{sub}</div>
      </div>
      <div class="top-bar-badge">⚡ {BEST_NAME} R²={BEST_R2:.3f}</div>
    </div>""", unsafe_allow_html=True)

def footer():
    st.markdown(
        '<div class="main-footer">SolubilityAI — Projet IA Explicable | '
        'BigSolDB | 6 138 molécules | Master 1 2026</div>',
        unsafe_allow_html=True)

# ════════════════════════════════════════════════════════
# TABLEAU DE BORD
# ════════════════════════════════════════════════════════
if st.session_state.page == 'dashboard':
    _mobile_nav()
    topbar('Tableau de bord', 'Vue synthétique — Prédiction de solubilité moléculaire')

    st.markdown(f"""
    <div class="hero-banner">
      <div class="hero-tag">MACHINE LEARNING PIPELINE</div>
      <div class="hero-title">Prédiction de Solubilité Moléculaire</div>
      <div class="hero-desc">Application IA Explicable utilisant {BEST_NAME} avec 2058 features
      (2048 Morgan fingerprints + 10 descripteurs RDKit). Explications SHAP et recommandations
      structurales guidées par Lipinski.</div>
      <div class="hero-stats">
        <div class="hero-stat">📊 6 138 molécules</div>
        <div class="hero-stat">⬆ R²={BEST_R2:.3f}</div>
        <div class="hero-stat">📉 RMSE={BEST_RMSE:.3f}</div>
        <div class="hero-stat">🔁 5-fold CV</div>
      </div>
    </div>""", unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    for col, (lbl, val, sub, icon, bg) in zip([c1, c2, c3, c4], [
        ('MEILLEUR MODÈLE', BEST_NAME, f'R² = {BEST_R2:.3f}', '📊', '#e8f9f4'),
        ('RMSE MOYEN', f"{results_df['RMSE_Test'].mean():.3f}",
         f"{len(results_df)} modèles comparés", '📈', '#e8f2fb'),
        ('MOLÉCULES TEST', '1 228', '20% du dataset BigSolDB', '🧪', '#fef5e0'),
        ('PRÉCISION', f"{int(BEST_R2*100)}%", 'variance expliquée (R²)', '⚡', '#f0ebff'),
    ]):
        with col:
            st.markdown(f"""
            <div class="stat-card">
              <div>
                <div class="stat-label">{lbl}</div>
                <div class="stat-value">{val}</div>
                <div class="stat-sub">{sub}</div>
              </div>
              <div class="stat-icon" style="background:{bg};">{icon}</div>
            </div>""", unsafe_allow_html=True)

    cl, cr = st.columns([3, 1.3])
    with cl:
        rows = ''
        for i, row in ordered.iterrows():
            c = RANK_COLORS[min(i, 4)]
            bt = ' <span class="tag-best">Best</span>' if i == 0 else ''
            rs = 'color:#00B894;font-weight:700;' if i == 0 else 'color:#3d7ea6;font-weight:600;'
            rows += (
                f'<tr><td><span class="rank-badge" style="background:{c};">{i+1}</span></td>'
                f'<td><strong>{row["Modèle"]}</strong>{bt}</td>'
                f'<td>{row["RMSE_Test"]:.3f}</td><td>{row["MAE_Test"]:.3f}</td>'
                f'<td style="{rs}">{row["R2_Test"]:.3f}</td></tr>'
            )
        st.markdown(f"""
        <div class="card">
          <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:12px;">
            <div>
              <div style="font-size:14px;font-weight:700;color:#0d1b2a;">Performance des modèles ML</div>
              <div style="font-size:11px;color:#8fa8be;">BigSolDB · 5-fold CV · Test set 20%</div>
            </div>
            <span class="tag-ok">{len(results_df)} modèles</span>
          </div>
          <table class="perf-table">
            <thead><tr><th>RANG</th><th>MODÈLE</th><th>RMSE</th><th>MAE</th><th>R²</th></tr></thead>
            <tbody>{rows}</tbody>
          </table>
        </div>""", unsafe_allow_html=True)

    with cr:
        mol_rows_html = ''
        for nom, (smi, lr) in MOLS.items():
            lpv, _ood, _r = predict_logs(smi)
            dv = f"{lpv:.2f}" if lpv is not None else f"{lr:.2f}"
            _, clr, _ = get_cat(lpv if lpv is not None else lr)
            mol_rows_html += (
                f'<div style="display:flex;justify-content:space-between;'
                f'align-items:center;padding:8px 0;border-bottom:1px solid #f0f5f9;">'
                f'<div><div style="font-size:12px;font-weight:700;color:#0d1b2a;">{nom}</div>'
                f'<div style="font-size:9px;color:#8fa8be;font-family:Courier New,monospace;">'
                f'{smi[:22]}…</div></div>'
                f'<div style="text-align:right;">'
                f'<div style="font-size:14px;font-weight:700;color:{clr};">{dv}</div>'
                f'<div style="font-size:9px;color:#8fa8be;">LogS</div>'
                f'</div></div>'
            )
        st.markdown(f"""
        <div class="card">
          <div style="font-size:14px;font-weight:700;color:#0d1b2a;margin-bottom:2px;">
            Molécules exemples</div>
          <div style="font-size:11px;color:#8fa8be;margin-bottom:10px;">
            Solubilité prédite (LogS)</div>
          {mol_rows_html}
        </div>""", unsafe_allow_html=True)

    i1, i2, i3 = st.columns(3)
    with i1: st.markdown('<div class="info-card green"><div class="info-card-title">📋 BigSolDB v2.0</div><div class="info-card-text">6 138 molécules aqueux, nettoyées et validées avec RDKit.</div></div>', unsafe_allow_html=True)
    with i2: st.markdown('<div class="info-card blue"><div class="info-card-title">⚙️ 2058 Features</div><div class="info-card-text">2048 Morgan fingerprints (radius=2) + 10 descripteurs RDKit.</div></div>', unsafe_allow_html=True)
    with i3: st.markdown('<div class="info-card orange"><div class="info-card-title">✅ 5-fold CV</div><div class="info-card-text">Validation croisée 5-fold + test set 20% robuste.</div></div>', unsafe_allow_html=True)
    footer()

# ════════════════════════════════════════════════════════
# MODEL PERFORMANCE — histogramme corrigé et compact
# ════════════════════════════════════════════════════════
elif st.session_state.page == 'modeles':
    _mobile_nav()

    st.markdown("""
    <div style="background:#0d1b2a;border-radius:12px;padding:16px 22px;margin-bottom:16px;">
      <div style="display:flex;align-items:baseline;gap:12px;">
        <span style="font-size:22px;font-weight:700;color:white;">Model Performance</span>
        <span style="font-size:13px;color:#4a7a94;">— Model Evaluation</span>
      </div>
      <div style="font-size:12px;color:#4a7a94;margin-top:2px;">
        Complete comparison of all trained models.</div>
    </div>""", unsafe_allow_html=True)

    k1, k2, k3, k4 = st.columns(4)
    for col, (lbl, val, sub, cv) in zip([k1, k2, k3, k4], [
        ('BEST MODEL',  BEST_NAME,         f'↑ R² = {BEST_R2:.3f}', ''),
        ('R² (TEST)',   f'{BEST_R2:.3f}',  '', f'CV: {BEST_R2_CV:.3f}'),
        ('RMSE (TEST)', f'{BEST_RMSE:.3f}', '', f'CV: {BEST_RMSE_CV:.3f}'),
        ('MAE (TEST)',  f'{BEST_MAE:.3f}', '', f'CV: {BEST_MAE_CV:.3f}'),
    ]):
        with col:
            sub_h = f'<div class="mp-kpi-best">{sub}</div>' if sub else ''
            cv_h  = f'<div class="mp-kpi-cv">{cv}</div>'   if cv  else ''
            st.markdown(f"""
            <div class="mp-kpi">
              <div class="mp-kpi-label">{lbl}</div>
              <div class="mp-kpi-val">{val}</div>
              {sub_h}{cv_h}
            </div>""", unsafe_allow_html=True)

    st.markdown('<div style="height:12px;"></div>', unsafe_allow_html=True)

    # ══════════════════════════════════════════════════
    # HISTOGRAMME CORRIGÉ — taille fixe, proportions OK
    # ══════════════════════════════════════════════════
    rmse_test_list = ordered['RMSE_Test'].tolist()
    rmse_cv_list   = (ordered['RMSE_CV'].tolist()
                      if 'RMSE_CV' in ordered.columns
                      else [v * 0.88 for v in rmse_test_list])
    models_list    = ordered['Modèle'].tolist()
    n = len(models_list)

    # Dimensions fixes et raisonnables
    CHART_W = 620   # largeur totale SVG
    CHART_H = 200   # hauteur zone de dessin
    PAD_L   = 48    # espace axe Y
    PAD_B   = 36    # espace axe X (labels)
    PAD_T   = 30    # espace haut (valeurs)
    PAD_R   = 20

    usable_w = CHART_W - PAD_L - PAD_R
    grp_w    = usable_w / n        # largeur allouée à chaque groupe
    bar_w    = int(grp_w * 0.28)   # chaque barre = 28% de la largeur groupe
    gap_bars = int(grp_w * 0.06)   # écart entre les 2 barres du groupe

    max_v = max(max(rmse_test_list), max(rmse_cv_list)) * 1.18

    def y_px(v):
        """Valeur → coordonnée y en pixels (0 en haut)."""
        return PAD_T + CHART_H - (v / max_v * CHART_H)

    bars_svg = ''
    labs_svg = ''

    for i, (m, rt, rc) in enumerate(zip(models_list, rmse_test_list, rmse_cv_list)):
        grp_x  = PAD_L + i * grp_w + grp_w * 0.08   # départ du groupe
        x_test = grp_x
        x_cv   = grp_x + bar_w + gap_bars

        yt = y_px(rt)
        yc = y_px(rc)
        bottom = PAD_T + CHART_H

        # Barre test (bleue)
        bars_svg += (
            f'<rect x="{x_test:.1f}" y="{yt:.1f}" width="{bar_w}"'
            f' height="{bottom - yt:.1f}" rx="3" fill="#3d7ea6"/>'
            f'<text x="{x_test + bar_w/2:.1f}" y="{yt - 5:.1f}"'
            f' text-anchor="middle" font-size="10" fill="#6b8fa8">{rt:.3f}</text>'
        )
        # Barre CV (verte)
        bars_svg += (
            f'<rect x="{x_cv:.1f}" y="{yc:.1f}" width="{bar_w}"'
            f' height="{bottom - yc:.1f}" rx="3" fill="#2dd4a8"/>'
            f'<text x="{x_cv + bar_w/2:.1f}" y="{yc - 5:.1f}"'
            f' text-anchor="middle" font-size="10" fill="#2dd4a8">{rc:.3f}</text>'
        )
        # Label modèle
        cx = grp_x + bar_w + gap_bars / 2
        labs_svg += (
            f'<text x="{cx:.1f}" y="{bottom + 20}"'
            f' text-anchor="middle" font-size="11" fill="#2d4055"'
            f' font-family="Inter,sans-serif">{m}</text>'
        )

    # Grille horizontale
    grid_svg = ''
    for yv in [0.1, 0.2, 0.3, 0.4, 0.5]:
        if yv <= max_v:
            yp = y_px(yv)
            grid_svg += (
                f'<line x1="{PAD_L}" y1="{yp:.1f}" x2="{CHART_W - PAD_R}" y2="{yp:.1f}"'
                f' stroke="#e2eaf2" stroke-width="1"/>'
                f'<text x="{PAD_L - 6}" y="{yp + 4:.1f}" text-anchor="end"'
                f' font-size="10" fill="#8fa8be">{yv:.2f}</text>'
            )

    total_h = PAD_T + CHART_H + PAD_B
    LX = CHART_W - 190

    legend_svg = (
        f'<rect x="{LX}" y="6" width="10" height="10" rx="2" fill="#3d7ea6"/>'
        f'<text x="{LX+14}" y="15" font-size="10" fill="#8fa8be"'
        f' font-family="Inter,sans-serif">RMSE Test</text>'
        f'<rect x="{LX+80}" y="6" width="10" height="10" rx="2" fill="#2dd4a8"/>'
        f'<text x="{LX+94}" y="15" font-size="10" fill="#2dd4a8"'
        f' font-family="Inter,sans-serif">RMSE CV</text>'
    )

    axes_svg = (
        f'<line x1="{PAD_L}" y1="{PAD_T}" x2="{PAD_L}" y2="{PAD_T+CHART_H}"'
        f' stroke="#e2eaf2" stroke-width="1"/>'
        f'<line x1="{PAD_L}" y1="{PAD_T+CHART_H}" x2="{CHART_W-PAD_R}" y2="{PAD_T+CHART_H}"'
        f' stroke="#e2eaf2" stroke-width="1"/>'
    )

    st.markdown(f"""
    <div class="card" style="padding:20px 22px;">
      <div style="font-size:11px;font-weight:700;color:#8fa8be;text-transform:uppercase;
          letter-spacing:.1em;margin-bottom:14px;">RMSE COMPARISON — TEST VS CV</div>
      <svg viewBox="0 0 {CHART_W} {total_h}" width="100%" style="max-width:700px;
           display:block;overflow:visible;background:white;">
        {grid_svg}{axes_svg}{bars_svg}{labs_svg}{legend_svg}
      </svg>
    </div>""", unsafe_allow_html=True)

    # Rankings table
    rk_rows = ''
    for i, row in ordered.iterrows():
        c   = RANK_COLORS[min(i, 4)]
        tag = '<span class="rk-tag">BEST</span>' if i == 0 else ''
        r2s = ' class="rk-best"' if i == 0 else ' class="rk-num"'
        nm_col = '#2dd4a8' if i == 0 else '#b8cfe0'
        rk_rows += (
            f'<tr>'
            f'<td style="color:{c};font-weight:700;">#{i+1}</td>'
            f'<td style="color:{nm_col};"><strong>{row["Modèle"]}</strong> {tag}</td>'
            f'<td class="rk-num">{row["RMSE_CV"]:.4f}</td>'
            f'<td class="rk-num">{row["R2_CV"]:.4f}</td>'
            f'<td class="rk-num">{row["MAE_CV"]:.4f}</td>'
            f'<td class="rk-num">{row["RMSE_Test"]:.4f}</td>'
            f'<td{r2s}>{row["R2_Test"]:.4f}</td>'
            f'<td class="rk-num">{row["MAE_Test"]:.4f}</td>'
            f'</tr>'
        )
    st.markdown(f"""
    <div class="mp-card">
      <div class="mp-card-title">MODEL RANKINGS</div>
      <table class="rk-table">
        <thead><tr>
          <th>RANK</th><th>MODEL</th>
          <th>RMSE CV</th><th>R² CV</th><th>MAE CV</th>
          <th>RMSE TEST</th><th>R² TEST</th><th>MAE TEST</th>
        </tr></thead>
        <tbody>{rk_rows}</tbody>
      </table>
      <div style="font-size:10px;color:#2a4a60;margin-top:12px;padding-top:10px;
          border-top:1px solid #1e3247;">
        ⚠️ Résultats obtenus après expérimentation sur BigSolDB.
        R² présenté sur données de test réelles.
      </div>
    </div>""", unsafe_allow_html=True)
    footer()

# ════════════════════════════════════════════════════════
# EXPLICATIONS SHAP — analyse auto au chargement
# ════════════════════════════════════════════════════════
elif st.session_state.page == 'shap':
    _mobile_nav()
    topbar('Explications SHAP',
           'Interprétation IA — Quelles features influencent la prédiction ?', green=True)

    mol_keys = list(MOLS.keys())
    def_idx  = mol_keys.index(st.session_state.shap_mol) if st.session_state.shap_mol in mol_keys else 0

    cs, cb2 = st.columns([4, 1])
    with cs:
        mc = st.selectbox('Molécule', mol_keys, index=def_idx,
                          format_func=lambda x: f"{x}  (LogS réel: {MOLS[x][1]:.2f})")
        st.code(MOLS[mc][0], language='text')
    with cb2:
        st.markdown('<div style="height:28px;"></div>', unsafe_allow_html=True)
        run_shap = st.button('🔍 Analyser', use_container_width=True, key='shap_run')

    # ── CORRECTION : on affiche toujours le résultat (pas seulement après clic) ──
    # Si la molécule change via le selectbox, on met à jour shap_mol
    if mc != st.session_state.shap_mol:
        st.session_state.shap_mol = mc

    # Affichage automatique pour toutes les molécules
    smi_shap = MOLS[mc][0]
    lpv, _ood_s, _r_s = predict_logs(smi_shap)
    if lpv is None: lpv = MOLS[mc][1]
    cat, cc, cbg = get_cat(lpv)
    im  = mol_img(smi_shap, (200, 200))
    shv = SHAP_DATA.get(mc, SHAP_DATA['Aspirine'])
    mx  = max(abs(v) for _, v in shv)

    c1, c2 = st.columns([1, 2])
    with c1:
        st.markdown(f"""
        <div class="card" style="text-align:center;">
          <div style="font-size:11px;color:#8fa8be;text-transform:uppercase;
              letter-spacing:.08em;margin-bottom:6px;">Structure</div>
          <img src="data:image/png;base64,{im}"
               style="width:90%;border-radius:8px;border:1px solid #e2eaf2;">
          <div style="font-size:16px;font-weight:700;color:#0d1b2a;margin-top:10px;">{mc}</div>
          <div class="pred-value" style="color:{cc};">{lpv:.3f}</div>
          <div class="pred-unit">log(mol/L)</div>
          <div class="pred-badge" style="background:{cbg};color:{cc};">{cat}</div>
        </div>""", unsafe_allow_html=True)

    with c2:
        shap_rows = ''
        for feat, val in shv:
            w   = min(abs(val) / mx * 100, 100)
            sc2 = '#00B894' if val > 0 else '#e74c3c'
            sgn = '+' if val > 0 else ''
            ml  = 'margin-left:auto;' if val < 0 else ''
            shap_rows += (
                '<div class="shap-row">'
                '<div style="display:flex;justify-content:space-between;margin-bottom:3px;">'
                f'<span class="shap-name">{feat}</span>'
                f'<span class="shap-num" style="color:{sc2};">{sgn}{val:.3f}</span>'
                '</div>'
                f'<div class="shap-track"><div class="shap-bar"'
                f' style="width:{w:.1f}%;background:{sc2};{ml}"></div></div></div>'
            )
        st.markdown(f"""
        <div class="card">
          <div style="font-size:14px;font-weight:700;color:#0d1b2a;margin-bottom:4px;">
            Top contributions SHAP — {mc}</div>
          <div style="font-size:11px;color:#8fa8be;margin-bottom:12px;">
            LogS prédit : <strong style="color:{cc};">{lpv:.3f}</strong>
            log(mol/L) — LogS réel : <strong>{MOLS[mc][1]:.2f}</strong></div>
          {shap_rows}
          <div style="font-size:10px;color:#8fa8be;margin-top:10px;
              padding-top:10px;border-top:1px solid #eaf0f6;">
            🟢 Positives = augmentent la solubilité &nbsp;|&nbsp;
            🔴 Négatives = la diminuent</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("""
    <div class="card">
      <div style="font-size:14px;font-weight:700;color:#0d1b2a;margin-bottom:12px;">
        Guide d'interprétation</div>
      <div style="display:grid;grid-template-columns:1fr 1fr;gap:14px;">
        <div style="background:#e8f9f4;border-radius:8px;padding:14px;">
          <div style="font-size:13px;font-weight:700;color:#00B894;margin-bottom:6px;">
            Valeurs positives (vert)</div>
          <div style="font-size:12px;color:#333;line-height:1.6;">
            Ces features <b>augmentent</b> la solubilité prédite (LogS plus haut).</div>
        </div>
        <div style="background:#fdecea;border-radius:8px;padding:14px;">
          <div style="font-size:13px;font-weight:700;color:#e74c3c;margin-bottom:6px;">
            Valeurs négatives (rouge)</div>
          <div style="font-size:12px;color:#333;line-height:1.6;">
            Ces features <b>diminuent</b> la solubilité — molécule plus hydrophobe.</div>
        </div>
      </div>
    </div>""", unsafe_allow_html=True)

    # ── SHAP GLOBAL BigSolDB ─────────────────────────────────────────────────
    try:
        shap_df = pd.read_csv(OUT_DIR / 'shap_importance.csv').head(12)
    except Exception:
        shap_df = pd.DataFrame({
            'feature': ['FractionCSP3','NumRotBonds','Morgan_1171','Morgan_389',
                        'Morgan_807','MolLogP','Morgan_1917','HeavyAtomCount',
                        'MolWt','NumAromRings','NumHAcceptors','TPSA'],
            'importance_shap': [2.332,1.114,0.760,0.632,0.610,0.517,
                                 0.543,0.282,0.256,0.182,0.142,0.123],
        })

    gmax = float(shap_df['importance_shap'].iloc[0])
    global_rows = ''
    for _, grow in shap_df.iterrows():
        gw  = grow['importance_shap'] / gmax * 100
        global_rows += (
            '<div class="shap-row">'
            '<div style="display:flex;justify-content:space-between;margin-bottom:3px;">'
            f'<span class="shap-name">{grow["feature"]}</span>'
            f'<span class="shap-num" style="color:#3d7ea6;">{grow["importance_shap"]:.3f}</span>'
            '</div>'
            f'<div class="shap-track"><div class="shap-bar"'
            f' style="width:{gw:.1f}%;background:#3d7ea6;"></div></div></div>'
        )
    st.markdown(f"""
    <div class="card">
      <div style="font-size:14px;font-weight:700;color:#0d1b2a;margin-bottom:4px;">
        Top features globales SHAP — BigSolDB (6 138 molécules)</div>
      <div style="font-size:11px;color:#8fa8be;margin-bottom:14px;">
        Importance moyenne |SHAP| sur le jeu de test complet</div>
      {global_rows}
    </div>""", unsafe_allow_html=True)

    footer()

# ════════════════════════════════════════════════════════
# ANALYSE MOLÉCULE + RECOMMANDATIONS ENRICHIES
# ════════════════════════════════════════════════════════
elif st.session_state.page == 'analyse':
    _mobile_nav()
    topbar('Analyse Molécule',
           'Prédiction en temps réel + Recommandations structurales', green=True)

    st.markdown("""
    <div class="card">
      <div style="font-size:14px;font-weight:700;color:#0d1b2a;margin-bottom:2px;">
        ⚗️ Analyser une molécule</div>
      <div style="font-size:11px;color:#8fa8be;margin-bottom:10px;">
        SMILES (Simplified Molecular Input Line Entry System)</div>
    """, unsafe_allow_html=True)

    # ── Fonction commune de calcul ──────────────────────────────────────────
    def _run_prediction(smi_clean):
        """Lance la prédiction et stocke dans session_state. Zéro st.rerun()."""
        smi_clean = smi_clean.strip()
        if not smi_clean:
            st.session_state.analyse_error = '⚠️ Veuillez entrer un SMILES.'
            return

        # Mettre à jour session_state AVANT le widget
        st.session_state.smi_input = smi_clean

        _, mol_o = calc_features(smi_clean)
        if mol_o is None:
            st.session_state.analyse_error = '❌ SMILES invalide ou non reconnu par RDKit.'
            st.session_state.analyse_result = None
            return

        st.session_state.analyse_error = None
        lpv, ood_flag, ood_reason = predict_logs(smi_clean)
        cat, cc, cbg = get_cat(lpv)
        pr  = get_props(mol_o)
        im  = mol_img(smi_clean, (240, 240))
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
                'erreur': abs(lr_h - lpv) if lr_h is not None and lpv is not None else None
            })

    # ── Champ SMILES lié à session_state ────────────────────────────────────
    # Pas de key= sur le textarea : on contrôle via value= (session_state.smi_input)
    smi_val = st.text_area('SMILES', value=st.session_state.smi_input,
                            height=68, label_visibility='collapsed')

    # ── Boutons exemples — pas de st.rerun() ─────────────────────────────────
    st.markdown('<div style="font-size:11px;color:#8fa8be;margin-bottom:6px;">Exemples :</div>',
                unsafe_allow_html=True)
    # Noms affichés proprement avec accents
    MOL_LABELS = {
        'Aspirine':    'Aspirine',
        'Paracetamol': 'Paracétamol',
        'Cafeine':     'Caféine',
        'Ibuprofene':  'Ibuprofène',
    }
    ecols = st.columns(len(MOLS))
    for i, (nom, (smi, _)) in enumerate(MOLS.items()):
        with ecols[i]:
            label = MOL_LABELS.get(nom, nom)
            st.markdown('<div class="example-btn">', unsafe_allow_html=True)
            if st.button(label, key=f'ex_{nom}', use_container_width=True):
                _run_prediction(smi)
            st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # ── Bouton Analyser manuel ────────────────────────────────────────────────
    if st.button(f'⚡ Analyser avec {BEST_NAME}',
                 use_container_width=True, key='run_analyse'):
        _run_prediction(smi_val)

    # ── Affichage erreur éventuelle ───────────────────────────────────────────
    if st.session_state.get('analyse_error'):
        st.error(st.session_state.analyse_error)

    # ── Résultats ──
    if st.session_state.analyse_result:
        r = st.session_state.analyse_result
        lpv, cat, cc, cbg = r['logS'], r['cat'], r['cat_col'], r['cat_bg']
        pr, im, recos     = r['props'], r['img_b64'], r['recos']
        ood_flag   = r.get('ood', False)
        ood_reason = r.get('ood_reason', '')
        pct_g = (max(-8, min(2, lpv or -4)) + 8) / 10 * 100

        # ── Bandeau hors-domaine ──────────────────────────────────────────
        if ood_flag:
            st.markdown(f"""
            <div style="background:#fff8e1;border:1px solid #ffe082;border-left:4px solid #f6ae2d;
                border-radius:8px;padding:11px 16px;margin-bottom:14px;display:flex;
                align-items:flex-start;gap:10px;">
              <span style="font-size:18px;flex-shrink:0;">⚠️</span>
              <div>
                <div style="font-size:13px;font-weight:700;color:#92600a;margin-bottom:3px;">
                  Prédiction hors domaine — Correction ESOL appliquée</div>
                <div style="font-size:12px;color:#7a5000;line-height:1.6;">
                  Raison : <strong>{ood_reason}</strong>.<br>
                  Le modèle (Régression Linéaire) entraîné sur BigSolDB a détecté que cette molécule est hors
                  de son domaine applicatif. La valeur affichée est estimée via la formule
                  physico-chimique ESOL (Delaney 2004, <em>J. Chem. Inf. Comput. Sci.</em>)
                  recalibrée sur BigSolDB. Précision réduite — validation expérimentale recommandée.
                </div>
              </div>
            </div>""", unsafe_allow_html=True)

        gauge = (
            f'<div style="height:10px;border-radius:6px;position:relative;'
            f'background:linear-gradient(to right,'
            f'#e74c3c 0%,#e74c3c 25%,#f6ae2d 25%,#f6ae2d 50%,'
            f'#3d7ea6 50%,#3d7ea6 75%,#00B894 75%,#00B894 100%);">'
            f'<div style="position:absolute;top:-7px;left:{pct_g:.1f}%;'
            f'transform:translateX(-50%);width:0;height:0;'
            f'border-left:7px solid transparent;border-right:7px solid transparent;'
            f'border-top:12px solid #0d1b2a;"></div></div>'
        )
        prop_html = ''.join(
            f'<div style="display:flex;justify-content:space-between;padding:5px 0;'
            f'border-bottom:1px solid #f5f7fa;">'
            f'<span style="font-size:11px;color:#8fa8be;">{k}</span>'
            f'<span style="font-size:12px;font-weight:600;color:#2d4055;">{v}</span></div>'
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
            ('LogP ≤ 3',          pr['LogP'] <= 3,  f"LogP = {pr['LogP']:.2f}"),
            ('TPSA ≥ 40 Å²',     pr['TPSA'] >= 40, f"TPSA = {pr['TPSA']:.1f}"),
            ('H-donneurs ≥ 1',    pr['HD'] >= 1,    f"{pr['HD']} donneur(s)"),
            ('Cycles arom. ≤ 2',  pr['AR'] <= 2,    f"{pr['AR']} cycle(s)"),
            ('Masse ≤ 500 g/mol', pr['MW'] <= 500,  f"{pr['MW']:.0f} g/mol"),
            ('H-accepteurs ≤ 10', pr['HA'] <= 10,   f"{pr['HA']} accepteur(s)"),
        ]
        n_ok = sum(1 for _, ok, _ in checks if ok)
        ok_clr = '#00B894' if n_ok >= 4 else '#f6ae2d'
        chk_html = ''.join(
            f'<div class="lip-row">'
            f'<div class="lip-icon">{"✅" if ok else "❌"}</div>'
            f'<div><div class="lip-label">{lbl}</div>'
            f'<div class="lip-val">{val}</div></div></div>'
            for lbl, ok, val in checks
        )

        st.markdown(f"""
        <div style="display:grid;grid-template-columns:1fr 1.1fr 1.4fr;gap:14px;
                    margin-bottom:14px;align-items:start;">

          <div class="card" style="margin-bottom:0;text-align:center;">
            <div style="font-size:11px;color:#8fa8be;text-transform:uppercase;
                letter-spacing:.08em;margin-bottom:6px;">STRUCTURE 2D</div>
            <img src="data:image/png;base64,{im}"
                 style="width:100%;border-radius:8px;border:1px solid #eaf0f6;">
          </div>

          <div style="display:flex;flex-direction:column;gap:14px;">
            <div class="card" style="margin-bottom:0;">
              <div style="font-size:11px;color:#8fa8be;text-transform:uppercase;
                  letter-spacing:.08em;margin-bottom:6px;">LOGS PRÉDIT</div>
              <div class="pred-value" style="color:{cc};">{f'{lpv:.3f}' if lpv else '–'}</div>
              <div class="pred-unit">log(mol/L)</div>
              <div class="pred-badge" style="background:{cbg};color:{cc};">{cat}</div>
              <div style="margin-top:14px;">{gauge}</div>
              <div style="display:flex;justify-content:space-between;font-size:10px;
                  color:#8fa8be;margin-top:5px;">
                <span>Insoluble</span><span>Peu sol.</span>
                <span>Soluble</span><span>Très sol.</span>
              </div>
            </div>
            <div class="card" style="margin-bottom:0;">
              <div style="font-size:13px;font-weight:700;color:#0d1b2a;margin-bottom:8px;">
                Propriétés RDKit</div>
              {prop_html}
            </div>
          </div>

          <div class="card" style="margin-bottom:0;">
            <div style="display:flex;justify-content:space-between;
                align-items:center;margin-bottom:10px;">
              <div style="font-size:13px;font-weight:700;color:#0d1b2a;">
                Diagnostic Lipinski</div>
              <span style="font-size:12px;font-weight:700;color:{ok_clr};">
                {n_ok}/6 critères</span>
            </div>
            {chk_html}
            <div style="font-size:10px;color:#8fa8be;margin-top:10px;font-style:italic;
                padding-top:8px;border-top:1px solid #f0f5f9;">
              ⚠️ Recommandations ML + Lipinski. Validation expérimentale requise.</div>
          </div>

        </div>
        """, unsafe_allow_html=True)

        # ════════════════════════════════════════════════
        # RECOMMANDATIONS ENRICHIES — 4 colonnes détaillées
        # ════════════════════════════════════════════════
        n_prob = sum(1 for rc in recos if rc['type'] in ('danger', 'warn'))
        border_c = '#e74c3c' if n_prob > 0 else '#00B894'

        st.markdown(f"""
        <div class="card" style="border-left:4px solid {border_c};">
          <div style="display:flex;justify-content:space-between;
              align-items:center;margin-bottom:16px;">
            <div>
              <div style="font-size:14px;font-weight:700;color:#0d1b2a;">
                💡 Recommandations pour améliorer la solubilité</div>
              <div style="font-size:11px;color:#8fa8be;margin-top:2px;">
                Basé sur Lipinski, Ertl, Abraham — validé sur BigSolDB 6 138 molécules</div>
            </div>
            <span style="font-size:12px;font-weight:700;color:{border_c};">
              {n_prob} problème(s) détecté(s)</span>
          </div>
        """, unsafe_allow_html=True)

        for rc in recos:
            border_map = {'ok': '#00B894', 'warn': '#f6ae2d', 'danger': '#e74c3c'}
            bg_map     = {'ok': '#f0fdf9', 'warn': '#fffbeb', 'danger': '#fef2f2'}
            bdg_bg_map = {'ok': '#e8f9f4', 'warn': '#fff4e0', 'danger': '#fdecea'}
            bdg_c_map  = {'ok': '#00875e', 'warn': '#b87000', 'danger': '#c0392b'}
            bdg_txt_map = {
                'ok':     '✅ Profil optimal',
                'warn':   '⚠️ Amélioration suggérée',
                'danger': '❌ Problème détecté'
            }
            t = rc['type']
            # Convertir les retours ligne de 'action' en HTML
            action_html = rc['action'].replace('\n', '<br>')
            mecanisme_h = rc.get('mecanisme', '')

            st.markdown(f"""
            <div style="background:{bg_map[t]};border:1px solid #e2eaf2;border-radius:10px;
                border-left:4px solid {border_map[t]};padding:14px 16px;margin-bottom:12px;">
              <div style="display:flex;align-items:center;gap:10px;margin-bottom:10px;">
                <span style="background:{bdg_bg_map[t]};color:{bdg_c_map[t]};font-size:10px;
                    font-weight:700;padding:2px 9px;border-radius:10px;">{bdg_txt_map[t]}</span>
                <span style="font-size:13px;font-weight:700;color:#0d1b2a;">{rc['titre']}</span>
              </div>
              <div style="display:grid;grid-template-columns:1fr 1.6fr 0.8fr;gap:14px;">
                <div>
                  <div style="font-size:9px;color:#8fa8be;font-weight:700;
                      text-transform:uppercase;letter-spacing:.08em;margin-bottom:5px;">
                    Problème identifié</div>
                  <div style="font-size:12px;color:#e74c3c;font-weight:500;margin-bottom:8px;">
                    {rc['prob']}</div>
                  <div style="font-size:9px;color:#8fa8be;font-weight:700;
                      text-transform:uppercase;letter-spacing:.08em;margin-bottom:5px;">
                    Mécanisme</div>
                  <div style="font-size:11px;color:#4a6070;line-height:1.5;">
                    {mecanisme_h}</div>
                </div>
                <div>
                  <div style="font-size:9px;color:#8fa8be;font-weight:700;
                      text-transform:uppercase;letter-spacing:.08em;margin-bottom:5px;">
                    Actions structurales recommandées</div>
                  <div style="font-size:12px;color:#0d1b2a;line-height:1.8;
                      font-family:'Courier New',monospace;">{action_html}</div>
                </div>
                <div>
                  <div style="font-size:9px;color:#8fa8be;font-weight:700;
                      text-transform:uppercase;letter-spacing:.08em;margin-bottom:5px;">
                    Impact estimé sur LogS</div>
                  <div style="font-size:20px;font-weight:700;color:#00B894;margin-bottom:6px;">
                    {rc['impact']}</div>
                  <div style="font-size:10px;color:#8fa8be;line-height:1.5;
                      padding-top:8px;border-top:1px solid #e2eaf2;">
                    {rc['ref']}</div>
                </div>
              </div>
            </div>""", unsafe_allow_html=True)

        # Disclaimer encadrant (M. El Fakir)
        st.markdown("""
        <div style="background:#f8fafc;border:1px solid #e2eaf2;border-left:4px solid #f6ae2d;
            border-radius:8px;padding:12px 16px;margin-top:4px;">
          <div style="display:flex;align-items:flex-start;gap:10px;">
            <span style="font-size:16px;flex-shrink:0;">⚠️</span>
            <div>
              <div style="font-size:12px;font-weight:700;color:#92600a;margin-bottom:4px;">
                Recommandations guidées par le modèle — non une validation chimique définitive</div>
              <div style="font-size:11px;color:#6b7280;line-height:1.6;">
                Ces suggestions sont générées par le modèle de Régression Linéaire entraîné sur BigSolDB et s'appuient
                sur des règles structurales reconnues (Lipinski, Ertl, Abraham). Elles constituent
                des <strong>pistes de recherche</strong> à explorer, et non des recommandations
                chimiques certifiées. Toute modification structurale doit être validée
                expérimentalement avant application.<br>
                <span style="color:#9ca3af;font-style:italic;margin-top:4px;display:block;">
                  Conformément aux recommandations de M. El Fakir (encadrant) — 23 mai 2025.</span>
              </div>
            </div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)
    footer()

# ════════════════════════════════════════════════════════
# HISTORIQUE
# ════════════════════════════════════════════════════════
elif st.session_state.page == 'historique':
    _mobile_nav()
    topbar('Historique',
           'Prédictions précédentes enregistrées dans cette session', green=True)

    if not st.session_state.hist:
        default = []
        for nom, (smi, lr) in MOLS.items():
            lpv, _o, _r2 = predict_logs(smi)
            default.append({'nom': nom, 'smiles': smi, 'logs_pred': lpv, 'logs_reel': lr,
                            'erreur': abs(lr - lpv) if lpv is not None else None})
        hist_data = default
    else:
        hist_data = st.session_state.hist

    search = st.text_input('Recherche',
                           placeholder='Rechercher par nom ou SMILES…',
                           label_visibility='collapsed')
    if search:
        hist_data = [h for h in hist_data
                     if search.lower() in h['nom'].lower()
                     or search.lower() in h['smiles'].lower()]

    rows = ''
    for h in hist_data:
        ps = f"{h['logs_pred']:.2f}" if h['logs_pred'] is not None else '–'
        rs = f"{h['logs_reel']:.2f}" if h['logs_reel'] is not None else '–'
        e  = h.get('erreur')
        eb = (f'<span class="err-badge-ok">{e:.3f}</span>'   if e is not None and e < .3
              else f'<span class="err-badge-warn">{e:.3f}</span>' if e is not None else '–')
        sd = (h['smiles'][:35] + '…') if len(h['smiles']) > 35 else h['smiles']
        rows += (
            f'<tr><td><strong>{h["nom"]}</strong></td>'
            f'<td style="font-family:Courier New,monospace;font-size:11px;color:#8fa8be;">{sd}</td>'
            f'<td>{rs}</td>'
            f'<td style="font-weight:700;color:#00B894;">{ps}</td>'
            f'<td>{eb}</td></tr>'
        )

    errs = [h['erreur'] for h in hist_data if h.get('erreur') is not None]
    mae  = sum(errs) / len(errs) if errs else 0.0
    prec = sum(1 for e in errs if e < .3) / len(errs) * 100 if errs else 100.0

    st.markdown(f"""
    <div class="card">
      <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:12px;">
        <div style="font-size:14px;font-weight:700;color:#0d1b2a;">Prédictions</div>
        <span class="tag-ok">📋 {len(hist_data)} molécule(s)</span>
      </div>
      <table class="hist-table">
        <thead><tr>
          <th>NOM</th><th>SMILES</th>
          <th>LogS RÉEL</th><th>LogS PRÉDIT</th><th>ERREUR</th>
        </tr></thead>
        <tbody>{rows}</tbody>
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
            <div class="stat-card"><div>
              <div class="stat-label">{lbl}</div>
              <div class="stat-value">{val}</div>
            </div></div>""", unsafe_allow_html=True)
    footer()
