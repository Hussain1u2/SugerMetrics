import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import joblib
import warnings
from scipy.stats import gaussian_kde

warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SugarMetrics",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────────────────────────────────────
# CSS  –  clean Power-BI dark theme
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

html, body, .stApp {
    background: #111 !important;
    font-family: 'Inter', 'Segoe UI', sans-serif !important;
    color: #ddd !important;
}
.main .block-container { padding: 0 1.6rem 2rem !important; max-width: 100% !important; }

[data-testid="stSidebar"] {
    background: #181818 !important;
    border-right: 1px solid #242424 !important;
    min-width: 220px !important; max-width: 220px !important;
}
[data-testid="stSidebar"] > div:first-child { padding: 0 !important; }
[data-testid="stSidebarNav"] { display: none !important; }

.topbar {
    background: #181818; border-bottom: 1px solid #242424;
    padding: 0.75rem 1.6rem; margin: 0 -1.6rem 1.5rem;
    display: flex; align-items: center; gap: 0.75rem;
}
.topbar-accent { width: 3px; height: 1.1rem; border-radius: 2px; background: #0078d4; flex-shrink: 0; }
.topbar-title  { font-size: 0.95rem; font-weight: 600; color: #fff; }
.topbar-sub    { font-size: 0.72rem; color: #555; margin-left: auto; }

.nav-brand { padding: 1.2rem 1rem 0.9rem; border-bottom: 1px solid #242424; }
.nav-brand-icon  { font-size: 1.5rem; }
.nav-brand-name  { font-size: 0.92rem; font-weight: 700; color: #fff; margin-top: 0.25rem; }
.nav-brand-sub   { font-size: 0.68rem; color: #444; }
.nav-group-label {
    font-size: 0.6rem; font-weight: 600; color: #3a3a3a;
    text-transform: uppercase; letter-spacing: 1.5px;
    padding: 0.9rem 1rem 0.25rem;
}

[data-testid="stSidebar"] .stRadio { padding: 0 !important; }
[data-testid="stSidebar"] .stRadio > label { display: none !important; }
[data-testid="stSidebar"] .stRadio > div  { gap: 2px !important; padding: 0 6px !important; }
[data-testid="stSidebar"] .stRadio div[role="radio"] {
    background: transparent !important; border: none !important;
    border-radius: 5px !important; padding: 0.5rem 0.9rem !important;
    font-size: 0.82rem !important; color: #888 !important; transition: all .12s;
}
[data-testid="stSidebar"] .stRadio div[role="radio"]:hover  { background: #242424 !important; color: #ddd !important; }
[data-testid="stSidebar"] .stRadio div[role="radio"][aria-checked="true"] {
    background: #0078d4 !important; color: #fff !important; font-weight: 600 !important;
}
[data-testid="stSidebar"] .stRadio div[role="radio"] p { margin: 0 !important; color: inherit !important; }

.kpi {
    background: #181818; border: 1px solid #242424; border-radius: 7px;
    padding: 0.9rem 1rem 0.85rem; position: relative; overflow: hidden;
}
.kpi-bar { position: absolute; top: 0; left: 0; width: 3px; height: 100%; background: var(--c, #0078d4); }
.kpi-lbl  { font-size: 0.65rem; color: #555; text-transform: uppercase; letter-spacing: .9px; margin-bottom: 0.3rem; }
.kpi-val  { font-size: 1.55rem; font-weight: 700; color: var(--c, #0078d4); line-height: 1.1; }
.kpi-sub  { font-size: 0.67rem; color: #444; margin-top: 0.18rem; }
.kpi-ico  { position: absolute; right: 0.9rem; top: 50%; transform: translateY(-50%); font-size: 1.6rem; opacity: 0.1; }

.cc { background: #181818; border: 1px solid #242424; border-radius: 7px; padding: 1rem 1.1rem 0.85rem; margin-bottom: 0.7rem; }
.cc-title { font-size: 0.68rem; font-weight: 600; color: #666; text-transform: uppercase; letter-spacing: .9px; margin-bottom: 0.75rem; }

.sec {
    font-size: 0.65rem; font-weight: 600; color: #444;
    text-transform: uppercase; letter-spacing: 1.2px;
    border-bottom: 1px solid #1e1e1e;
    padding-bottom: 0.4rem; margin: 1.1rem 0 0.8rem;
}

.res-card {
    background: #181818; border-radius: 7px;
    padding: 1.4rem 1.2rem; text-align: center;
    border: 1px solid var(--bc, #333); border-top: 3px solid var(--bc, #333);
}
.res-label  { font-size: 0.78rem; font-weight: 600; color: var(--bc); margin-bottom: 0.3rem; }
.res-pct    { font-size: 2.8rem; font-weight: 800; color: var(--bc); line-height: 1; }
.res-sub    { font-size: 0.68rem; color: #555; margin-top: 0.3rem; }
.res-badge  {
    display: inline-block; padding: 0.22rem 0.9rem; border-radius: 20px;
    font-size: 0.72rem; font-weight: 600; margin-top: 0.6rem;
    background: rgba(255,255,255,0.05); border: 1px solid var(--bc); color: var(--bc);
}

.stNumberInput label, .stTextInput label, .stSelectbox label { color: #777 !important; font-size: 0.73rem !important; }
.stNumberInput > div > div > input,
.stTextInput   > div > div > input {
    background: #1e1e1e !important; border: 1px solid #2e2e2e !important;
    border-radius: 5px !important; color: #ddd !important; font-size: 0.84rem !important;
}
.stNumberInput > div > div > input:focus { border-color: #0078d4 !important; }
.stSelectbox > div > div { background: #1e1e1e !important; border: 1px solid #2e2e2e !important; border-radius: 5px !important; }

.stButton > button {
    background: #0078d4 !important; color: #fff !important; border: none !important;
    border-radius: 5px !important; font-weight: 600 !important; font-size: 0.83rem !important;
    padding: 0.5rem 1.1rem !important; transition: background .15s !important;
}
.stButton > button:hover { background: #106ebe !important; }
.stDownloadButton > button {
    background: #107c41 !important; color: #fff !important; border: none !important;
    border-radius: 5px !important; font-weight: 600 !important;
}

[data-testid="metric-container"] {
    background: #181818 !important; border: 1px solid #242424 !important;
    border-radius: 6px !important; padding: 0.7rem !important;
}
[data-testid="metric-container"] label { color: #666 !important; font-size: 0.7rem !important; }
.stDataFrame  { border: 1px solid #242424 !important; border-radius: 7px !important; }
.streamlit-expanderHeader {
    background: #181818 !important; border: 1px solid #242424 !important;
    border-radius: 6px !important; color: #aaa !important;
}
[data-testid="stFileUploader"] {
    background: #181818 !important; border: 2px dashed #242424 !important; border-radius: 7px !important;
}
::-webkit-scrollbar { width: 4px; height: 4px; }
::-webkit-scrollbar-track { background: #111; }
::-webkit-scrollbar-thumb { background: #2e2e2e; border-radius: 2px; }
hr { border-color: #1e1e1e !important; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
FEAT_ALL = ['Pregnancies','Glucose','BloodPressure','SkinThickness',
            'Insulin','BMI','DiabetesPedigreeFunction','Age']

ZERO_IMPUTE_COLS = ['Glucose','BloodPressure','SkinThickness','Insulin','BMI']

MEDIANS = {
    'Pregnancies': 3, 'Glucose': 117.0, 'BloodPressure': 72.0,
    'SkinThickness': 23.0, 'Insulin': 30.5, 'BMI': 32.0,
    'DiabetesPedigreeFunction': 0.372, 'Age': 29,
}

BG   = '#181818'
LN   = '#242424'
TXT  = '#cccccc'
BLUE = '#0078d4'
RED  = '#c0392b'
GRN  = '#27ae60'
ORG  = '#d4a017'
PUR  = '#8e44ad'

# ─────────────────────────────────────────────────────────────────────────────
# SAMPLE CSV  (embedded — used in Bulk Scanner download button)
# ─────────────────────────────────────────────────────────────────────────────
SAMPLE_CSV_DATA = """\
PatientID,Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age
P001,6,148,72,35,0,33.6,0.627,50
P002,1,85,66,29,0,26.6,0.351,31
P003,8,183,64,0,0,23.3,0.672,32
P004,1,89,66,23,94,28.1,0.167,21
P005,0,137,40,35,168,43.1,2.288,33
P006,5,116,74,0,0,25.6,0.201,30
P007,3,78,50,32,88,31.0,0.248,26
P008,10,115,0,0,0,35.3,0.134,29
P009,2,197,70,45,543,30.5,0.158,53
P010,8,125,96,0,0,0.0,0.232,54
P011,4,110,92,0,0,37.6,0.191,30
P012,10,168,74,0,0,38.0,0.537,34
P013,10,139,80,0,0,27.1,1.441,57
P014,1,189,60,23,846,30.1,0.398,59
P015,5,166,72,19,175,25.8,0.587,51
P016,7,100,0,0,0,30.0,0.484,32
P017,0,118,84,47,230,45.8,0.551,31
P018,7,107,74,0,0,29.6,0.254,31
P019,1,103,30,38,83,43.3,0.183,33
P020,1,115,70,30,96,34.6,0.529,32
"""

# ─────────────────────────────────────────────────────────────────────────────
# LOAD ARTIFACTS
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    model    = joblib.load('best_model.pkl')
    scaler   = joblib.load('scaler.pkl')
    features = joblib.load('features.pkl')
    return model, scaler, features

@st.cache_data
def load_data():
    df      = pd.read_csv('diabetes_clean.csv')
    results = pd.read_csv('model_results.csv')
    return df, results

_err = ""
try:
    model, scaler, features = load_artifacts()
    df, results_df = load_data()
    READY = True
except Exception as e:
    READY = False; _err = str(e)

# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def fig(w=8, h=4):
    f, a = plt.subplots(figsize=(w, h))
    f.patch.set_facecolor(BG); a.set_facecolor(BG)
    for s in a.spines.values(): s.set_color(LN)
    a.tick_params(colors='#666', labelsize=8)
    a.xaxis.label.set_color('#666'); a.yaxis.label.set_color('#666')
    a.title.set_color(TXT)
    a.grid(axis='y', color=LN, lw=0.5); a.set_axisbelow(True)
    return f, a

def figp(w=5.5, h=5):
    f, a = plt.subplots(figsize=(w, h), subplot_kw=dict(polar=True))
    f.patch.set_facecolor(BG); a.set_facecolor(BG)
    a.grid(color=LN, alpha=0.8); a.spines['polar'].set_color(LN)
    return f, a

def kpi_card(col, lbl, val, sub, color, icon):
    col.markdown(f"""
    <div class="kpi">
        <div class="kpi-bar" style="--c:{color};"></div>
        <div class="kpi-lbl">{lbl}</div>
        <div class="kpi-val" style="--c:{color};">{val}</div>
        <div class="kpi-sub">{sub}</div>
        <div class="kpi-ico">{icon}</div>
    </div>""", unsafe_allow_html=True)

def sec(t):
    st.markdown(f'<div class="sec">{t}</div>', unsafe_allow_html=True)

def sp(h=0.6):
    st.markdown(f"<div style='height:{h}rem'></div>", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div class="nav-brand">
        <div class="nav-brand-icon">🩺</div>
        <div class="nav-brand-name">SugarMetrics</div>
        <div class="nav-brand-sub">Diabetes Analytics</div>
    </div>
    <div class="nav-group-label">Pages</div>
    """, unsafe_allow_html=True)

    page = st.radio("_", [
        "🔬  Risk Prediction",
        "📊  Data Insights",
        "📈  Model Performance",
        "📂  Bulk Scanner",
        "ℹ️   About",
    ], label_visibility="hidden")

    st.markdown("""
    <div style="margin-top:auto; padding:1rem 1rem 0.5rem;
                font-size:0.63rem; color:#333; line-height:2;
                border-top:1px solid #1e1e1e; margin-top:1.5rem;">
        RF · GridSearchCV Tuned<br>
        StandardScaler · SelectKBest(6)<br>
        768 records · SMOTE balanced
    </div>""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# TOP BAR
# ─────────────────────────────────────────────────────────────────────────────
_meta = {
    "🔬  Risk Prediction":   ("Risk Prediction",   "StandardScaler → SelectKBest(6) → Random Forest"),
    "📊  Data Insights":     ("Data Insights",     "Pima Indians Diabetes · 768 records · 8 clinical features"),
    "📈  Model Performance": ("Model Performance", "10 ML models · GridSearchCV tuned · 10-fold CV"),
    "📂  Bulk Scanner":      ("Bulk Scanner",      "Batch predict — CSV · JSON · XLSX · with auto preprocessing"),
    "ℹ️   About":             ("About",             "Pipeline details · Preprocessing · Feature selection"),
}
pt, ps = _meta.get(page, ("SugarMetrics",""))
st.markdown(f"""
<div class="topbar">
    <div class="topbar-accent"></div>
    <div class="topbar-title">{pt}</div>
    <div class="topbar-sub">{ps}</div>
</div>""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# GUARD
# ─────────────────────────────────────────────────────────────────────────────
if not READY and page != "ℹ️   About":
    st.error(f"Could not load model artifacts: {_err}")
    st.info("Place `best_model.pkl`, `scaler.pkl`, `features.pkl`, `diabetes_clean.csv` "
            "and `model_results.csv` in the same directory.")
    st.stop()

# ═════════════════════════════════════════════════════════════════════════════
# PAGE 1  ─  RISK PREDICTION
# ═════════════════════════════════════════════════════════════════════════════
if page == "🔬  Risk Prediction":

    col_form, col_out = st.columns([1, 1], gap="large")

    with col_form:
        sec("Patient Parameters")
        st.markdown('<div style="background:#181818;border:1px solid #242424;border-radius:7px;padding:1rem 1.1rem;">', unsafe_allow_html=True)
        st.markdown(f"""
        <div style="background:rgba(0,120,212,0.07);border:1px solid #0078d4;border-radius:5px;
                    padding:0.5rem 0.8rem;margin-bottom:0.85rem;font-size:0.73rem;color:#888;">
            <span style="color:#0078d4;font-weight:600;">Pipeline:</span>
            StandardScaler → SelectKBest(6) → Random Forest
            &nbsp;·&nbsp; Zeros auto-imputed with training medians
        </div>""", unsafe_allow_html=True)

        ra, rb = st.columns(2)
        with ra:
            pregnancies = st.number_input("Pregnancies",        min_value=0,    max_value=17,   value=1,     step=1)
            blood_pres  = st.number_input("Blood Pressure mmHg",min_value=0,    max_value=122,  value=70,    step=1,
                                          help="0 will be imputed with training median (72 mmHg)")
            insulin     = st.number_input("Insulin μU/mL",      min_value=0,    max_value=846,  value=80,    step=1,
                                          help="0 will be imputed with training median (30.5 μU/mL)")
            dpf         = st.number_input("Pedigree Function",  min_value=0.078,max_value=2.420,value=0.350, step=0.001, format="%.3f")
        with rb:
            glucose    = st.number_input("Glucose mg/dL",       min_value=0,    max_value=199,  value=110,   step=1,
                                         help="0 will be imputed with training median (117 mg/dL)")
            skin_thick = st.number_input("Skin Thickness mm",   min_value=0,    max_value=99,   value=20,    step=1,
                                         help="0 will be imputed with training median (23 mm)")
            bmi        = st.number_input("BMI kg/m²",           min_value=0.0,  max_value=67.0, value=28.0,  step=0.1,  format="%.1f",
                                         help="0 will be imputed with training median (32.0 kg/m²)")
            age        = st.number_input("Age years",            min_value=21,   max_value=81,   value=33,    step=1)
        st.markdown('</div>', unsafe_allow_html=True)
        sp(0.4)
        st.button("Analyse Risk →", use_container_width=True)

    raw_vals = {
        'Pregnancies': pregnancies, 'Glucose': glucose, 'BloodPressure': blood_pres,
        'SkinThickness': skin_thick, 'Insulin': insulin, 'BMI': bmi,
        'DiabetesPedigreeFunction': dpf, 'Age': age
    }
    clean_vals = {f: (MEDIANS[f] if (f in ZERO_IMPUTE_COLS and v == 0) else v)
                  for f, v in raw_vals.items()}
    inp    = pd.DataFrame([clean_vals])[FEAT_ALL]
    scaled = pd.DataFrame(scaler.transform(inp), columns=FEAT_ALL)
    pred   = model.predict(scaled[features])[0]
    prob   = model.predict_proba(scaled[features])[0]
    pct    = prob[1]*100

    imputed = [f for f, v in raw_vals.items() if f in ZERO_IMPUTE_COLS and v == 0]
    if imputed:
        st.warning(f"⚠ Zero values detected in: **{', '.join(imputed)}** — replaced with training medians before prediction.")

    rc  = RED  if pct>=60 else (ORG if pct>=30 else GRN)
    rlb = "HIGH RISK" if pct>=60 else ("MODERATE RISK" if pct>=30 else "LOW RISK")

    with col_out:
        sec("Prediction Result")
        lbl = "⚠ Diabetes Risk Detected" if pred==1 else "✓ Low Diabetes Risk"
        st.markdown(f"""
        <div class="res-card" style="--bc:{rc};">
            <div class="res-label">{lbl}</div>
            <div class="res-pct">{pct:.1f}%</div>
            <div class="res-sub">Diabetes probability · Random Forest model</div>
            <div class="res-badge">{rlb}</div>
        </div>""", unsafe_allow_html=True)
        sp(0.6)

        f_g, a_g = plt.subplots(figsize=(6, 1.2))
        f_g.patch.set_facecolor(BG); a_g.set_facecolor(BG)
        a_g.barh([0],[100],color='#222',height=0.55)
        a_g.barh([0],[pct], color=rc,   height=0.55, alpha=0.9)
        for thresh, c in [(30,ORG),(60,RED)]:
            a_g.axvline(thresh, color=c, lw=1, ls='--', alpha=0.6)
        for x, t, c in [(15,'Low',GRN),(45,'Moderate',ORG),(80,'High',RED)]:
            a_g.text(x, 0, t, ha='center', va='center', color=c, fontsize=8.5, fontweight='600')
        a_g.set_xlim(0,100); a_g.set_yticks([])
        for s in a_g.spines.values(): s.set_visible(False)
        a_g.tick_params(colors='#555', labelsize=7.5)
        plt.tight_layout(pad=0.1)
        st.pyplot(f_g, use_container_width=True); plt.close()

    sp()

    sec("Patient Health Summary")
    g   = clean_vals['Glucose']
    b   = clean_vals['BMI']
    bp  = clean_vals['BloodPressure']
    ins = clean_vals['Insulin']

    k1,k2,k3,k4 = st.columns(4)
    kpi_card(k1,"Diabetes Risk Score",  f"{pct:.1f}%",  rlb,                                                                rc,   "🎯")
    kpi_card(k2,"Glucose Level",        f"{g:.0f}",     "Normal 70–100 mg/dL · "+("✓ Normal" if 70<=g<=100 else "⚠ Elevated"),  BLUE if 70<=g<=100 else ORG, "🍬")
    kpi_card(k3,"BMI",                  f"{b:.1f}",     "Healthy 18.5–24.9 · "+("✓ Normal" if 18.5<=b<=24.9 else "⚠ Review"),   GRN  if 18.5<=b<=24.9 else ORG, "⚖️")
    kpi_card(k4,"Blood Pressure",       f"{bp:.0f}",    "Normal 60–80 mmHg · "+("✓ Normal" if 60<=bp<=80 else "⚠ Check"),       GRN  if 60<=bp<=80 else ORG, "💉")

    sp()
    sec("Active Model Features")
    feat_cols_ui = st.columns(8)
    for i, feat in enumerate(FEAT_ALL):
        is_active = feat in list(features)
        color = GRN if is_active else "#333"
        label = "✓ Active" if is_active else "○ Unused"
        feat_cols_ui[i].markdown(f"""
        <div style="background:#181818;border:1px solid {color};border-radius:5px;
                    padding:0.45rem 0.5rem;text-align:center;">
            <div style="font-size:0.62rem;font-weight:600;color:{color};">{label}</div>
            <div style="font-size:0.7rem;color:#aaa;margin-top:0.15rem;">{feat}</div>
        </div>""", unsafe_allow_html=True)

    sp()
    sec("Clinical Analysis")
    cr, cf = st.columns(2)

    with cr:
        feat_r = ['Glucose','BMI','Age','Insulin','BloodPressure','Pregnancies']
        p_v = [(clean_vals['Glucose']/199),(clean_vals['BMI']/67),(age/81),
               (clean_vals['Insulin']/846),(clean_vals['BloodPressure']/122),(pregnancies/17)]
        d_v = [(df[df['Outcome']==1][f].mean()-df[f].min())/(df[f].max()-df[f].min()) for f in feat_r]
        h_v = [(df[df['Outcome']==0][f].mean()-df[f].min())/(df[f].max()-df[f].min()) for f in feat_r]
        ang = np.linspace(0,2*np.pi,len(feat_r),endpoint=False).tolist()+[0]
        ext = lambda l: l+l[:1]

        f_r, a_r = figp(5.5, 4.8)
        a_r.plot(ang,ext(p_v),'o-',color=BLUE,lw=2,ms=4,label='Patient',zorder=3)
        a_r.fill(ang,ext(p_v),alpha=0.15,color=BLUE)
        a_r.plot(ang,ext(d_v),'s--',color=RED,lw=1.4,ms=3,label='Avg Diabetic',zorder=2)
        a_r.plot(ang,ext(h_v),'^--',color=GRN,lw=1.4,ms=3,label='Avg Healthy',zorder=2)
        a_r.set_xticks(ang[:-1])
        a_r.set_xticklabels(feat_r,color='#888',fontsize=8,fontweight='500')
        a_r.set_yticklabels([])
        a_r.set_title('Patient vs Population Profile',color=TXT,fontweight='600',pad=14,fontsize=9)
        a_r.legend(loc='upper right',bbox_to_anchor=(1.4,1.12),fontsize=7.5,
                   facecolor=BG,edgecolor=LN,labelcolor='#bbb')
        plt.tight_layout()
        st.markdown('<div class="cc"><div class="cc-title">Feature Radar (6 active + 2 context features)</div>', unsafe_allow_html=True)
        st.pyplot(f_r, use_container_width=True); plt.close()
        st.markdown('</div>', unsafe_allow_html=True)

    with cf:
        fi = pd.Series(model.feature_importances_, index=features).sort_values()
        f_fi, a_fi = fig(5.5, 4.8)
        a_fi.grid(axis='x',color=LN,lw=0.5); a_fi.grid(axis='y',visible=False)
        clrs = [RED if f==fi.idxmax() else BLUE for f in fi.index]
        bars = a_fi.barh(fi.index, fi.values, color=clrs, height=0.5, edgecolor=BG)
        for b, v in zip(bars, fi.values):
            a_fi.text(b.get_width()+0.002, b.get_y()+b.get_height()/2,
                      f'{v:.3f}', va='center', fontsize=8, color='#aaa', fontweight='500')
        a_fi.set_xlabel('Importance Score')
        a_fi.set_title('Feature Importance (Top driver highlighted)',fontsize=9,fontweight='600')
        plt.tight_layout()
        st.markdown('<div class="cc"><div class="cc-title">Feature Importance</div>', unsafe_allow_html=True)
        st.pyplot(f_fi, use_container_width=True); plt.close()
        st.markdown('</div>', unsafe_allow_html=True)

# ═════════════════════════════════════════════════════════════════════════════
# PAGE 2  ─  DATA INSIGHTS
# ═════════════════════════════════════════════════════════════════════════════
elif page == "📊  Data Insights":

    total   = len(df)
    d_count = int(df['Outcome'].sum())
    h_count = total - d_count
    mean_gl = df['Glucose'].mean()
    mean_bm = df['BMI'].mean()

    k1,k2,k3,k4 = st.columns(4)
    kpi_card(k1,"Total Patients",   f"{total:,}",                     f"{d_count} diabetic · {h_count} healthy", BLUE, "👥")
    kpi_card(k2,"Diabetic Rate",    f"{d_count/total*100:.1f}%",      f"{d_count} out of {total} patients",       RED,  "📊")
    kpi_card(k3,"Mean Glucose",     f"{mean_gl:.0f} mg/dL",           "dataset average across all records",       ORG,  "🍬")
    kpi_card(k4,"Mean BMI",         f"{mean_bm:.1f}",                 "dataset average · healthy range 18.5–24.9",GRN,  "⚖️")

    sp()
    sec("Feature Explorer")
    sel = st.selectbox("Select feature:", FEAT_ALL, index=1, label_visibility="collapsed")
    sp(0.3)

    cl, cr = st.columns(2)

    with cl:
        f_d, a_d = fig(6.5, 4)
        a_d.grid(axis='x', visible=False)
        for lbl, c, nm in [(0,GRN,'Healthy'),(1,RED,'Diabetic')]:
            sub = df[df['Outcome']==lbl][sel].dropna()
            a_d.hist(sub, bins=22, alpha=0.45, color=c, label=nm, density=True, edgecolor=BG)
            kx = np.linspace(sub.min(), sub.max(), 200)
            a_d.plot(kx, gaussian_kde(sub)(kx), color=c, lw=2)
        a_d.set_xlabel(sel); a_d.set_ylabel('Density')
        a_d.legend(facecolor=BG, edgecolor=LN, labelcolor='#bbb', fontsize=8, framealpha=0.9)
        a_d.set_title(f'{sel} — Distribution by Outcome', fontsize=9, fontweight='600')
        plt.tight_layout()
        st.markdown('<div class="cc"><div class="cc-title">Distribution</div>', unsafe_allow_html=True)
        st.pyplot(f_d, use_container_width=True); plt.close()
        st.markdown('</div>', unsafe_allow_html=True)

    with cr:
        f_b, a_b = fig(6.5, 4)
        a_b.grid(axis='x', visible=False)
        bp = a_b.boxplot(
            [df[df['Outcome']==0][sel].dropna(), df[df['Outcome']==1][sel].dropna()],
            labels=['Healthy','Diabetic'], patch_artist=True, widths=0.42,
            medianprops=dict(color='#fff',lw=1.5),
            flierprops=dict(marker='o',markerfacecolor='#444',markersize=3,linestyle='none')
        )
        for patch, c in zip(bp['boxes'],[GRN,RED]):
            patch.set_facecolor(c); patch.set_alpha(0.55)
        for el in ['whiskers','caps']:
            for it in bp[el]: it.set_color('#555')
        Q1,Q3 = df[sel].quantile([0.25,0.75]); iqr=Q3-Q1
        n_out = int(((df[sel]<Q1-1.5*iqr)|(df[sel]>Q3+1.5*iqr)).sum())
        a_b.set_title(f'{sel} — Box Plot · {n_out} outliers', fontsize=9, fontweight='600')
        plt.tight_layout()
        st.markdown('<div class="cc"><div class="cc-title">Box Plot</div>', unsafe_allow_html=True)
        st.pyplot(f_b, use_container_width=True); plt.close()
        st.markdown('</div>', unsafe_allow_html=True)

    sec("Correlation & Statistics")
    ch, cs = st.columns([3,2])

    with ch:
        corr = df.corr(numeric_only=True)
        f_c, a_c = fig(8, 5)
        cmap = sns.diverging_palette(10,145,s=80,l=40,as_cmap=True)
        sns.heatmap(corr, annot=True, fmt='.2f', cmap=cmap, center=0, vmin=-1, vmax=1,
                    ax=a_c, linewidths=0.4, linecolor='#111',
                    annot_kws={'fontsize':7.5,'fontweight':'500'},
                    cbar_kws={'shrink':0.78})
        a_c.tick_params(colors='#888', labelsize=8)
        a_c.set_title('Pearson Correlation Matrix', fontsize=9, fontweight='600')
        plt.tight_layout()
        st.markdown('<div class="cc"><div class="cc-title">Correlation Heatmap</div>', unsafe_allow_html=True)
        st.pyplot(f_c, use_container_width=True); plt.close()
        st.markdown('</div>', unsafe_allow_html=True)

    with cs:
        f_p, a_p = plt.subplots(figsize=(4.5,3.5))
        f_p.patch.set_facecolor(BG); a_p.set_facecolor(BG)
        wedges, texts, autos = a_p.pie(
            [h_count, d_count], labels=['Healthy','Diabetic'],
            colors=[GRN, RED], autopct='%1.1f%%', startangle=90,
            textprops={'color':'#bbb','fontsize':8.5},
            wedgeprops={'edgecolor':'#111','linewidth':2.5},
            pctdistance=0.78
        )
        for at in autos: at.set_fontweight('700')
        a_p.add_patch(plt.Circle((0,0),0.55,color=BG))
        a_p.text(0,0,f'{d_count/total*100:.0f}%\nDiabetic',
                 ha='center',va='center',fontsize=10,fontweight='700',color=RED)
        a_p.set_title('Outcome Split', color=TXT, fontsize=9, fontweight='600')
        plt.tight_layout()
        st.markdown('<div class="cc"><div class="cc-title">Outcome Distribution</div>', unsafe_allow_html=True)
        st.pyplot(f_p, use_container_width=True); plt.close()
        st.markdown('</div>', unsafe_allow_html=True)

        sp(0.4)
        st.markdown('<div class="cc"><div class="cc-title">Descriptive Statistics</div>', unsafe_allow_html=True)
        st.dataframe(df[FEAT_ALL].describe().round(2), use_container_width=True, height=210)
        st.markdown('</div>', unsafe_allow_html=True)

# ═════════════════════════════════════════════════════════════════════════════
# PAGE 3  ─  MODEL PERFORMANCE
# ═════════════════════════════════════════════════════════════════════════════
elif page == "📈  Model Performance":

    best_row  = results_df.loc[results_df['AUC-ROC'].idxmax()]
    best_name = best_row['Model']
    best_acc  = results_df['Accuracy'].max()
    best_auc  = results_df['AUC-ROC'].max()
    best_f1   = results_df['F1-Score'].max()

    k1,k2,k3,k4 = st.columns(4)
    kpi_card(k1,"Best Model",    best_name,           "ranked by AUC-ROC score",    PUR, "🏆")
    kpi_card(k2,"Best Accuracy", f"{best_acc:.2f}%",  "highest classification rate", GRN, "🎯")
    kpi_card(k3,"Best AUC-ROC",  f"{best_auc:.2f}%",  "area under ROC curve",        BLUE,"📈")
    kpi_card(k4,"Best F1-Score", f"{best_f1:.2f}%",   "precision-recall harmonic",   ORG, "⚖️")

    sp()
    sec("Leaderboard")
    disp = results_df.sort_values('AUC-ROC',ascending=False).reset_index(drop=True)
    disp.index += 1

    def _sn(v):
        if not isinstance(v,float): return ''
        if v>=95: return f'color:{GRN};font-weight:600'
        if v>=90: return f'color:{ORG};font-weight:600'
        return f'color:{RED}'

    ncols = [c for c in disp.columns if c!='Model']
    st.dataframe(disp.style.map(_sn, subset=ncols), use_container_width=True, height=340)
    sp()

    sec("Visual Comparison")
    ca, cb, cc = st.columns(3)

    with ca:
        sr = disp.sort_values('Accuracy')
        f_a, a_a = fig(5, 5.5)
        a_a.grid(axis='x',color=LN,lw=0.5); a_a.grid(axis='y',visible=False)
        clr_a = [RED if v==sr['Accuracy'].max() else BLUE for v in sr['Accuracy']]
        bars_a = a_a.barh(sr['Model'], sr['Accuracy'], color=clr_a, height=0.5, edgecolor=BG)
        for b, v in zip(bars_a, sr['Accuracy']):
            a_a.text(b.get_width()+0.4, b.get_y()+b.get_height()/2,
                     f'{v:.1f}', va='center', fontsize=7.5, color='#aaa', fontweight='500')
        a_a.set_xlabel('Accuracy (%)'); a_a.set_xlim(0,115)
        a_a.set_title('Accuracy', fontsize=9, fontweight='600')
        plt.tight_layout()
        st.markdown('<div class="cc"><div class="cc-title">Accuracy Ranking</div>', unsafe_allow_html=True)
        st.pyplot(f_a, use_container_width=True); plt.close()
        st.markdown('</div>', unsafe_allow_html=True)

    with cb:
        sr2 = disp.sort_values('AUC-ROC')
        f_b2, a_b2 = fig(5, 5.5)
        a_b2.grid(axis='x',color=LN,lw=0.5); a_b2.grid(axis='y',visible=False)
        clr_b = [ORG if v==sr2['AUC-ROC'].max() else '#1a4f7a' for v in sr2['AUC-ROC']]
        bars_b = a_b2.barh(sr2['Model'], sr2['AUC-ROC'], color=clr_b, height=0.5, edgecolor=BG)
        for b, v in zip(bars_b, sr2['AUC-ROC']):
            a_b2.text(b.get_width()+0.06, b.get_y()+b.get_height()/2,
                      f'{v:.1f}', va='center', fontsize=7.5, color='#aaa', fontweight='500')
        a_b2.set_xlabel('AUC-ROC (%)'); a_b2.set_xlim(80,108)
        a_b2.set_title('AUC-ROC', fontsize=9, fontweight='600')
        plt.tight_layout()
        st.markdown('<div class="cc"><div class="cc-title">AUC-ROC Ranking</div>', unsafe_allow_html=True)
        st.pyplot(f_b2, use_container_width=True); plt.close()
        st.markdown('</div>', unsafe_allow_html=True)

    with cc:
        f_sc, a_sc = fig(5, 5.5)
        a_sc.grid(color=LN, lw=0.5)
        cmap_sc = plt.cm.tab10(np.linspace(0,1,len(disp)))
        for i, (_,row) in enumerate(disp.iterrows()):
            a_sc.scatter(row['F1-Score'],row['AUC-ROC'],
                         s=90, color=cmap_sc[i-1], zorder=5, edgecolors='#333', lw=0.8)
            a_sc.annotate(row['Model'],(row['F1-Score'],row['AUC-ROC']),
                          textcoords='offset points',xytext=(4,2),fontsize=6.5,color='#888')
        a_sc.set_xlabel('F1-Score (%)'); a_sc.set_ylabel('AUC-ROC (%)')
        a_sc.set_title('F1 vs AUC-ROC', fontsize=9, fontweight='600')
        plt.tight_layout()
        st.markdown('<div class="cc"><div class="cc-title">F1 vs AUC-ROC</div>', unsafe_allow_html=True)
        st.pyplot(f_sc, use_container_width=True); plt.close()
        st.markdown('</div>', unsafe_allow_html=True)

    sp()
    sec("Top 5 Radar")
    _, rad_col, _ = st.columns([1,2,1])
    with rad_col:
        top5 = disp.head(5)
        mets = ['Accuracy','Precision','Recall','F1-Score','AUC-ROC']
        ang_r = np.linspace(0,2*np.pi,len(mets),endpoint=False).tolist()+[0]
        rcols = [BLUE,GRN,ORG,RED,PUR]

        f_rad, a_rad = figp(7, 5.5)
        for i, (_,row) in enumerate(top5.iterrows()):
            v = [row[m] for m in mets]+[row[mets[0]]]
            a_rad.plot(ang_r,v,'o-',color=rcols[i],lw=1.8,ms=4,label=row['Model'])
            a_rad.fill(ang_r,v,alpha=0.05,color=rcols[i])
        a_rad.set_xticks(ang_r[:-1])
        a_rad.set_xticklabels(mets,color='#888',fontsize=8.5,fontweight='500')
        a_rad.set_yticklabels([]); a_rad.set_ylim(85,101)
        a_rad.set_title('Top 5 Models — Multi-Metric Radar',color=TXT,fontweight='600',pad=16,fontsize=9)
        a_rad.legend(loc='upper right',bbox_to_anchor=(1.38,1.15),fontsize=8,
                     facecolor=BG,edgecolor=LN,labelcolor='#ccc')
        plt.tight_layout()
        st.markdown('<div class="cc"><div class="cc-title">Multi-Metric Comparison</div>', unsafe_allow_html=True)
        st.pyplot(f_rad, use_container_width=True); plt.close()
        st.markdown('</div>', unsafe_allow_html=True)

# ═════════════════════════════════════════════════════════════════════════════
# PAGE 4  ─  BULK SCANNER
# ═════════════════════════════════════════════════════════════════════════════
elif page == "📂  Bulk Scanner":

    ALIASES = {
        'Pregnancies':             ['pregnancies','preg','pregnancy','num_pregnancies'],
        'Glucose':                 ['glucose','glu','blood_glucose','plasma_glucose','glucose_mg'],
        'BloodPressure':           ['bloodpressure','blood_pressure','bp','diastolic','diastolic_bp'],
        'SkinThickness':           ['skinthickness','skin_thickness','skin','triceps','skin_fold'],
        'Insulin':                 ['insulin','serum_insulin','insulin_level','insulin_2hr'],
        'BMI':                     ['bmi','body_mass_index','bodymassindex'],
        'DiabetesPedigreeFunction':['diabetespedigreefunction','pedigree','dpf',
                                    'diabetes_pedigree','pedigree_function'],
        'Age':                     ['age','patient_age','age_years'],
    }

    def auto_map(cols):
        cn = {c.lower().replace(' ','_').replace('-','_'): c for c in cols}
        m  = {}
        for feat, als in ALIASES.items():
            hit = next((cn[a] for a in als if a in cn), None)
            if hit is None and feat.lower() in cn: hit = cn[feat.lower()]
            m[feat] = hit
        return m

    # ── SAMPLE CSV DOWNLOAD PANEL ─────────────────────────────────────────────
    st.markdown(f"""
    <div style="background:#181818;border:1px solid #242424;border-radius:7px;
                padding:1rem 1.2rem;margin-bottom:1rem;
                display:flex;align-items:center;gap:1.5rem;flex-wrap:wrap;">
        <div style="flex:1;min-width:200px;">
            <div style="font-size:0.72rem;font-weight:600;color:#888;margin-bottom:0.2rem;">
                📄 Don't have a file yet?
            </div>
            <div style="font-size:0.68rem;color:#555;line-height:1.7;">
                Download the sample CSV (20 patients · all 8 features · real Pima-style values including zeros
                for imputation testing) — fill in your own data and re-upload.
            </div>
        </div>
        <div style="display:flex;flex-direction:column;gap:0.35rem;font-size:0.65rem;color:#444;">
            <span>✓ PatientID + 8 clinical features</span>
            <span>✓ Includes realistic zero values</span>
            <span>✓ Mix of diabetic &amp; healthy profiles</span>
        </div>
    </div>""", unsafe_allow_html=True)

    st.download_button(
        label="⬇️  Download Sample CSV (20 patients)",
        data=SAMPLE_CSV_DATA.encode('utf-8'),
        file_name="SugarMetrics_sample_patients.csv",
        mime="text/csv",
        help="20 anonymised patients with all 8 clinical features — use as a template",
        use_container_width=False,
    )

    sp(0.5)

    # ── FILE UPLOADER ─────────────────────────────────────────────────────────
    uploaded = st.file_uploader(
        "Upload patient dataset (CSV · JSON · XLSX)",
        type=["csv","json","xlsx"]
    )

    if uploaded is None:
        st.markdown("""
        <div style="background:#181818;border:2px dashed #242424;border-radius:8px;
                    padding:3rem 2rem;text-align:center;margin-top:1rem;">
            <div style="font-size:2.4rem;margin-bottom:0.6rem;">📂</div>
            <div style="font-size:0.95rem;font-weight:600;color:#bbb;margin-bottom:0.35rem;">
                No file uploaded yet</div>
            <div style="color:#444;font-size:0.8rem;line-height:1.9;max-width:420px;margin:0 auto;">
                Drop a <strong style="color:#777;">CSV, JSON or XLSX</strong> file above.<br>
                Columns are auto-matched to the 8 clinical features.<br>
                Missing features are filled with training-set medians.<br>
                <span style="color:#2e2e2e;">← Use the sample CSV above to get started.</span>
            </div>
        </div>""", unsafe_allow_html=True)

    else:
        try:
            ext = uploaded.name.rsplit('.',1)[-1].lower()
            if   ext=='csv':  raw_df = pd.read_csv(uploaded)
            elif ext=='json': raw_df = pd.read_json(uploaded)
            elif ext=='xlsx': raw_df = pd.read_excel(uploaded)
            else: st.error("Unsupported format."); st.stop()
        except Exception as e:
            st.error(f"Could not read file: {e}"); st.stop()

        st.markdown(f"""
        <div style="background:#181818;border:1px solid #242424;border-radius:7px;
                    padding:0.65rem 1rem;margin-bottom:0.8rem;
                    display:flex;align-items:center;gap:1.5rem;">
            <span style="color:{GRN};font-weight:600;font-size:0.82rem;">✓ {uploaded.name}</span>
            <span style="color:#555;font-size:0.75rem;">{len(raw_df):,} rows · {len(raw_df.columns)} columns</span>
        </div>""", unsafe_allow_html=True)

        with st.expander("Preview — first 5 rows"):
            st.dataframe(raw_df.head(), use_container_width=True)

        sec("Column Mapping")
        st.caption("Auto-detected — override if needed")

        auto_m  = auto_map(raw_df.columns.tolist())
        opts    = ["— skip —"] + raw_df.columns.tolist()
        map_res = {}
        fc      = st.columns(4)
        for i,feat in enumerate(FEAT_ALL):
            det = auto_m.get(feat)
            idx = opts.index(det) if det in opts else 0
            with fc[i%4]:
                ch = st.selectbox(feat, options=opts, index=idx, key=f"bm_{feat}")
                map_res[feat] = None if ch=="— skip —" else ch

        mapped   = [f for f,c in map_res.items() if c]
        unmapped = [f for f,c in map_res.items() if not c]

        m1,m2 = st.columns(2)
        m1.markdown(f"""<div style="background:rgba(39,174,96,.07);border:1px solid {GRN};
            border-radius:5px;padding:0.55rem 0.9rem;font-size:0.77rem;">
            <span style="color:{GRN};font-weight:600;">✓ Mapped ({len(mapped)}/8):</span>
            <span style="color:#888;"> {', '.join(mapped) or '—'}</span></div>""",
            unsafe_allow_html=True)
        bc2 = RED if unmapped else '#333'
        m2.markdown(f"""<div style="background:rgba(192,57,43,.06);border:1px solid {bc2};
            border-radius:5px;padding:0.55rem 0.9rem;font-size:0.77rem;">
            <span style="color:{bc2};font-weight:600;">⚠ Missing ({len(unmapped)}):</span>
            <span style="color:#888;"> {', '.join(unmapped) if unmapped else 'None — all present!'}</span></div>""",
            unsafe_allow_html=True)

        if unmapped:
            st.info("Missing features will be filled with training-set medians.")

        sp(0.5)
        if st.button("Run Bulk Prediction →", use_container_width=True):
            with st.spinner("Preprocessing → Scaling → Predicting…"):
                ib = pd.DataFrame(index=raw_df.index)
                for feat in FEAT_ALL:
                    col2 = map_res.get(feat)
                    if col2 and col2 in raw_df.columns:
                        vals = pd.to_numeric(raw_df[col2], errors='coerce').fillna(MEDIANS[feat])
                        if feat in ZERO_IMPUTE_COLS:
                            vals = vals.replace(0, MEDIANS[feat])
                            vals = vals.where(vals > 0, MEDIANS[feat])
                        ib[feat] = vals
                    else:
                        ib[feat] = MEDIANS[feat]

                sc2    = pd.DataFrame(scaler.transform(ib[FEAT_ALL]),
                                      columns=FEAT_ALL, index=ib.index)
                preds  = model.predict(sc2[features])
                probas = model.predict_proba(sc2[features])[:,1]*100

                out = raw_df.copy().reset_index(drop=True)
                out.insert(0,'Risk_Level',
                           pd.cut(probas,bins=[-1,30,60,101],
                                  labels=['Low','Moderate','High']).astype(str))
                out.insert(0,'Prediction',
                           ['Diabetic' if p==1 else 'Non-Diabetic' for p in preds])
                out.insert(0,'Diabetes_Prob_%', probas.round(2))

            sec("Batch Results")
            nt  = len(out)
            nd  = int((preds==1).sum())
            nh  = nt-nd
            avg = probas.mean()
            hir = int((probas>=60).sum())

            bk1,bk2,bk3,bk4,bk5 = st.columns(5)
            kpi_card(bk1,"Scanned",       f"{nt:,}",         "total patient records",         BLUE,"👥")
            kpi_card(bk2,"Diabetic",       nd,                f"{nd/nt*100:.1f}% of batch",    RED, "🔴")
            kpi_card(bk3,"Non-Diabetic",   nh,                f"{nh/nt*100:.1f}% of batch",    GRN, "🟢")
            kpi_card(bk4,"Avg Risk Score", f"{avg:.1f}%",     "mean probability",              ORG, "📊")
            kpi_card(bk5,"High-Risk",      hir,               "probability ≥ 60%",             RED, "⚠️")

            sp()
            sec("Risk Distribution")
            bc1, bc2_col = st.columns(2)

            with bc1:
                f_h, a_h = fig(6,3.8)
                a_h.grid(axis='x',visible=False)
                n_arr, bins_arr, _ = a_h.hist(probas,bins=26,color=BLUE,edgecolor=BG,alpha=0.8)
                a_h.axvline(30,color=ORG,lw=1.2,ls='--',label='30% threshold')
                a_h.axvline(60,color=RED,lw=1.2,ls='--',label='60% threshold')
                a_h.fill_betweenx([0,n_arr.max()*1.05],  0, 30,alpha=0.04,color=GRN)
                a_h.fill_betweenx([0,n_arr.max()*1.05], 30, 60,alpha=0.04,color=ORG)
                a_h.fill_betweenx([0,n_arr.max()*1.05], 60,100,alpha=0.04,color=RED)
                a_h.set_xlabel('Diabetes Probability (%)'); a_h.set_ylabel('Patient Count')
                a_h.set_title('Score Distribution', fontsize=9, fontweight='600')
                a_h.legend(facecolor=BG,edgecolor=LN,labelcolor='#bbb',fontsize=8)
                plt.tight_layout()
                st.markdown('<div class="cc"><div class="cc-title">Probability Histogram</div>', unsafe_allow_html=True)
                st.pyplot(f_h, use_container_width=True); plt.close()
                st.markdown('</div>', unsafe_allow_html=True)

            with bc2_col:
                lc = int((probas<30).sum())
                mc = int(((probas>=30)&(probas<60)).sum())
                hc = int((probas>=60).sum())
                f_d2, a_d2 = plt.subplots(figsize=(6,3.8))
                f_d2.patch.set_facecolor(BG); a_d2.set_facecolor(BG)
                wedges2,_,auts2 = a_d2.pie(
                    [lc,mc,hc], labels=['Low','Moderate','High'],
                    colors=[GRN,ORG,RED], autopct='%1.1f%%', startangle=140,
                    textprops={'color':'#bbb','fontsize':9},
                    wedgeprops={'edgecolor':'#111','linewidth':2},
                    pctdistance=0.75
                )
                for at in auts2: at.set_fontweight('700')
                a_d2.add_patch(plt.Circle((0,0),0.52,color=BG))
                a_d2.text(0,0,f'{hc}\nHigh Risk',ha='center',va='center',
                          fontsize=9,fontweight='700',color=RED)
                a_d2.set_title('Risk Level Split', color=TXT, fontsize=9, fontweight='600')
                plt.tight_layout()
                st.markdown('<div class="cc"><div class="cc-title">Risk Level Breakdown</div>', unsafe_allow_html=True)
                st.pyplot(f_d2, use_container_width=True); plt.close()
                st.markdown('</div>', unsafe_allow_html=True)

            sec("Full Results")

            def _sp2(v):
                if v=='Diabetic':     return f'color:{RED};font-weight:600'
                if v=='Non-Diabetic': return f'color:{GRN};font-weight:600'
                return ''

            def _sp3(v):
                if not isinstance(v,float): return ''
                if v>=60: return f'color:{RED};font-weight:600'
                if v>=30: return f'color:{ORG};font-weight:600'
                return f'color:{GRN};font-weight:600'

            st.dataframe(
                out.style.map(_sp2,subset=['Prediction']).map(_sp3,subset=['Diabetes_Prob_%']),
                use_container_width=True, height=380
            )

            csv_out = out.to_csv(index=False).encode('utf-8')
            st.download_button(
                "⬇️  Download Results (CSV)",
                data=csv_out,
                file_name=f"SugarMetrics_{uploaded.name.rsplit('.',1)[0]}_Results.csv",
                mime='text/csv', use_container_width=True
            )

# ═════════════════════════════════════════════════════════════════════════════
# PAGE 5  ─  ABOUT
# ═════════════════════════════════════════════════════════════════════════════
elif page == "ℹ️   About":

    st.markdown("""
    <div style="background:#181818;border:1px solid #242424;border-radius:7px;
                padding:1.6rem 1.8rem;margin-bottom:1rem;border-left:3px solid #0078d4;">
        <div style="font-size:1.3rem;font-weight:700;color:#fff;margin-bottom:0.35rem;">🩺 SugarMetrics</div>
        <div style="color:#888;font-size:0.85rem;line-height:1.8;max-width:700px;">
            An AI-powered diabetes early-detection dashboard built on a complete 3-week ML pipeline.
            Trained on the Pima Indians Diabetes Dataset (768 records, 8 clinical features)
            using 10 scikit-learn models with full preprocessing, oversampling, feature selection,
            and GridSearchCV hyperparameter tuning.
        </div>
    </div>""", unsafe_allow_html=True)

    sec("ML Pipeline — 5 Stages")
    s1,s2,s3,s4,s5 = st.columns(5)
    for col, num, title, color, steps in [
        (s1,"①","Data Prep",   BLUE, ["Load Pima CSV","Zero-value imputation","IQR outlier capping","Remove duplicates","Log transforms"]),
        (s2,"②","Feature Eng", ORG,  ["MinMax normalise","StandardScaler","Interaction features","Binning (BMI/Age)","SelectKBest ANOVA"]),
        (s3,"③","Balancing",   PUR,  ["SMOTE-style oversample","50/50 class balance","80-20 stratified split","10-fold CV","RFE feature check"]),
        (s4,"④","Modelling",   GRN,  ["10 sklearn models","GridSearchCV / RandomSearch","Best: Random Forest","Tuned hyperparams","Saved as .pkl"]),
        (s5,"⑤","Evaluation",  RED,  ["Accuracy · Precision","Recall · F1-Score","AUC-ROC · ROC curve","Confusion matrices","Permutation importance"]),
    ]:
        li = "".join(f"<li style='color:#777;font-size:0.78rem;line-height:2;'>{s}</li>" for s in steps)
        col.markdown(f"""
        <div style="background:#181818;border:1px solid #242424;border-radius:7px;
                    padding:1rem;border-top:3px solid {color};height:100%;">
            <div style="font-size:1.1rem;color:{color};font-weight:700;margin-bottom:0.3rem;">{num}</div>
            <div style="font-size:0.8rem;font-weight:600;color:#ccc;margin-bottom:0.5rem;">{title}</div>
            <ul style="margin:0;padding-left:1rem;">{li}</ul>
        </div>""", unsafe_allow_html=True)

    sp()
    sec("Key Design Decisions")
    d1, d2, d3 = st.columns(3)
    for col, title, color, items in [
        (d1,"Preprocessing Details", BLUE, [
            "Zeros imputed in: Glucose, BloodPressure, SkinThickness, Insulin, BMI",
            "IQR capping on Glucose, Insulin, BMI, DiabetesPedigreeFunction",
            "StandardScaler (mean=0, std=1) for all 8 features",
            "SMOTE-style oversampling → balanced 50/50 dataset",
        ]),
        (d2,"Feature Selection", GRN, [
            "SelectKBest with ANOVA F-test (f_classif)",
            "Top 6 of 8 features selected",
            "Typical: Glucose, BMI, Age, DPF, Insulin, Pregnancies",
            "Loaded from features.pkl — exact set is pipeline-determined",
            "2 lowest-scored features excluded from model",
        ]),
        (d3,"Model & Deployment", ORG, [
            "Best model: Random Forest (GridSearchCV tuned)",
            "Grid searched: n_estimators, max_depth, min_samples_split, max_features",
            "Validated: 10-fold stratified CV on balanced data",
            "Artifacts: best_model.pkl · scaler.pkl · features.pkl",
            "Dashboard data: diabetes_clean.csv · model_results.csv",
        ]),
    ]:
        li = "".join(f"<li style='color:#777;font-size:0.8rem;line-height:2.1;'>{i}</li>" for i in items)
        col.markdown(f"""
        <div style="background:#181818;border:1px solid #242424;border-radius:7px;
                    padding:1.2rem;border-top:3px solid {color};height:100%;">
            <div style="font-size:0.82rem;font-weight:600;color:{color};margin-bottom:0.6rem;">{title}</div>
            <ul style="margin:0;padding-left:1rem;">{li}</ul>
        </div>""", unsafe_allow_html=True)

    sp()
    sec("10 Models Trained")
    models_info = [
        ("Logistic Regression",  "Baseline · lbfgs solver · GridSearchCV C",    BLUE),
        ("Decision Tree",        "max_depth=6 · min_samples_split=10",           GRN),
        ("Random Forest ⭐",     "Best model · 200 estimators · GridSearchCV",   ORG),
        ("Gradient Boosting",    "200 estimators · lr=0.1 · RandomizedSearch",   PUR),
        ("XGBoost-style GBM",    "300 estimators · lr=0.05 · subsample=0.8",     RED),
        ("SVM",                  "RBF kernel · probability=True · GridSearchCV", BLUE),
        ("KNN",                  "k=9 · distance weights · Minkowski metric",    GRN),
        ("Neural Network (ANN)", "128→64→32 · ReLU · Adam · early stopping",     ORG),
        ("LightGBM-style",       "HistGradientBoosting · GridSearchCV tuned",    PUR),
        ("AdaBoost",             "200 estimators · lr=0.5 · SAMME algorithm",    RED),
    ]
    mc = st.columns(5)
    for i, (name, desc, color) in enumerate(models_info):
        mc[i%5].markdown(f"""
        <div style="background:#181818;border:1px solid #242424;border-radius:6px;
                    padding:0.75rem 0.8rem;margin-bottom:0.5rem;border-left:2px solid {color};">
            <div style="font-size:0.75rem;font-weight:600;color:{color};">{name}</div>
            <div style="font-size:0.67rem;color:#555;margin-top:0.2rem;">{desc}</div>
        </div>""", unsafe_allow_html=True)

    sp()
    st.markdown("""
    <div style="background:rgba(192,57,43,.06);border:1px solid #c0392b;
                border-radius:7px;padding:0.85rem 1.1rem;">
        <span style="color:#c0392b;font-size:0.78rem;font-weight:600;">⚠ Medical Disclaimer &nbsp;—&nbsp;</span>
        <span style="color:#666;font-size:0.78rem;">
        For educational and academic purposes only. Not a substitute for professional medical advice.
        Always consult a qualified healthcare provider for clinical decisions.
        </span>
    </div>""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<hr>
<div style="text-align:center;color:#2e2e2e;font-size:0.7rem;padding:0.6rem 0;">
    🩺 SugarMetrics &nbsp;·&nbsp; Data Science & Machine Learning
    &nbsp;·&nbsp; Python · Scikit-Learn · Streamlit
</div>""", unsafe_allow_html=True)
