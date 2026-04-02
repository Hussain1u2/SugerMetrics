
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import joblib
import warnings
from scipy import stats
from sklearn.metrics import (confusion_matrix, roc_curve, roc_auc_score,
                              accuracy_score, precision_score, recall_score, f1_score)

warnings.filterwarnings('ignore')

# ── PAGE CONFIG ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SugarMetrics — Early Diabetes Prediction",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── CUSTOM CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Teal/Indigo Background ───────────────────────────────────── */
.stApp { background: linear-gradient(135deg, #0d1b2a 0%, #112240 60%, #0d2137 100%); }
.main  { background-color: transparent; }

/* ── Sidebar styling ──────────────────────────────────────────── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d2137 0%, #0a1628 100%) !important;
    border-right: 1px solid #1e3a5f;
}

/* ── Hero Banner ──────────────────────────────────────────────── */
.hero-banner {
    background: linear-gradient(135deg, #0f7ea6 0%, #0db8b0 60%, #00c9a7 100%);
    padding: 2.4rem 2rem; border-radius: 18px; margin-bottom: 1.8rem;
    box-shadow: 0 8px 32px rgba(13,184,176,0.35); text-align: center;
}
.hero-title { font-size: 2.8rem; font-weight: 800; color: white; margin: 0; letter-spacing: -1px; }
.hero-sub   { font-size: 1.05rem; color: rgba(255,255,255,0.92); margin-top: 0.5rem; }
.hero-badge {
    display: inline-block; background: rgba(255,255,255,0.18);
    border: 1px solid rgba(255,255,255,0.35); border-radius: 20px;
    padding: 4px 14px; font-size: 0.8rem; margin: 0.5rem 0.3rem 0;
    color: white; font-weight: 600;
}

/* ── KPI Cards ────────────────────────────────────────────────── */
.kpi-card {
    background: linear-gradient(135deg, #112240 0%, #1a3557 100%);
    border: 1px solid #1e4976; border-radius: 16px; padding: 1.6rem 1.4rem;
    text-align: center; box-shadow: 0 4px 20px rgba(0,0,0,0.4);
    margin-bottom: 0.5rem;
}
.kpi-icon  { font-size: 2rem; margin-bottom: 0.3rem; }
.kpi-value { font-size: 2.5rem; font-weight: 800; margin: 0.2rem 0; line-height: 1; }
.kpi-label { font-size: 0.75rem; color: #7eb8d4; text-transform: uppercase; letter-spacing: 1.2px; }
.kpi-sub   { font-size: 0.78rem; color: #7eb8d4; margin-top: 0.4rem; }

/* ── Section Headers ──────────────────────────────────────────── */
.section-hdr {
    background: linear-gradient(90deg, rgba(13,184,176,0.22), transparent);
    border-left: 4px solid #0db8b0; padding: 0.65rem 1rem;
    border-radius: 0 8px 8px 0; margin: 1.4rem 0 1rem;
    font-size: 1.1rem; font-weight: 700; color: #e0f4f7;
}

/* ── Prediction Result ────────────────────────────────────────── */
.result-diabetic {
    background: rgba(231,76,60,0.10); border: 2px solid #e74c3c;
    border-radius: 16px; padding: 2rem; text-align: center; margin: 1rem 0;
}
.result-healthy {
    background: rgba(13,184,176,0.10); border: 2px solid #0db8b0;
    border-radius: 16px; padding: 2rem; text-align: center; margin: 1rem 0;
}
.result-title { font-size: 1.9rem; font-weight: 800; margin-bottom: 0.4rem; }
.result-prob  { font-size: 3rem; font-weight: 800; line-height: 1.1; }
.result-sub   { font-size: 0.9rem; color: #7eb8d4; margin-top: 0.4rem; }
.risk-badge   {
    display: inline-block; padding: 0.4rem 1.4rem;
    border-radius: 20px; font-size: 1.05rem; font-weight: 700; margin-top: 0.8rem;
}

/* ── Metric Boxes ─────────────────────────────────────────────── */
[data-testid="metric-container"] {
    background: #112240; border-radius: 10px;
    padding: 0.9rem; border: 1px solid #1e4976;
}

/* ── Tabs ─────────────────────────────────────────────────────── */
.stTabs [data-baseweb="tab"]        { border-radius: 8px 8px 0 0; color: #7eb8d4; }
.stTabs [aria-selected="true"]      { background: #0db8b0 !important; color: white !important; }

/* ── Text input styling ───────────────────────────────────────── */
.stTextInput > div > div > input {
    background: #112240 !important;
    border: 1px solid #1e4976 !important;
    border-radius: 8px !important;
    color: #e0f4f7 !important;
    font-size: 0.95rem !important;
}
.stTextInput > div > div > input:focus {
    border-color: #0db8b0 !important;
    box-shadow: 0 0 0 2px rgba(13,184,176,0.25) !important;
}
.stTextInput label { color: #c5e8f0 !important; font-size: 0.88rem !important; }

/* ── Number input styling ─────────────────────────────────────── */
.stNumberInput > div > div > input {
    background: #112240 !important;
    border: 1px solid #1e4976 !important;
    color: #e0f4f7 !important;
}

/* ── Scrollbar ────────────────────────────────────────────────── */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: #0d1b2a; }
::-webkit-scrollbar-thumb { background: #1e4976; border-radius: 3px; }
</style>
""", unsafe_allow_html=True)

# ── LOAD ARTIFACTS ────────────────────────────────────────────────────────────
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

try:
    model, scaler, features = load_artifacts()
    df, results_df = load_data()
    READY = True
except Exception as e:
    st.error(f"⚠️ Artifact load error: {e}")
    READY = False

FEAT_ALL = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
            'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
PALETTE  = ['#2ecc71', '#e74c3c']

# ── CHART HELPERS ─────────────────────────────────────────────────────────────
def dark_fig(w=10, h=5):
    fig, ax = plt.subplots(figsize=(w, h))
    fig.patch.set_facecolor('#112240')
    ax.set_facecolor('#112240')
    for sp in ax.spines.values():
        sp.set_color('#1e4976')
    ax.tick_params(colors='#7eb8d4', labelsize=9)
    ax.xaxis.label.set_color('#7eb8d4')
    ax.yaxis.label.set_color('#7eb8d4')
    ax.title.set_color('#f0f2f6')
    return fig, ax

def dark_figs(rows, cols, w=14, h=5):
    fig, axes = plt.subplots(rows, cols, figsize=(w, h))
    fig.patch.set_facecolor('#112240')
    axes_flat = axes.flatten() if hasattr(axes, 'flatten') else [axes]
    for ax in axes_flat:
        ax.set_facecolor('#112240')
        for sp in ax.spines.values():
            sp.set_color('#1e4976')
        ax.tick_params(colors='#7eb8d4', labelsize=9)
        ax.xaxis.label.set_color('#7eb8d4')
        ax.yaxis.label.set_color('#7eb8d4')
        ax.title.set_color('#f0f2f6')
    return fig, axes

# ── SIDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding:1rem 0 0.5rem;'>
        <span style='font-size:3rem;'>🩺</span><br>
        <span style='font-size:1.35rem; font-weight:800; color:#0db8b0;'>SugarMetrics</span><br>
        <span style='font-size:0.8rem; color:#7eb8d4;'>Diabetes Risk Predictor</span>
    </div>
    <hr style='border-color:#1e4976; margin:0.8rem 0;'>
    <div style='font-size:0.82rem; font-weight:700; color:#e0f4f7; margin-bottom:0.8rem;'>
        📋 Patient Health Parameters
    </div>
    """, unsafe_allow_html=True)

    def _int_input(label, default, lo, hi):
        raw = st.text_input(label, value=str(default), key=label)
        try:
            val = int(float(raw))
            val = max(lo, min(hi, val))
        except (ValueError, TypeError):
            st.caption(f"⚠️ Enter a number ({lo}–{hi})")
            val = default
        return val

    def _float_input(label, default, lo, hi, fmt="%.2f"):
        raw = st.text_input(label, value=fmt % default, key=label)
        try:
            val = float(raw)
            val = max(lo, min(hi, val))
        except (ValueError, TypeError):
            st.caption(f"⚠️ Enter a number ({lo}–{hi})")
            val = default
        return val

    pregnancies = _int_input  ("🤰 Pregnancies  (0–17)",          1,   0,   17)
    glucose     = _int_input  ("🍬 Glucose mg/dL  (44–199)",     110,  44,  199)
    blood_pres  = _int_input  ("💉 Blood Pressure mmHg  (24–122)", 70,  24,  122)
    skin_thick  = _int_input  ("📏 Skin Thickness mm  (7–99)",     20,   7,   99)
    insulin     = _int_input  ("💊 Insulin μU/mL  (14–846)",       80,  14,  846)
    bmi         = _float_input("⚖️ BMI kg/m²  (10.0–67.0)",      28.0, 10.0, 67.0, "%.1f")
    dpf         = _float_input("🧬 Pedigree Function  (0.078–2.42)", 0.35, 0.078, 2.42, "%.3f")
    age         = _int_input  ("🎂 Age years  (21–81)",            33,  21,   81)

    st.markdown("<hr style='border-color:#1e4976;'>", unsafe_allow_html=True)
    st.markdown("""
    <div style='font-size:0.72rem; color:#7eb8d4; text-align:center; line-height:1.8;'>
        ⚡ Powered by Random Forest ML<br>
        🎓 SugarMetrics Project<br>
        📊 768 Pima Indians Records<br>
        🏆 Best AUC-ROC: 99.7%
    </div>
    """, unsafe_allow_html=True)

# ── HERO HEADER ───────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-banner">
    <div class="hero-title">🩺 SugarMetrics</div>
    <div class="hero-sub">AI-Powered Early Diabetes Detection & Risk Assessment Platform</div>
    <span class="hero-badge">🤖 10 ML Models</span>
    <span class="hero-badge">📊 768 Records</span>
    <span class="hero-badge">🎯 96.2% Accuracy</span>
    <span class="hero-badge">📈 99.7% AUC-ROC</span>
</div>
""", unsafe_allow_html=True)

# ── NAVIGATION TABS ───────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "🔬 Prediction",
    "📊 Data Insights",
    "📈 Model Performance",
    "ℹ️ About"
])

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — PREDICTION
# ═══════════════════════════════════════════════════════════════════════════════
with tab1:
    if not READY:
        st.warning("Model artifacts not found. Please run the ML pipeline first.")
        st.stop()

    # ── KPI Row ───────────────────────────────────────────────────────────────
    total     = len(df)
    diabetic  = int(df['Outcome'].sum())
    best_acc  = float(results_df['Accuracy'].max())
    best_auc  = float(results_df['AUC-ROC'].max())

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""<div class="kpi-card">
            <div class="kpi-icon">👥</div>
            <div class="kpi-value" style="color:#3498db;">{total}</div>
            <div class="kpi-label">Total Patients</div>
            <div class="kpi-sub">{diabetic} Diabetic · {total-diabetic} Healthy</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""<div class="kpi-card">
            <div class="kpi-icon">🔴</div>
            <div class="kpi-value" style="color:#e74c3c;">{diabetic/total*100:.1f}%</div>
            <div class="kpi-label">Diabetic Prevalence</div>
            <div class="kpi-sub">{diabetic} out of {total} patients</div>
        </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown(f"""<div class="kpi-card">
            <div class="kpi-icon">🎯</div>
            <div class="kpi-value" style="color:#2ecc71;">{best_acc:.1f}%</div>
            <div class="kpi-label">Best Model Accuracy</div>
            <div class="kpi-sub">Gradient Boosting · Tuned</div>
        </div>""", unsafe_allow_html=True)
    with c4:
        st.markdown(f"""<div class="kpi-card">
            <div class="kpi-icon">📈</div>
            <div class="kpi-value" style="color:#f39c12;">{best_auc:.1f}%</div>
            <div class="kpi-label">Best AUC-ROC</div>
            <div class="kpi-sub">XGBoost-style model</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Build prediction ──────────────────────────────────────────────────────
    input_df = pd.DataFrame([[pregnancies, glucose, blood_pres, skin_thick,
                               insulin, bmi, dpf, age]], columns=FEAT_ALL)
    input_scaled = pd.DataFrame(scaler.transform(input_df), columns=FEAT_ALL)
    input_final  = input_scaled[features]
    prediction   = model.predict(input_final)[0]
    prob         = model.predict_proba(input_final)[0]
    diabetes_pct = prob[1] * 100

    risk_color = '#e74c3c' if diabetes_pct >= 60 else ('#f39c12' if diabetes_pct >= 30 else '#2ecc71')
    risk_label = ('🔴 HIGH RISK'    if diabetes_pct >= 60 else
                  '🟡 MODERATE RISK' if diabetes_pct >= 30 else '🟢 LOW RISK')
    risk_bg    = ('rgba(231,76,60,0.15)'  if diabetes_pct >= 60 else
                  'rgba(243,156,18,0.15)' if diabetes_pct >= 30 else 'rgba(46,204,113,0.15)')

    col_pred, col_chart = st.columns([1, 1])

    # ── Prediction card ───────────────────────────────────────────────────────
    with col_pred:
        st.markdown('<div class="section-hdr">🔬 Prediction Result</div>', unsafe_allow_html=True)

        if prediction == 1:
            st.markdown(f"""
            <div class="result-diabetic">
                <div class="result-title" style="color:#e74c3c;">⚠️ Diabetes Risk Detected</div>
                <div class="result-prob" style="color:#e74c3c;">{diabetes_pct:.1f}%</div>
                <div class="result-sub">Probability of Diabetes · Random Forest</div>
                <div class="risk-badge" style="background:{risk_bg}; color:{risk_color}; border:1px solid {risk_color};">
                    {risk_label}
                </div>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="result-healthy">
                <div class="result-title" style="color:#2ecc71;">✅ Low Diabetes Risk</div>
                <div class="result-prob" style="color:#2ecc71;">{diabetes_pct:.1f}%</div>
                <div class="result-sub">Probability of Diabetes · Random Forest</div>
                <div class="risk-badge" style="background:{risk_bg}; color:{risk_color}; border:1px solid {risk_color};">
                    {risk_label}
                </div>
            </div>""", unsafe_allow_html=True)

        # Risk probability bar
        fig_bar, ax_bar = dark_fig(6, 2.2)
        ax_bar.barh([''], [100], color='#1a3557', height=0.4)
        ax_bar.barh([''], [diabetes_pct], color=risk_color, height=0.4, alpha=0.85)
        ax_bar.axvline(30, color='#f39c12', lw=1.5, ls='--', alpha=0.7)
        ax_bar.axvline(60, color='#e74c3c', lw=1.5, ls='--', alpha=0.7)
        for x, label, col in [(15, 'Low', '#2ecc71'), (45, 'Moderate', '#f39c12'), (80, 'High', '#e74c3c')]:
            ax_bar.text(x, 0, label, ha='center', va='center', color=col,
                        fontweight='bold', fontsize=11)
        ax_bar.set_xlim(0, 100)
        ax_bar.set_title(f'Diabetes Risk Score: {diabetes_pct:.1f}%', fontsize=12, fontweight='bold')
        ax_bar.set_xlabel('Risk Probability (%)')
        ax_bar.set_yticks([])
        plt.tight_layout()
        st.pyplot(fig_bar, use_container_width=True)
        plt.close()

        # Clinical benchmarks
        st.markdown("**📌 Clinical Benchmarks:**")
        bc1, bc2 = st.columns(2)
        norms = {'Glucose': (70, 100, 'mg/dL', glucose),
                 'BMI':     (18.5, 24.9, 'kg/m²', bmi),
                 'BloodP.': (60, 80, 'mmHg', blood_pres),
                 'Insulin': (16, 166, 'μU/mL', insulin)}
        for i, (name, (lo, hi, unit, val)) in enumerate(norms.items()):
            status = "✅" if lo <= val <= hi else "⚠️"
            label  = "Normal" if lo <= val <= hi else "Abnormal"
            (bc1 if i % 2 == 0 else bc2).metric(f"{status} {name}", f"{val:.1f} {unit}", label)

    # ── Radar chart ───────────────────────────────────────────────────────────
    with col_chart:
        st.markdown('<div class="section-hdr">📡 Patient vs Population</div>', unsafe_allow_html=True)

        feat_r = ['Glucose', 'BMI', 'Age', 'Insulin', 'BloodPressure', 'Pregnancies']
        p_vals = [(glucose/199), (bmi/67), (age/81), (insulin/846),
                  (blood_pres/122), (pregnancies/17)]
        d_avg  = [(df[df['Outcome']==1][f].mean()-df[f].min())/(df[f].max()-df[f].min())
                  for f in feat_r]
        h_avg  = [(df[df['Outcome']==0][f].mean()-df[f].min())/(df[f].max()-df[f].min())
                  for f in feat_r]

        angles = np.linspace(0, 2*np.pi, len(feat_r), endpoint=False).tolist()
        angles += angles[:1]
        def ext(lst): return lst + lst[:1]

        fig_r, ax_r = plt.subplots(figsize=(6, 5.5), subplot_kw=dict(polar=True))
        fig_r.patch.set_facecolor('#112240')
        ax_r.set_facecolor('#112240')

        ax_r.plot(angles, ext(p_vals), 'o-', color='#3498db', lw=2.5, label='Patient')
        ax_r.fill(angles, ext(p_vals), alpha=0.2, color='#3498db')
        ax_r.plot(angles, ext(d_avg),  's--', color='#e74c3c', lw=1.8, label='Avg Diabetic')
        ax_r.fill(angles, ext(d_avg),  alpha=0.08, color='#e74c3c')
        ax_r.plot(angles, ext(h_avg),  '^--', color='#2ecc71', lw=1.8, label='Avg Healthy')

        ax_r.set_xticks(angles[:-1])
        ax_r.set_xticklabels(feat_r, color='#cdd5e0', fontsize=10, fontweight='bold')
        ax_r.set_yticklabels([])
        ax_r.grid(color='#1e4976', alpha=0.6)
        ax_r.spines['polar'].set_color('#1e4976')
        ax_r.set_title('Normalized Feature Comparison\n(Patient vs Diabetic/Healthy Avg)',
                        color='white', fontweight='bold', pad=18, fontsize=11)
        ax_r.legend(loc='upper right', bbox_to_anchor=(1.35, 1.15), fontsize=9,
                    facecolor='#112240', edgecolor='#1e4976', labelcolor='white')
        plt.tight_layout()
        st.pyplot(fig_r, use_container_width=True)
        plt.close()

        # Feature contribution bar
        st.markdown("**🎯 Feature Importance (Random Forest):**")
        fi = pd.Series(model.feature_importances_, index=features).sort_values()
        fi_colors = plt.cm.RdYlGn(np.linspace(0.15, 0.9, len(fi)))

        fig_fi, ax_fi = dark_fig(7, 3.5)
        bars = ax_fi.barh(fi.index, fi.values, color=fi_colors, height=0.55, edgecolor='#112240')
        for bar, val in zip(bars, fi.values):
            ax_fi.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height()/2,
                       f'{val:.3f}', va='center', fontweight='bold',
                       color='white', fontsize=9)
        ax_fi.set_title('Feature Importance (Model Weights)', fontweight='bold', fontsize=11)
        ax_fi.set_xlabel('Importance Score')
        plt.tight_layout()
        st.pyplot(fig_fi, use_container_width=True)
        plt.close()

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — DATA INSIGHTS
# ═══════════════════════════════════════════════════════════════════════════════
with tab2:
    if not READY:
        st.stop()

    st.markdown('<div class="section-hdr">📊 Dataset Overview</div>', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Records",     len(df))
    c2.metric("Feature Count",     len(df.columns) - 1)
    c3.metric("Diabetic Cases",    int(df['Outcome'].sum()))
    c4.metric("Prevalence Rate",   f"{df['Outcome'].mean()*100:.1f}%")

    # Feature selector
    sel = st.selectbox("🔍 Select feature to explore:", FEAT_ALL, index=1)
    col_l, col_r = st.columns(2)

    # Distribution histograms
    with col_l:
        st.markdown('<div class="section-hdr">📈 Feature Distribution</div>', unsafe_allow_html=True)
        fig, ax = dark_fig(7, 4)
        for label, color, name in zip([0, 1], PALETTE, ['Non-Diabetic', 'Diabetic']):
            ax.hist(df[df['Outcome'] == label][sel], bins=25, alpha=0.65,
                    color=color, label=name, density=True, edgecolor='#112240')
        from scipy.stats import gaussian_kde
        for label, color in zip([0, 1], PALETTE):
            data_kde = df[df['Outcome'] == label][sel].dropna()
            kde_x = np.linspace(data_kde.min(), data_kde.max(), 200)
            kde   = gaussian_kde(data_kde)
            ax.plot(kde_x, kde(kde_x), color=color, lw=2.5)
        ax.set_title(f'{sel} Distribution by Outcome', fontsize=12, fontweight='bold')
        ax.set_xlabel(sel); ax.set_ylabel('Density')
        ax.legend(facecolor='#112240', edgecolor='#1e4976', labelcolor='white')
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()

    # Boxplot
    with col_r:
        st.markdown('<div class="section-hdr">📦 Boxplot Analysis</div>', unsafe_allow_html=True)
        fig, ax = dark_fig(7, 4)
        bp_data = [df[df['Outcome'] == 0][sel], df[df['Outcome'] == 1][sel]]
        bp = ax.boxplot(bp_data, labels=['Non-Diabetic', 'Diabetic'],
                        patch_artist=True, notch=False, widths=0.5)
        for patch, color in zip(bp['boxes'], PALETTE):
            patch.set_facecolor(color); patch.set_alpha(0.75)
        for elem in ['whiskers', 'caps', 'medians', 'fliers']:
            for it in bp[elem]: it.set_color('#cdd5e0')
        Q1, Q3 = df[sel].quantile([0.25, 0.75])
        iqr = Q3 - Q1
        outliers = int(((df[sel] < Q1 - 1.5*iqr) | (df[sel] > Q3 + 1.5*iqr)).sum())
        ax.set_title(f'{sel} Boxplot · IQR Outliers: {outliers}', fontsize=12, fontweight='bold')
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()

    # Correlation heatmap
    st.markdown('<div class="section-hdr">🔥 Feature Correlation Heatmap</div>', unsafe_allow_html=True)
    fig, ax = dark_fig(10, 6)
    corr = df.corr()
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='RdYlGn', center=0,
                vmin=-1, vmax=1, ax=ax, linewidths=0.5, linecolor='#0f1117',
                annot_kws={'fontsize': 8, 'fontweight': 'bold'},
                cbar_kws={'shrink': 0.8})
    ax.set_title('Feature Correlation Matrix', fontsize=13, fontweight='bold')
    ax.tick_params(colors='#cdd5e0', labelsize=9)
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close()

    # Summary stats table
    st.markdown('<div class="section-hdr">📋 Descriptive Statistics</div>', unsafe_allow_html=True)
    st.dataframe(df.describe().round(3), use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 — MODEL PERFORMANCE
# ═══════════════════════════════════════════════════════════════════════════════
with tab3:
    if not READY:
        st.stop()

    st.markdown('<div class="section-hdr">🏆 ML Model Leaderboard</div>', unsafe_allow_html=True)

    # Styled dataframe
    def highlight(val):
        if isinstance(val, float):
            if val >= 95: return 'color: #2ecc71; font-weight: bold'
            if val >= 90: return 'color: #f39c12; font-weight: bold'
            return 'color: #e74c3c'
        return ''

    disp = results_df.sort_values('AUC-ROC', ascending=False).reset_index(drop=True)
    disp.index += 1
    st.dataframe(
        disp.style.map(highlight, subset=['Accuracy','Precision','Recall','F1-Score','AUC-ROC']),
        use_container_width=True, height=370
    )

    col_a, col_b = st.columns(2)

    # Accuracy bar chart
    with col_a:
        st.markdown('<div class="section-hdr">📊 Accuracy Comparison</div>', unsafe_allow_html=True)
        fig, ax = dark_fig(7, 5.5)
        sorted_r  = disp.sort_values('Accuracy')
        bar_cols  = plt.cm.RdYlGn(np.linspace(0.2, 0.9, len(sorted_r)))
        bars = ax.barh(sorted_r['Model'], sorted_r['Accuracy'],
                       color=bar_cols, height=0.55, edgecolor='#112240')
        for bar, val in zip(bars, sorted_r['Accuracy']):
            ax.text(bar.get_width() + 0.2, bar.get_y() + bar.get_height()/2,
                    f'{val:.1f}%', va='center', fontweight='bold', color='white', fontsize=9)
        ax.set_title('Model Accuracy Comparison', fontweight='bold', fontsize=12)
        ax.set_xlabel('Accuracy (%)'); ax.set_xlim(0, 115)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()

    # AUC-ROC bar chart
    with col_b:
        st.markdown('<div class="section-hdr">📈 AUC-ROC Comparison</div>', unsafe_allow_html=True)
        fig, ax = dark_fig(7, 5.5)
        sorted_r2 = disp.sort_values('AUC-ROC')
        bar_cols2 = plt.cm.Blues(np.linspace(0.3, 0.9, len(sorted_r2)))
        bars2 = ax.barh(sorted_r2['Model'], sorted_r2['AUC-ROC'],
                        color=bar_cols2, height=0.55, edgecolor='#112240')
        for bar, val in zip(bars2, sorted_r2['AUC-ROC']):
            ax.text(bar.get_width() + 0.05, bar.get_y() + bar.get_height()/2,
                    f'{val:.1f}%', va='center', fontweight='bold', color='white', fontsize=9)
        ax.set_title('AUC-ROC Comparison', fontweight='bold', fontsize=12)
        ax.set_xlabel('AUC-ROC (%)'); ax.set_xlim(80, 108)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()

    # Multi-metric radar chart
    st.markdown('<div class="section-hdr">🕸️ Multi-Metric Radar — Top 5 Models</div>', unsafe_allow_html=True)
    col_rad, col_fi = st.columns(2)

    with col_rad:
        top5 = disp.head(5)
        metrics_r = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC-ROC']
        angles = np.linspace(0, 2*np.pi, len(metrics_r), endpoint=False).tolist()
        angles += angles[:1]
        radar_colors = ['#2ecc71', '#3498db', '#f39c12', '#9b59b6', '#1abc9c']

        fig_rad, ax_rad = plt.subplots(figsize=(7, 6.5), subplot_kw=dict(polar=True))
        fig_rad.patch.set_facecolor('#112240')
        ax_rad.set_facecolor('#112240')
        for i, (_, row) in enumerate(top5.iterrows()):
            vals = [row[m] for m in metrics_r] + [row[metrics_r[0]]]
            ax_rad.plot(angles, vals, 'o-', color=radar_colors[i], lw=2.5, label=row['Model'])
            ax_rad.fill(angles, vals, alpha=0.07, color=radar_colors[i])
        ax_rad.set_xticks(angles[:-1])
        ax_rad.set_xticklabels(metrics_r, color='white', fontsize=10, fontweight='bold')
        ax_rad.set_yticklabels([])
        ax_rad.grid(color='#1e4976', alpha=0.6)
        ax_rad.spines['polar'].set_color('#1e4976')
        ax_rad.set_ylim(85, 101)
        ax_rad.set_title('Top 5 Models — Performance Radar', color='white',
                          fontweight='bold', pad=20, fontsize=11)
        ax_rad.legend(loc='upper right', bbox_to_anchor=(1.45, 1.2), fontsize=9,
                      facecolor='#112240', edgecolor='#1e4976', labelcolor='white')
        plt.tight_layout()
        st.pyplot(fig_rad, use_container_width=True)
        plt.close()

    # F1 vs AUC scatter
    with col_fi:
        st.markdown('<div class="section-hdr">🎯 F1-Score vs AUC-ROC</div>', unsafe_allow_html=True)
        fig_sc, ax_sc = dark_fig(7, 6)
        scatter_colors = plt.cm.Set2(np.linspace(0, 1, len(disp)))
        for i, (_, row) in enumerate(disp.iterrows()):
            ax_sc.scatter(row['F1-Score'], row['AUC-ROC'], s=180,
                          color=scatter_colors[i-1], zorder=5,
                          edgecolors='white', linewidths=1.5)
            ax_sc.annotate(row['Model'], (row['F1-Score'], row['AUC-ROC']),
                           textcoords='offset points', xytext=(6, 4),
                           fontsize=8, color='#cdd5e0')
        ax_sc.set_xlabel('F1-Score (%)', fontsize=11)
        ax_sc.set_ylabel('AUC-ROC (%)', fontsize=11)
        ax_sc.set_title('F1-Score vs AUC-ROC Trade-off', fontsize=12, fontweight='bold')
        plt.tight_layout()
        st.pyplot(fig_sc, use_container_width=True)
        plt.close()

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 4 — ABOUT
# ═══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown("""
    <div style="background:#112240; border-radius:16px; padding:2.2rem;
                border:1px solid #1e4976; margin-bottom:1.5rem;">
        <h2 style="color:#0db8b0; margin-top:0; font-size:1.8rem;">🩺 SugarMetrics</h2>
        <p style="color:#cdd5e0; line-height:1.9; font-size:0.95rem;">
            SugarMetrics is an AI-powered diabetes early detection platform developed as a project
            in Data Science & Machine Learning. It leverages ensemble machine learning methods trained on
            768 Pima Indians Diabetes records to predict the risk of diabetes from 8 clinical indicators.
        </p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        <div style="background:#112240; border-radius:14px; padding:1.6rem; border:1px solid #1e4976; height:100%;">
            <h3 style="color:#0db8b0; margin-top:0;">🔬 Technology Stack</h3>
            <ul style="color:#cdd5e0; line-height:2.2; font-size:0.9rem;">
                <li>Python 3.x</li>
                <li>Scikit-Learn</li>
                <li>Pandas · NumPy · SciPy</li>
                <li>Matplotlib · Seaborn</li>
                <li>Streamlit</li>
                <li>Joblib (model persistence)</li>
            </ul>
        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div style="background:#112240; border-radius:14px; padding:1.6rem; border:1px solid #1e4976; height:100%;">
            <h3 style="color:#2ecc71; margin-top:0;">🤖 Models Trained</h3>
            <ul style="color:#cdd5e0; line-height:2.2; font-size:0.9rem;">
                <li>Logistic Regression</li>
                <li>Decision Tree</li>
                <li>Random Forest ⭐</li>
                <li>Gradient Boosting 🏆</li>
                <li>XGBoost-style GBM</li>
                <li>SVM · KNN · ANN</li>
                <li>LightGBM-style · AdaBoost</li>
            </ul>
        </div>""", unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div style="background:#112240; border-radius:14px; padding:1.6rem; border:1px solid #1e4976; height:100%;">
            <h3 style="color:#f39c12; margin-top:0;">📊 Performance</h3>
            <ul style="color:#cdd5e0; line-height:2.2; font-size:0.9rem;">
                <li>Best Accuracy: 96.22%</li>
                <li>Best AUC-ROC: 99.70%</li>
                <li>Best F1-Score: 96.26%</li>
                <li>Best Precision: 94.74%</li>
                <li>Best Recall: 97.83%</li>
                <li>CV (10-fold) Validated</li>
            </ul>
        </div>""", unsafe_allow_html=True)

    st.markdown("""
    <div style="background:rgba(231,76,60,0.1); border:1px solid #e74c3c; border-radius:12px;
                padding:1.2rem 1.5rem; margin-top:1.2rem;">
        <strong style="color:#e74c3c;">⚠️ Medical Disclaimer:</strong>
        <span style="color:#7eb8d4; font-size:0.88rem;">
        This tool is developed for educational and project purposes only.
        It is NOT a substitute for professional medical advice, diagnosis, or treatment.
        Always consult a qualified healthcare provider for clinical decisions.
        </span>
    </div>
    """, unsafe_allow_html=True)

# ── FOOTER ────────────────────────────────────────────────────────────────────
st.markdown("""
<hr style='border-color:#1e4976; margin-top:3rem;'>
<div style='text-align:center; color:#7eb8d4; font-size:0.82rem; padding:1rem 0;'>
    🩺 SugarMetrics · Data Science & Machine Learning ·
    Built with ❤️ using Python, Scikit-Learn & Streamlit
</div>
""", unsafe_allow_html=True)
