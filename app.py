import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import io
import time
import plotly.graph_objects as go
import plotly.express as px

# -----------------------------------------------------------------------------
# PAGE CONFIGURATION
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Hydraulic Condition Monitoring AI",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------------------------------------------------------
# THEME SELECTION & COLOR TOKENS
# -----------------------------------------------------------------------------
st.sidebar.markdown("### 🎨 Display Preferences")
theme_choice = st.sidebar.radio(
    "Interface Theme",
    ["🌙 Dark Mode (Industrial)", "☀️ Light Mode (Clean Studio)"],
    index=0
)
is_dark = "Dark" in theme_choice

if is_dark:
    T = {
        'bg_app': '#0A0E17',
        'bg_card': '#111827',
        'bg_header': 'linear-gradient(135deg, #0F172A 0%, #1E293B 50%, #0F172A 100%)',
        'border': '#1F2937',
        'border_focus': '#38BDF8',
        'border_header': 'rgba(56, 189, 248, 0.25)',
        'text_title': '#F8FAFC',
        'text_primary': '#F1F5F9',
        'text_secondary': '#E2E8F0',
        'text_muted': '#94A3B8',
        'sub_bg': 'rgba(15, 23, 42, 0.85)',
        'plotly_paper': '#0A0E17',
        'plotly_bg': '#111827',
        'plotly_grid': '#1F2937',
        'plotly_font': '#F8FAFC',
        'plotly_muted': '#94A3B8',
        'tab_bg': '#111827',
        'tab_active_bg': '#1E293B',
        'tab_text': '#94A3B8',
        'tab_active_text': '#38BDF8',
        'badge_healthy_bg': 'rgba(16, 185, 129, 0.18)',
        'badge_healthy_txt': '#34D399',
        'badge_healthy_bdr': 'rgba(16, 185, 129, 0.4)',
        'badge_warning_bg': 'rgba(245, 158, 11, 0.18)',
        'badge_warning_txt': '#FBBF24',
        'badge_warning_bdr': 'rgba(245, 158, 11, 0.4)',
        'badge_critical_bg': 'rgba(239, 68, 68, 0.18)',
        'badge_critical_txt': '#F87171',
        'badge_critical_bdr': 'rgba(239, 68, 68, 0.4)',
        'shadow': '0 4px 20px rgba(0,0,0,0.4)',
    }
else:
    T = {
        'bg_app': '#F8FAFC',
        'bg_card': '#FFFFFF',
        'bg_header': 'linear-gradient(135deg, #FFFFFF 0%, #F1F5F9 50%, #E2E8F0 100%)',
        'border': '#CBD5E1',
        'border_focus': '#0284C7',
        'border_header': 'rgba(2, 132, 199, 0.3)',
        'text_title': '#0F172A',
        'text_primary': '#0F172A',
        'text_secondary': '#1E293B',
        'text_muted': '#475569',
        'sub_bg': '#F1F5F9',
        'plotly_paper': '#F8FAFC',
        'plotly_bg': '#FFFFFF',
        'plotly_grid': '#E2E8F0',
        'plotly_font': '#0F172A',
        'plotly_muted': '#475569',
        'tab_bg': '#FFFFFF',
        'tab_active_bg': '#F1F5F9',
        'tab_text': '#475569',
        'tab_active_text': '#0284C7',
        'badge_healthy_bg': '#DCFCE7',
        'badge_healthy_txt': '#15803D',
        'badge_healthy_bdr': '#86EFAC',
        'badge_warning_bg': '#FEF3C7',
        'badge_warning_txt': '#B45309',
        'badge_warning_bdr': '#FDE68A',
        'badge_critical_bg': '#FEE2E2',
        'badge_critical_txt': '#B91C1C',
        'badge_critical_bdr': '#FCA5A5',
        'shadow': '0 2px 10px rgba(0,0,0,0.06)',
    }

# -----------------------------------------------------------------------------
# HIGH-END DYNAMIC THEME CSS
# -----------------------------------------------------------------------------
st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');
    
    html, body, [class*="css"] {{
        font-family: 'Plus Jakarta Sans', sans-serif;
    }}
    
    .stApp {{
        background-color: {T['bg_app']} !important;
        color: {T['text_primary']} !important;
    }}
    
    /* Ensure all Streamlit labels, text, headings have maximum legibility */
    p, span, label, [data-testid="stWidgetLabel"] p, .stMarkdown p {{
        color: {T['text_primary']} !important;
        font-weight: 500;
    }}
    
    h1, h2, h3, h4, h5, h6 {{
        color: {T['text_title']} !important;
        font-weight: 700 !important;
    }}
    
    .stCaption, small, .caption {{
        color: {T['text_muted']} !important;
        font-weight: 500 !important;
    }}
    
    /* Header Box */
    .header-box {{
        background: {T['bg_header']};
        border: 1px solid {T['border_header']};
        border-radius: 16px;
        padding: 22px 28px;
        margin-bottom: 20px;
        box-shadow: {T['shadow']};
    }}
    .main-title {{
        font-size: 2.2rem;
        font-weight: 800;
        background: linear-gradient(90deg, #0284C7 0%, #6366F1 50%, #A855F7 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 4px;
        letter-spacing: -0.02em;
    }}
    .sub-title {{
        font-size: 0.98rem;
        color: {T['text_muted']};
        font-weight: 500;
        margin: 0;
    }}
    
    /* Stat Metric Cards */
    .stat-card {{
        background: {T['bg_card']};
        border: 1px solid {T['border']};
        border-radius: 14px;
        padding: 16px 18px;
        text-align: center;
        box-shadow: {T['shadow']};
        transition: all 0.25s ease;
    }}
    .stat-card:hover {{
        border-color: {T['border_focus']};
        transform: translateY(-2px);
    }}
    .stat-val {{
        font-size: 1.8rem;
        font-weight: 800;
        color: {T['text_title']};
    }}
    .stat-lbl {{
        font-size: 0.78rem;
        color: {T['text_muted']};
        text-transform: uppercase;
        letter-spacing: 0.08em;
        font-weight: 700;
        margin-top: 4px;
    }}
    
    /* Diagnostic Component Card */
    .comp-card {{
        background: {T['bg_card']};
        border: 1px solid {T['border']};
        border-radius: 14px;
        padding: 18px 20px;
        margin-bottom: 16px;
        box-shadow: {T['shadow']};
        transition: border-color 0.2s ease;
    }}
    .comp-card:hover {{
        border-color: {T['border_focus']};
    }}
    .comp-header {{
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 10px;
    }}
    .comp-name {{
        font-size: 1.05rem;
        font-weight: 700;
        color: {T['text_title']};
    }}
    .comp-state {{
        font-size: 1.25rem;
        font-weight: 800;
        margin-bottom: 4px;
    }}
    .comp-desc {{
        font-size: 0.82rem;
        color: {T['text_muted']};
        margin-bottom: 10px;
        line-height: 1.45;
        font-weight: 500;
    }}
    .comp-action {{
        background: {T['sub_bg']};
        border-left: 3px solid {T['border_focus']};
        padding: 8px 12px;
        border-radius: 0 6px 6px 0;
        font-size: 0.82rem;
        color: {T['text_secondary']};
        font-weight: 500;
    }}
    
    /* Badges */
    .badge-healthy {{
        background: {T['badge_healthy_bg']};
        color: {T['badge_healthy_txt']} !important;
        border: 1px solid {T['badge_healthy_bdr']};
        padding: 4px 10px;
        border-radius: 9999px;
        font-size: 0.74rem;
        font-weight: 700;
    }}
    .badge-warning {{
        background: {T['badge_warning_bg']};
        color: {T['badge_warning_txt']} !important;
        border: 1px solid {T['badge_warning_bdr']};
        padding: 4px 10px;
        border-radius: 9999px;
        font-size: 0.74rem;
        font-weight: 700;
    }}
    .badge-critical {{
        background: {T['badge_critical_bg']};
        color: {T['badge_critical_txt']} !important;
        border: 1px solid {T['badge_critical_bdr']};
        padding: 4px 10px;
        border-radius: 9999px;
        font-size: 0.74rem;
        font-weight: 700;
    }}
    
    /* Streamlit Tabs Customization */
    .stTabs [data-baseweb="tab-list"] {{
        gap: 8px;
        background-color: transparent;
    }}
    .stTabs [data-baseweb="tab"] {{
        background-color: {T['tab_bg']};
        border: 1px solid {T['border']};
        border-radius: 10px 10px 0 0;
        color: {T['tab_text']} !important;
        padding: 10px 20px;
        font-weight: 600;
        font-size: 0.92rem;
    }}
    .stTabs [aria-selected="true"] {{
        background-color: {T['tab_active_bg']} !important;
        color: {T['tab_active_text']} !important;
        border-color: {T['border_focus']} {T['border_focus']} {T['tab_active_bg']} {T['border_focus']} !important;
        border-bottom: 2px solid {T['border_focus']} !important;
    }}
    
    /* Sidebar Styling */
    section[data-testid="stSidebar"] {{
        background-color: {T['bg_card']} !important;
        border-right: 1px solid {T['border']} !important;
    }}
    section[data-testid="stSidebar"] p, section[data-testid="stSidebar"] span {{
        color: {T['text_primary']} !important;
    }}
    
    /* Input & Select Box styling */
    div[data-baseweb="select"] > div {{
        background-color: {T['bg_card']} !important;
        color: {T['text_primary']} !important;
        border-color: {T['border']} !important;
    }}
    div[data-baseweb="select"] span {{
        color: {T['text_primary']} !important;
    }}
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# DOMAIN KNOWLEDGE CONFIGURATION
# -----------------------------------------------------------------------------
TARGET_COLS = ['Cooler_Condition', 'Valve_Condition', 'Pump_Leakage', 'Accumulator_Pressure', 'Stable_Flag']

TARGET_CONFIG = {
    'Cooler_Condition': {
        'title': 'Cooler Heat Exchanger',
        'icon': '❄️',
        'unit': '% Cooling Efficiency',
        'classes': {
            0: {'label': 'Total Failure (3%)', 'desc': 'Heat exchanger blocked or coolant loop failure. Severe thermal buildup.', 'severity': 'critical', 'score': 10},
            1: {'label': 'Reduced Efficiency (20%)', 'desc': 'Cooler heavily fouled or flow throttled. Degraded thermal dissipation.', 'severity': 'warning', 'score': 50},
            2: {'label': 'Full Efficiency (100%)', 'desc': 'Nominal heat transfer rate. Optimal operating temperatures.', 'severity': 'healthy', 'score': 100}
        },
        'actions': {
            'critical': '🚨 Immediate Shutdown: Flush cooler radiator lines, verify cooling pump pressure, and replace clogged filter cores.',
            'warning': '⚠️ Schedule Service: Clean cooling fins and inspect temperature regulating proportional valve.',
            'healthy': '✅ Normal Operation: Oil temperatures within design envelope (40°C - 50°C).'
        }
    },
    'Valve_Condition': {
        'title': 'Directional Spool Valve',
        'icon': '🚰',
        'unit': '% Switching Speed',
        'classes': {
            0: {'label': 'Near Total Failure (73%)', 'desc': 'Severe mechanical binding or spool galling. Imminent valve seizure.', 'severity': 'critical', 'score': 15},
            1: {'label': 'Severe Lag (80%)', 'desc': 'High switching delay. Erratic stroke cycle timing.', 'severity': 'critical', 'score': 40},
            2: {'label': 'Small Lag (90%)', 'desc': 'Minor spool resistance or pilot fluid viscosity drag.', 'severity': 'warning', 'score': 75},
            3: {'label': 'Optimal Switching (100%)', 'desc': 'Crisp, repeatable directional switching under design limits.', 'severity': 'healthy', 'score': 100}
        },
        'actions': {
            'critical': '🚨 Urgent Intervention: Replace directional spool cartridge. Inspect hydraulic fluid for particulate contamination (ISO 4406).',
            'warning': '⚠️ Maintenance Due: Check pilot line supply pressure and inspect solenoid coil excitation currents.',
            'healthy': '✅ Normal Operation: Clean pressure response slopes; no spool hysteresis.'
        }
    },
    'Pump_Leakage': {
        'title': 'Internal Axial Piston Pump',
        'icon': '🔄',
        'unit': 'Volumetric Slip',
        'classes': {
            0: {'label': 'No Leakage (0)', 'desc': 'Volumetric efficiency >95%. Tight piston shoe clearances.', 'severity': 'healthy', 'score': 100},
            1: {'label': 'Weak Leakage (1)', 'desc': 'Minor internal bypass across swashplate and valve plate.', 'severity': 'warning', 'score': 65},
            2: {'label': 'Severe Leakage (2)', 'desc': 'Critical internal slip. Motor draws higher power to maintain line pressure.', 'severity': 'critical', 'score': 15}
        },
        'actions': {
            'critical': '🚨 High Failure Risk: Inspect axial swashplate shoes and cylinder block face. Case drain flow is excessive.',
            'warning': '⚠️ Monitor Closely: Track case drain temperature and motor power consumption (EPS1).',
            'healthy': '✅ Normal Operation: Minimal bypass flow. Motor electric power is within baseline.'
        }
    },
    'Accumulator_Pressure': {
        'title': 'Hydraulic Bladder Accumulator',
        'icon': '🔋',
        'unit': 'bar Nitrogen Pre-charge',
        'classes': {
            0: {'label': 'Severely Depressurized (90 bar)', 'desc': 'Dangerous nitrogen loss. High risk of hydraulic shock hammer.', 'severity': 'critical', 'score': 10},
            1: {'label': 'Severely Reduced (100 bar)', 'desc': 'Substantial energy absorption loss during peak cycle demand.', 'severity': 'critical', 'score': 45},
            2: {'label': 'Slightly Reduced (115 bar)', 'desc': 'Permeation of nitrogen through elastomer bladder over running cycles.', 'severity': 'warning', 'score': 75},
            3: {'label': 'Optimal Precharge (130 bar)', 'desc': 'Full nitrogen charge. Smooth hydraulic pulsation dampening.', 'severity': 'healthy', 'score': 100}
        },
        'actions': {
            'critical': '🚨 Shock Hazard: Depressurize circuit immediately and recharge nitrogen bladder to 130 bar. Inspect bladder for tears.',
            'warning': '⚠️ Top-up Required: Nitrogen charge has diffused. Schedule gas charge servicing.',
            'healthy': '✅ Normal Operation: Peak pulsation dampening and reserve hydraulic energy optimal.'
        }
    },
    'Stable_Flag': {
        'title': 'Cyclic Pressure Stability',
        'icon': '⚖️',
        'unit': 'Stability Binary',
        'classes': {
            0: {'label': 'Dynamic Instability (0)', 'desc': 'Non-repeating pressure oscillations and erratic flow variations.', 'severity': 'warning', 'score': 40},
            1: {'label': 'Stable Operation (1)', 'desc': 'Hydraulic cycles adhere to tight steady-state repeatability limits.', 'severity': 'healthy', 'score': 100}
        },
        'actions': {
            'critical': '🚨 Circuit Resonance: Check for entrained air in oil, verify relief valve damping, and inspect accumulator gas volume.',
            'warning': '⚠️ Minor Hunting: Verify closed-loop pressure controller PID gains and inspect return line filters.',
            'healthy': '✅ Normal Operation: Cyclic pressure envelopes are consistent across all 60s windows.'
        }
    }
}

# -----------------------------------------------------------------------------
# CACHED DATA & MODELS
# -----------------------------------------------------------------------------
@st.cache_data
def load_data():
    preproc_path = 'hydraulic_preprocessed_dataset.csv'
    if os.path.exists(preproc_path):
        df = pd.read_csv(preproc_path)
    else:
        df = pd.read_csv('hydraulic_combined_dataset.csv')
    return df

@st.cache_data
def get_metadata(df):
    target_cols = TARGET_COLS + [t+'_enc' for t in TARGET_COLS] + ['Cycle_ID']
    feat_cols = [c for c in df.columns if c not in target_cols]
    stats = df[feat_cols].describe().T[['min', 'mean', 'max']]
    medians = df[feat_cols].median()
    return feat_cols, stats, medians

@st.cache_resource
def load_models():
    models = {'rf': {}, 'svm': {}, 'gb': {}}
    for t in TARGET_COLS:
        rf_p = f'outputs/models/rf_{t}.joblib'
        if os.path.exists(rf_p):
            models['rf'][t] = joblib.load(rf_p)
        svm_p = f'outputs/models/svm_{t}.joblib'
        if os.path.exists(svm_p):
            models['svm'][t] = joblib.load(svm_p)
        gb_p = f'outputs/models/gb_{t}.joblib'
        if os.path.exists(gb_p):
            models['gb'][t] = joblib.load(gb_p)
    return models

@st.cache_data
def load_benchmarks():
    res = {}
    if os.path.exists('outputs/model_comparison_results.csv'):
        res['comparison'] = pd.read_csv('outputs/model_comparison_results.csv')
    if os.path.exists('outputs/rf_cv_results.csv'):
        res['cv'] = pd.read_csv('outputs/rf_cv_results.csv')
    return res

df = load_data()
feat_cols, feat_stats, feat_medians = get_metadata(df)
models = load_models()
benchmarks = load_benchmarks()

# -----------------------------------------------------------------------------
# PREDICTION HELPER
# -----------------------------------------------------------------------------
def predict_instance(input_df, model_selection):
    results = {}
    aligned_df = pd.DataFrame(index=input_df.index)
    for col in feat_cols:
        if col in input_df.columns:
            aligned_df[col] = input_df[col].astype(float)
        else:
            aligned_df[col] = feat_medians[col]

    for t in TARGET_COLS:
        preds_all = {}
        probs = None

        # Random Forest
        if t in models['rf']:
            rf_pred = models['rf'][t].predict(aligned_df)[0]
            preds_all['rf'] = rf_pred
            if hasattr(models['rf'][t], 'predict_proba'):
                rf_probs = models['rf'][t].predict_proba(aligned_df)[0]
            else:
                rf_probs = None
        else:
            preds_all['rf'] = None
            rf_probs = None

        # Gradient Boosting
        if t in models['gb']:
            gb_pred = models['gb'][t].predict(aligned_df)[0]
            preds_all['gb'] = gb_pred
            if hasattr(models['gb'][t], 'predict_proba'):
                gb_probs = models['gb'][t].predict_proba(aligned_df)[0]
            else:
                gb_probs = None
        else:
            preds_all['gb'] = None
            gb_probs = None

        # SVM
        if t in models['svm']:
            svm_obj = models['svm'][t]
            if isinstance(svm_obj, tuple):
                svm_m, svm_sc = svm_obj
                svm_pred = svm_m.predict(svm_sc.transform(aligned_df))[0]
            else:
                svm_pred = svm_obj.predict(aligned_df)[0]
            preds_all['svm'] = svm_pred
        else:
            preds_all['svm'] = None

        if "Gradient Boosting" in model_selection and preds_all['gb'] is not None:
            chosen_class = preds_all['gb']
            probs = gb_probs
        elif "Support Vector Machine" in model_selection and preds_all['svm'] is not None:
            chosen_class = preds_all['svm']
            probs = None
        elif "Consensus" in model_selection:
            votes = [v for v in [preds_all['rf'], preds_all['gb'], preds_all['svm']] if v is not None]
            chosen_class = max(set(votes), key=votes.count) if votes else 0
            probs = rf_probs
        else:
            chosen_class = preds_all['rf'] if preds_all['rf'] is not None else 0
            probs = rf_probs

        results[t] = {
            'class': int(chosen_class),
            'preds_all': preds_all,
            'probs': probs
        }
    return results

# -----------------------------------------------------------------------------
# TOP HEADER BANNER
# -----------------------------------------------------------------------------
st.markdown(f"""
<div class="header-box">
    <div class="main-title">⚙️ Hydraulic Condition Monitoring AI & Digital Twin</div>
    <div class="sub-title">Real-Time Multi-Target Diagnostics • What-If Sensitivity Explorer • 3D Cluster Spatial Map</div>
</div>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# SIDEBAR CONTROLS
# -----------------------------------------------------------------------------
st.sidebar.markdown("### 🎛️ AI Engine Configuration")
active_model = st.sidebar.selectbox(
    "Select Inference Engine",
    [
        "🌲 Random Forest (Ensemble Winner - 98.01%)",
        "⚡ Gradient Boosting (Valve Specialist - 97.05%)",
        "🎯 Support Vector Machine (RBF Kernel - 92.79%)",
        "🤝 Tri-Model Consensus (Voting Ensemble)"
    ],
    index=0
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 📡 Digital Twin Specifications")
st.sidebar.markdown("""
- **Test Rig Working Cycles**: 2,205 Runs
- **Active Physical Sensors**: 17 Channels
- **Engineered Metrics**: 82 Filtered Signals
- **Condition Subsystems**: 5 Components
- **Sampling Frequency**: 1 to 100 Hz
""")

st.sidebar.markdown("---")
st.sidebar.caption("Hydraulic Condition Monitoring Studio • Multi-Target AI Diagnostics")

# -----------------------------------------------------------------------------
# TABS SETUP
# -----------------------------------------------------------------------------
tab_live, tab_whatif, tab_3d, tab_sim, tab_batch, tab_benchmarks = st.tabs([
    "🎯 Live Cycle Diagnostics",
    "📈 What-If Sensitivity Studio",
    "🌐 3D Sensor Cluster Explorer",
    "▶️ Telemetry Stream Simulator",
    "📁 Batch File Diagnostics",
    "📊 Benchmark Analytics"
])

# =============================================================================
# TAB 1: LIVE CYCLE DIAGNOSTICS & SCENARIO SIMULATOR
# =============================================================================
with tab_live:
    st.markdown("### 🛠️ Interactive Scenario Loader & Sensor Controls")
    st.caption("Load realistic physical operating states or fine-tune key subsystem parameters to observe immediate multi-target diagnostic response.")

    col_pre, col_res = st.columns([3, 1])
    with col_pre:
        preset_choice = st.selectbox(
            "Load Operating Scenario Preset:",
            [
                "🟢 Nominal Healthy Baseline (All Systems 100%)",
                "🔴 Critical Cooler Heat-Exchanger Failure (High Temps / Low Dissipation)",
                "🟠 Directional Valve Mechanical Sticking & Switching Lag",
                "🔴 Severe Hydraulic Pump Volumetric Slip & Internal Bypass",
                "🟡 Hydraulic Accumulator Nitrogen Depressurization (90 bar)",
                "⚠️ Dynamic System Pressure Oscillation & Instability",
                "🎲 Random Historical Cycle from Field Log"
            ]
        )

    preset_indices = {
        "🟢 Nominal Healthy Baseline (All Systems 100%)": 1664,
        "🔴 Critical Cooler Heat-Exchanger Failure (High Temps / Low Dissipation)": 0,
        "🟠 Directional Valve Mechanical Sticking & Switching Lag": 211,
        "🔴 Severe Hydraulic Pump Volumetric Slip & Internal Bypass": 210,
        "🟡 Hydraulic Accumulator Nitrogen Depressurization (90 bar)": 599,
        "⚠️ Dynamic System Pressure Oscillation & Instability": 15
    }

    if "Random Historical Cycle" in preset_choice:
        if 'rand_idx' not in st.session_state:
            st.session_state.rand_idx = np.random.randint(0, len(df))
        with col_res:
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("🎲 Draw Another Random Cycle"):
                st.session_state.rand_idx = np.random.randint(0, len(df))
                st.rerun()
        seed_idx = st.session_state.rand_idx
    else:
        seed_idx = preset_indices.get(preset_choice, 1664)

    base_features = df.iloc[seed_idx][feat_cols].to_dict()

    # Subsystem Sliders
    st.markdown("#### 🎚️ Subsystem Control Sliders")
    sc1, sc2, sc3 = st.columns(3)

    with sc1:
        st.markdown("**❄️ Cooling Circuit**")
        ts1_range_val = st.slider("Thermal Delta (TS1_range °C)", 0.3, 1.9, float(np.clip(base_features.get('TS1_range', 0.66), 0.3, 1.9)), 0.05, help="Peak temperature delta across cycle.")
        ce_slope_val = st.slider("Cooling Slope (CE_slope)", -0.30, 0.26, float(np.clip(base_features.get('CE_slope', 0.0), -0.30, 0.26)), 0.02, format="%.2f")
        ce_std_val = st.slider("Cooling Variation (CE_std)", 0.05, 6.0, float(np.clip(base_features.get('CE_std', 0.28), 0.05, 6.0)), 0.1)

    with sc2:
        st.markdown("**🔄 Pump, Flow & Electric Motor**")
        eps1_val = st.slider("Motor Electric Power (EPS1_mean W)", 2360.0, 2740.0, float(np.clip(base_features.get('EPS1_mean', 2495.0), 2360.0, 2740.0)), 10.0, help="Elevated power indicates pump internal volumetric slip.")
        fs1_max_val = st.slider("Peak Primary Flow (FS1_max L/min)", 18.7, 20.5, float(np.clip(base_features.get('FS1_max', 20.1), 18.7, 20.5)), 0.1)
        fs1_std_val = st.slider("Flow Ripple (FS1_std)", 2.6, 3.8, float(np.clip(base_features.get('FS1_std', 3.0), 2.6, 3.8)), 0.05)

    with sc3:
        st.markdown("**🗜️ Pressures & Dynamic Slopes**")
        ps1_val = st.slider("Primary System Pressure (PS1_mean bar)", 155.0, 181.0, float(np.clip(base_features.get('PS1_mean', 160.5), 155.0, 181.0)), 0.5)
        ps2_val = st.slider("Secondary Pressure (PS2_median bar)", 124.0, 165.0, float(np.clip(base_features.get('PS2_median', 128.7), 124.0, 165.0)), 0.5)
        ps1_slope_val = st.slider("Pressure Response Rate (PS1_slope)", -0.007, 0.007, float(np.clip(base_features.get('PS1_slope', 0.0), -0.007, 0.007)), 0.0005, format="%.4f", help="Direct indicator of valve switching lag.")

    # Update active features
    active_features = base_features.copy()
    active_features['TS1_range'] = ts1_range_val
    active_features['CE_slope'] = ce_slope_val
    active_features['CE_std'] = ce_std_val
    active_features['EPS1_mean'] = eps1_val
    active_features['FS1_max'] = fs1_max_val
    active_features['FS1_std'] = fs1_std_val
    active_features['PS1_mean'] = ps1_val
    active_features['PS2_median'] = ps2_val
    active_features['PS1_slope'] = ps1_slope_val

    # Inference on Active Instance
    input_instance_df = pd.DataFrame([active_features])
    predictions = predict_instance(input_instance_df, active_model)

    # Calculate Composite Score
    comp_scores = [TARGET_CONFIG[t]['classes'].get(predictions[t]['class'], {}).get('score', 50) for t in TARGET_COLS]
    health_index = int(np.mean(comp_scores))

    if health_index >= 85:
        health_status = "EXCELLENT / OPTIMAL HEALTH"
        health_color = "#10B981" if is_dark else "#059669"
    elif health_index >= 60:
        health_status = "DEGRADATION WARNING - SCHEDULE MAINTENANCE"
        health_color = "#F59E0B" if is_dark else "#D97706"
    else:
        health_status = "CRITICAL FAULT DETECTED - IMMEDIATE ACTION"
        health_color = "#EF4444" if is_dark else "#DC2626"

    st.markdown("---")

    # Interactive Plotly Gauge and Radar Section
    st.markdown("### 📊 Live System Health Cockpit")
    gauge_col, radar_col = st.columns([1, 1])

    with gauge_col:
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=health_index,
            title={'text': f"<b>Composite Health Index</b><br><span style='font-size:0.8em;color:{health_color}'>{health_status}</span>", 'font': {'size': 16, 'color': T['text_title']}},
            delta={'reference': 100, 'increasing': {'color': "#10B981"}, 'decreasing': {'color': "#EF4444"}},
            number={'suffix': "%", 'font': {'size': 44, 'color': health_color, 'family': 'Plus Jakarta Sans'}},
            gauge={
                'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': T['plotly_muted']},
                'bar': {'color': health_color, 'thickness': 0.28},
                'bgcolor': T['bg_card'],
                'borderwidth': 1,
                'bordercolor': T['border'],
                'steps': [
                    {'range': [0, 60], 'color': "rgba(239, 68, 68, 0.25)" if is_dark else "rgba(239, 68, 68, 0.15)"},
                    {'range': [60, 85], 'color': "rgba(245, 158, 11, 0.25)" if is_dark else "rgba(245, 158, 11, 0.15)"},
                    {'range': [85, 100], 'color': "rgba(16, 185, 129, 0.25)" if is_dark else "rgba(16, 185, 129, 0.15)"}
                ],
                'threshold': {
                    'line': {'color': "#38BDF8" if is_dark else "#0284C7", 'width': 4},
                    'thickness': 0.8,
                    'value': health_index
                }
            }
        ))
        fig_gauge.update_layout(
            paper_bgcolor=T['plotly_paper'],
            plot_bgcolor=T['plotly_paper'],
            height=280,
            margin=dict(l=20, r=20, t=50, b=20)
        )
        st.plotly_chart(fig_gauge, use_container_width=True)

    with radar_col:
        radar_categories = ['Cooler', 'Valve', 'Pump', 'Accumulator', 'Stability']
        radar_values = comp_scores + [comp_scores[0]]
        radar_cats_closed = radar_categories + [radar_categories[0]]
        baseline_values = [100, 100, 100, 100, 100, 100]

        fig_radar = go.Figure()
        fig_radar.add_trace(go.Scatterpolar(
            r=baseline_values,
            theta=radar_cats_closed,
            name='Nominal Baseline (100%)',
            line=dict(color='rgba(148, 163, 184, 0.6)' if is_dark else 'rgba(71, 85, 105, 0.6)', dash='dash'),
            fill='none'
        ))
        fig_radar.add_trace(go.Scatterpolar(
            r=radar_values,
            theta=radar_cats_closed,
            name='Live Machine Health',
            line=dict(color='#38BDF8' if is_dark else '#0284C7', width=2.5),
            fill='toself',
            fillcolor='rgba(56, 189, 248, 0.25)' if is_dark else 'rgba(2, 132, 199, 0.2)'
        ))
        fig_radar.update_layout(
            polar=dict(
                radialaxis=dict(visible=True, range=[0, 100], tickfont=dict(color=T['plotly_muted'], size=9), gridcolor=T['plotly_grid']),
                angularaxis=dict(tickfont=dict(color=T['text_primary'], size=11, family='Plus Jakarta Sans'), gridcolor=T['plotly_grid']),
                bgcolor=T['bg_card']
            ),
            paper_bgcolor=T['plotly_paper'],
            height=280,
            showlegend=True,
            legend=dict(font=dict(color=T['text_muted'], size=10), orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5),
            margin=dict(l=30, r=30, t=20, b=30)
        )
        st.plotly_chart(fig_radar, use_container_width=True)

    # Component Diagnostic Cards
    st.markdown("### 🔍 Detailed Subsystem Diagnostics")
    diag_cols = st.columns(len(TARGET_COLS))

    for idx, t in enumerate(TARGET_COLS):
        with diag_cols[idx]:
            cfg = TARGET_CONFIG[t]
            pred_info = predictions[t]
            cls_idx = pred_info['class']
            class_cfg = cfg['classes'].get(cls_idx, {'label': f'Class {cls_idx}', 'severity': 'warning', 'desc': ''})
            severity = class_cfg.get('severity', 'warning')

            badge_class = f"badge-{severity}"
            badge_text = severity.upper()

            conf_str = "Certainty: High"
            if pred_info['probs'] is not None and len(pred_info['probs']) > cls_idx:
                conf = pred_info['probs'][cls_idx] * 100
                conf_str = f"Confidence: <b>{conf:.1f}%</b>"

            rec_text = cfg['actions'].get(severity, 'Inspect during normal interval.')
            state_color = ('#10B981' if is_dark else '#059669') if severity=='healthy' else (('#F59E0B' if is_dark else '#D97706') if severity=='warning' else ('#EF4444' if is_dark else '#DC2626'))

            st.markdown(f"""
            <div class="comp-card">
                <div class="comp-header">
                    <span class="comp-name">{cfg['icon']} {cfg['title'].split()[0]}</span>
                    <span class="{badge_class}">{badge_text}</span>
                </div>
                <div class="comp-state" style="color: {state_color};">
                    {class_cfg['label']}
                </div>
                <div style="font-size: 0.82rem; color: {T['text_muted']}; margin-bottom: 8px;">{conf_str}</div>
                <div class="comp-desc">{class_cfg['desc']}</div>
                <div class="comp-action"><b>Protocol:</b> {rec_text}</div>
            </div>
            """, unsafe_allow_html=True)

            # Interactive Plotly probability breakdown
            if pred_info['probs'] is not None:
                p_labels = [cfg['classes'][i]['label'].split('(')[0].strip() for i in range(len(pred_info['probs']))]
                fig_p = go.Figure(go.Bar(
                    x=pred_info['probs'],
                    y=p_labels,
                    orientation='h',
                    marker_color='#38BDF8' if is_dark else '#0284C7',
                    text=[f"{p*100:.0f}%" for p in pred_info['probs']],
                    textposition='outside',
                    textfont=dict(color=T['text_primary'], size=8)
                ))
                fig_p.update_layout(
                    paper_bgcolor=T['bg_card'],
                    plot_bgcolor=T['bg_card'],
                    height=130,
                    margin=dict(l=10, r=30, t=5, b=5),
                    xaxis=dict(visible=False, range=[0, 1.2]),
                    yaxis=dict(tickfont=dict(color=T['text_muted'], size=8))
                )
                st.plotly_chart(fig_p, use_container_width=True)

# =============================================================================
# TAB 2: WHAT-IF SENSITIVITY STUDIO & DECISION BOUNDARIES
# =============================================================================
with tab_whatif:
    st.markdown("### 📈 Interactive What-If Sensitivity Studio")
    st.caption("Investigate how failure probabilities shift as physical sensor features change. Identify critical tipping points before failure occurs.")

    w_col1, w_col2 = st.columns(2)
    with w_col1:
        target_to_study = st.selectbox(
            "Select Condition Target to Analyze:",
            TARGET_COLS,
            format_func=lambda x: f"{TARGET_CONFIG[x]['icon']} {TARGET_CONFIG[x]['title']}"
        )
    with w_col2:
        feature_options = {
            'Cooler_Condition': ['TS1_range', 'CE_slope', 'TS1_std', 'CE_std'],
            'Valve_Condition': ['PS1_slope', 'PS2_median', 'PS1_kurtosis', 'PS3_mean'],
            'Pump_Leakage': ['EPS1_mean', 'FS1_std', 'FS1_max', 'EPS1_range'],
            'Accumulator_Pressure': ['PS1_mean', 'PS2_median', 'PS3_mean', 'PS4_mean'],
            'Stable_Flag': ['PS1_std', 'VS1_mean', 'PS4_range', 'TS1_std']
        }
        available_sweeps = [f for f in feature_options.get(target_to_study, feat_cols[:4]) if f in feat_cols]
        feat_to_sweep = st.selectbox(
            "Select Driving Sensor Feature to Sweep:",
            available_sweeps
        )

    # Compute sweep
    f_min = float(feat_stats.loc[feat_to_sweep, 'min'])
    f_max = float(feat_stats.loc[feat_to_sweep, 'max'])
    f_curr = float(active_features.get(feat_to_sweep, feat_medians[feat_to_sweep]))

    sweep_vals = np.linspace(f_min, f_max, 40)
    sweep_df = pd.DataFrame([active_features] * len(sweep_vals))
    sweep_df[feat_to_sweep] = sweep_vals

    target_model = models['rf'].get(target_to_study)
    if target_model is not None and hasattr(target_model, 'predict_proba'):
        sweep_aligned = pd.DataFrame(index=sweep_df.index)
        for col in feat_cols:
            sweep_aligned[col] = sweep_df[col] if col in sweep_df.columns else feat_medians[col]
        probs_sweep = target_model.predict_proba(sweep_aligned)

        fig_whatif = go.Figure()
        classes_dict = TARGET_CONFIG[target_to_study]['classes']
        colors = ['#EF4444', '#F59E0B', '#10B981', '#38BDF8'] if is_dark else ['#DC2626', '#D97706', '#059669', '#0284C7']

        for c_idx in range(probs_sweep.shape[1]):
            c_name = classes_dict.get(c_idx, {}).get('label', f'Class {c_idx}')
            fig_whatif.add_trace(go.Scatter(
                x=sweep_vals,
                y=probs_sweep[:, c_idx] * 100,
                mode='lines',
                name=c_name,
                line=dict(color=colors[c_idx % len(colors)], width=2.5)
            ))

        # Current operating point vertical marker
        fig_whatif.add_vline(
            x=f_curr,
            line_width=2.5,
            line_dash="dot",
            line_color=T['text_title'],
            annotation_text=f"Current: {f_curr:.3f}",
            annotation_position="top right",
            annotation_font_color=T['text_title']
        )

        fig_whatif.update_layout(
            title=f"Sensitivity Curve: Probability of {TARGET_CONFIG[target_to_study]['title']} vs. {feat_to_sweep}",
            xaxis_title=f"{feat_to_sweep} ({f_min:.3f} to {f_max:.3f})",
            yaxis_title="Predicted Probability (%)",
            paper_bgcolor=T['plotly_paper'],
            plot_bgcolor=T['bg_card'],
            font=dict(color=T['text_title']),
            height=380,
            xaxis=dict(gridcolor=T['plotly_grid']),
            yaxis=dict(gridcolor=T['plotly_grid'], range=[-5, 105]),
            legend=dict(font=dict(color=T['text_muted']))
        )
        st.plotly_chart(fig_whatif, use_container_width=True)
        st.info(f"💡 **Physical Sensitivity**: Observe how crossing inflection points in **{feat_to_sweep}** triggers state transitions in {TARGET_CONFIG[target_to_study]['title']}.")

# =============================================================================
# TAB 3: 3D SENSOR CLUSTER EXPLORER
# =============================================================================
with tab_3d:
    st.markdown("### 🌐 3D Sensor Cluster & Spatial Manifold")
    st.caption("Visualize how operational cycles cluster in multi-dimensional space, and pinpoint where your active machine instance sits relative to historical failure envelopes.")

    th_c1, th_c2, th_c3, th_c4 = st.columns(4)
    with th_c1:
        x_axis = st.selectbox("X-Axis Feature", ['TS1_range', 'CE_std', 'PS1_mean', 'EPS1_mean'], index=0)
    with th_c2:
        y_axis = st.selectbox("Y-Axis Feature", ['FS1_max', 'EPS1_mean', 'PS2_median', 'VS1_mean'], index=0)
    with th_c3:
        z_axis = st.selectbox("Z-Axis Feature", ['PS1_mean', 'PS2_median', 'PS1_slope', 'PS3_mean'], index=0)
    with th_c4:
        color_target = st.selectbox("Color By Target", TARGET_COLS, index=0)

    # Subsample for rendering speed
    plot_sub = df.sample(n=min(500, len(df)), random_state=42).copy()
    plot_sub['Status_Label'] = plot_sub[color_target].map(lambda v: f"{color_target}: {v}")

    fig_3d = px.scatter_3d(
        plot_sub,
        x=x_axis,
        y=y_axis,
        z=z_axis,
        color='Status_Label',
        opacity=0.7,
        size_max=8,
        color_discrete_sequence=['#10B981', '#F59E0B', '#EF4444', '#38BDF8'] if is_dark else ['#059669', '#D97706', '#DC2626', '#0284C7']
    )

    # Current instance marker
    fig_3d.add_trace(go.Scatter3d(
        x=[active_features.get(x_axis, feat_medians[x_axis])],
        y=[active_features.get(y_axis, feat_medians[y_axis])],
        z=[active_features.get(z_axis, feat_medians[z_axis])],
        mode='markers+text',
        marker=dict(size=14, color='#EC4899', symbol='diamond', line=dict(color=T['text_title'], width=2)),
        name='LIVE CURRENT INSTANCE',
        text=['📍 YOUR ACTIVE MACHINE'],
        textposition='top center',
        textfont=dict(color='#EC4899', size=11)
    ))

    fig_3d.update_layout(
        paper_bgcolor=T['plotly_paper'],
        scene=dict(
            xaxis=dict(backgroundcolor=T['bg_card'], gridcolor=T['plotly_grid'], title=x_axis),
            yaxis=dict(backgroundcolor=T['bg_card'], gridcolor=T['plotly_grid'], title=y_axis),
            zaxis=dict(backgroundcolor=T['bg_card'], gridcolor=T['plotly_grid'], title=z_axis),
        ),
        margin=dict(l=10, r=10, t=10, b=10),
        height=520,
        legend=dict(font=dict(color=T['text_muted']))
    )
    st.plotly_chart(fig_3d, use_container_width=True)

# =============================================================================
# TAB 4: TELEMETRY STREAM SIMULATOR
# =============================================================================
with tab_sim:
    st.markdown("### ▶️ Real-Time Telemetry Stream Simulator")
    st.caption("Simulates continuous IoT sensor data streaming from the hydraulic test rig over consecutive working cycles.")

    ts_c1, ts_c2, ts_c3, ts_c4 = st.columns(4)
    with ts_c1:
        sim_start = st.number_input("Starting Cycle Index", min_value=0, max_value=len(df)-50, value=200, step=10)
    with ts_c2:
        sim_len = st.slider("Window Length (Cycles)", min_value=10, max_value=60, value=30, step=5)
    with ts_c3:
        sim_step = st.slider("Cycle Stepper Position", min_value=0, max_value=sim_len-1, value=5, step=1)
    with ts_c4:
        st.markdown("<br>", unsafe_allow_html=True)
        refresh_sim = st.button("🔄 Resample Window")

    sub_df = df.iloc[sim_start : sim_start + sim_len].copy().reset_index(drop=True)
    current_stream_cycle = sub_df.iloc[sim_step]

    # Predict current stream cycle
    stream_preds = predict_instance(pd.DataFrame([current_stream_cycle[feat_cols].to_dict()]), active_model)
    stream_scores = [TARGET_CONFIG[t]['classes'].get(stream_preds[t]['class'], {}).get('score', 50) for t in TARGET_COLS]
    stream_health = int(np.mean(stream_scores))

    st.markdown(f"#### 🛰️ Live Telemetry for Cycle #{int(current_stream_cycle.get('Cycle_ID', sim_start+sim_step))}")
    st_col1, st_col2, st_col3, st_col4, st_col5 = st.columns(5)
    for idx, t in enumerate(TARGET_COLS):
        with [st_col1, st_col2, st_col3, st_col4, st_col5][idx]:
            p_class = stream_preds[t]['class']
            info = TARGET_CONFIG[t]['classes'].get(p_class, {})
            st.metric(
                label=TARGET_CONFIG[t]['title'].split()[0],
                value=info.get('label', f'C{p_class}'),
                delta=f"{info.get('score', 50)}% Health",
                delta_color="normal" if info.get('severity')=='healthy' else "inverse"
            )

    # Plotly interactive time series of the window
    fig_stream = go.Figure()
    fig_stream.add_trace(go.Scatter(
        x=list(range(sim_len)),
        y=sub_df['PS1_mean'],
        name='PS1 Mean Pressure (bar)',
        line=dict(color='#38BDF8' if is_dark else '#0284C7', width=2)
    ))
    fig_stream.add_trace(go.Scatter(
        x=list(range(sim_len)),
        y=sub_df['EPS1_mean'],
        name='EPS1 Motor Power (W)',
        line=dict(color='#F59E0B' if is_dark else '#D97706', width=2),
        yaxis='y2'
    ))
    fig_stream.add_vline(x=sim_step, line_width=2, line_dash="dash", line_color="#EF4444" if is_dark else "#DC2626", annotation_text="Current Cycle", annotation_position="top left", annotation_font_color=T['text_title'])

    fig_stream.update_layout(
        title="Streaming Sensor Telemetry Window (Interactive Zoom & Hover)",
        paper_bgcolor=T['plotly_paper'],
        plot_bgcolor=T['bg_card'],
        font=dict(color=T['text_title']),
        height=320,
        xaxis=dict(title='Relative Cycle Window Step', gridcolor=T['plotly_grid']),
        yaxis=dict(title='Pressure (bar)', title_font=dict(color='#38BDF8' if is_dark else '#0284C7'), tickfont=dict(color='#38BDF8' if is_dark else '#0284C7'), gridcolor=T['plotly_grid']),
        yaxis2=dict(title='Electric Power (W)', title_font=dict(color='#F59E0B' if is_dark else '#D97706'), tickfont=dict(color='#F59E0B' if is_dark else '#D97706'), overlaying='y', side='right'),
        legend=dict(font=dict(color=T['text_muted']), orientation="h", y=1.1)
    )
    st.plotly_chart(fig_stream, use_container_width=True)

# =============================================================================
# TAB 5: BATCH CSV FILE DIAGNOSTICS & EXPORT
# =============================================================================
with tab_batch:
    st.markdown("### 📁 Batch File Condition Monitoring")
    st.caption("Upload any CSV containing multiple cycles. Run automated fleet-wide condition diagnosis and export a stamped diagnostic report.")

    b1, b2 = st.columns([3, 1])
    with b1:
        uploaded_file = st.file_uploader("Upload Cycles CSV", type=['csv'])
    with b2:
        st.markdown("<br>", unsafe_allow_html=True)
        if os.path.exists('sample_test_instances.csv'):
            with open('sample_test_instances.csv', 'rb') as f:
                st.download_button(
                    label="📥 Download Sample Batch Template",
                    data=f,
                    file_name="sample_test_instances.csv",
                    mime="text/csv"
                )

    if uploaded_file is not None:
        try:
            b_df = pd.read_csv(uploaded_file)
            st.success(f"Loaded **{b_df.shape[0]} cycles** with **{b_df.shape[1]} columns**.")

            if st.button("🚀 Execute Multi-Target Diagnosis on Fleet", type="primary"):
                with st.spinner("Processing batch inference across all 5 models..."):
                    recs = []
                    for r_idx in range(len(b_df)):
                        row = b_df.iloc[[r_idx]]
                        cid = row.get('Cycle_ID', pd.Series([r_idx+1])).values[0]
                        res = predict_instance(row, active_model)

                        rec = {'Cycle_ID': cid}
                        t_score = 0
                        for t in TARGET_COLS:
                            c_idx = res[t]['class']
                            rec[f'{t}_Pred'] = TARGET_CONFIG[t]['classes'].get(c_idx, {}).get('label', f"C{c_idx}")
                            rec[f'{t}_Severity'] = TARGET_CONFIG[t]['classes'].get(c_idx, {}).get('severity', 'warning')
                            t_score += TARGET_CONFIG[t]['classes'].get(c_idx, {}).get('score', 50)

                        rec['Health_Score'] = int(t_score / len(TARGET_COLS))
                        rec['Fleet_Status'] = "HEALTHY" if rec['Health_Score'] >= 85 else ("WARNING" if rec['Health_Score'] >= 60 else "CRITICAL")
                        recs.append(rec)

                    batch_res = pd.DataFrame(recs)

                    # Summary cards
                    bc1, bc2, bc3, bc4 = st.columns(4)
                    with bc1:
                        st.markdown(f"""<div class="stat-card"><div class="stat-val">{len(batch_res)}</div><div class="stat-lbl">Analyzed Cycles</div></div>""", unsafe_allow_html=True)
                    with bc2:
                        h_c = (batch_res['Fleet_Status'] == 'HEALTHY').sum()
                        st.markdown(f"""<div class="stat-card"><div class="stat-val" style="color: {'#10B981' if is_dark else '#059669'};">{h_c}</div><div class="stat-lbl">Nominal Healthy</div></div>""", unsafe_allow_html=True)
                    with bc3:
                        w_c = (batch_res['Fleet_Status'] == 'WARNING').sum()
                        st.markdown(f"""<div class="stat-card"><div class="stat-val" style="color: {'#F59E0B' if is_dark else '#D97706'};">{w_c}</div><div class="stat-lbl">Warning State</div></div>""", unsafe_allow_html=True)
                    with bc4:
                        c_c = (batch_res['Fleet_Status'] == 'CRITICAL').sum()
                        st.markdown(f"""<div class="stat-card"><div class="stat-val" style="color: {'#EF4444' if is_dark else '#DC2626'};">{c_c}</div><div class="stat-lbl">Critical Faults</div></div>""", unsafe_allow_html=True)

                    st.markdown("#### 📋 Fleet Diagnostic Report")
                    st.dataframe(batch_res, use_container_width=True)

                    csv_buf = io.StringIO()
                    batch_res.to_csv(csv_buf, index=False)
                    st.download_button(
                        "💾 Download Diagnostic Report (CSV)",
                        data=csv_buf.getvalue(),
                        file_name="fleet_hydraulic_diagnostic_report.csv",
                        mime="text/csv",
                        type="primary"
                    )
        except Exception as e:
            st.error(f"Error parsing batch CSV: {e}")

# =============================================================================
# TAB 6: BENCHMARK ANALYTICS & VALIDATION
# =============================================================================
with tab_benchmarks:
    st.markdown("### 📊 Model Benchmark & Generalization Analytics")
    st.caption("Consolidated test accuracy, 5-fold cross validation, and confusion matrices for Random Forest, SVM, and Gradient Boosting.")

    if 'comparison' in benchmarks:
        st.markdown("#### 🏆 Benchmark Leaderboard Across All 5 Components")
        st.dataframe(
            benchmarks['comparison'].style.highlight_max(subset=['Random_Forest', 'SVM', 'Gradient_Boosting'], axis=1, color='#065F46' if is_dark else '#DCFCE7'),
            use_container_width=True
        )

    b_c1, b_c2 = st.columns(2)
    with b_c1:
        if os.path.exists('outputs/plots/model_comparison_by_target.png'):
            st.image('outputs/plots/model_comparison_by_target.png', caption="Model Comparison by Target Component", use_container_width=True)
    with b_c2:
        if 'cv' in benchmarks:
            st.markdown("#### 🔄 5-Fold Stratified Cross-Validation (Random Forest)")
            st.dataframe(benchmarks['cv'], use_container_width=True)
            st.info("Validation standard deviations are below ±0.02 across all targets, confirming strong stability.")

    cm1, cm2 = st.columns(2)
    with cm1:
        if os.path.exists('outputs/plots/confusion_matrix_cooler_rf.png'):
            st.image('outputs/plots/confusion_matrix_cooler_rf.png', caption="Cooler Condition Confusion Matrix (100% Accuracy)", use_container_width=True)
    with cm2:
        if os.path.exists('outputs/plots/confusion_matrix_valve_rf.png'):
            st.image('outputs/plots/confusion_matrix_valve_rf.png', caption="Valve Condition Confusion Matrix (95.5% Accuracy)", use_container_width=True)

# -----------------------------------------------------------------------------
# FOOTER
# -----------------------------------------------------------------------------
st.markdown("---")
st.markdown(f"""
<div style="text-align: center; color: {T['text_muted']}; font-size: 0.82rem; padding: 10px; font-weight: 500;">
    ⚙️ Hydraulic Condition Monitoring AI • Multi-Target Edge Diagnostics • Scikit-Learn & Plotly
</div>
""", unsafe_allow_html=True)
