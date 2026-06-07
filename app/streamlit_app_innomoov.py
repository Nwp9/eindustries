"""
Interface Streamlit — InnoMoov
Solution IA de maintenance prédictive par prédiction de RUL

À placer idéalement dans : app/streamlit_app_innomoov.py

Structure attendue :

MaintenancePredictive_Moteurs/
├── app/
│   └── streamlit_app_innomoov.py
├── data/
│   └── raw/
│       ├── test_FD001.txt
│       └── RUL_FD001.txt
├── models/
│   ├── innomoov_best_rul_model.h5
│   ├── innomoov_rul_model_metadata.json
│   └── innomoov_rul_scaler.joblib
└── src/
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from matplotlib.patches import Wedge

try:
    import tensorflow as tf
except Exception:
    tf = None

# -----------------------------------------------------------------------------
# Compatibilité Keras : anciens modèles GRU sauvegardés avec time_major
# -----------------------------------------------------------------------------
if tf is not None:
    class GRUCompat(tf.keras.layers.GRU):
        @classmethod
        def from_config(cls, config):
            config.pop("time_major", None)
            return super().from_config(config)


# -----------------------------------------------------------------------------
# Configuration générale
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="InnoMoov | Maintenance prédictive RUL",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

INNO_TEAL = "#00796B"
INNO_DARK = "#0F172A"
INNO_ORANGE = "#F97316"
INNO_GREEN = "#16A34A"
INNO_RED = "#DC2626"

# -----------------------------------------------------------------------------
# Icônes SVG InnoMoov
# -----------------------------------------------------------------------------
ICON_LOGO = """
<svg viewBox="0 0 24 24" aria-hidden="true">
  <path d="M12 2.6 20 7v10l-8 4.4L4 17V7l8-4.4Z" fill="none" stroke="currentColor" stroke-width="1.8"/>
  <path d="M8.2 15.3V8.7h2.2l1.6 3.5 1.6-3.5h2.2v6.6" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"/>
</svg>
"""

ICON_ENGINE = """
<svg viewBox="0 0 24 24" aria-hidden="true">
  <path d="M4 10h3l2-3h4l2 3h2.5A2.5 2.5 0 0 1 20 12.5V17h-3l-1.5 2h-7L7 17H4v-7Z" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linejoin="round"/>
  <path d="M7 13h3m3 0h3M10 7V4h4v3" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round"/>
</svg>
"""

ICON_RUL = """
<svg viewBox="0 0 24 24" aria-hidden="true">
  <path d="M4 17c3.5-7 7-2 10.5-9" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"/>
  <path d="M14 8h5v5" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
  <circle cx="4" cy="17" r="1.4" fill="currentColor"/>
  <circle cx="10" cy="13" r="1.4" fill="currentColor"/>
</svg>
"""

ICON_SHIELD = """
<svg viewBox="0 0 24 24" aria-hidden="true">
  <path d="M12 3.2 19 6v5.5c0 4.2-2.8 7.9-7 9.3-4.2-1.4-7-5.1-7-9.3V6l7-2.8Z" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linejoin="round"/>
  <path d="m8.5 12 2.2 2.2 4.8-5" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"/>
</svg>
"""

ICON_DECISION = """
<svg viewBox="0 0 24 24" aria-hidden="true">
  <path d="M8 4h8l2 2v14H6V6l2-2Z" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linejoin="round"/>
  <path d="M9 10h6M9 14h6M9 18h4M10 4v3h4V4" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round"/>
</svg>
"""

ICON_SIGNAL = """
<svg viewBox="0 0 24 24" aria-hidden="true">
  <path d="M3 13h3l2-6 4 12 3-9 2 3h4" fill="none" stroke="currentColor" stroke-width="1.9" stroke-linecap="round" stroke-linejoin="round"/>
</svg>
"""

ICON_CLOUD = """
<svg viewBox="0 0 24 24" aria-hidden="true">
  <path d="M7 18h10.5a3.5 3.5 0 0 0 .4-7 5.6 5.6 0 0 0-10.8-1.6A4.3 4.3 0 0 0 7 18Z" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linejoin="round"/>
  <path d="M12 12v5m0 0-2-2m2 2 2-2" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"/>
</svg>
"""

ICON_FILTER = """
<svg viewBox="0 0 24 24" aria-hidden="true">
  <path d="M4 6h16l-6 7v5l-4 2v-7L4 6Z" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linejoin="round"/>
</svg>
"""

ICON_BRAIN = """
<svg viewBox="0 0 24 24" aria-hidden="true">
  <path d="M9 4a3 3 0 0 0-3 3v1a3 3 0 0 0 0 6v1a3 3 0 0 0 5 2.2V4.8A3 3 0 0 0 9 4Zm6 0a3 3 0 0 1 3 3v1a3 3 0 0 1 0 6v1a3 3 0 0 1-5 2.2V4.8A3 3 0 0 1 15 4Z" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linejoin="round"/>
  <path d="M8 9h3m2 0h3M8 14h3m2 0h3" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round"/>
</svg>
"""

ICON_DASHBOARD = """
<svg viewBox="0 0 24 24" aria-hidden="true">
  <path d="M4 19V5h16v14H4Z" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linejoin="round"/>
  <path d="M8 16v-4m4 4V8m4 8v-6" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round"/>
</svg>
"""

ICON_CHECK = """
<svg viewBox="0 0 24 24" aria-hidden="true">
  <path d="M20 7 10 17l-5-5" fill="none" stroke="currentColor" stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round"/>
</svg>
"""

ICON_WARNING = """
<svg viewBox="0 0 24 24" aria-hidden="true">
  <path d="M12 3 22 20H2L12 3Z" fill="none" stroke="currentColor" stroke-width="1.9" stroke-linejoin="round"/>
  <path d="M12 9v5m0 3h.01" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round"/>
</svg>
"""

ICON_INFO = """
<svg viewBox="0 0 24 24" aria-hidden="true">
  <circle cx="12" cy="12" r="9" fill="none" stroke="currentColor" stroke-width="1.8"/>
  <path d="M12 10v6m0-9h.01" fill="none" stroke="currentColor" stroke-width="2.1" stroke-linecap="round"/>
</svg>
"""

# -----------------------------------------------------------------------------
# Chemins projet
# -----------------------------------------------------------------------------
def get_project_root() -> Path:
    """Retourne la racine projet en partant du fichier Streamlit."""
    current_file = Path(__file__).resolve()

    # Cas standard : app/streamlit_app_innomoov.py
    if current_file.parent.name.lower() == "app":
        return current_file.parent.parent

    # Cas où le script est lancé depuis la racine
    return current_file.parent


PROJECT_ROOT = get_project_root()
RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw"
MODELS_PATH = PROJECT_ROOT / "models"

MODEL_PATH = MODELS_PATH / "innomoov_best_rul_model.h5"
METADATA_PATH = MODELS_PATH / "innomoov_rul_model_metadata.json"
SCALER_PATH = MODELS_PATH / "innomoov_rul_scaler.joblib"


# -----------------------------------------------------------------------------
# CSS InnoMoov
# -----------------------------------------------------------------------------
def inject_css() -> None:
    st.markdown(
        f"""
        <style>
        .stApp {{
            background:
                radial-gradient(circle at top left, rgba(0,121,107,0.13), transparent 30%),
                radial-gradient(circle at top right, rgba(249,115,22,0.10), transparent 28%),
                linear-gradient(135deg, #f8fafc 0%, #eef7f6 45%, #ffffff 100%);
            color: {INNO_DARK};
        }}

        .block-container {{
            padding-top: 1.2rem;
            padding-bottom: 2rem;
            max-width: 1420px;
        }}

        h1, h2, h3 {{
            color: {INNO_DARK};
            font-weight: 850;
            letter-spacing: -0.035em;
        }}

        p, label, span, li {{
            color: #1f2937;
        }}

        section[data-testid="stSidebar"] {{
            background:
                linear-gradient(180deg, #003C3C 0%, #005F5B 52%, #0f172a 100%);
            border-right: 1px solid rgba(255,255,255,0.18);
        }}

        section[data-testid="stSidebar"] * {{
            color: white !important;
        }}

        .sidebar-logo {{
            display: flex;
            align-items: center;
            gap: 10px;
            margin: 0.4rem 0 1.2rem 0;
            padding-bottom: 1rem;
            border-bottom: 1px solid rgba(255,255,255,0.18);
        }}

        .sidebar-logo-icon {{
            width: 36px;
            height: 36px;
            color: #2dd4bf;
        }}

        .sidebar-logo-icon svg,
        .icon-svg svg,
        .kpi-icon svg,
        .status-icon svg,
        .hero-step-icon svg,
        .rule-dot-svg svg {{
            width: 100%;
            height: 100%;
            display: block;
        }}

        .sidebar-logo-text {{
            font-size: 1.45rem;
            font-weight: 900;
            letter-spacing: -0.03em;
        }}

        .sidebar-info-card {{
            background: rgba(255,255,255,0.10);
            border: 1px solid rgba(255,255,255,0.16);
            border-radius: 16px;
            padding: 14px 16px;
            margin-top: 14px;
        }}

        .sidebar-footer {{
            margin-top: 2rem;
            padding-top: 1.2rem;
            border-top: 1px solid rgba(255,255,255,0.16);
            font-size: 0.88rem;
            color: rgba(255,255,255,0.78);
        }}

        div[data-testid="stMetric"] {{
            background: rgba(255,255,255,0.96);
            border: 1px solid rgba(0, 121, 107, 0.16);
            padding: 18px 20px;
            border-radius: 20px;
            box-shadow: 0 12px 30px rgba(15, 23, 42, 0.08);
        }}

        div[data-testid="stMetric"] label {{
            color: #475569 !important;
            font-size: 0.88rem;
            font-weight: 750;
        }}

        div[data-testid="stMetric"] div {{
            color: {INNO_DARK} !important;
        }}

        .hero-card {{
            position: relative;
            overflow: hidden;
            background:
                linear-gradient(135deg, #003C3C 0%, #00796B 55%, #00A896 100%);
            border-radius: 28px;
            padding: 36px 42px;
            box-shadow: 0 22px 50px rgba(0, 121, 107, 0.26);
            color: white;
            margin-bottom: 1.5rem;
        }}

        .hero-card::after {{
            content: "";
            position: absolute;
            right: -90px;
            top: -90px;
            width: 290px;
            height: 290px;
            background: rgba(255,255,255,0.12);
            border-radius: 50%;
        }}

        .hero-illustration {{
            position: absolute;
            right: 42px;
            top: 34px;
            width: 250px;
            height: 190px;
            opacity: 0.18;
            color: #e6fffb;
            pointer-events: none;
        }}

        .hero-illustration svg {{
            width: 100%;
            height: 100%;
        }}

        .hero-kicker {{
            display: inline-block;
            padding: 7px 15px;
            border-radius: 999px;
            background: rgba(255,255,255,0.16);
            border: 1px solid rgba(255,255,255,0.28);
            color: white;
            font-size: 0.76rem;
            font-weight: 850;
            text-transform: uppercase;
            letter-spacing: 0.09em;
            margin-bottom: 14px;
        }}

        .hero-title {{
            font-size: 2.45rem;
            line-height: 1.08;
            font-weight: 950;
            margin: 0 0 14px 0;
            color: white;
            max-width: 900px;
            position: relative;
            z-index: 2;
        }}

        .hero-subtitle {{
            font-size: 1.02rem;
            line-height: 1.7;
            max-width: 900px;
            color: rgba(255,255,255,0.93);
            margin-bottom: 18px;
            position: relative;
            z-index: 2;
        }}

        .hero-badges {{
            display: flex;
            flex-wrap: wrap;
            gap: 10px;
            margin-top: 16px;
            position: relative;
            z-index: 2;
        }}

        .hero-badge {{
            display: inline-flex;
            align-items: center;
            gap: 8px;
            background: rgba(255,255,255,0.15);
            border: 1px solid rgba(255,255,255,0.24);
            color: white;
            border-radius: 999px;
            padding: 8px 13px;
            font-size: 0.82rem;
            font-weight: 800;
        }}

        .icon-svg {{
            width: 18px;
            height: 18px;
            color: currentColor;
            flex-shrink: 0;
        }}

        .mini-flow {{
            display: flex;
            flex-wrap: wrap;
            align-items: center;
            gap: 9px;
            margin-top: 16px;
            position: relative;
            z-index: 2;
        }}

        .flow-step {{
            display: inline-flex;
            align-items: center;
            gap: 8px;
            background: rgba(255,255,255,0.14);
            border: 1px solid rgba(255,255,255,0.22);
            border-radius: 12px;
            padding: 8px 11px;
            color: white;
            font-size: 0.8rem;
            font-weight: 750;
        }}

        .hero-step-icon {{
            width: 18px;
            height: 18px;
            color: white;
            flex-shrink: 0;
        }}

        .flow-arrow {{
            color: rgba(255,255,255,0.84);
            font-weight: 900;
        }}

        .section-title {{
            font-size: 1.35rem;
            font-weight: 900;
            color: {INNO_DARK};
            margin: 1.2rem 0 0.8rem 0;
        }}

        .card {{
            background: rgba(255,255,255,0.96);
            border: 1px solid rgba(0,121,107,0.15);
            border-radius: 22px;
            padding: 20px 24px;
            margin: 12px 0;
            box-shadow: 0 10px 28px rgba(15, 23, 42, 0.06);
        }}

        .muted {{
            color: #64748b;
            font-size: 0.92rem;
            font-weight: 500;
        }}

        .kpi-card {{
            background: rgba(255,255,255,0.98);
            border: 1px solid rgba(15, 23, 42, 0.08);
            border-radius: 16px;
            padding: 14px 18px;
            box-shadow: 0 10px 28px rgba(15, 23, 42, 0.06);
            min-height: 82px;
            display: flex;
            align-items: center;
            gap: 14px;
        }}

        .kpi-icon {{
            width: 54px;
            height: 54px;
            min-width: 54px;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            padding: 13px;
            font-weight: 800;
            margin-bottom: 0;
            flex-shrink: 0;
        }}
        
        .kpi-icon-blue {{
            background: rgba(59,130,246,0.10);
            color: #2563eb;
        }}

        .kpi-icon-green {{
            background: rgba(34,197,94,0.10);
            color: #16a34a;
        }}

        .kpi-icon-teal {{
            background: rgba(20,184,166,0.10);
            color: #0f766e;
        }}

        .kpi-icon-purple {{
            background: rgba(139,92,246,0.10);
            color: #7c3aed;
        }}

        .kpi-text {{
            display: flex;
            flex-direction: column;
            justify-content: center;
            min-width: 0;
            max-width: 100%;
        }}
        
        .kpi-label {{
            font-size: 0.92rem;
            color: #64748b;
            font-weight: 650;
            margin-bottom: 6px;
            white-space: nowrap;
        }}

        .kpi-value {{
            font-size: 1.18rem;
            font-weight: 850;
            color: #0f172a;
            line-height: 1.15;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }}

        .kpi-value-strong {{
            font-size: 1.65rem;
            font-weight: 850;
            color: #0f172a;
            line-height: 1.08;
            white-space: nowrap;
        }}
        
        .kpi-unit {{
            font-size: 1.05rem;
            color: #0f766e;
            font-weight: 850;
        }}
 
        .status-card {{
            border-radius: 22px;
            padding: 22px 24px;
            display: flex;
            align-items: center;
            gap: 18px;
            border: 1px solid rgba(15,23,42,0.08);
            box-shadow: 0 10px 28px rgba(15,23,42,0.06);
            margin-top: 8px;
            margin-bottom: 14px;
        }}

        .status-icon {{
            width: 62px;
            height: 62px;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            padding: 17px;
            flex-shrink: 0;
        }}

        .status-success {{
            background:
                radial-gradient(circle at right, rgba(34,197,94,0.08), transparent 30%),
                linear-gradient(135deg, rgba(34,197,94,0.12), rgba(34,197,94,0.04));
            border-left: 5px solid #16a34a;
        }}

        .status-success .status-icon {{
            background: #16a34a;
            color: white;
        }}

        .status-warning {{
            background:
                radial-gradient(circle at right, rgba(249,115,22,0.09), transparent 30%),
                linear-gradient(135deg, rgba(249,115,22,0.14), rgba(249,115,22,0.04));
            border-left: 5px solid #f97316;
        }}

        .status-warning .status-icon {{
            background: #f97316;
            color: white;
        }}

        .status-danger {{
            background:
                radial-gradient(circle at right, rgba(220,38,38,0.09), transparent 30%),
                linear-gradient(135deg, rgba(220,38,38,0.14), rgba(220,38,38,0.04));
            border-left: 5px solid #dc2626;
        }}

        .status-danger .status-icon {{
            background: #dc2626;
            color: white;
        }}

        .status-title {{
            font-size: 1.15rem;
            font-weight: 900;
            margin-bottom: 5px;
            color: #0f172a;
        }}

        .status-text {{
            font-size: 0.98rem;
            color: #334155;
            margin-bottom: 6px;
        }}

        div[data-testid="stVerticalBlockBorderWrapper"] {{
            background: #ffffff !important;
            border: 1px solid rgba(15, 23, 42, 0.08) !important;
            border-radius: 18px !important;
            box-shadow: 0 12px 30px rgba(15, 23, 42, 0.08) !important;
            padding: 18px 20px 16px 20px !important;
            min-height: 330px !important;
            overflow: hidden !important;
        }}

        div[data-testid="stVerticalBlockBorderWrapper"] > div {{
            background: transparent !important;
            border: none !important;
            box-shadow: none !important;
            padding: 0 !important;
            min-height: auto !important;
        }}

        div[data-testid="stVerticalBlockBorderWrapper"] h3 {{
            font-size: 1.05rem !important;
            font-weight: 850 !important;
            color: #0f172a !important;
            margin-top: 0 !important;
            margin-bottom: 0.8rem !important;
            line-height: 1.25 !important;
        }}

        div[data-testid="stVerticalBlockBorderWrapper"] .stPlotlyChart,
        div[data-testid="stVerticalBlockBorderWrapper"] .stPyplot {{
            margin-top: 0.2rem !important;
            margin-bottom: 0.2rem !important;
        }}

        div[data-testid="stVerticalBlockBorderWrapper"] p {{
            font-size: 0.92rem !important;
            line-height: 1.45 !important;
        }}

        div[data-testid="stVerticalBlockBorderWrapper"] .card {{
            box-shadow: none !important;
            border: 1px solid rgba(15,23,42,0.10) !important;
            border-radius: 14px !important;
            background: #ffffff !important;
        }}

        .rule-item {{
            display: grid;
            grid-template-columns: 18px 1fr auto;
            align-items: center;
            gap: 10px;
            padding: 12px 14px;
            margin: 8px 0;
            border: 1px solid rgba(15,23,42,0.10);
            border-radius: 14px;
            background: #ffffff;
        }}

        .rule-dot-svg {{
            width: 13px;
            height: 13px;
            color: currentColor;
        }}

        .rule-red {{ color: #dc2626; }}
        .rule-orange {{ color: #ea580c; }}
        .rule-green {{ color: #16a34a; }}

       /* Correction layout header */
        .hero-card {{
            min-height: 255px;
        }}

        .hero-content {{
            position: relative;
            z-index: 2;
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 28px;
        }}

        .hero-left {{
            flex: 1;
            min-width: 0;
            max-width: 860px;
        }}

        .hero-visual {{
            position: absolute;
            right: 30px;
            top: 30px;
            width: 330px;
            height: 220px;
            opacity: 0.38;
            pointer-events: none;
            z-index: 1;
        }}

        .hero-visual svg {{
            width: 100%;
            height: 100%;
        }}

        @media (max-width: 1100px) {{
            .hero-visual {{
                display: none;
            }}
        }}

        /* Onglets style dashboard */
        div[data-testid="stTabs"] {{
            margin-top: -0.2rem !important;
        }}

        div[data-testid="stTabs"] [data-baseweb="tab-list"] {{
            background: rgba(255,255,255,0.96) !important;
            border: 1px solid rgba(15,23,42,0.08) !important;
            border-radius: 12px !important;
            box-shadow: 0 10px 26px rgba(15,23,42,0.06) !important;
            padding: 0 10px !important;
            gap: 0 !important;
            width: fit-content !important;
            margin-bottom: 1rem !important;
        }}

        div[data-testid="stTabs"] [data-baseweb="tab"] {{
            height: 52px !important;
            padding: 0 26px !important;
            font-size: 1.05rem !important;
            font-weight: 750 !important;
            color: #334155 !important;
            border-right: 1px solid rgba(15,23,42,0.08) !important;
        }}

        div[data-testid="stTabs"] [data-baseweb="tab"] p {{
            font-size: 1.05rem !important;
            font-weight: 750 !important;
        }}

        div[data-testid="stTabs"] [data-baseweb="tab"]:last-child {{
            border-right: none !important;
        }}

        div[data-testid="stTabs"] [aria-selected="true"] {{
            color: #00796B !important;
            background: transparent !important;
        }}

        div[data-testid="stTabs"] [aria-selected="true"] p {{
            color: #00796B !important;
            font-weight: 900 !important;
        }}

        div[data-testid="stTabs"] [data-baseweb="tab-highlight"] {{
            background-color: #00796B !important;
            height: 3px !important;
            border-radius: 999px !important;
        }}

        /* Titre de section plus compact */
        .section-title {{
            margin-top: 0.4rem !important;
            margin-bottom: 0.8rem !important;
        }}

        /* Cartes basses : jauge, capteurs, décision */
        .panel-header {{
            font-size: 1.05rem;
            font-weight: 900;
            color: #0f172a;
            margin: 0 0 0.8rem 0;
            line-height: 1.25;
        }}
    
        .panel-subtitle {{
            font-size: 0.78rem;
            font-weight: 600;
            color: #64748b;
            margin-left: 4px;
        }}

        .decision-rule {{
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 14px;
            padding: 13px 14px;
            margin-bottom: 10px;
            border: 1px solid rgba(15,23,42,0.10);
            border-radius: 12px;
            background: #ffffff;
        }}

        .decision-left {{
            display: flex;
            align-items: flex-start;
            gap: 10px;
        }}

        .decision-dot {{
            width: 11px;
            height: 11px;
            border-radius: 50%;
            margin-top: 5px;
            flex-shrink: 0;
        }}

        .red-dot {{
            background: #ef4444;
        }}

        .orange-dot {{
            background: #f97316;
        }}

        .green-dot {{
            background: #22c55e;
        }}

        .decision-threshold {{
            font-size: 0.92rem;
            font-weight: 900;
            color: #0f172a;
            line-height: 1.2;
        }}

        .decision-risk {{
            font-size: 0.84rem;
            color: #475569;
            margin-top: 2px;
        }}

        .decision-action {{
            font-size: 0.86rem;
            font-weight: 900;
            text-align: right;
            white-space: nowrap;
        }}

        .red-text {{
            color: #dc2626;
        }}

        .orange-text {{
            color: #ea580c;
        }}

        .green-text {{
            color: #16a34a;
        }}

        div[data-testid="stVerticalBlockBorderWrapper"] .stPyplot {{
            display: flex !important;
            justify-content: center !important;
        }}

        div[data-testid="stVerticalBlockBorderWrapper"] img {{
            border-radius: 10px !important;
        }}


        </style>
        """,
        unsafe_allow_html=True,
    )


inject_css()


# -----------------------------------------------------------------------------
# Fonctions data et prédiction
# -----------------------------------------------------------------------------
FD_COLUMNS = (
    ["engine_id", "cycle"]
    + [f"setting_{i}" for i in range(1, 4)]
    + [f"sensor_{i}" for i in range(1, 22)]
)


@st.cache_resource
def load_model_and_assets() -> Tuple[object, Dict, object]:
    """Charge le modèle Keras, les métadonnées et le scaler."""
    if tf is None:
        raise RuntimeError("TensorFlow n'est pas installé dans cet environnement.")

    missing = [str(p) for p in [MODEL_PATH, METADATA_PATH, SCALER_PATH] if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Fichiers modèle manquants :\n" + "\n".join(missing)
        )

    model = tf.keras.models.load_model(
        MODEL_PATH,
        compile=False,
        custom_objects={"GRU": GRUCompat}
    )

    metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    scaler = joblib.load(SCALER_PATH)

    return model, metadata, scaler

@st.cache_data
def load_fd001_test_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Charge les fichiers test_FD001 et RUL_FD001."""
    test_path = RAW_DATA_PATH / "test_FD001.txt"
    rul_path = RAW_DATA_PATH / "RUL_FD001.txt"

    if not test_path.exists():
        raise FileNotFoundError(f"Fichier introuvable : {test_path}")

    if not rul_path.exists():
        raise FileNotFoundError(f"Fichier introuvable : {rul_path}")

    test_df = pd.read_csv(test_path, sep=r"\s+", header=None, names=FD_COLUMNS)
    rul_df = pd.read_csv(rul_path, sep=r"\s+", header=None, names=["RUL_reelle_fin_test"])

    rul_df["engine_id"] = np.arange(1, len(rul_df) + 1)

    return test_df, rul_df


def scale_test_data(test_df: pd.DataFrame, feature_cols: List[str], scaler) -> pd.DataFrame:
    """Applique le scaler sauvegardé aux variables utilisées par le modèle."""
    df_scaled = test_df.copy()

    missing_features = [col for col in feature_cols if col not in df_scaled.columns]
    if missing_features:
        raise ValueError(f"Colonnes absentes des données de test : {missing_features}")

    df_scaled[feature_cols] = scaler.transform(df_scaled[feature_cols])

    return df_scaled


def maintenance_decision(rul_pred: float) -> Tuple[str, str, str, str]:
    """Traduit une RUL prédite en niveau de risque et décision de maintenance."""
    if rul_pred <= 30:
        return (
            "Risque élevé",
            "Maintenance préventive à planifier",
            "Intervention à court terme recommandée pour éviter un arrêt non planifié.",
            "risk-high",
        )

    if rul_pred <= 60:
        return (
            "Risque modéré",
            "Surveillance renforcée",
            "Suivre l'évolution des capteurs et préparer une intervention si la tendance se dégrade.",
            "risk-medium",
        )

    return (
        "Risque faible",
        "Production autorisée",
        "Fonctionnement acceptable, aucune action immédiate requise.",
        "risk-low",
    )


def predict_engine_rul(
    engine_id: int,
    test_scaled: pd.DataFrame,
    feature_cols: List[str],
    model,
    metadata: Dict,
) -> Dict:
    """Prédit la RUL à partir des derniers cycles du moteur sélectionné."""
    sequence_length = int(metadata.get("sequence_length", 30))
    rul_max = float(metadata.get("rul_max", 125))
    model_name = metadata.get("best_model_name", "GRU")

    engine_data = test_scaled[test_scaled["engine_id"] == engine_id].sort_values("cycle")

    if len(engine_data) < sequence_length:
        raise ValueError(
            f"Le moteur {engine_id} ne dispose que de {len(engine_data)} cycles, "
            f"alors que le modèle attend {sequence_length} cycles."
        )

    last_sequence = engine_data[feature_cols].values[-sequence_length:]
    x_input = np.expand_dims(last_sequence, axis=0)

    # Sécurité si un modèle MLP a été sauvegardé
    if str(model_name).lower().startswith("mlp"):
        x_input = x_input.reshape(1, -1)

    pred = float(model.predict(x_input, verbose=0).flatten()[0])
    pred = float(np.clip(pred, 0, rul_max))

    risk, decision, recommendation, css_class = maintenance_decision(pred)

    return {
        "engine_id": engine_id,
        "predicted_rul": round(pred, 2),
        "risk": risk,
        "decision": decision,
        "recommendation": recommendation,
        "css_class": css_class,
        "model_name": model_name,
        "sequence_length": sequence_length,
    }


# -----------------------------------------------------------------------------
# Visualisations
# -----------------------------------------------------------------------------
def plot_rul_gauge(rul: float, rul_max: float = 125) -> plt.Figure:
    """Crée une jauge semi-circulaire RUL : faible RUL = risque élevé."""
    display_max = max(150, float(rul_max))
    rul_display = float(np.clip(rul, 0, display_max))

    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    ax.set_aspect("equal")
    ax.axis("off")

    def value_to_angle(value: float) -> float:
        return 180 - (value / display_max) * 180

    # Segments : rouge 0-30, orange 30-60, vert 60-display_max
    segments = [
        (0, 30, INNO_RED, "Élevé", "≤ 30"),
        (30, 60, INNO_ORANGE, "Modéré", "30 < RUL ≤ 60"),
        (60, display_max, INNO_GREEN, "Faible", "> 60"),
    ]

    outer_r = 1.0
    width = 0.18

    for start, end, color, label, sublabel in segments:
        theta1 = value_to_angle(end)
        theta2 = value_to_angle(start)
        wedge = Wedge(
            center=(0, 0),
            r=outer_r,
            theta1=theta1,
            theta2=theta2,
            width=width,
            facecolor=color,
            edgecolor="white",
            linewidth=2,
            alpha=0.92,
        )
        ax.add_patch(wedge)

        mid = (start + end) / 2
        angle = np.deg2rad(value_to_angle(mid))
        ax.text(
            0.67 * np.cos(angle),
            0.67 * np.sin(angle) - 0.03,
            label,
            ha="center",
            va="center",
            fontsize=10,
            fontweight="bold",
            color=color,
        )
        ax.text(
            0.48 * np.cos(angle),
            0.48 * np.sin(angle) - 0.08,
            sublabel,
            ha="center",
            va="center",
            fontsize=8,
            color="#334155",
        )

    # Aiguille
    angle = np.deg2rad(value_to_angle(rul_display))
    ax.plot(
        [0, 0.88 * np.cos(angle)],
        [0, 0.88 * np.sin(angle)],
        color="#111827",
        linewidth=4,
        solid_capstyle="round",
        zorder=5,
    )
    ax.scatter([0], [0], s=70, color="#111827", zorder=6)

    # Bornes
    ax.text(-1.04, -0.05, "0", ha="center", va="center", fontsize=10, color="#334155")
    ax.text(1.04, -0.05, f"{int(display_max)}", ha="center", va="center", fontsize=10, color="#334155")

    # Valeur centrale
    ax.text(
        0,
        -0.44,
        f"{rul:.1f} cycles",
        ha="center",
        va="center",
        fontsize=18,
        fontweight="bold",
        color=INNO_GREEN if rul > 60 else INNO_ORANGE if rul > 30 else INNO_RED,
    )
    ax.text(
        0,
        -0.59,
        "RUL prédite",
        ha="center",
        va="center",
        fontsize=9,
        color="#475569",
    )

    ax.set_xlim(-1.15, 1.15)
    ax.set_ylim(-0.7, 1.15)

    return fig



def plot_sensors(engine_df: pd.DataFrame, selected_sensors: List[str]) -> plt.Figure:
    """Trace l'évolution des capteurs sélectionnés."""
    fig, ax = plt.subplots(figsize=(10, 4.2))

    for sensor in selected_sensors:
        ax.plot(engine_df["cycle"], engine_df[sensor], label=sensor, linewidth=1.8)

    ax.set_title("Évolution des signaux capteurs", fontweight="bold", color=INNO_DARK)
    ax.set_xlabel("Cycle")
    ax.set_ylabel("Valeur capteur")
    ax.grid(alpha=0.22)
    ax.legend(ncol=2)

    return fig


# -----------------------------------------------------------------------------
# Interface Streamlit
# -----------------------------------------------------------------------------
st.markdown(
"""
<div class="hero-card">
<div class="hero-content">

<div class="hero-left">
<div class="hero-kicker">INNOMOOV · IA INDUSTRIELLE · MAINTENANCE PRÉDICTIVE</div>

<div class="hero-title">
Prédiction de durée de vie restante des équipements
</div>

<div class="hero-subtitle">
Cette interface estime la <b>RUL — Remaining Useful Life</b> d’un équipement critique
à partir des derniers cycles de fonctionnement. L’objectif est de transformer les signaux
capteurs en une décision claire : <b>produire, surveiller ou intervenir</b>.
</div>

<div class="hero-badges">
<span class="hero-badge">▣ Modèle GRU</span>
<span class="hero-badge">⌁ Séquences temporelles</span>
<span class="hero-badge">◉ OPC UA / MES ready</span>
<span class="hero-badge">✓ Aide à la décision</span>
</div>

<div class="mini-flow">
<span class="flow-step">⌁ Capteurs</span>
<span class="flow-arrow">→</span>
<span class="flow-step">☁ OPC UA / MQTT</span>
<span class="flow-arrow">→</span>
<span class="flow-step">▽ Prétraitement</span>
<span class="flow-arrow">→</span>
<span class="flow-step">◌ Modèle IA</span>
<span class="flow-arrow">→</span>
<span class="flow-step">▥ Dashboard maintenance</span>
</div>
</div>

<div class="hero-visual">
<svg viewBox="0 0 340 220" fill="none">
<path d="M40 40H120M120 40L160 80H240M70 145H150M150 145L185 120H285" stroke="rgba(255,255,255,0.18)" stroke-width="2"/>
<circle cx="40" cy="40" r="4" fill="rgba(255,255,255,0.35)"/>
<circle cx="120" cy="40" r="4" fill="rgba(255,255,255,0.35)"/>
<circle cx="240" cy="80" r="4" fill="rgba(255,255,255,0.35)"/>
<circle cx="70" cy="145" r="4" fill="rgba(255,255,255,0.35)"/>
<circle cx="285" cy="120" r="4" fill="rgba(255,255,255,0.35)"/>

<path d="M165 70L210 105" stroke="rgba(255,255,255,0.28)" stroke-width="12" stroke-linecap="round"/>
<path d="M210 105L255 72" stroke="rgba(255,255,255,0.28)" stroke-width="12" stroke-linecap="round"/>
<circle cx="158" cy="65" r="26" stroke="rgba(255,255,255,0.30)" stroke-width="8"/>
<circle cx="210" cy="105" r="24" stroke="rgba(255,255,255,0.30)" stroke-width="8"/>
<circle cx="260" cy="70" r="24" stroke="rgba(255,255,255,0.30)" stroke-width="8"/>
<path d="M210 128V178" stroke="rgba(255,255,255,0.28)" stroke-width="14" stroke-linecap="round"/>
<path d="M178 190H275" stroke="rgba(255,255,255,0.28)" stroke-width="16" stroke-linecap="round"/>
<path d="M150 205H300" stroke="rgba(255,255,255,0.18)" stroke-width="18" stroke-linecap="round"/>
</svg>
</div>

</div>
</div>
""",
    unsafe_allow_html=True,
)

try:
    model, metadata, scaler = load_model_and_assets()
    test_df, rul_df = load_fd001_test_data()

    feature_cols = metadata.get("feature_cols")
    if not feature_cols:
        raise ValueError("La clé 'feature_cols' est absente du fichier de métadonnées.")

    test_scaled = scale_test_data(test_df, feature_cols, scaler)

    model_name = metadata.get("best_model_name", "GRU")
    seq_len = int(metadata.get("sequence_length", 30))
    rul_max = float(metadata.get("rul_max", 125))
    metrics = metadata.get("metrics", {})

except Exception as app_error:
    st.error("Impossible d'initialiser l'application.")
    st.exception(app_error)
    st.stop()


# -----------------------------------------------------------------------------
# Sidebar
# -----------------------------------------------------------------------------
st.sidebar.markdown(
    f"""
    <div class="sidebar-logo">
        <div class="sidebar-logo-icon">{ICON_LOGO}</div>
        <div class="sidebar-logo-text">InnoMoov</div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.sidebar.markdown("## ⚙️ Paramètres")
st.sidebar.markdown("**Projet :** InnoMoov")
st.sidebar.markdown(f"**Modèle chargé :** {model_name}")
st.sidebar.markdown(f"**Fenêtre temporelle :** {seq_len} cycles")

with st.sidebar.expander("📁 Chemins chargés"):
    st.write("Projet :", str(PROJECT_ROOT))
    st.write("Modèle :", str(MODEL_PATH))
    st.write("Métadonnées :", str(METADATA_PATH))
    st.write("Scaler :", str(SCALER_PATH))
    st.write("Données :", str(RAW_DATA_PATH))

engine_ids = sorted(test_df["engine_id"].unique())

selected_engine = st.sidebar.selectbox(
    "Choisir un moteur du jeu de test",
    engine_ids,
    index=0,
)

engine_df = test_df[test_df["engine_id"] == selected_engine].sort_values("cycle").copy()

true_rul_row = rul_df[rul_df["engine_id"] == selected_engine]
true_rul = None if true_rul_row.empty else int(true_rul_row["RUL_reelle_fin_test"].iloc[0])

st.sidebar.markdown(
    f"""
    <div class="sidebar-info-card">
        <div style="display:flex; align-items:center; gap:10px; margin-bottom:8px;">
            <span class="icon-svg" style="color:#2dd4bf;">{ICON_ENGINE}</span>
            <b>Moteur sélectionné : {selected_engine}</b>
        </div>
        <div>Cycles disponibles : <b>{len(engine_df)}</b></div>
        <div>Dernier cycle : <b>{int(engine_df['cycle'].max())}</b></div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.sidebar.markdown(
    """
    <div class="sidebar-footer">
        <b>InnoMoov Platform</b><br>
        IA industrielle · Maintenance prédictive<br>
        <span style="opacity:0.68;">v1.0.0</span>
    </div>
    """,
    unsafe_allow_html=True,
)


# -----------------------------------------------------------------------------
# Onglets
# -----------------------------------------------------------------------------
tab_prediction, tab_capteurs, tab_donnees, tab_modele = st.tabs(
    [
        "⌁  Prédiction RUL",
        "⌇  Signaux capteurs",
        "▤  Données moteur",
        "ⓘ  À propos du modèle",
    ]
)

with tab_prediction:
    st.markdown(
        '<div class="section-title">Résultat de la prédiction maintenance</div>',
        unsafe_allow_html=True,
    )

    result = predict_engine_rul(
        selected_engine,
        test_scaled,
        feature_cols,
        model,
        metadata,
    )

    if result["risk"] == "Risque faible":
        status_class = "status-success"
        status_icon = ICON_CHECK
        risk_color = "#166534"
        decision_color = "#166534"
    elif result["risk"] == "Risque modéré":
        status_class = "status-warning"
        status_icon = ICON_INFO
        risk_color = "#ea580c"
        decision_color = "#ea580c"
    else:
        status_class = "status-danger"
        status_icon = ICON_WARNING
        risk_color = "#dc2626"
        decision_color = "#dc2626"

    k1, k2, k3, k4 = st.columns(4)

    with k1:
        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-icon kpi-icon-blue">{ICON_ENGINE}</div>
                <div class="kpi-text">
                    <div class="kpi-label">Moteur</div>
                    <div class="kpi-value-strong">{selected_engine}</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with k2:
        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-icon kpi-icon-teal">{ICON_RUL}</div>
                <div class="kpi-text">
                    <div class="kpi-label">RUL prédite</div>
                    <div class="kpi-value-strong">
                        {result['predicted_rul']} <span class="kpi-unit">cycles</span>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with k3:
        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-icon kpi-icon-green">{ICON_SHIELD}</div>
                <div class="kpi-text">
                    <div class="kpi-label">Niveau de risque</div>
                    <div class="kpi-value" style="color:{risk_color};">
                        {result['risk']}
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with k4:
        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-icon kpi-icon-purple">{ICON_DECISION}</div>
                <div class="kpi-text">
                    <div class="kpi-label">Décision</div>
                    <div class="kpi-value" style="color:{decision_color};">
                        {result['decision']}
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown(
        f"""
        <div class="status-card {status_class}">
            <div class="status-icon">{status_icon}</div>
            <div>
                <div class="status-title">{result['risk']} — {result['decision']}</div>
                <div class="status-text">{result['recommendation']}</div>
                <div class="muted">
                    La décision est calculée à partir des seuils :
                    RUL ≤ 30, 30 &lt; RUL ≤ 60, RUL &gt; 60.
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    p1, p2, p3 = st.columns([1.15, 1.45, 0.9])

    with p1:
        with st.container(border=True):
            st.markdown(
                """
                <div class="panel-header">
                    Lecture du risque selon la durée de vie restante (RUL)
                </div>
                """,
                unsafe_allow_html=True,
            )

            fig = plot_rul_gauge(result["predicted_rul"], rul_max=rul_max)
            st.pyplot(fig, use_container_width=True)

    with p2:
        with st.container(border=True):
            st.markdown(
                """
                <div class="panel-header">
                    Évolution des signaux capteurs
                    <span class="panel-subtitle">(derniers cycles)</span>
                </div>
                """,
                unsafe_allow_html=True,
            )

            sensors_for_panel = [
                s for s in ["sensor_11", "sensor_12", "sensor_4"]
                if s in engine_df.columns
            ]

            sensor_labels = {
                "sensor_11": "Vibration (mm/s)",
                "sensor_12": "Température (°C)",
                "sensor_4": "Courant (A)",
            }

            if sensors_for_panel:
                fig, ax = plt.subplots(figsize=(8.8, 3.8))

                for sensor in sensors_for_panel:
                    series = engine_df[sensor].astype(float)

                    if series.max() != series.min():
                        plot_series = 20 + 70 * (series - series.min()) / (series.max() - series.min())
                    else:
                        plot_series = series

                    ax.plot(
                        engine_df["cycle"],
                        plot_series,
                        label=sensor_labels.get(sensor, sensor),
                        linewidth=1.9,
                    )

                ax.set_xlabel("Cycle")
                ax.set_ylabel("Indice capteur")
                ax.grid(alpha=0.18)
                ax.legend(
                    loc="upper center",
                    ncol=len(sensors_for_panel),
                    fontsize=8,
                    frameon=False,
                )
                ax.spines[["top", "right"]].set_visible(False)

                st.pyplot(fig, use_container_width=True)
            else:
                st.info("Capteurs indisponibles pour l’affichage.")

    with p3:
        with st.container(border=True):
            st.markdown(
                """
                <div class="panel-header">
                    Décision opérationnelle
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.markdown(
                """
                <div class="muted" style="margin-bottom:14px;">
                    Règles de décision basées sur la durée de vie restante (RUL) prédite.
                </div>

                <div class="decision-rule">
                    <div class="decision-left">
                        <span class="decision-dot red-dot"></span>
                        <div>
                            <div class="decision-threshold">RUL ≤ 30</div>
                            <div class="decision-risk">Risque élevé</div>
                        </div>
                    </div>
                    <div class="decision-action red-text">Intervention</div>
                </div>

                <div class="decision-rule">
                    <div class="decision-left">
                        <span class="decision-dot orange-dot"></span>
                        <div>
                            <div class="decision-threshold">30 &lt; RUL ≤ 60</div>
                            <div class="decision-risk">Risque modéré</div>
                        </div>
                    </div>
                    <div class="decision-action orange-text">Surveillance</div>
                </div>

                <div class="decision-rule">
                    <div class="decision-left">
                        <span class="decision-dot green-dot"></span>
                        <div>
                            <div class="decision-threshold">RUL &gt; 60</div>
                            <div class="decision-risk">Risque faible</div>
                        </div>
                    </div>
                    <div class="decision-action green-text">Production autorisée</div>
                </div>

                <div class="muted" style="margin-top:14px; display:flex; align-items:center; gap:8px;">
                    <span style="display:inline-flex; width:18px; height:18px; border-radius:50%; border:1px solid #94a3b8; align-items:center; justify-content:center; font-size:12px; color:#64748b;">i</span>
                    Ces seuils sont paramétrables selon vos besoins.
                </div>
                """,
                unsafe_allow_html=True,
            )
   
    # p1, p2, p3 = st.columns([1.15, 1.45, 0.9])

    # with p1:
    #     with st.container(border=True):
    #         st.markdown("### Lecture du risque selon la durée de vie restante (RUL)")

    #         fig = plot_rul_gauge(result["predicted_rul"], rul_max=rul_max)
    #         st.pyplot(fig, use_container_width=True)

    # with p2:
    #     with st.container(border=True):
    #         st.markdown("### Évolution des signaux capteurs")

    #         sensors_for_panel = [
    #             s for s in ["sensor_11", "sensor_12", "sensor_4"]
    #             if s in engine_df.columns
    #         ]

    #         sensor_labels = {
    #             "sensor_11": "Vibration (mm/s)",
    #             "sensor_12": "Température (°C)",
    #             "sensor_4": "Courant (A)",
    #         }

    #         if sensors_for_panel:
    #             fig, ax = plt.subplots(figsize=(8, 3.6))

    #             # Normalisation visuelle uniquement pour le dashboard
    #             for sensor in sensors_for_panel:
    #                 series = engine_df[sensor].astype(float)
    #                 if series.max() != series.min():
    #                     plot_series = 20 + 70 * (series - series.min()) / (series.max() - series.min())
    #                 else:
    #                     plot_series = series
    #                 ax.plot(
    #                     engine_df["cycle"],
    #                     plot_series,
    #                     label=sensor_labels.get(sensor, sensor),
    #                     linewidth=1.8,
    #                 )

    #             ax.set_xlabel("Cycle")
    #             ax.set_ylabel("Indice capteur")
    #             ax.grid(alpha=0.22)
    #             ax.legend(
    #                 loc="upper center",
    #                 ncol=len(sensors_for_panel),
    #                 fontsize=8,
    #                 frameon=False,
    #             )
    #             ax.spines[["top", "right"]].set_visible(False)

    #             st.pyplot(fig, use_container_width=True)
    #         else:
    #             st.info("Capteurs indisponibles pour l’affichage.")
    
    # with p3:
    #     with st.container(border=True):
    #         st.markdown("### Décision opérationnelle")

    #         st.markdown(
    #             """
    #             <div class="muted" style="margin-bottom:12px;">
    #                 Règles de décision basées sur la durée de vie restante (RUL) prédite.
    #             </div>

    #             <div class="card" style="padding:12px 14px; margin:8px 0; border-radius:14px;">
    #                 <div style="display:flex; align-items:flex-start; gap:10px;">
    #                     <div style="width:10px; height:10px; border-radius:50%; background:#e74c3c; margin-top:6px;"></div>
    #                     <div style="flex:1;">
    #                         <div style="font-weight:800; color:#0f172a;">RUL ≤ 30</div>
    #                         <div class="muted">Risque élevé</div>
    #                     </div>
    #                     <div style="font-weight:800; color:#dc2626;">Intervention</div>
    #                 </div>
    #             </div>

    #             <div class="card" style="padding:12px 14px; margin:8px 0; border-radius:14px;">
    #                 <div style="display:flex; align-items:flex-start; gap:10px;">
    #                     <div style="width:10px; height:10px; border-radius:50%; background:#f39c12; margin-top:6px;"></div>
    #                     <div style="flex:1;">
    #                         <div style="font-weight:800; color:#0f172a;">30 &lt; RUL ≤ 60</div>
    #                         <div class="muted">Risque modéré</div>
    #                     </div>
    #                     <div style="font-weight:800; color:#ea580c;">Surveillance</div>
    #                 </div>
    #             </div>

    #             <div class="card" style="padding:12px 14px; margin:8px 0; border-radius:14px;">
    #                 <div style="display:flex; align-items:flex-start; gap:10px;">
    #                     <div style="width:10px; height:10px; border-radius:50%; background:#27ae60; margin-top:6px;"></div>
    #                     <div style="flex:1;">
    #                         <div style="font-weight:800; color:#0f172a;">RUL &gt; 60</div>
    #                         <div class="muted">Risque faible</div>
    #                     </div>
    #                     <div style="font-weight:800; color:#16a34a;">Production autorisée</div>
    #                 </div>
    #             </div>

    #             <div class="muted" style="margin-top:10px; display:flex; align-items:center; gap:8px;">
    #                 <span style="display:inline-flex; width:18px; height:18px; border-radius:50%; border:1px solid #94a3b8; align-items:center; justify-content:center; font-size:12px; color:#64748b;">i</span>
    #                 Ces seuils sont paramétrables selon vos besoins.
    #             </div>
    #             """,
    #             unsafe_allow_html=True,
    #         )

with tab_capteurs:
    st.markdown(
        '<div class="section-title">Évolution détaillée des signaux capteurs</div>',
        unsafe_allow_html=True,
    )

    default_sensors = [
        s for s in ["sensor_11", "sensor_12", "sensor_4", "sensor_7"]
        if s in engine_df.columns
    ]

    sensor_options = [c for c in engine_df.columns if c.startswith("sensor_")]

    selected_sensors = st.multiselect(
        "Choisir les capteurs à afficher",
        sensor_options,
        default=default_sensors[:2] if default_sensors else sensor_options[:2],
    )

    if selected_sensors:
        fig = plot_sensors(engine_df, selected_sensors)
        st.pyplot(fig, use_container_width=True)
    else:
        st.info("Sélectionne au moins un capteur à afficher.")

    st.markdown(
        """
        <div class="card">
            <b>Interprétation :</b> les variations progressives des capteurs sont utilisées par le modèle IA
            pour apprendre la dynamique de dégradation de l'équipement sur les derniers cycles.
        </div>
        """,
        unsafe_allow_html=True,
    )


with tab_donnees:
    st.markdown(
        '<div class="section-title">Données du moteur sélectionné</div>',
        unsafe_allow_html=True,
    )

    d1, d2, d3, d4 = st.columns(4)

    with d1:
        st.metric("Moteur", selected_engine)

    with d2:
        st.metric("Cycles disponibles", len(engine_df))

    with d3:
        st.metric("Dernier cycle", int(engine_df["cycle"].max()))

    with d4:
        if true_rul is not None:
            st.metric("RUL réelle fin test", f"{true_rul} cycles")
        else:
            st.metric("RUL réelle fin test", "N/A")

    st.dataframe(engine_df.tail(15), use_container_width=True)


with tab_modele:
    st.markdown(
        '<div class="section-title">À propos du modèle IA</div>',
        unsafe_allow_html=True,
    )

    m1, m2, m3 = st.columns(3)

    with m1:
        st.metric("Modèle retenu", model_name)

    with m2:
        st.metric(
            "MAE",
            f"{metrics.get('mae', 0):.2f} cycles" if metrics else "N/A",
        )

    with m3:
        st.metric(
            "R²",
            f"{metrics.get('r2', 0):.4f}" if metrics else "N/A",
        )

    st.markdown(
        f"""
        <div class="card">
            <h3>Principe de la solution</h3>
            <p>
                Le modèle analyse les <b>{seq_len} derniers cycles</b> d'un moteur
                afin d'estimer sa durée de vie restante.
                La sortie du modèle est ensuite convertie en niveau de risque
                et en décision de maintenance.
            </p>
            <ul>
                <li><b>Données terrain</b> : signaux capteurs et conditions de fonctionnement.</li>
                <li><b>Prétraitement</b> : sélection des variables et standardisation.</li>
                <li><b>Modèle IA</b> : {model_name} pour prédire la RUL.</li>
                <li><b>Décision</b> : production, surveillance ou intervention.</li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="footer-note">
            Intégration cible :
            Machines / capteurs → OPC UA / MQTT → UNS / MES →
            Prétraitement → Modèle IA RUL → Dashboard maintenance.
        </div>
        """,
        unsafe_allow_html=True,
    )