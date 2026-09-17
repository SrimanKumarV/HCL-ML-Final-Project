# ⚙️ Hydraulic System Multi-Target Condition Monitoring AI

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.35%2B-FF4B4B.svg)](https://streamlit.io/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3%2B-F7931E.svg)](https://scikit-learn.org/)
[![Plotly](https://img.shields.io/badge/Plotly-7.1%2B-3F4F75.svg)](https://plotly.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An industrial-grade predictive maintenance system and interactive digital twin for multi-component condition monitoring of hydraulic test-rig power units. Built using sensor data from 2,205 working cycles across 17 physical channels.

---

## 🌟 Key Features

- **Multi-Target Classification**: Simultaneously diagnoses 5 independent subsystems:
  - ❄️ **Cooler Condition** (Full Efficiency, Reduced, Total Failure) — *100.0% Accuracy*
  - 🚰 **Directional Valve** (Optimal, Small Lag, Severe Lag, Near Failure) — *97.1% Accuracy*
  - 🔄 **Hydraulic Pump Leakage** (No Leakage, Weak Slip, Severe Slip) — *99.6% Accuracy*
  - 🔋 **Accumulator Pressure** (130 bar, 115 bar, 100 bar, 90 bar) — *97.5% Accuracy*
  - ⚖️ **System Stability Flag** (Stable, Dynamic Instability) — *97.5% Accuracy*
- **Interactive Digital Twin Dashboard (`app.py`)**:
  - Speedometer Gauge & 5-Axis Radar Health Polygon
  - What-If Sensitivity Studio with decision boundary transition curves
  - 3D Sensor Cluster Explorer with live instance tracking
  - Continuous Telemetry Stream Simulator with dual-axis strip charts
  - Batch CSV Fleet Diagnostics with automated report generation
- **High-Contrast Dual Theme**: Complete support for both Industrial Dark Mode and Clean Studio Light Mode.

---

## 📊 Benchmark Results

| Condition Target | Random Forest | SVM (RBF) | Gradient Boosting | Winning Model |
| :--- | :---: | :---: | :---: | :---: |
| **Cooler Condition** | **1.0000** | **1.0000** | **1.0000** | **Tie (100.0%)** |
| **Valve Condition** | 0.9546 | 0.7982 | **0.9705** | **Gradient Boosting (97.1%)** |
| **Pump Leakage** | **0.9955** | 0.9864 | 0.9887 | **Random Forest (99.6%)** |
| **Accumulator Pressure** | **0.9751** | 0.8934 | 0.9637 | **Random Forest (97.5%)** |
| **Stable Flag** | **0.9751** | 0.9615 | 0.9728 | **Random Forest (97.5%)** |
| **Mean Accuracy** | **0.9801 (98.01%)** | **0.9279 (92.79%)** | **0.9791 (97.91%)** | **Random Forest** |

---

## 🚀 Quick Start (Local)

1. **Clone the repository:**
   ```bash
   git clone https://github.com/SrimanKumarV/HCL-ML-Final-Project.git
   cd HCL-ML-Final-Project
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Launch the interactive Streamlit dashboard:**
   ```bash
   streamlit run app.py
   ```

4. **Re-train or evaluate models:**
   ```bash
   python run_pipeline.py
   ```

---

## 🌐 Deploy to Streamlit Cloud

1. Fork or push this repository to your GitHub account.
2. Sign in to [share.streamlit.io](https://share.streamlit.io/) with your GitHub account.
3. Click **"New app"**, then select:
   - **Repository:** `SrimanKumarV/HCL-ML-Final-Project`
   - **Branch:** `main`
   - **Main file path:** `app.py`
4. Click **Deploy!**

---

## 🛡️ Prevent Sleep on Free Tier

Streamlit Community Cloud puts apps to sleep after inactivity. This repository includes two built-in solutions:
1. **GitHub Actions Auto-Pinger (`.github/workflows/keep_alive.yml`)**: Pings your app URL twice daily. Add your deployed app URL to repository secrets as `STREAMLIT_APP_URL`.
2. **External Uptime Monitor**: Add your URL to [Cron-job.org](https://cron-job.org/) or [UptimeRobot](https://uptimerobot.com/) to ping every 30 minutes.
