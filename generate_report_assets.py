"""
Generate publication-quality diagrams and visualizations for the
Hydraulic Condition Monitoring AI & Digital Twin Project Report.
Uses dark navy / cyan / teal industrial aesthetic.
"""

import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import pandas as pd

# Create output folder
ASSET_DIR = r'e:\ML - HCL\Final Project\outputs\report_assets'
os.makedirs(ASSET_DIR, exist_ok=True)

# Theme Colors
NAVY_BG = '#0A0E17'
CARD_BG = '#111827'
CARD_BORDER = '#1E293B'
CYAN_ACCENT = '#38BDF8'
TEAL_ACCENT = '#14B8A6'
GREEN_HEALTHY = '#10B981'
AMBER_WARNING = '#F59E0B'
RED_CRITICAL = '#EF4444'
TEXT_WHITE = '#F8FAFC'
TEXT_MUTED = '#94A3B8'
BLUE_BOX = '#1E3A8A'

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']

# =============================================================================
# 1. SYSTEM ARCHITECTURE DIAGRAM
# =============================================================================
def make_system_architecture():
    fig, ax = plt.subplots(figsize=(14, 8), facecolor=NAVY_BG)
    ax.set_facecolor(NAVY_BG)
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 8)
    ax.axis('off')

    # Title
    ax.text(7, 7.6, 'END-TO-END SYSTEM ARCHITECTURE', 
            ha='center', va='center', color=CYAN_ACCENT, fontsize=16, fontweight='bold')
    ax.text(7, 7.25, 'Physical Rig → Telemetry → Feature Engineering → Multi-Target AI → Interactive Digital Twin',
            ha='center', va='center', color=TEXT_MUTED, fontsize=10)

    # 6 Major Stages
    stages = [
        ('1. PHYSICAL TEST RIG', 
         ['Hydraulic Power Pack', 'Axial Piston Pump', 'Directional Spool Valve', 'Bladder Accumulator', 'Cooler Heat Exchanger'],
         0.6, CYAN_ACCENT),
        ('2. MULTI-SENSOR TELEMETRY',
         ['17 Physical Sensors', '6 Pressure (PS1-PS6)', '2 Volume Flow (FS1, FS2)', '4 Temperature (TS1-TS4)', 'Vibration, Efficiency, Power'],
         2.7, TEAL_ACCENT),
        ('3. FEATURE PIPELINE',
         ['60s Cycle Segmentation', '171 Statistical Features', 'Pearson Collinearity Filter', '|r| > 0.97 Cutoff Applied', '88 High-Info Features'],
         4.8, '#6366F1'),
        ('4. MULTI-TARGET ML',
         ['5 Specialized Model Heads', 'Random Forest (300 Trees)', 'Gradient Boosting (200 Trees)', 'Support Vector Machine (RBF)', 'Stratified 5-Fold CV'],
         6.9, '#8B5CF6'),
        ('5. DIAGNOSTIC ENGINE',
         ['Cooler (3 Classes)', 'Valve (4 Classes)', 'Pump Leakage (3 Classes)', 'Accumulator (4 Classes)', 'Stability Flag (2 Classes)'],
         9.0, AMBER_WARNING),
        ('6. DIGITAL TWIN APP',
         ['Streamlit Dashboard', 'Live Diagnostics Cockpit', 'What-If Sensitivity Studio', '3D Cluster & Stream Sim', 'Batch File Diagnostics'],
         11.1, GREEN_HEALTHY)
    ]

    for title, items, x_pos, accent_color in stages:
        # Box background
        rect = patches.FancyBboxPatch((x_pos, 1.2), 1.9, 5.5,
                                      boxstyle="round,pad=0.1,rounding_size=0.15",
                                      facecolor=CARD_BG, edgecolor=accent_color, linewidth=2)
        ax.add_patch(rect)

        # Header Pill
        head_rect = patches.FancyBboxPatch((x_pos + 0.05, 6.0), 1.8, 0.6,
                                           boxstyle="round,pad=0.05,rounding_size=0.1",
                                           facecolor=accent_color, edgecolor='none')
        ax.add_patch(head_rect)
        ax.text(x_pos + 0.95, 6.3, title, ha='center', va='center', color=NAVY_BG, fontsize=8.5, fontweight='bold')

        # Items
        y = 5.4
        for item in items:
            bullet = patches.Circle((x_pos + 0.25, y), 0.04, facecolor=accent_color)
            ax.add_patch(bullet)
            ax.text(x_pos + 0.38, y, item, ha='left', va='center', color=TEXT_WHITE, fontsize=7.5)
            y -= 0.85

        # Connecting Arrow to next stage
        if x_pos < 10.0:
            ax.annotate('', xy=(x_pos + 2.05, 3.9), xytext=(x_pos + 1.9, 3.9),
                        arrowprops=dict(arrowstyle="->", color=CYAN_ACCENT, lw=2.5))

    # Bottom summary badge
    summary_rect = patches.FancyBboxPatch((1.5, 0.3), 11.0, 0.6,
                                          boxstyle="round,pad=0.08,rounding_size=0.1",
                                          facecolor='#0F172A', edgecolor=CYAN_ACCENT, linewidth=1)
    ax.add_patch(summary_rect)
    ax.text(7, 0.6, 'Data Flow: 2,205 Test Cycles × 17 Sensors (1-100 Hz) → 88 Features → 15 Production Models → Real-Time Telemetry & Sensitivity AI',
            ha='center', va='center', color=CYAN_ACCENT, fontsize=9, fontweight='bold')

    plt.tight_layout()
    plt.savefig(os.path.join(ASSET_DIR, 'system_architecture.png'), dpi=300, facecolor=NAVY_BG)
    plt.close()
    print('Generated system_architecture.png')

# =============================================================================
# 2. TRADITIONAL VS AI MONITORING COMPARISON
# =============================================================================
def make_traditional_vs_ai():
    fig, ax = plt.subplots(figsize=(13, 6.5), facecolor=NAVY_BG)
    ax.set_facecolor(NAVY_BG)
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 6.5)
    ax.axis('off')

    ax.text(6.5, 6.1, 'MONITORING PARADIGM COMPARISON', ha='center', va='center', color=CYAN_ACCENT, fontsize=15, fontweight='bold')
    ax.text(6.5, 5.75, 'Traditional Rule-Based Monitoring vs. Multi-Target AI Condition Monitoring', ha='center', va='center', color=TEXT_MUTED, fontsize=9.5)

    # Left Box: Traditional
    rect_trad = patches.FancyBboxPatch((0.8, 0.8), 5.3, 4.6, boxstyle="round,pad=0.1,rounding_size=0.15",
                                       facecolor=CARD_BG, edgecolor=RED_CRITICAL, linewidth=2)
    ax.add_patch(rect_trad)
    ax.text(3.45, 5.0, 'TRADITIONAL METHODOLOGY', ha='center', va='center', color=RED_CRITICAL, fontsize=12, fontweight='bold')
    ax.text(3.45, 4.65, 'Rule-Based Thresholds & Static Alarm Limits', ha='center', va='center', color=TEXT_MUTED, fontsize=8.5)

    trad_points = [
        'Single-Sensor Thresholds: Triggered only when extreme limits exceeded',
        'Blind to Slow Degradation: Misses progressive pump slip & cooler fouling',
        'High False Alarm Rate: Spurious alarms from innocent transient spikes',
        'Isolated Telemetry: Does not fuse pressure, flow, and thermal signals',
        'Reactive Maintenance: Repairs executed after secondary mechanical damage',
        'Fixed Maintenance Schedules: Time-based servicing leads to unnecessary costs'
    ]
    y = 4.1
    for pt in trad_points:
        bullet = patches.RegularPolygon((1.3, y), 3, radius=0.08, facecolor=RED_CRITICAL)
        ax.add_patch(bullet)
        ax.text(1.55, y, pt, ha='left', va='center', color='#E2E8F0', fontsize=8)
        y -= 0.6

    # VS circle in center
    vs_circle = patches.Circle((6.5, 3.1), 0.45, facecolor='#1E293B', edgecolor=CYAN_ACCENT, linewidth=2)
    ax.add_patch(vs_circle)
    ax.text(6.5, 3.1, 'VS', ha='center', va='center', color=TEXT_WHITE, fontsize=11, fontweight='bold')

    # Right Box: AI & Digital Twin
    rect_ai = patches.FancyBboxPatch((6.9, 0.8), 5.3, 4.6, boxstyle="round,pad=0.1,rounding_size=0.15",
                                    facecolor=CARD_BG, edgecolor=GREEN_HEALTHY, linewidth=2)
    ax.add_patch(rect_ai)
    ax.text(9.55, 5.0, 'AI & DIGITAL TWIN METHODOLOGY', ha='center', va='center', color=GREEN_HEALTHY, fontsize=12, fontweight='bold')
    ax.text(9.55, 4.65, 'Multi-Sensor ML & Continuous Degradation Tracking', ha='center', va='center', color=TEXT_MUTED, fontsize=8.5)

    ai_points = [
        'Multi-Sensor Fusion: Fuses 17 pressure, flow, temp & power channels',
        'Early Incipient Detection: Catches wear at 10-20% degradation stage',
        'Exceptional Accuracy: 95.5% - 100.0% accuracy across 5 target subsystems',
        'Cross-Component Synergy: Detects simultaneous independent degradation',
        'Condition-Based Maintenance: Actions suggested based on exact wear stage',
        'Digital Twin Sensitivity: Interactive What-If exploration of failure risks'
    ]
    y = 4.1
    for pt in ai_points:
        bullet = patches.RegularPolygon((7.4, y), 3, radius=0.08, facecolor=GREEN_HEALTHY)
        ax.add_patch(bullet)
        ax.text(7.65, y, pt, ha='left', va='center', color='#E2E8F0', fontsize=8)
        y -= 0.6

    plt.tight_layout()
    plt.savefig(os.path.join(ASSET_DIR, 'traditional_vs_ai.png'), dpi=300, facecolor=NAVY_BG)
    plt.close()
    print('Generated traditional_vs_ai.png')

# =============================================================================
# 3. FEATURE ENGINEERING & COLLINEARITY REDUCTION PIPELINE
# =============================================================================
def make_feature_engineering_pipeline():
    fig, ax = plt.subplots(figsize=(13, 7), facecolor=NAVY_BG)
    ax.set_facecolor(NAVY_BG)
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 7)
    ax.axis('off')

    ax.text(6.5, 6.6, 'FEATURE ENGINEERING & COLLINEARITY FILTERING PIPELINE', 
            ha='center', va='center', color=CYAN_ACCENT, fontsize=15, fontweight='bold')
    ax.text(6.5, 6.25, 'Transforming High-Frequency Sensor Signals into High-Information Predictive Descriptors',
            ha='center', va='center', color=TEXT_MUTED, fontsize=9.5)

    steps = [
        ('Raw Signal Feeds', ['17 Sensors', '100 Hz (PS1-6, EPS1)', '10 Hz (FS1-2)', '1 Hz (TS1-4, VS1, CE, CP, SE)', '60s Repetitive Cycles'], 0.6, CYAN_ACCENT),
        ('Cycle Segmentation', ['60-Second Cycles', '2,205 Physical Runs', 'Phase Alignment', 'Repetitive Duty Cycle', 'Signal Synchronization'], 3.1, TEAL_ACCENT),
        ('Statistical Profiling', ['Mean, Median', 'Std Dev, Variance', 'Min, Max, Peak Range', 'Transient Slopes (d/dt)', '171 Total Features'], 5.6, '#6366F1'),
        ('Collinearity Pruning', ['Pearson Matrix (|r|)', 'Cutoff: |r| > 0.97', 'Upper Triangle Scan', 'Redundant Signals Purged', '83 Features Dropped'], 8.1, AMBER_WARNING),
        ('Final Feature Matrix', ['88 Optimal Features', '100% Signal Coverage', 'Zero Redundancy', 'Multi-Scale Time Windows', 'Ready for ML Ensembles'], 10.6, GREEN_HEALTHY)
    ]

    for title, items, x_pos, col in steps:
        rect = patches.FancyBboxPatch((x_pos, 1.4), 2.0, 4.4,
                                      boxstyle="round,pad=0.1,rounding_size=0.15",
                                      facecolor=CARD_BG, edgecolor=col, linewidth=2)
        ax.add_patch(rect)

        pill = patches.FancyBboxPatch((x_pos + 0.05, 5.2), 1.9, 0.5,
                                     boxstyle="round,pad=0.05,rounding_size=0.1",
                                     facecolor=col, edgecolor='none')
        ax.add_patch(pill)
        ax.text(x_pos + 1.0, 5.45, title, ha='center', va='center', color=NAVY_BG, fontsize=8.5, fontweight='bold')

        y = 4.7
        for it in items:
            bullet = patches.Circle((x_pos + 0.25, y), 0.04, facecolor=col)
            ax.add_patch(bullet)
            ax.text(x_pos + 0.38, y, it, ha='left', va='center', color=TEXT_WHITE, fontsize=7.5)
            y -= 0.75

        if x_pos < 10.0:
            ax.annotate('', xy=(x_pos + 2.15, 3.6), xytext=(x_pos + 2.0, 3.6),
                        arrowprops=dict(arrowstyle="->", color=CYAN_ACCENT, lw=2.5))

    # Bottom technical footnote card
    note_rect = patches.FancyBboxPatch((1.0, 0.3), 11.0, 0.7,
                                       boxstyle="round,pad=0.08,rounding_size=0.1",
                                       facecolor='#0F172A', edgecolor=CYAN_ACCENT, linewidth=1)
    ax.add_patch(note_rect)
    ax.text(6.5, 0.65, 'Mathematical Rule: For any pair (f_i, f_j) where Pearson |r(f_i, f_j)| > 0.97, the collinear column is removed,\n'
                       'eliminating multi-collinearity inflation while retaining 100% of physical sensor subsystem representation.',
            ha='center', va='center', color='#E2E8F0', fontsize=8.5)

    plt.tight_layout()
    plt.savefig(os.path.join(ASSET_DIR, 'feature_engineering_pipeline.png'), dpi=300, facecolor=NAVY_BG)
    plt.close()
    print('Generated feature_engineering_pipeline.png')

# =============================================================================
# 4. MULTI-TARGET PREDICTION ARCHITECTURE
# =============================================================================
def make_multi_target_architecture():
    fig, ax = plt.subplots(figsize=(14, 8), facecolor=NAVY_BG)
    ax.set_facecolor(NAVY_BG)
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 8)
    ax.axis('off')

    ax.text(7, 7.6, 'MULTI-TARGET MACHINE LEARNING PREDICTION ARCHITECTURE', 
            ha='center', va='center', color=CYAN_ACCENT, fontsize=15, fontweight='bold')
    ax.text(7, 7.25, 'Single 88-Feature Vector Dispatched to 5 Independent Specialized Classifier Heads',
            ha='center', va='center', color=TEXT_MUTED, fontsize=9.5)

    # Input Vector Box
    in_rect = patches.FancyBboxPatch((0.6, 2.5), 2.2, 3.0,
                                     boxstyle="round,pad=0.1,rounding_size=0.15",
                                     facecolor=CARD_BG, edgecolor=CYAN_ACCENT, linewidth=2)
    ax.add_patch(in_rect)
    ax.text(1.7, 5.1, 'INPUT FEATURE\nVECTOR', ha='center', va='center', color=CYAN_ACCENT, fontsize=11, fontweight='bold')
    ax.text(1.7, 4.3, '88 Engineered\nSensor Metrics\n\n(17 Channels:\nPS1-6, FS1-2,\nTS1-4, VS1,\nCE, CP, EPS1)',
            ha='center', va='center', color=TEXT_WHITE, fontsize=8)

    # 5 Target Subsystem Heads
    targets = [
        ('Cooler Condition', 'Random Forest (300 Trees)', ['100% (Full)', '20% (Reduced)', '3% (Failure)'], '100.0% Acc', 5.8, CYAN_ACCENT),
        ('Valve Condition', 'Gradient Boosting (200 Trees)', ['100% (Optimal)', '90% (Small Lag)', '80% (Severe)', '73% (Critical)'], '97.05% Acc', 4.6, TEAL_ACCENT),
        ('Pump Leakage', 'Random Forest (300 Trees)', ['0 (No Leakage)', '1 (Weak Leakage)', '2 (Severe Bypass)'], '99.55% Acc', 3.4, '#8B5CF6'),
        ('Accumulator Pressure', 'Random Forest (300 Trees)', ['130 bar (Optimal)', '115 bar (Slight)', '100 bar (Reduced)', '90 bar (Danger)'], '97.51% Acc', 2.2, AMBER_WARNING),
        ('Stability Flag', 'Random Forest (300 Trees)', ['1 (Stable Cycle)', '0 (Dynamic Instability)'], '97.51% Acc', 1.0, GREEN_HEALTHY)
    ]

    for title, model_desc, classes, acc_str, y_pos, accent in targets:
        # Arrow from input
        ax.annotate('', xy=(3.6, y_pos + 0.4), xytext=(2.9, 4.0),
                    arrowprops=dict(arrowstyle="->", color=accent, lw=1.8))

        # Model Head Box
        box = patches.FancyBboxPatch((3.7, y_pos), 4.2, 0.9,
                                    boxstyle="round,pad=0.08,rounding_size=0.1",
                                    facecolor=CARD_BG, edgecolor=accent, linewidth=1.5)
        ax.add_patch(box)
        ax.text(3.9, y_pos + 0.65, title, ha='left', va='center', color=accent, fontsize=9.5, fontweight='bold')
        ax.text(3.9, y_pos + 0.25, model_desc, ha='left', va='center', color=TEXT_MUTED, fontsize=8)

        # Accuracy Badge
        badge = patches.FancyBboxPatch((7.0, y_pos + 0.2), 0.8, 0.5,
                                      boxstyle="round,pad=0.05,rounding_size=0.08",
                                      facecolor=accent, edgecolor='none')
        ax.add_patch(badge)
        ax.text(7.4, y_pos + 0.45, acc_str, ha='center', va='center', color=NAVY_BG, fontsize=7.5, fontweight='bold')

        # Arrow to Class Output
        ax.annotate('', xy=(8.7, y_pos + 0.45), xytext=(8.0, y_pos + 0.45),
                    arrowprops=dict(arrowstyle="->", color=accent, lw=2))

        # Class Output Card
        out_box = patches.FancyBboxPatch((8.8, y_pos), 4.5, 0.9,
                                         boxstyle="round,pad=0.08,rounding_size=0.1",
                                         facecolor='#0F172A', edgecolor=accent, linewidth=1)
        ax.add_patch(out_box)
        ax.text(9.0, y_pos + 0.65, 'Target States / Classes:', ha='left', va='center', color=TEXT_MUTED, fontsize=7.5)
        cls_txt = ' | '.join(classes)
        ax.text(9.0, y_pos + 0.25, cls_txt, ha='left', va='center', color=TEXT_WHITE, fontsize=7.5, fontweight='bold')

    plt.tight_layout()
    plt.savefig(os.path.join(ASSET_DIR, 'ml_pipeline.png'), dpi=300, facecolor=NAVY_BG)
    plt.close()
    print('Generated ml_pipeline.png')

# =============================================================================
# 5. DIGITAL TWIN ARCHITECTURE
# =============================================================================
def make_digital_twin_architecture():
    fig, ax = plt.subplots(figsize=(13, 7.5), facecolor=NAVY_BG)
    ax.set_facecolor(NAVY_BG)
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 7.5)
    ax.axis('off')

    ax.text(6.5, 7.1, '5-LAYER DIGITAL TWIN ARCHITECTURE FOR HYDRAULIC SYSTEMS', 
            ha='center', va='center', color=CYAN_ACCENT, fontsize=15, fontweight='bold')
    ax.text(6.5, 6.75, 'Bi-Directional Telemetry, AI Diagnostic Synchronization & Interactive Sensitivity Testing',
            ha='center', va='center', color=TEXT_MUTED, fontsize=9.5)

    layers = [
        ('LAYER 1: PHYSICAL TEST RIG', 
         ['Hydraulic Power Unit', 'Constant Displacement Pump', 'Proportional Valve', 'Cooling System', 'Piston Accumulator'], 
         5.6, CYAN_ACCENT),
        ('LAYER 2: SENSING & TELEMETRY', 
         ['17 Physical Sensor Transducers', '1-100 Hz Continuous Data Stream', 'Cycle Boundary Segmentation', 'Signal Conditioning & Calibration'], 
         4.3, TEAL_ACCENT),
        ('LAYER 3: DIGITAL STATE SHADOW', 
         ['88 Normalized Feature Vector', 'Dynamic Telemetry Synchronization', 'Rolling History Buffers', 'Statistical Drift Monitoring'], 
         3.0, '#6366F1'),
        ('LAYER 4: AI DIAGNOSTIC ENGINE', 
         ['15 Production ML Models (RF, SVM, GB)', 'Real-Time Multi-Target Inference', 'Dynamic Confidence Probabilities', 'Automated Health Scoring (0-100)'], 
         1.7, AMBER_WARNING),
        ('LAYER 5: DIGITAL COCKPIT & WHAT-IF', 
         ['Streamlit Industrial UI', 'Interactive Scenario Sliders', '40-Step Sensitivity Sweeps', 'Component Maintenance Workorders'], 
         0.4, GREEN_HEALTHY)
    ]

    for title, items, y_pos, col in layers:
        rect = patches.FancyBboxPatch((1.2, y_pos), 10.6, 1.05,
                                      boxstyle="round,pad=0.08,rounding_size=0.1",
                                      facecolor=CARD_BG, edgecolor=col, linewidth=1.8)
        ax.add_patch(rect)

        # Title Pill
        pill = patches.FancyBboxPatch((1.3, y_pos + 0.65), 3.2, 0.35,
                                     boxstyle="round,pad=0.05,rounding_size=0.08",
                                     facecolor=col, edgecolor='none')
        ax.add_patch(pill)
        ax.text(2.9, y_pos + 0.82, title, ha='center', va='center', color=NAVY_BG, fontsize=8, fontweight='bold')

        # Bullets
        x_bullet = 4.8
        for it in items:
            ax.text(x_bullet, y_pos + 0.82 if x_bullet < 8 else y_pos + 0.35, f"• {it}", 
                    ha='left', va='center', color=TEXT_WHITE, fontsize=7.5)
            x_bullet += 3.8
            if x_bullet > 11:
                x_bullet = 1.6

        # Bi-directional arrows between layers
        if y_pos > 0.5:
            ax.annotate('', xy=(6.5, y_pos), xytext=(6.5, y_pos - 0.25),
                        arrowprops=dict(arrowstyle="<->", color=CYAN_ACCENT, lw=2))

    plt.tight_layout()
    plt.savefig(os.path.join(ASSET_DIR, 'digital_twin_architecture.png'), dpi=300, facecolor=NAVY_BG)
    plt.close()
    print('Generated digital_twin_architecture.png')

# =============================================================================
# 6. STREAMLIT DASHBOARD ARCHITECTURE
# =============================================================================
def make_dashboard_architecture():
    fig, ax = plt.subplots(figsize=(14, 7.5), facecolor=NAVY_BG)
    ax.set_facecolor(NAVY_BG)
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 7.5)
    ax.axis('off')

    ax.text(7, 7.1, 'STREAMLIT APPLICATION MODULAR ARCHITECTURE', 
            ha='center', va='center', color=CYAN_ACCENT, fontsize=15, fontweight='bold')
    ax.text(7, 6.75, '6 Comprehensive Diagnostic & Simulation Modules Powered by Cached AI Engines',
            ha='center', va='center', color=TEXT_MUTED, fontsize=9.5)

    # Core UI Controller Box
    ctrl_rect = patches.FancyBboxPatch((0.8, 5.3), 12.4, 1.1,
                                       boxstyle="round,pad=0.08,rounding_size=0.1",
                                       facecolor=CARD_BG, edgecolor=CYAN_ACCENT, linewidth=1.5)
    ax.add_patch(ctrl_rect)
    ax.text(1.1, 6.0, 'CORE APPLICATION CONTROLLER & STATE ENGINE (app.py)', 
            ha='left', va='center', color=CYAN_ACCENT, fontsize=10, fontweight='bold')
    ax.text(1.1, 5.6, 'Theme Engine (Dark/Light) | @st.cache_resource Models (RF, SVM, GB) | @st.cache_data Preprocessed Telemetry | Model Consensus Logic',
            ha='left', va='center', color=TEXT_WHITE, fontsize=8.5)

    # 6 Modules
    modules = [
        ('1. Live Cycle Diagnostics', ['Operating Scenario Presets', 'Subsystem Sliders', 'Cockpit KPI Cards', 'Composite Health Score'], 0.8, CYAN_ACCENT),
        ('2. What-If Sensitivity', ['40-Step Feature Sweeps', 'Probability Curves', 'Phase-Change Limits', 'Early Tipping Triggers'], 2.9, TEAL_ACCENT),
        ('3. 3D Cluster Explorer', ['3D Scatter Feature Space', 'Manifold Separation', 'Target Colormap Fusion', 'Interactive Rotation'], 5.0, '#6366F1'),
        ('4. Telemetry Streamer', ['Sequential Cycle Playback', 'Continuous IoT Streaming', 'Dynamic Sparklines', 'Transient Anomaly Alert'], 7.1, '#8B5CF6'),
        ('5. Batch Diagnostics', ['CSV Log File Upload', 'Parallel Multi-Row Infer', 'Automated Scoring Table', 'Downloadable CSV Report'], 9.2, AMBER_WARNING),
        ('6. Benchmark Analytics', ['Model Comparison Matrix', '5-Fold CV Distributions', 'Confusion Matrix Views', 'Feature Importance Ranks'], 11.3, GREEN_HEALTHY)
    ]

    for title, items, x_pos, col in modules:
        rect = patches.FancyBboxPatch((x_pos, 0.8), 1.9, 4.1,
                                      boxstyle="round,pad=0.08,rounding_size=0.12",
                                      facecolor=CARD_BG, edgecolor=col, linewidth=1.5)
        ax.add_patch(rect)

        pill = patches.FancyBboxPatch((x_pos + 0.05, 4.3), 1.8, 0.5,
                                     boxstyle="round,pad=0.05,rounding_size=0.08",
                                     facecolor=col, edgecolor='none')
        ax.add_patch(pill)
        ax.text(x_pos + 0.95, 4.55, title, ha='center', va='center', color=NAVY_BG, fontsize=8, fontweight='bold')

        y = 3.9
        for it in items:
            bullet = patches.Circle((x_pos + 0.25, y), 0.04, facecolor=col)
            ax.add_patch(bullet)
            ax.text(x_pos + 0.38, y, it, ha='left', va='center', color=TEXT_WHITE, fontsize=7.5)
            y -= 0.8

        # Arrow from controller
        ax.annotate('', xy=(x_pos + 0.95, 4.95), xytext=(x_pos + 0.95, 5.3),
                    arrowprops=dict(arrowstyle="->", color=col, lw=1.8))

    plt.tight_layout()
    plt.savefig(os.path.join(ASSET_DIR, 'dashboard_architecture.png'), dpi=300, facecolor=NAVY_BG)
    plt.close()
    print('Generated dashboard_architecture.png')

# =============================================================================
# 7. WHAT-IF SCENARIO WORKFLOW & USER INTERACTION
# =============================================================================
def make_whatif_workflow():
    fig, ax = plt.subplots(figsize=(13, 6), facecolor=NAVY_BG)
    ax.set_facecolor(NAVY_BG)
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 6)
    ax.axis('off')

    ax.text(6.5, 5.6, 'WHAT-IF SENSITIVITY STUDIO EXECUTION WORKFLOW', 
            ha='center', va='center', color=CYAN_ACCENT, fontsize=15, fontweight='bold')
    ax.text(6.5, 5.25, 'Algorithmic Sequence for Simulating Degradation Dynamics & Decision Boundaries',
            ha='center', va='center', color=TEXT_MUTED, fontsize=9.5)

    steps = [
        ('1. Select Target', 'User chooses subsystem to analyze (Cooler, Valve, Pump, Acc, Stability)', 0.6, CYAN_ACCENT),
        ('2. Choose Driving Metric', 'Select key sensor feature (e.g. TS1_range, EPS1_mean, PS1_slope)', 3.1, TEAL_ACCENT),
        ('3. 40-Step Grid Sweep', 'Synthesize 40 vectors from min to max, aligning with median baselines', 5.6, '#6366F1'),
        ('4. Model Inference', 'Call predict_proba() across all 40 aligned scenario vectors', 8.1, AMBER_WARNING),
        ('5. Sensitivity Plot & Alert', 'Render multi-class curves, current operating point, & warning triggers', 10.6, GREEN_HEALTHY)
    ]

    for title, desc, x_pos, col in steps:
        rect = patches.FancyBboxPatch((x_pos, 1.2), 2.0, 3.5,
                                      boxstyle="round,pad=0.08,rounding_size=0.12",
                                      facecolor=CARD_BG, edgecolor=col, linewidth=1.8)
        ax.add_patch(rect)

        pill = patches.FancyBboxPatch((x_pos + 0.05, 4.1), 1.9, 0.5,
                                     boxstyle="round,pad=0.05,rounding_size=0.08",
                                     facecolor=col, edgecolor='none')
        ax.add_patch(pill)
        ax.text(x_pos + 1.0, 4.35, title, ha='center', va='center', color=NAVY_BG, fontsize=8.5, fontweight='bold')

        ax.text(x_pos + 1.0, 2.7, desc, ha='center', va='center', color=TEXT_WHITE, fontsize=8, wrap=True)

        if x_pos < 10.0:
            ax.annotate('', xy=(x_pos + 2.15, 2.9), xytext=(x_pos + 2.0, 2.9),
                        arrowprops=dict(arrowstyle="->", color=CYAN_ACCENT, lw=2.5))

    # Bottom Callout
    c_rect = patches.FancyBboxPatch((1.0, 0.2), 11.0, 0.7,
                                    boxstyle="round,pad=0.08,rounding_size=0.1",
                                    facecolor='#0F172A', edgecolor=GREEN_HEALTHY, linewidth=1)
    ax.add_patch(c_rect)
    ax.text(6.5, 0.55, 'Operational Benefit: Allows hydraulic engineers to test "what-if" operating conditions in software,\n'
                       'predicting exactly when a component will transition into a critical failure state without physical machine stress.',
            ha='center', va='center', color='#E2E8F0', fontsize=8.5)

    plt.tight_layout()
    plt.savefig(os.path.join(ASSET_DIR, 'whatif_workflow.png'), dpi=300, facecolor=NAVY_BG)
    plt.close()
    print('Generated whatif_workflow.png')

# =============================================================================
# 8. TARGET CLASS DISTRIBUTIONS PLOT
# =============================================================================
def make_target_distributions():
    fig, axes = plt.subplots(1, 5, figsize=(16, 4.5), facecolor=NAVY_BG)
    fig.suptitle('DISTRIBUTION OF GROUND-TRUTH TEST RIG RUNS ACROSS 5 TARGET SUBSYSTEMS (N = 2,205 Cycles)',
                 color=CYAN_ACCENT, fontsize=13, fontweight='bold', y=1.03)

    target_data = [
        ('Cooler Condition', {'100%': 741, '20%': 732, '3%': 732}, [GREEN_HEALTHY, AMBER_WARNING, RED_CRITICAL]),
        ('Valve Condition', {'100%': 1125, '90%': 360, '80%': 360, '73%': 360}, [GREEN_HEALTHY, CYAN_ACCENT, AMBER_WARNING, RED_CRITICAL]),
        ('Pump Leakage', {'None (0)': 1221, 'Weak (1)': 492, 'Severe (2)': 492}, [GREEN_HEALTHY, AMBER_WARNING, RED_CRITICAL]),
        ('Accumulator Pressure', {'130 bar': 599, '115 bar': 399, '100 bar': 399, '90 bar': 808}, [GREEN_HEALTHY, CYAN_ACCENT, AMBER_WARNING, RED_CRITICAL]),
        ('Stable Flag', {'Stable (1)': 756, 'Unstable (0)': 1449}, [GREEN_HEALTHY, RED_CRITICAL])
    ]

    for idx, (title, data_dict, colors) in enumerate(target_data):
        ax = axes[idx]
        ax.set_facecolor(CARD_BG)
        bars = ax.bar(list(data_dict.keys()), list(data_dict.values()), color=colors, edgecolor=CARD_BORDER, linewidth=1.2, width=0.6)
        ax.set_title(title, color=TEXT_WHITE, fontsize=10, fontweight='bold', pad=10)
        ax.tick_params(colors=TEXT_MUTED, labelsize=7.5)
        ax.set_xticklabels(list(data_dict.keys()), rotation=30, ha='right', color=TEXT_WHITE)
        ax.grid(axis='y', color='#1F2937', linestyle='--', alpha=0.7)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color(CARD_BORDER)
        ax.spines['bottom'].set_color(CARD_BORDER)
        ax.set_ylim(0, 1600)

        for bar in bars:
            h = bar.get_height()
            pct = (h / 2205) * 100
            ax.text(bar.get_x() + bar.get_width()/2., h + 30, f'{h}\n({pct:.1f}%)',
                    ha='center', va='bottom', color=TEXT_WHITE, fontsize=7, fontweight='bold')

    plt.tight_layout()
    plt.savefig(os.path.join(ASSET_DIR, 'target_distributions.png'), dpi=300, facecolor=NAVY_BG)
    plt.close()
    print('Generated target_distributions.png')

# =============================================================================
# 9. RANDOM FOREST CROSS-VALIDATION PERFORMANCE
# =============================================================================
def make_cv_performance():
    fig, ax = plt.subplots(figsize=(10, 5), facecolor=NAVY_BG)
    ax.set_facecolor(CARD_BG)

    targets = ['Cooler Condition', 'Pump Leakage', 'Accumulator Pressure', 'Stable Flag', 'Valve Condition']
    means = [0.9972, 0.9932, 0.9717, 0.9581, 0.9365]
    stds = [0.0018, 0.0046, 0.0095, 0.0138, 0.0194]
    colors = [GREEN_HEALTHY, TEAL_ACCENT, CYAN_ACCENT, '#8B5CF6', AMBER_WARNING]

    y_pos = np.arange(len(targets))
    bars = ax.barh(y_pos, [m * 100 for m in means], xerr=[s * 100 for s in stds],
                   color=colors, edgecolor=CARD_BORDER, height=0.55,
                   capsize=6, error_kw=dict(ecolor=TEXT_WHITE, lw=1.8, capthick=1.8))

    ax.set_yticks(y_pos)
    ax.set_yticklabels(targets, color=TEXT_WHITE, fontsize=9.5, fontweight='bold')
    ax.set_xlim(88, 101)
    ax.set_xlabel('5-Fold Stratified Cross-Validation Accuracy (%) ± 1 Std Dev', color=TEXT_MUTED, fontsize=9, labelpad=8)
    ax.set_title('RANDOM FOREST 5-FOLD CROSS-VALIDATION ACCURACY & FOLD STABILITY',
                 color=CYAN_ACCENT, fontsize=12, fontweight='bold', pad=12)
    ax.grid(axis='x', color='#1F2937', linestyle='--', alpha=0.7)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color(CARD_BORDER)
    ax.spines['bottom'].set_color(CARD_BORDER)
    ax.tick_params(colors=TEXT_MUTED)

    for idx, bar in enumerate(bars):
        w = bar.get_width()
        m_val = means[idx] * 100
        s_val = stds[idx] * 100
        ax.text(w + 0.8, bar.get_y() + bar.get_height()/2., f'{m_val:.2f}% ± {s_val:.2f}%',
                ha='left', va='center', color=TEXT_WHITE, fontsize=8.5, fontweight='bold')

    plt.tight_layout()
    plt.savefig(os.path.join(ASSET_DIR, 'rf_cv_performance.png'), dpi=300, facecolor=NAVY_BG)
    plt.close()
    print('Generated rf_cv_performance.png')

# Run all generator functions
if __name__ == '__main__':
    print('Generating all report diagram assets...')
    make_system_architecture()
    make_traditional_vs_ai()
    make_feature_engineering_pipeline()
    make_multi_target_architecture()
    make_digital_twin_architecture()
    make_dashboard_architecture()
    make_whatif_workflow()
    make_target_distributions()
    make_cv_performance()
    print('All report assets successfully generated!')
