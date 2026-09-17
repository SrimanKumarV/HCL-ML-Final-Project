import os
import sys
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

def main():
    print("=" * 70)
    print("HYDRAULIC CONDITION MONITORING - PIPELINE EXECUTION")
    print("=" * 70)
    start_time = time.time()

    # 1. Load Data
    data_path = 'hydraulic_combined_dataset.csv'
    print(f"\n[1/7] Loading dataset from: {data_path}")
    df = pd.read_csv(data_path)
    print(f"Dataset shape: {df.shape[0]} cycles (rows), {df.shape[1]} columns")
    print(f"Missing values: {df.isnull().sum().sum()}")
    print(f"Duplicate rows: {df.duplicated().sum()}")

    TARGET_COLS = ['Cooler_Condition', 'Valve_Condition', 'Pump_Leakage', 'Accumulator_Pressure', 'Stable_Flag']
    FEATURE_COLS = [c for c in df.columns if c not in TARGET_COLS + ['Cycle_ID']]
    print(f"Raw sensor feature columns: {len(FEATURE_COLS)}")
    print(f"Target condition columns: {len(TARGET_COLS)} ({', '.join(TARGET_COLS)})")

    # 2. Variance and Correlation Filtering
    print("\n[2/7] Preprocessing & Feature Selection...")
    variances = df[FEATURE_COLS].var()
    zero_var_cols = variances[variances < 1e-8].index.tolist()
    print(f"Zero/near-zero variance features: {len(zero_var_cols)}")

    corr_matrix = df[FEATURE_COLS].corr().abs()
    upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [col for col in upper_tri.columns if any(upper_tri[col] > 0.97)]
    print(f"Features dropped due to collinearity (|r| > 0.97): {len(to_drop)}")
    
    FEATURE_COLS_REDUCED = [c for c in FEATURE_COLS if c not in to_drop]
    print(f"Remaining high-value feature count: {len(FEATURE_COLS_REDUCED)}")

    # 3. Label Encoding
    encoders = {}
    df_encoded = df.copy()
    for t in TARGET_COLS:
        le = LabelEncoder()
        df_encoded[t + '_enc'] = le.fit_transform(df[t])
        encoders[t] = le
        mapping = dict(zip(le.classes_, le.transform(le.classes_)))
        print(f"  Target '{t}': {mapping}")

    # Save preprocessed dataset
    final_cols = ['Cycle_ID'] + FEATURE_COLS_REDUCED + TARGET_COLS + [t+'_enc' for t in TARGET_COLS]
    df_final = df_encoded[final_cols]
    df_final.to_csv('hydraulic_preprocessed_dataset.csv', index=False)
    print("Saved preprocessed dataset -> 'hydraulic_preprocessed_dataset.csv'")

    # Create directories for artifacts
    os.makedirs('outputs/plots', exist_ok=True)
    os.makedirs('outputs/models', exist_ok=True)

    # 4. Correlation Heatmap Plot
    plt.figure(figsize=(10, 8))
    sns.heatmap(df[FEATURE_COLS[:20]].corr(), cmap='coolwarm', center=0)
    plt.title('Correlation Heatmap (First 20 Engineered Features)')
    plt.tight_layout()
    plt.savefig('outputs/plots/correlation_first_20_features.png', dpi=200)
    plt.close()

    # 5. Model Training & Evaluation Across All 5 Targets
    print("\n[3/7] Setting up Stratified Splits (80/20 train/test)...")
    X = df_encoded[FEATURE_COLS_REDUCED]
    splits = {}
    scaled_splits = {}
    for t in TARGET_COLS:
        y_t = df_encoded[t + '_enc']
        Xtr, Xte, ytr, yte = train_test_split(X, y_t, test_size=0.2, random_state=42, stratify=y_t)
        splits[t] = (Xtr, Xte, ytr, yte)
        
        sc = StandardScaler()
        Xtr_s = sc.fit_transform(Xtr)
        Xte_s = sc.transform(Xte)
        scaled_splits[t] = (Xtr_s, Xte_s, ytr, yte, sc)

    print("\n[4/7] Training & Evaluating Random Forest (300 estimators)...")
    rf_models = {}
    rf_results = []
    cv_results = []

    for t in TARGET_COLS:
        Xtr, Xte, ytr, yte = splits[t]
        model = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)
        model.fit(Xtr, ytr)
        pred = model.predict(Xte)
        acc = accuracy_score(yte, pred)
        rf_models[t] = model
        rf_results.append({'Target': t, 'Test_Accuracy': round(acc, 4)})
        
        # 5-fold CV
        scores = cross_val_score(model, Xtr, ytr, cv=5, scoring='accuracy')
        cv_results.append({
            'Target': t,
            'CV_Mean_Accuracy': round(scores.mean(), 4),
            'CV_Std': round(scores.std(), 4)
        })
        print(f"  {t:22s} | Test Acc: {acc:.4f} | 5-Fold CV: {scores.mean():.4f} +/- {scores.std():.4f}")
        joblib.dump(model, f'outputs/models/rf_{t}.joblib')

    rf_summary = pd.DataFrame(rf_results)
    cv_summary = pd.DataFrame(cv_results)

    # Feature Importance for Cooler Condition
    importances = pd.Series(rf_models['Cooler_Condition'].feature_importances_, index=FEATURE_COLS_REDUCED)
    top15 = importances.sort_values(ascending=False).head(15)
    plt.figure(figsize=(9, 6))
    top15.sort_values().plot(kind='barh', color='steelblue')
    plt.title('Top 15 Most Important Features - Cooler Condition (Random Forest)')
    plt.xlabel('Gini Importance')
    plt.tight_layout()
    plt.savefig('outputs/plots/rf_feature_importance_cooler.png', dpi=200)
    plt.close()

    # Confusion Matrices for Cooler and Valve
    for target_name, cmap, fname in [
        ('Cooler_Condition', 'Blues', 'outputs/plots/confusion_matrix_cooler_rf.png'),
        ('Valve_Condition', 'Greens', 'outputs/plots/confusion_matrix_valve_rf.png')
    ]:
        Xtr, Xte, ytr, yte = splits[target_name]
        pred = rf_models[target_name].predict(Xte)
        cm = confusion_matrix(yte, pred)
        classes = [str(c) for c in encoders[target_name].classes_]
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap=cmap, xticklabels=classes, yticklabels=classes)
        plt.xlabel('Predicted Class')
        plt.ylabel('Actual Class')
        plt.title(f'Confusion Matrix: {target_name} (Random Forest)')
        plt.tight_layout()
        plt.savefig(fname, dpi=200)
        plt.close()

    print("\n[5/7] Training & Evaluating Support Vector Machine (RBF kernel, C=10)...")
    svm_models = {}
    svm_results = []
    for t in TARGET_COLS:
        Xtr_s, Xte_s, ytr, yte, sc = scaled_splits[t]
        model = SVC(kernel='rbf', C=10, gamma='scale', random_state=42)
        model.fit(Xtr_s, ytr)
        pred = model.predict(Xte_s)
        acc = accuracy_score(yte, pred)
        svm_models[t] = model
        svm_results.append({'Target': t, 'Test_Accuracy': round(acc, 4)})
        print(f"  {t:22s} | Test Acc: {acc:.4f}")
        joblib.dump((model, sc), f'outputs/models/svm_{t}.joblib')

    svm_summary = pd.DataFrame(svm_results)

    print("\n[6/7] Training & Evaluating Gradient Boosting (200 trees, lr=0.1, depth=3)...")
    gb_models = {}
    gb_results = []
    for t in TARGET_COLS:
        Xtr, Xte, ytr, yte = splits[t]
        model = GradientBoostingClassifier(n_estimators=200, learning_rate=0.1, max_depth=3, random_state=42)
        model.fit(Xtr, ytr)
        pred = model.predict(Xte)
        acc = accuracy_score(yte, pred)
        gb_models[t] = model
        gb_results.append({'Target': t, 'Test_Accuracy': round(acc, 4)})
        print(f"  {t:22s} | Test Acc: {acc:.4f}")
        joblib.dump(model, f'outputs/models/gb_{t}.joblib')

    gb_summary = pd.DataFrame(gb_results)

    # 6. Consolidated Model Comparison
    print("\n[7/7] Generating Consolidated Performance Reports & Comparison Plots...")
    consolidated = rf_summary.rename(columns={'Test_Accuracy': 'Random_Forest'}).merge(
        svm_summary.rename(columns={'Test_Accuracy': 'SVM'}), on='Target').merge(
        gb_summary.rename(columns={'Test_Accuracy': 'Gradient_Boosting'}), on='Target'
    )
    consolidated['Best_Model'] = consolidated[['Random_Forest', 'SVM', 'Gradient_Boosting']].idxmax(axis=1)

    print("\n" + "="*70)
    print("CONSOLIDATED TEST ACCURACY BY TARGET:")
    print("="*70)
    print(consolidated.to_string(index=False))

    overall = consolidated[['Random_Forest', 'SVM', 'Gradient_Boosting']].mean().round(4).sort_values(ascending=False)
    print("\n" + "="*70)
    print("OVERALL MEAN ACCURACY ACROSS ALL 5 TARGETS:")
    print("="*70)
    print(overall.to_frame('Mean_Accuracy').to_string())

    print("\n" + "="*70)
    print("5-FOLD CROSS VALIDATION FOR RANDOM FOREST:")
    print("="*70)
    print(cv_summary.to_string(index=False))

    # Bar chart comparison
    plot_df = consolidated.set_index('Target')[['Random_Forest', 'SVM', 'Gradient_Boosting']]
    plt.figure(figsize=(11, 6))
    ax = plot_df.plot(kind='bar', figsize=(11, 6), color=['forestgreen', 'steelblue', 'darkorange'], width=0.8)
    plt.ylabel('Test Accuracy')
    plt.title('Random Forest vs SVM vs Gradient Boosting - Accuracy per Target')
    plt.ylim(0.75, 1.03)
    plt.xticks(rotation=15, ha='right')
    plt.legend(title='Model')
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    for p in ax.patches:
        height = p.get_height()
        if height > 0:
            ax.annotate(f"{height:.3f}",
                        (p.get_x() + p.get_width() / 2., height),
                        ha='center', va='bottom', fontsize=8, rotation=0, xytext=(0, 2),
                        textcoords='offset points')
    plt.tight_layout()
    plt.savefig('outputs/plots/model_comparison_by_target.png', dpi=200)
    plt.close()

    # Save summary tables to CSV
    consolidated.to_csv('outputs/model_comparison_results.csv', index=False)
    cv_summary.to_csv('outputs/rf_cv_results.csv', index=False)

    elapsed = time.time() - start_time
    print(f"\nPipeline successfully completed in {elapsed:.2f} seconds!")
    print("Artifacts generated:")
    print("  - Preprocessed dataset: hydraulic_preprocessed_dataset.csv")
    print("  - High-res plots: outputs/plots/")
    print("  - Saved model binaries: outputs/models/")
    print("  - Metrics CSVs: outputs/model_comparison_results.csv, outputs/rf_cv_results.csv")
    print("="*70)

if __name__ == '__main__':
    main()
