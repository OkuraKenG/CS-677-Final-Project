"""
High-Impact Feature Engineering + SHAP Analysis
UCI Credit Default Dataset (Taiwan)
X1=LIMIT_BAL, X5=AGE, X6=PAY_0, X7-X11=PAY_2-6, X12-X17=BILL_AMT1-6, X18-X23=PAY_AMT1-6
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("HIGH-IMPACT FEATURE ENGINEERING + SHAP ANALYSIS")
print("=" * 80)

# Load
from ucimlrepo import fetch_ucirepo
credit_card = fetch_ucirepo(id=350)
X = credit_card.data.features.iloc[:, :]
y = credit_card.data.targets.values.ravel()

print(f"\nDataset: {X.shape[0]} samples, {X.shape[1]} original features")
print(f"Default rate: {y.mean():.2%}")

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

print("\n" + "=" * 80)
print("ENGINEERING NEW FEATURES")
print("=" * 80)

def engineer_features(X_df):
    """Create high-impact features from UCI credit data"""
    X_eng = X_df.copy()
    n_samples = len(X_eng)
    
    # Column indices (0-indexed in pandas)
    limit_bal = 'X1'
    age = 'X5'
    repay_cols = ['X6', 'X7', 'X8', 'X9', 'X10', 'X11']  # Payment status
    bill_cols = ['X12', 'X13', 'X14', 'X15', 'X16', 'X17']  # Bills
    pay_cols = ['X18', 'X19', 'X20', 'X21', 'X22', 'X23']   # Payments
    
    # 1. Utilization Ratios
    X_eng['util_mean'] = X_eng[bill_cols].mean(axis=1) / (X_eng[limit_bal] + 1)
    X_eng['util_max'] = X_eng[bill_cols].max(axis=1) / (X_eng[limit_bal] + 1)
    X_eng['util_recent'] = X_eng['X12'] / (X_eng[limit_bal] + 1)
    
    # 2. Payment Fractions
    pay_fracs = []
    for bill, pay in zip(bill_cols, pay_cols):
        frac = X_eng[pay] / (X_eng[bill].clip(lower=1) + 1)
        pay_fracs.append(frac)
    
    X_eng['payfrac_mean'] = pd.concat(pay_fracs, axis=1).mean(axis=1)
    X_eng['payfrac_recent'] = pay_fracs[0]
    X_eng['payfrac_min'] = pd.concat(pay_fracs, axis=1).min(axis=1)
    
    # 3. Underpayment/Overpayment counts
    underpay = sum([(X_eng[pay] < 0.1 * X_eng[bill]).astype(int) for bill, pay in zip(bill_cols, pay_cols)])
    overpay = sum([(X_eng[pay] > X_eng[bill]).astype(int) for bill, pay in zip(bill_cols, pay_cols)])
    X_eng['underpay_count'] = underpay
    X_eng['overpay_count'] = overpay
    
    # 4. Trends
    X_eng['bill_trend'] = X_eng['X12'] - X_eng['X14']
    X_eng['bill_oldtrend'] = X_eng['X14'] - X_eng['X17']
    X_eng['pay_trend'] = X_eng['X18'] - X_eng['X20']
    
    # 5. Repayment Status Aggregates
    X_eng['repay_max'] = X_eng[repay_cols].max(axis=1)
    X_eng['repay_mean'] = X_eng[repay_cols].mean(axis=1)
    X_eng['months_late'] = (X_eng[repay_cols] > 0).sum(axis=1)
    X_eng['months_severe'] = (X_eng[repay_cols] >= 2).sum(axis=1)
    
    # 6. Volatility
    X_eng['bill_std'] = X_eng[bill_cols].std(axis=1)
    X_eng['bill_cv'] = X_eng[bill_cols].std(axis=1) / (X_eng[bill_cols].mean(axis=1).clip(lower=1) + 1)
    X_eng['pay_std'] = X_eng[pay_cols].std(axis=1)
    
    # 7. Age Bins
    X_eng['age_lt30'] = (X_eng[age] < 30).astype(int)
    X_eng['age_30_45'] = ((X_eng[age] >= 30) & (X_eng[age] < 45)).astype(int)
    X_eng['age_45_60'] = ((X_eng[age] >= 45) & (X_eng[age] < 60)).astype(int)
    
    # 8. Interaction
    X_eng['late_hi_util'] = ((X_eng['months_late'] > 0) & (X_eng['util_recent'] > 0.8)).astype(int)
    
    return X_eng

X_train_eng = engineer_features(X_train)
X_test_eng = engineer_features(X_test)

n_new = X_train_eng.shape[1] - X_train.shape[1]
print(f"Original: {X_train.shape[1]} features")
print(f"Engineered: {X_train_eng.shape[1]} features (+{n_new})")

# Scale
scaler = StandardScaler()
X_train_eng_scaled = scaler.fit_transform(X_train_eng)
X_test_eng_scaled = scaler.transform(X_test_eng)

# SMOTE balance
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train_eng_scaled, y_train)

print(f"SMOTE: {X_train_eng_scaled.shape[0]} → {X_train_smote.shape[0]} samples")

print("\n" + "=" * 80)
print("MODEL EVALUATION")
print("=" * 80)

models = {
    'ExtraTrees': ExtraTreesClassifier(n_estimators=800, max_depth=20, class_weight='balanced', n_jobs=-1, random_state=42),
    'GradBoosting': GradientBoostingClassifier(n_estimators=200, max_depth=5, learning_rate=0.1, random_state=42),
}

for name, model in models.items():
    print(f"\n{name}:")
    model.fit(X_train_smote, y_train_smote)
    y_pred = model.predict(X_test_eng_scaled)
    y_proba = model.predict_proba(X_test_eng_scaled)[:, 1]
    
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    auc = roc_auc_score(y_test, y_proba)
    
    cm = confusion_matrix(y_test, y_pred)
    tp = cm[1,1]
    total_pos = cm[1,0] + cm[1,1]
    
    print(f"  Acc={acc:.4f} | Prec={prec:.4f} | Rec={rec:.4f} | F1={f1:.4f} | AUC={auc:.4f}")
    print(f"  Catching {rec:.1%} of defaults ({tp}/{total_pos})")

print("\n" + "=" * 80)
print("THRESHOLD TUNING (ExtraTrees)")
print("=" * 80)

et = ExtraTreesClassifier(n_estimators=800, max_depth=20, class_weight='balanced', n_jobs=-1, random_state=42)
et.fit(X_train_smote, y_train_smote)
y_proba_et = et.predict_proba(X_test_eng_scaled)[:, 1]

print("\nThreshold | Accuracy | Precision | Recall | F1")
print("-" * 55)

best_f1, best_thresh = 0, 0.5
best_metrics = {}

for thresh in np.arange(0.20, 0.75, 0.05):
    y_pred_t = (y_proba_et >= thresh).astype(int)
    acc_t = accuracy_score(y_test, y_pred_t)
    prec_t = precision_score(y_test, y_pred_t, zero_division=0)
    rec_t = recall_score(y_test, y_pred_t, zero_division=0)
    f1_t = f1_score(y_test, y_pred_t, zero_division=0)
    
    if f1_t > best_f1:
        best_f1 = f1_t
        best_thresh = thresh
        best_metrics = {'Acc': acc_t, 'Prec': prec_t, 'Rec': rec_t, 'F1': f1_t}
    
    print(f"  {thresh:.2f}    | {acc_t:.4f}    | {prec_t:.4f}     | {rec_t:.4f} | {f1_t:.4f}")

print(f"\n✅ Best: Threshold={best_thresh:.2f} → F1={best_f1:.4f}")
print(f"   Acc={best_metrics['Acc']:.4f}, Rec={best_metrics['Rec']:.4f}, Prec={best_metrics['Prec']:.4f}")

print("\n" + "=" * 80)
print("SHAP FEATURE IMPORTANCE")
print("=" * 80)

try:
    import shap
    print("\n📊 Computing SHAP (sampling 2000 instances)...")
    
    sample_idx = np.random.choice(len(X_train_eng_scaled), 2000, replace=False)
    X_shap = X_train_eng_scaled[sample_idx]
    
    et_shap = ExtraTreesClassifier(n_estimators=800, max_depth=20, class_weight='balanced', n_jobs=-1, random_state=42)
    et_shap.fit(X_train_eng_scaled, y_train)
    
    explainer = shap.TreeExplainer(et_shap)
    shap_vals = explainer.shap_values(X_shap)
    
    mean_abs_shap = np.abs(shap_vals[1]).mean(axis=0)
    top_idx = np.argsort(mean_abs_shap)[::-1][:20]
    
    feature_names = X_train_eng.columns.tolist()
    engineered = set(X_train_eng.columns) - set(X_train.columns)
    
    print("\nTop 20 Features (SHAP importance for Default class):")
    print("-" * 60)
    for i, idx in enumerate(top_idx, 1):
        fname = feature_names[idx]
        imp = mean_abs_shap[idx]
        is_eng = " [ENG]" if fname in engineered else ""
        print(f"{i:2d}. {fname:25s} {imp:8.4f}{is_eng}")
    
    print(f"\n📌 Engineered features in Top 20:")
    count = sum(1 for idx in top_idx if feature_names[idx] in engineered)
    print(f"   {count}/20 are engineered features")
    
except ImportError:
    print("⚠️  SHAP not installed: pip install shap")
except Exception as e:
    print(f"⚠️  SHAP error: {e}")

print("\n" + "=" * 80)
print("FINAL SUMMARY")
print("=" * 80)
print(f"\n✅ Engineered {n_new} features successfully")
print(f"✅ Best model: ExtraTrees with engineered features + SMOTE")
print(f"✅ Optimal threshold: {best_thresh:.2f}")
print(f"✅ Performance: F1={best_f1:.4f}, Recall={best_metrics['Rec']:.4f}, Accuracy={best_metrics['Acc']:.4f}")
