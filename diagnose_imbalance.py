"""
Diagnose why "no sampling" is winning - check confusion matrices
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier, ExtraTreesClassifier, RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from ucimlrepo import fetch_ucirepo

print("=" * 80)
print("DIAGNOSING IMBALANCE ISSUE - CONFUSION MATRIX ANALYSIS")
print("=" * 80)

# Load data
credit_card = fetch_ucirepo(id=350)
X = credit_card.data.features.iloc[:, 1:]  # Drop ID
y = credit_card.data.targets.values.ravel()

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.3, random_state=42, stratify=y
)

print(f"\nClass distribution:")
print(f"  Train: {np.bincount(y_train)} ({y_train.mean():.2%} default)")
print(f"  Test:  {np.bincount(y_test)} ({y_test.mean():.2%} default)")

# Baseline: Always predict majority class
baseline_pred = np.zeros_like(y_test)
baseline_acc = accuracy_score(y_test, baseline_pred)
print(f"\n🚨 BASELINE (always predict 0): {baseline_acc:.4f}")

print("\n" + "=" * 80)
print("TESTING GRADBOOSTING")
print("=" * 80)

# 1. GradBoosting without sampling
print("\n1. GradBoosting (NO SAMPLING)")
gb_none = GradientBoostingClassifier(n_estimators=200, max_depth=5, learning_rate=0.1, random_state=42)
gb_none.fit(X_train, y_train)
y_pred_none = gb_none.predict(X_test)

print(f"Accuracy: {accuracy_score(y_test, y_pred_none):.4f}")
print("\nConfusion Matrix:")
cm_none = confusion_matrix(y_test, y_pred_none)
print(f"  TN={cm_none[0,0]:5d} | FP={cm_none[0,1]:5d}")
print(f"  FN={cm_none[1,0]:5d} | TP={cm_none[1,1]:5d}")
print(f"\n  Predicting mostly 0 (non-default)? {(y_pred_none == 0).sum() / len(y_pred_none):.2%}")
print(f"  Predicting mostly 1 (default)? {(y_pred_none == 1).sum() / len(y_pred_none):.2%}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred_none, target_names=['Non-Default', 'Default']))

# 2. GradBoosting with SMOTE
print("\n" + "-" * 80)
print("2. GradBoosting (WITH SMOTE)")
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
gb_smote = GradientBoostingClassifier(n_estimators=200, max_depth=5, learning_rate=0.1, random_state=42)
gb_smote.fit(X_train_smote, y_train_smote)
y_pred_smote = gb_smote.predict(X_test)

print(f"Accuracy: {accuracy_score(y_test, y_pred_smote):.4f}")
print("\nConfusion Matrix:")
cm_smote = confusion_matrix(y_test, y_pred_smote)
print(f"  TN={cm_smote[0,0]:5d} | FP={cm_smote[0,1]:5d}")
print(f"  FN={cm_smote[1,0]:5d} | TP={cm_smote[1,1]:5d}")
print(f"\n  Predicting mostly 0 (non-default)? {(y_pred_smote == 0).sum() / len(y_pred_smote):.2%}")
print(f"  Predicting mostly 1 (default)? {(y_pred_smote == 1).sum() / len(y_pred_smote):.2%}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred_smote, target_names=['Non-Default', 'Default']))

# 3. ExtraTrees with RandomUndersampling (best F1 from earlier)
print("\n" + "=" * 80)
print("3. ExtraTrees (WITH RANDOM UNDERSAMPLING) - Best F1 earlier")
print("=" * 80)
rus = RandomUnderSampler(random_state=42)
X_train_rus, y_train_rus = rus.fit_resample(X_train, y_train)
et_rus = ExtraTreesClassifier(n_estimators=800, max_depth=20, class_weight='balanced', n_jobs=-1, random_state=42)
et_rus.fit(X_train_rus, y_train_rus)
y_pred_rus = et_rus.predict(X_test)

print(f"Accuracy: {accuracy_score(y_test, y_pred_rus):.4f}")
print("\nConfusion Matrix:")
cm_rus = confusion_matrix(y_test, y_pred_rus)
print(f"  TN={cm_rus[0,0]:5d} | FP={cm_rus[0,1]:5d}")
print(f"  FN={cm_rus[1,0]:5d} | TP={cm_rus[1,1]:5d}")
print(f"\n  Predicting mostly 0 (non-default)? {(y_pred_rus == 0).sum() / len(y_pred_rus):.2%}")
print(f"  Predicting mostly 1 (default)? {(y_pred_rus == 1).sum() / len(y_pred_rus):.2%}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred_rus, target_names=['Non-Default', 'Default']))

# Summary comparison
print("\n" + "=" * 80)
print("SUMMARY COMPARISON")
print("=" * 80)

def get_metrics(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    return {
        'Accuracy': (tp + tn) / (tp + tn + fp + fn),
        'Precision': tp / (tp + fp) if (tp + fp) > 0 else 0,
        'Recall': tp / (tp + fn) if (tp + fn) > 0 else 0,
        'Specificity': tn / (tn + fp) if (tn + fp) > 0 else 0,
        'F1': 2*tp / (2*tp + fp + fn) if (2*tp + fp + fn) > 0 else 0,
        'TP': tp,
        'TN': tn,
        'FP': fp,
        'FN': fn,
    }

baseline_metrics = get_metrics(y_test, baseline_pred)
none_metrics = get_metrics(y_test, y_pred_none)
smote_metrics = get_metrics(y_test, y_pred_smote)
rus_metrics = get_metrics(y_test, y_pred_rus)

comparison = pd.DataFrame({
    'Baseline (always 0)': baseline_metrics,
    'GradBoost (none)': none_metrics,
    'GradBoost (SMOTE)': smote_metrics,
    'ExtraTrees (RUS)': rus_metrics,
})

print(comparison.T.to_string())

print("\n" + "=" * 80)
print("🚨 DIAGNOSIS")
print("=" * 80)

if none_metrics['Recall'] < 0.5:
    print("\n⚠️  WARNING: 'No sampling' model has LOW RECALL!")
    print(f"   Only catching {none_metrics['Recall']:.1%} of actual defaults")
    print(f"   Missing {none_metrics['FN']} out of {none_metrics['TP'] + none_metrics['FN']} defaults!")
    print("\n   This is why accuracy is misleading for imbalanced data.")
    print("   The model is just good at predicting the majority class (non-default).")

if smote_metrics['Recall'] > none_metrics['Recall']:
    print(f"\n✅ SMOTE improves recall: {smote_metrics['Recall']:.1%} vs {none_metrics['Recall']:.1%}")
    print(f"   Catches {smote_metrics['TP'] - none_metrics['TP']} more defaults!")
    print(f"   Trade-off: Accuracy drops {none_metrics['Accuracy'] - smote_metrics['Accuracy']:.3f}")

print("\n" + "=" * 80)
print("RECOMMENDATION")
print("=" * 80)
print("\nFor credit default prediction, we care MORE about:")
print("  - RECALL (catching defaults) - cost of missing a default is HIGH")
print("  - F1 score (balance precision/recall)")
print("\nAccuracy is MISLEADING because:")
print("  - Dataset is 78% non-default")
print("  - Predicting 'non-default' for everyone gives 78% accuracy!")
print(f"  - Baseline (always 0): {baseline_acc:.1%} accuracy")
print(f"  - GradBoost (none): {none_metrics['Accuracy']:.1%} accuracy")
print(f"    → Only {none_metrics['Accuracy'] - baseline_acc:.1%} better than dumb baseline!")

print("\n📊 BETTER METRIC RANKING:")
ranking = pd.DataFrame({
    'Model': ['Baseline', 'GradBoost (none)', 'GradBoost (SMOTE)', 'ExtraTrees (RUS)'],
    'Accuracy': [baseline_metrics['Accuracy'], none_metrics['Accuracy'], smote_metrics['Accuracy'], rus_metrics['Accuracy']],
    'Recall': [baseline_metrics['Recall'], none_metrics['Recall'], smote_metrics['Recall'], rus_metrics['Recall']],
    'F1': [baseline_metrics['F1'], none_metrics['F1'], smote_metrics['F1'], rus_metrics['F1']],
})
print("\n" + ranking.sort_values('F1', ascending=False).to_string(index=False))

print("\n🎯 WINNER (by F1): ", ranking.loc[ranking['F1'].idxmax(), 'Model'])
