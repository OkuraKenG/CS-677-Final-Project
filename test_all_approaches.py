"""
Comprehensive test of all sampling + model combinations for credit default prediction.
Runs in parallel/batch to find the best approach quickly.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier, VotingClassifier, StackingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

# Sampling techniques
from imblearn.over_sampling import SMOTE, ADASYN, BorderlineSMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTETomek

import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("COMPREHENSIVE APPROACH TESTING FOR CREDIT DEFAULT PREDICTION")
print("=" * 80)

# Load data (assuming preprocessed data from notebook)
print("\n[1/4] Loading data...")
try:
    from ucimlrepo import fetch_ucirepo
    credit_card = fetch_ucirepo(id=350)
    X = credit_card.data.features
    y = credit_card.data.targets.values.ravel()
except:
    print("ERROR: Could not load UCI dataset. Make sure ucimlrepo is installed.")
    exit(1)

# Basic preprocessing
X = X.iloc[:, 1:]  # Drop ID column if present
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.3, random_state=42, stratify=y
)

print(f"  Train shape: {X_train.shape}, Test shape: {X_test.shape}")
print(f"  Default rate (train): {y_train.mean():.2%}, (test): {y_test.mean():.2%}")

# Define sampling techniques
samplers = {
    "None": None,
    "SMOTE": SMOTE(random_state=42),
    "ADASYN": ADASYN(random_state=42),
    "BorderlineSMOTE": BorderlineSMOTE(random_state=42),
    "SMOTETomek": SMOTETomek(random_state=42),
    "RandomUndersampling": RandomUnderSampler(random_state=42),
}

# Define classifiers
classifiers = {
    "LR": LogisticRegression(max_iter=1000, random_state=42),
    "SVM": SVC(kernel='rbf', probability=True, random_state=42),
    "RF": RandomForestClassifier(n_estimators=200, max_depth=20, class_weight='balanced', n_jobs=-1, random_state=42),
    "ExtraTrees": ExtraTreesClassifier(n_estimators=800, max_depth=20, class_weight='balanced', n_jobs=-1, random_state=42),
    "GradBoosting": GradientBoostingClassifier(n_estimators=200, max_depth=5, learning_rate=0.1, random_state=42),
}

results = []

print("\n" + "=" * 80)
print("[2/4] Testing all combinations (sampling + classifier)...")
print("=" * 80)

for sampler_name, sampler in samplers.items():
    for clf_name, clf in classifiers.items():
        # Apply sampling if specified
        if sampler is not None:
            X_train_res, y_train_res = sampler.fit_resample(X_train, y_train)
        else:
            X_train_res, y_train_res = X_train, y_train
        
        # Train classifier
        clf.fit(X_train_res, y_train_res)
        
        # Predict
        y_pred = clf.predict(X_test)
        y_proba = clf.predict_proba(X_test)[:, 1]
        
        # Metrics
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        auc = roc_auc_score(y_test, y_proba)
        
        results.append({
            'Sampler': sampler_name,
            'Classifier': clf_name,
            'Accuracy': acc,
            'Precision': prec,
            'Recall': rec,
            'F1': f1,
            'AUC': auc,
        })
        
        print(f"{sampler_name:20} + {clf_name:15} | Acc: {acc:.4f} | Prec: {prec:.4f} | Rec: {rec:.4f} | F1: {f1:.4f} | AUC: {auc:.4f}")

# Convert results to DataFrame and sort by accuracy
results_df = pd.DataFrame(results)
results_df_sorted = results_df.sort_values('Accuracy', ascending=False)

print("\n" + "=" * 80)
print("[3/4] TOP 10 COMBINATIONS BY ACCURACY")
print("=" * 80)
print(results_df_sorted.head(10).to_string(index=False))

print("\n" + "=" * 80)
print("[4/4] TOP PERFORMERS BY METRIC")
print("=" * 80)
print("\nBest Accuracy:")
best_acc = results_df_sorted.iloc[0]
print(f"  {best_acc['Sampler']:20} + {best_acc['Classifier']:15} = {best_acc['Accuracy']:.4f}")

best_by_metric = {
    'Precision': results_df.loc[results_df['Precision'].idxmax()],
    'Recall': results_df.loc[results_df['Recall'].idxmax()],
    'F1': results_df.loc[results_df['F1'].idxmax()],
    'AUC': results_df.loc[results_df['AUC'].idxmax()],
}

for metric, row in best_by_metric.items():
    print(f"\nBest {metric}:")
    print(f"  {row['Sampler']:20} + {row['Classifier']:15} = {row[metric]:.4f}")

# Now test ensemble approaches
print("\n" + "=" * 80)
print("BONUS: TESTING ENSEMBLE APPROACHES (Voting & Stacking)")
print("=" * 80)

# Use best sampling method found
best_sampler_name = results_df_sorted.iloc[0]['Sampler']
best_sampler = samplers[best_sampler_name]

if best_sampler is not None:
    X_train_res, y_train_res = best_sampler.fit_resample(X_train, y_train)
else:
    X_train_res, y_train_res = X_train, y_train

print(f"\nUsing best sampler: {best_sampler_name}")

# Voting Classifier
base_clfs = [
    ('lr', LogisticRegression(max_iter=1000, random_state=42)),
    ('rf', RandomForestClassifier(n_estimators=200, max_depth=20, class_weight='balanced', n_jobs=-1, random_state=42)),
    ('et', ExtraTreesClassifier(n_estimators=800, max_depth=20, class_weight='balanced', n_jobs=-1, random_state=42)),
    ('gb', GradientBoostingClassifier(n_estimators=200, max_depth=5, learning_rate=0.1, random_state=42)),
]

print("\n1. Voting Classifier (Hard)...")
voting_hard = VotingClassifier(estimators=base_clfs, voting='hard', n_jobs=-1)
voting_hard.fit(X_train_res, y_train_res)
y_pred_vh = voting_hard.predict(X_test)
acc_vh = accuracy_score(y_test, y_pred_vh)
f1_vh = f1_score(y_test, y_pred_vh)
print(f"   Accuracy: {acc_vh:.4f}, F1: {f1_vh:.4f}")

print("\n2. Voting Classifier (Soft)...")
voting_soft = VotingClassifier(estimators=base_clfs, voting='soft', n_jobs=-1)
voting_soft.fit(X_train_res, y_train_res)
y_pred_vs = voting_soft.predict(X_test)
y_proba_vs = voting_soft.predict_proba(X_test)[:, 1]
acc_vs = accuracy_score(y_test, y_pred_vs)
f1_vs = f1_score(y_test, y_pred_vs)
auc_vs = roc_auc_score(y_test, y_proba_vs)
print(f"   Accuracy: {acc_vs:.4f}, F1: {f1_vs:.4f}, AUC: {auc_vs:.4f}")

print("\n3. Stacking Classifier...")
stacking = StackingClassifier(
    estimators=base_clfs,
    final_estimator=LogisticRegression(max_iter=1000, random_state=42),
    cv=5,
    n_jobs=-1
)
stacking.fit(X_train_res, y_train_res)
y_pred_stack = stacking.predict(X_test)
y_proba_stack = stacking.predict_proba(X_test)[:, 1]
acc_stack = accuracy_score(y_test, y_pred_stack)
f1_stack = f1_score(y_test, y_pred_stack)
auc_stack = roc_auc_score(y_test, y_proba_stack)
print(f"   Accuracy: {acc_stack:.4f}, F1: {f1_stack:.4f}, AUC: {auc_stack:.4f}")

# Threshold tuning for best model
print("\n" + "=" * 80)
print("THRESHOLD TUNING ON BEST MODEL")
print("=" * 80)

best_model_row = results_df_sorted.iloc[0]
best_sampler_name = best_model_row['Sampler']
best_clf_name = best_model_row['Classifier']

sampler = samplers[best_sampler_name]
clf = classifiers[best_clf_name]

if sampler is not None:
    X_train_res, y_train_res = sampler.fit_resample(X_train, y_train)
else:
    X_train_res, y_train_res = X_train, y_train

clf.fit(X_train_res, y_train_res)
y_proba = clf.predict_proba(X_test)[:, 1]

print(f"\nModel: {best_sampler_name} + {best_clf_name}")
print("\nThreshold | Accuracy | Precision | Recall | F1")
print("-" * 50)

best_f1_thresh = 0.5
best_f1_score = 0

for threshold in np.arange(0.1, 0.9, 0.05):
    y_pred_thresh = (y_proba >= threshold).astype(int)
    acc_t = accuracy_score(y_test, y_pred_thresh)
    prec_t = precision_score(y_test, y_pred_thresh, zero_division=0)
    rec_t = recall_score(y_test, y_pred_thresh, zero_division=0)
    f1_t = f1_score(y_test, y_pred_thresh, zero_division=0)
    
    if f1_t > best_f1_score:
        best_f1_score = f1_t
        best_f1_thresh = threshold
    
    print(f"  {threshold:.2f}    | {acc_t:.4f}    | {prec_t:.4f}     | {rec_t:.4f} | {f1_t:.4f}")

print(f"\nBest threshold (F1): {best_f1_thresh:.2f} with F1={best_f1_score:.4f}")

print("\n" + "=" * 80)
print("ANALYSIS COMPLETE!")
print("=" * 80)
print("\nSummary:")
print(f"  Best single model: {best_sampler_name} + {best_clf_name}")
print(f"    - Accuracy: {best_model_row['Accuracy']:.4f}")
print(f"    - Best ensemble (soft voting): Accuracy {acc_vs:.4f}")
print(f"    - Best ensemble (stacking): Accuracy {acc_stack:.4f}")
print(f"\nRecommendation: Use {best_clf_name} with {best_sampler_name} sampling, tuned at threshold {best_f1_thresh:.2f}")
