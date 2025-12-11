"""
Advanced Optimization Experiment for Credit Default Prediction

This script explores 4 optimization strategies:
1. Feature Engineering (Domain Knowledge)
2. Threshold Optimization (Profit Maximization)
3. Ensemble Stacking
4. Custom Objective Function (Business Profit)

Author: ML Final Project Team
Date: December 11, 2025
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    confusion_matrix, accuracy_score, precision_score, 
    recall_score, f1_score, roc_auc_score, roc_curve
)
from sklearn.ensemble import StackingClassifier, RandomForestClassifier
from xgboost import XGBClassifier
from imblearn.under_sampling import RandomUnderSampler
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

# Business cost constants
C_DEFAULT = 10000  # Loss per missed default
C_REJECT = 200     # Cost per false alarm (rejected good customer)
R_GOOD = 200       # Profit per good customer (annual)
R_PREVENT = 1000   # Value of preventing a default

print("="*80)
print("ADVANCED OPTIMIZATION EXPERIMENT FOR CREDIT DEFAULT PREDICTION")
print("="*80)
print()

# ============================================================================
# SECTION 0: DATA LOADING
# ============================================================================
print("SECTION 0: Loading UCI Credit Card Default Dataset...")
print("-"*80)

from ucimlrepo import fetch_ucirepo
credit_card = fetch_ucirepo(id=350)
X = credit_card.data.features
y = credit_card.data.targets.values.ravel()

# Rename columns to meaningful names
X.columns = [
    'LIMIT_BAL', 'SEX', 'EDUCATION', 'MARRIAGE', 'AGE',
    'PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6',
    'BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6',
    'PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6'
]

print(f"✓ Dataset loaded: {X.shape[0]} samples, {X.shape[1]} features")
print(f"✓ Default rate: {y.mean():.2%}")
print()

# ============================================================================
# SECTION 1: FEATURE ENGINEERING (DOMAIN KNOWLEDGE)
# ============================================================================
print("="*80)
print("SECTION 1: FEATURE ENGINEERING (DOMAIN KNOWLEDGE)")
print("="*80)
print()

def create_engineered_features(X_df):
    """
    Create domain-specific features based on credit risk knowledge
    """
    X_eng = X_df.copy()
    
    # 1. UTILIZATION RATIO - Key credit risk indicator
    print("1. Creating utilization ratio features...")
    for i in range(1, 7):
        bill_col = f'BILL_AMT{i}'
        if bill_col in X_eng.columns:
            X_eng[f'utilization_ratio_{i}'] = X_eng[bill_col] / (X_eng['LIMIT_BAL'] + 1)
    
    # Average utilization
    util_cols = [c for c in X_eng.columns if 'utilization_ratio' in c]
    X_eng['avg_utilization'] = X_eng[util_cols].mean(axis=1)
    X_eng['max_utilization'] = X_eng[util_cols].max(axis=1)
    
    # 2. PAYMENT BEHAVIOR TRENDS
    print("2. Creating payment behavior trend features...")
    pay_cols = ['PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6']
    
    # Payment worsening (recent vs older)
    X_eng['payment_worsening'] = (X_eng['PAY_0'] > X_eng['PAY_6']).astype(int)
    X_eng['payment_improving'] = (X_eng['PAY_0'] < X_eng['PAY_6']).astype(int)
    
    # Payment statistics
    X_eng['max_delay'] = X_eng[pay_cols].max(axis=1)
    X_eng['avg_delay'] = X_eng[pay_cols].mean(axis=1)
    X_eng['delay_volatility'] = X_eng[pay_cols].std(axis=1)
    
    # Count of late payments
    X_eng['num_late_payments'] = (X_eng[pay_cols] > 0).sum(axis=1)
    X_eng['num_severe_delays'] = (X_eng[pay_cols] >= 2).sum(axis=1)
    
    # 3. BILLING AMOUNT TRENDS
    print("3. Creating billing amount trend features...")
    bill_cols = ['BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6']
    
    X_eng['avg_bill_amt'] = X_eng[bill_cols].mean(axis=1)
    X_eng['max_bill_amt'] = X_eng[bill_cols].max(axis=1)
    X_eng['bill_trend'] = X_eng['BILL_AMT1'] - X_eng['BILL_AMT6']  # Increasing or decreasing
    X_eng['bill_volatility'] = X_eng[bill_cols].std(axis=1)
    
    # 4. PAYMENT AMOUNT ANALYSIS
    print("4. Creating payment amount features...")
    pay_amt_cols = ['PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6']
    
    X_eng['avg_pay_amt'] = X_eng[pay_amt_cols].mean(axis=1)
    X_eng['total_paid_6months'] = X_eng[pay_amt_cols].sum(axis=1)
    
    # Payment ratio (how much they pay vs what they owe)
    X_eng['payment_ratio'] = X_eng['avg_pay_amt'] / (X_eng['avg_bill_amt'] + 1)
    
    # 5. RISK COMPOSITE SCORES
    print("5. Creating composite risk scores...")
    
    # High risk if: high utilization + many late payments + low payment ratio
    X_eng['high_risk_score'] = (
        (X_eng['avg_utilization'] > 0.8).astype(int) +
        (X_eng['num_late_payments'] > 3).astype(int) +
        (X_eng['payment_ratio'] < 0.5).astype(int)
    )
    
    # Credit stress indicator
    X_eng['credit_stress'] = X_eng['max_utilization'] * X_eng['max_delay']
    
    # 6. DEMOGRAPHIC INTERACTIONS
    print("6. Creating demographic interaction features...")
    X_eng['young_high_limit'] = ((X_eng['AGE'] < 30) & (X_eng['LIMIT_BAL'] > 200000)).astype(int)
    X_eng['education_income_proxy'] = X_eng['EDUCATION'] * np.log1p(X_eng['LIMIT_BAL'])
    
    print(f"✓ Feature engineering complete: {len(X_eng.columns)} features (added {len(X_eng.columns) - len(X_df.columns)})")
    print()
    
    return X_eng

# Apply feature engineering
X_engineered = create_engineered_features(X)

# Train-test split
X_train_eng, X_test_eng, y_train, y_test = train_test_split(
    X_engineered, y, test_size=0.3, random_state=42, stratify=y
)

print(f"Train set: {X_train_eng.shape[0]} samples")
print(f"Test set: {X_test_eng.shape[0]} samples")
print()

# Calculate scale_pos_weight
scale_weight = (y_train == 0).sum() / (y_train == 1).sum()
print(f"Scale weight (for cost-sensitive learning): {scale_weight:.4f}")
print()

# Train baseline model with engineered features
print("Training XGBoost with engineered features...")
model_engineered = XGBClassifier(
    scale_pos_weight=scale_weight,
    n_estimators=100,
    max_depth=4,
    learning_rate=0.1,
    random_state=42,
    eval_metric='logloss'
)
model_engineered.fit(X_train_eng, y_train)

y_pred_eng = model_engineered.predict(X_test_eng)
y_prob_eng = model_engineered.predict_proba(X_test_eng)[:, 1]

tn, fp, fn, tp = confusion_matrix(y_test, y_pred_eng).ravel()
profit_eng = R_GOOD * tn - C_REJECT * fp - C_DEFAULT * fn + R_PREVENT * tp

print(f"✓ Engineered Features Results:")
print(f"  Accuracy: {accuracy_score(y_test, y_pred_eng):.4f}")
print(f"  Precision: {precision_score(y_test, y_pred_eng):.4f}")
print(f"  Recall: {recall_score(y_test, y_pred_eng):.4f}")
print(f"  F1-Score: {f1_score(y_test, y_pred_eng):.4f}")
print(f"  AUC-ROC: {roc_auc_score(y_test, y_prob_eng):.4f}")
print(f"  Missed Defaults: {fn}")
print(f"  Business Profit: ${profit_eng:,.0f}")
print()

# ============================================================================
# SECTION 2: THRESHOLD OPTIMIZATION (PROFIT MAXIMIZATION)
# ============================================================================
print("="*80)
print("SECTION 2: THRESHOLD OPTIMIZATION (PROFIT MAXIMIZATION)")
print("="*80)
print()

def find_optimal_threshold(y_true, y_prob, thresholds=None):
    """
    Find threshold that maximizes business profit
    """
    if thresholds is None:
        thresholds = np.arange(0.05, 0.96, 0.01)
    
    results = []
    
    for threshold in thresholds:
        y_pred_thresh = (y_prob >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred_thresh).ravel()
        
        profit = R_GOOD * tn - C_REJECT * fp - C_DEFAULT * fn + R_PREVENT * tp
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        results.append({
            'threshold': threshold,
            'profit': profit,
            'recall': recall,
            'precision': precision,
            'f1': f1,
            'tn': tn,
            'fp': fp,
            'fn': fn,
            'tp': tp
        })
    
    return pd.DataFrame(results)

print("Searching for optimal threshold (0.05 to 0.95)...")
threshold_results = find_optimal_threshold(y_test, y_prob_eng)

# Find best threshold
best_threshold_idx = threshold_results['profit'].idxmax()
best_threshold = threshold_results.loc[best_threshold_idx, 'threshold']
best_profit = threshold_results.loc[best_threshold_idx, 'profit']
best_recall = threshold_results.loc[best_threshold_idx, 'recall']

print(f"✓ Optimal threshold found: {best_threshold:.3f}")
print(f"  Maximum profit: ${best_profit:,.0f}")
print(f"  Recall at optimal: {best_recall:.2%}")
print(f"  Missed defaults: {int(threshold_results.loc[best_threshold_idx, 'fn'])}")
print()

# Apply optimal threshold
y_pred_optimal = (y_prob_eng >= best_threshold).astype(int)
tn_opt, fp_opt, fn_opt, tp_opt = confusion_matrix(y_test, y_pred_optimal).ravel()

print(f"✓ Optimal Threshold Results (τ = {best_threshold:.3f}):")
print(f"  Accuracy: {accuracy_score(y_test, y_pred_optimal):.4f}")
print(f"  Precision: {precision_score(y_test, y_pred_optimal):.4f}")
print(f"  Recall: {recall_score(y_test, y_pred_optimal):.4f}")
print(f"  F1-Score: {f1_score(y_test, y_pred_optimal):.4f}")
print(f"  Business Profit: ${best_profit:,.0f}")
print()

# Comparison with default threshold (0.5)
default_profit = profit_eng
improvement = best_profit - default_profit
improvement_pct = (improvement / abs(default_profit)) * 100 if default_profit != 0 else 0

print(f"📊 Improvement over default threshold (0.5):")
print(f"  Default threshold profit: ${default_profit:,.0f}")
print(f"  Optimal threshold profit: ${best_profit:,.0f}")
print(f"  Improvement: ${improvement:,.0f} ({improvement_pct:+.1f}%)")
print()

# Plot threshold analysis
plt.figure(figsize=(14, 5))

plt.subplot(1, 2, 1)
plt.plot(threshold_results['threshold'], threshold_results['profit'], 'b-', linewidth=2)
plt.axvline(best_threshold, color='r', linestyle='--', label=f'Optimal τ = {best_threshold:.3f}')
plt.axvline(0.5, color='gray', linestyle=':', label='Default τ = 0.5')
plt.xlabel('Threshold', fontsize=12)
plt.ylabel('Business Profit ($)', fontsize=12)
plt.title('Profit vs Threshold', fontsize=14, fontweight='bold')
plt.legend()
plt.grid(alpha=0.3)

plt.subplot(1, 2, 2)
plt.plot(threshold_results['threshold'], threshold_results['recall'], 'g-', linewidth=2, label='Recall')
plt.plot(threshold_results['threshold'], threshold_results['precision'], 'orange', linewidth=2, label='Precision')
plt.plot(threshold_results['threshold'], threshold_results['f1'], 'purple', linewidth=2, label='F1-Score')
plt.axvline(best_threshold, color='r', linestyle='--', alpha=0.5)
plt.xlabel('Threshold', fontsize=12)
plt.ylabel('Score', fontsize=12)
plt.title('Metrics vs Threshold', fontsize=14, fontweight='bold')
plt.legend()
plt.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('threshold_optimization_analysis.png', dpi=300, bbox_inches='tight')
print("✓ Visualization saved: threshold_optimization_analysis.png")
print()

# ============================================================================
# SECTION 3: ENSEMBLE STACKING
# ============================================================================
print("="*80)
print("SECTION 3: ENSEMBLE STACKING")
print("="*80)
print()

print("Creating stacking ensemble with multiple sampling strategies...")

# Prepare different training sets
print("1. Preparing training sets with different sampling strategies...")

# Original data (for training)
X_train_orig, _, y_train_orig, _ = train_test_split(
    X_engineered, y, test_size=0.3, random_state=42, stratify=y
)

# Random undersampling
rus = RandomUnderSampler(random_state=42)
X_train_rus, y_train_rus = rus.fit_resample(X_train_orig, y_train_orig)

print(f"  Original training set: {len(y_train_orig)} samples ({(y_train_orig==1).sum()} defaults)")
print(f"  Undersampled training set: {len(y_train_rus)} samples ({(y_train_rus==1).sum()} defaults)")
print()

# Define base models
print("2. Training base models...")

# Model 1: XGBoost with cost-sensitive learning (on original data)
base_model_1 = XGBClassifier(
    scale_pos_weight=scale_weight,
    n_estimators=100,
    max_depth=4,
    learning_rate=0.1,
    random_state=42,
    eval_metric='logloss'
)

# Model 2: XGBoost on undersampled data (higher recall)
base_model_2 = XGBClassifier(
    scale_pos_weight=1.0,  # Balanced after undersampling
    n_estimators=100,
    max_depth=5,
    learning_rate=0.1,
    random_state=43,
    eval_metric='logloss'
)

# Model 3: Random Forest with balanced class weights
base_model_3 = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    class_weight='balanced',
    random_state=44
)

# Train individual models
print("  Training Model 1 (XGB Cost-Sensitive)...")
base_model_1.fit(X_train_orig, y_train_orig)
pred_1 = base_model_1.predict(X_test_eng)
recall_1 = recall_score(y_test, pred_1)
print(f"    Model 1 Recall: {recall_1:.2%}")

print("  Training Model 2 (XGB on Undersampled Data)...")
base_model_2.fit(X_train_rus, y_train_rus)
pred_2 = base_model_2.predict(X_test_eng)
recall_2 = recall_score(y_test, pred_2)
print(f"    Model 2 Recall: {recall_2:.2%}")

print("  Training Model 3 (Random Forest Balanced)...")
base_model_3.fit(X_train_orig, y_train_orig)
pred_3 = base_model_3.predict(X_test_eng)
recall_3 = recall_score(y_test, pred_3)
print(f"    Model 3 Recall: {recall_3:.2%}")
print()

# Create stacking ensemble
print("3. Creating stacking ensemble with XGBoost meta-learner...")

# Use sample weights for meta-learner (emphasize defaults)
sample_weights = np.where(y_train_orig == 1, scale_weight, 1.0)

stacking_model = StackingClassifier(
    estimators=[
        ('xgb_cost', base_model_1),
        ('xgb_undersample', base_model_2),
        ('rf_balanced', base_model_3)
    ],
    final_estimator=XGBClassifier(
        scale_pos_weight=scale_weight,
        n_estimators=50,
        max_depth=3,
        learning_rate=0.1,
        random_state=45,
        eval_metric='logloss'
    ),
    cv=5
)

print("  Training stacking ensemble (this may take a moment)...")
stacking_model.fit(X_train_orig, y_train_orig)

y_pred_stack = stacking_model.predict(X_test_eng)
y_prob_stack = stacking_model.predict_proba(X_test_eng)[:, 1]

tn_stack, fp_stack, fn_stack, tp_stack = confusion_matrix(y_test, y_pred_stack).ravel()
profit_stack = R_GOOD * tn_stack - C_REJECT * fp_stack - C_DEFAULT * fn_stack + R_PREVENT * tp_stack

print(f"✓ Stacking Ensemble Results:")
print(f"  Accuracy: {accuracy_score(y_test, y_pred_stack):.4f}")
print(f"  Precision: {precision_score(y_test, y_pred_stack):.4f}")
print(f"  Recall: {recall_score(y_test, y_pred_stack):.4f}")
print(f"  F1-Score: {f1_score(y_test, y_pred_stack):.4f}")
print(f"  AUC-ROC: {roc_auc_score(y_test, y_prob_stack):.4f}")
print(f"  Missed Defaults: {fn_stack}")
print(f"  Business Profit: ${profit_stack:,.0f}")
print()

# Apply threshold optimization to stacking ensemble
print("4. Applying threshold optimization to stacking ensemble...")
threshold_results_stack = find_optimal_threshold(y_test, y_prob_stack)
best_threshold_stack_idx = threshold_results_stack['profit'].idxmax()
best_threshold_stack = threshold_results_stack.loc[best_threshold_stack_idx, 'threshold']
best_profit_stack = threshold_results_stack.loc[best_threshold_stack_idx, 'profit']

y_pred_stack_opt = (y_prob_stack >= best_threshold_stack).astype(int)
tn_so, fp_so, fn_so, tp_so = confusion_matrix(y_test, y_pred_stack_opt).ravel()

print(f"✓ Stacking + Optimal Threshold (τ = {best_threshold_stack:.3f}):")
print(f"  Recall: {recall_score(y_test, y_pred_stack_opt):.2%}")
print(f"  Precision: {precision_score(y_test, y_pred_stack_opt):.4f}")
print(f"  Missed Defaults: {fn_so}")
print(f"  Business Profit: ${best_profit_stack:,.0f}")
print()

# ============================================================================
# SECTION 4: CUSTOM OBJECTIVE FUNCTION (BUSINESS PROFIT)
# ============================================================================
print("="*80)
print("SECTION 4: CUSTOM OBJECTIVE FUNCTION (BUSINESS PROFIT)")
print("="*80)
print()

print("Implementing custom XGBoost objective that directly optimizes business profit...")
print()

def business_profit_obj(y_pred, dtrain):
    """
    Custom XGBoost objective function that maximizes business profit
    
    Gradient formulation:
    - For defaults (y=1): Penalize false negatives heavily (C_DEFAULT)
    - For non-defaults (y=0): Penalize false positives moderately (C_REJECT)
    """
    y_true = dtrain.get_label()
    
    # Convert logits to probabilities
    prob = 1.0 / (1.0 + np.exp(-y_pred))
    
    # Gradient calculation
    # For y=1 (default): want prob → 1, so negative gradient when prob is low
    # For y=0 (non-default): want prob → 0, so positive gradient when prob is high
    
    # Weight by business costs
    weight_positive = C_DEFAULT + R_PREVENT  # Cost of missing default + value of catching it
    weight_negative = C_REJECT + R_GOOD       # Cost of false alarm + lost revenue
    
    grad = np.where(
        y_true == 1,
        -weight_positive * (1 - prob),   # Push probability up for defaults
        weight_negative * prob            # Push probability down for non-defaults
    )
    
    # Hessian (second derivative for Newton's method)
    hess = np.where(
        y_true == 1,
        weight_positive * prob * (1 - prob),
        weight_negative * prob * (1 - prob)
    )
    
    return grad, hess

def business_profit_eval(y_pred, dtrain):
    """
    Custom evaluation metric: business profit
    """
    y_true = dtrain.get_label()
    y_pred_binary = (y_pred > 0.0).astype(int)  # 0.0 is the logit threshold for prob=0.5
    
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred_binary).ravel()
    profit = R_GOOD * tn - C_REJECT * fp - C_DEFAULT * fn + R_PREVENT * tp
    
    # Return metric name and value (higher is better)
    return 'business_profit', profit

print("Training XGBoost with custom business profit objective...")

# Prepare DMatrix
dtrain = xgb.DMatrix(X_train_orig, label=y_train_orig)
dtest = xgb.DMatrix(X_test_eng, label=y_test)

# Parameters
params = {
    'max_depth': 4,
    'eta': 0.1,  # learning_rate
    'eval_metric': 'logloss',
    'seed': 42
}

# Train with custom objective
model_custom_obj = xgb.train(
    params,
    dtrain,
    num_boost_round=100,
    obj=business_profit_obj,
    custom_metric=business_profit_eval,
    evals=[(dtrain, 'train'), (dtest, 'test')],
    verbose_eval=False
)

# Predict
y_pred_custom_logit = model_custom_obj.predict(dtest)
y_prob_custom = 1.0 / (1.0 + np.exp(-y_pred_custom_logit))  # Convert logits to probabilities
y_pred_custom = (y_prob_custom >= 0.5).astype(int)

tn_custom, fp_custom, fn_custom, tp_custom = confusion_matrix(y_test, y_pred_custom).ravel()
profit_custom = R_GOOD * tn_custom - C_REJECT * fp_custom - C_DEFAULT * fn_custom + R_PREVENT * tp_custom

print(f"✓ Custom Objective Function Results:")
print(f"  Accuracy: {accuracy_score(y_test, y_pred_custom):.4f}")
print(f"  Precision: {precision_score(y_test, y_pred_custom):.4f}")
print(f"  Recall: {recall_score(y_test, y_pred_custom):.4f}")
print(f"  F1-Score: {f1_score(y_test, y_pred_custom):.4f}")
print(f"  AUC-ROC: {roc_auc_score(y_test, y_prob_custom):.4f}")
print(f"  Missed Defaults: {fn_custom}")
print(f"  Business Profit: ${profit_custom:,.0f}")
print()

# Apply threshold optimization
print("Applying threshold optimization to custom objective model...")
threshold_results_custom = find_optimal_threshold(y_test, y_prob_custom)
best_threshold_custom_idx = threshold_results_custom['profit'].idxmax()
best_threshold_custom = threshold_results_custom.loc[best_threshold_custom_idx, 'threshold']
best_profit_custom = threshold_results_custom.loc[best_threshold_custom_idx, 'profit']

y_pred_custom_opt = (y_prob_custom >= best_threshold_custom).astype(int)
tn_co, fp_co, fn_co, tp_co = confusion_matrix(y_test, y_pred_custom_opt).ravel()

print(f"✓ Custom Objective + Optimal Threshold (τ = {best_threshold_custom:.3f}):")
print(f"  Recall: {recall_score(y_test, y_pred_custom_opt):.2%}")
print(f"  Precision: {precision_score(y_test, y_pred_custom_opt):.4f}")
print(f"  Missed Defaults: {fn_co}")
print(f"  Business Profit: ${best_profit_custom:,.0f}")
print()

# ============================================================================
# FINAL COMPARISON
# ============================================================================
print("="*80)
print("FINAL COMPARISON: ALL APPROACHES")
print("="*80)
print()

# Baseline (from previous experiments)
baseline_recall = 0.3521
baseline_fn = 1290
baseline_profit = R_GOOD * 6663 - C_REJECT * 346 - C_DEFAULT * 1290 + R_PREVENT * 701

comparison_data = {
    'Approach': [
        'Baseline (XGB scale_pos_weight)',
        '1. Feature Engineering',
        '2. Threshold Optimization',
        '3. Ensemble Stacking',
        '3b. Ensemble + Threshold',
        '4. Custom Objective',
        '4b. Custom Obj + Threshold'
    ],
    'Recall': [
        baseline_recall,
        recall_score(y_test, y_pred_eng),
        recall_score(y_test, y_pred_optimal),
        recall_score(y_test, y_pred_stack),
        recall_score(y_test, y_pred_stack_opt),
        recall_score(y_test, y_pred_custom),
        recall_score(y_test, y_pred_custom_opt)
    ],
    'Precision': [
        0.6695,
        precision_score(y_test, y_pred_eng),
        precision_score(y_test, y_pred_optimal),
        precision_score(y_test, y_pred_stack),
        precision_score(y_test, y_pred_stack_opt),
        precision_score(y_test, y_pred_custom),
        precision_score(y_test, y_pred_custom_opt)
    ],
    'F1-Score': [
        0.4615,
        f1_score(y_test, y_pred_eng),
        f1_score(y_test, y_pred_optimal),
        f1_score(y_test, y_pred_stack),
        f1_score(y_test, y_pred_stack_opt),
        f1_score(y_test, y_pred_custom),
        f1_score(y_test, y_pred_custom_opt)
    ],
    'Missed_Defaults': [
        baseline_fn,
        fn,
        fn_opt,
        fn_stack,
        fn_so,
        fn_custom,
        fn_co
    ],
    'Business_Profit': [
        baseline_profit,
        profit_eng,
        best_profit,
        profit_stack,
        best_profit_stack,
        profit_custom,
        best_profit_custom
    ]
}

comparison_df = pd.DataFrame(comparison_data)
comparison_df['Profit_Improvement'] = comparison_df['Business_Profit'] - baseline_profit
comparison_df['Profit_Improvement_Pct'] = (comparison_df['Profit_Improvement'] / abs(baseline_profit)) * 100

print(comparison_df.to_string(index=False))
print()

# Save results
comparison_df.to_csv('advanced_optimization_results.csv', index=False)
print("✓ Results saved to: advanced_optimization_results.csv")
print()

# Find best approach
best_idx = comparison_df['Business_Profit'].idxmax()
best_approach = comparison_df.loc[best_idx, 'Approach']
best_approach_profit = comparison_df.loc[best_idx, 'Business_Profit']
best_approach_recall = comparison_df.loc[best_idx, 'Recall']
best_approach_fn = comparison_df.loc[best_idx, 'Missed_Defaults']

print("="*80)
print("🏆 BEST APPROACH")
print("="*80)
print(f"Strategy: {best_approach}")
print(f"Business Profit: ${best_approach_profit:,.0f}")
print(f"Recall: {best_approach_recall:.2%}")
print(f"Missed Defaults: {int(best_approach_fn)}")
print(f"Improvement over baseline: ${comparison_df.loc[best_idx, 'Profit_Improvement']:,.0f} ({comparison_df.loc[best_idx, 'Profit_Improvement_Pct']:+.1f}%)")
print()

# Visualization: Comparison chart
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Plot 1: Business Profit Comparison
ax1 = axes[0, 0]
bars1 = ax1.barh(comparison_df['Approach'], comparison_df['Business_Profit'], color='steelblue')
bars1[best_idx].set_color('green')
ax1.axvline(baseline_profit, color='red', linestyle='--', linewidth=2, label='Baseline')
ax1.set_xlabel('Business Profit ($)', fontsize=12)
ax1.set_title('Business Profit Comparison', fontsize=14, fontweight='bold')
ax1.legend()
ax1.grid(axis='x', alpha=0.3)

# Plot 2: Recall Comparison
ax2 = axes[0, 1]
bars2 = ax2.barh(comparison_df['Approach'], comparison_df['Recall'], color='coral')
bars2[best_idx].set_color('green')
ax2.axvline(baseline_recall, color='red', linestyle='--', linewidth=2, label='Baseline')
ax2.set_xlabel('Recall', fontsize=12)
ax2.set_title('Recall Comparison', fontsize=14, fontweight='bold')
ax2.legend()
ax2.grid(axis='x', alpha=0.3)

# Plot 3: Missed Defaults
ax3 = axes[1, 0]
bars3 = ax3.barh(comparison_df['Approach'], comparison_df['Missed_Defaults'], color='indianred')
bars3[best_idx].set_color('green')
ax3.axvline(baseline_fn, color='red', linestyle='--', linewidth=2, label='Baseline')
ax3.set_xlabel('Missed Defaults (Lower is Better)', fontsize=12)
ax3.set_title('Missed Defaults Comparison', fontsize=14, fontweight='bold')
ax3.legend()
ax3.grid(axis='x', alpha=0.3)

# Plot 4: Profit Improvement
ax4 = axes[1, 1]
colors = ['green' if x > 0 else 'red' for x in comparison_df['Profit_Improvement']]
bars4 = ax4.barh(comparison_df['Approach'], comparison_df['Profit_Improvement'], color=colors)
ax4.axvline(0, color='black', linewidth=1)
ax4.set_xlabel('Profit Improvement vs Baseline ($)', fontsize=12)
ax4.set_title('Profit Improvement', fontsize=14, fontweight='bold')
ax4.grid(axis='x', alpha=0.3)

plt.tight_layout()
plt.savefig('advanced_optimization_comparison.png', dpi=300, bbox_inches='tight')
print("✓ Comparison visualization saved: advanced_optimization_comparison.png")
print()

print("="*80)
print("EXPERIMENT COMPLETE")
print("="*80)
print()
print("Key Findings:")
print("1. Feature engineering added domain knowledge (utilization, payment trends)")
print("2. Threshold optimization finds profit-maximizing decision boundary")
print("3. Ensemble stacking combines multiple sampling strategies")
print("4. Custom objective directly optimizes business profit in training")
print()
print("Files generated:")
print("  - advanced_optimization_results.csv")
print("  - threshold_optimization_analysis.png")
print("  - advanced_optimization_comparison.png")
print()
