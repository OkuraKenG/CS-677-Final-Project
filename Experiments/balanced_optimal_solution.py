"""
BALANCED OPTIMAL SOLUTION FINDER
=================================
Goal: Find ONE model with threshold that achieves:
- At least 80% recall (catch 80% of defaults)
- At least 80% specificity (approve 80% of non-defaults)
- Maximum business profit

This is the REAL-WORLD constraint!
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    confusion_matrix, accuracy_score, precision_score, 
    recall_score, f1_score, roc_auc_score, roc_curve
)
from sklearn.ensemble import StackingClassifier, RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE
from ucimlrepo import fetch_ucirepo
import warnings
warnings.filterwarnings('ignore')

# Business costs
C_DEFAULT = 10000
C_REJECT = 200
R_GOOD = 200
R_PREVENT = 1000

print("="*80)
print("FINDING THE BALANCED OPTIMAL SOLUTION")
print("="*80)
print()
print("Constraints:")
print("  1. Recall (TPR) >= 80% - Must catch at least 80% of defaults")
print("  2. Specificity (TNR) >= 80% - Must approve at least 80% of non-defaults")
print("  3. Maximize business profit")
print()
print("="*80)
print()

# ============================================================================
# LOAD AND PREPARE DATA
# ============================================================================
print("Loading data...")
credit_card = fetch_ucirepo(id=350)
X = credit_card.data.features
y = credit_card.data.targets.values.ravel()

X.columns = [
    'LIMIT_BAL', 'SEX', 'EDUCATION', 'MARRIAGE', 'AGE',
    'PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6',
    'BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6',
    'PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6'
]

# Feature engineering
def create_features(X_df):
    X_eng = X_df.copy()
    
    # Core utilization features
    for i in range(1, 7):
        X_eng[f'util_{i}'] = X_eng[f'BILL_AMT{i}'] / (X_eng['LIMIT_BAL'] + 1)
    X_eng['avg_util'] = X_eng[[f'util_{i}' for i in range(1, 7)]].mean(axis=1)
    X_eng['max_util'] = X_eng[[f'util_{i}' for i in range(1, 7)]].max(axis=1)
    
    # Payment behavior
    pay_cols = ['PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6']
    X_eng['max_delay'] = X_eng[pay_cols].max(axis=1)
    X_eng['avg_delay'] = X_eng[pay_cols].mean(axis=1)
    X_eng['num_late'] = (X_eng[pay_cols] > 0).sum(axis=1)
    X_eng['payment_trend'] = X_eng['PAY_0'] - X_eng['PAY_6']
    
    # Bill trends
    bill_cols = ['BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6']
    X_eng['avg_bill'] = X_eng[bill_cols].mean(axis=1)
    X_eng['bill_trend'] = X_eng['BILL_AMT1'] - X_eng['BILL_AMT6']
    
    # Payment amounts
    pay_amt_cols = ['PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6']
    X_eng['avg_payment'] = X_eng[pay_amt_cols].mean(axis=1)
    X_eng['payment_ratio'] = X_eng['avg_payment'] / (X_eng['avg_bill'] + 1)
    
    # Risk score
    X_eng['risk_score'] = (
        (X_eng['avg_util'] > 0.8).astype(int) * 3 +
        (X_eng['num_late'] > 3).astype(int) * 2 +
        (X_eng['payment_ratio'] < 0.5).astype(int)
    )
    
    # Fill any NaN values
    X_eng = X_eng.fillna(0)
    
    return X_eng

X_eng = create_features(X)
X_train, X_test, y_train, y_test = train_test_split(
    X_eng, y, test_size=0.3, random_state=42, stratify=y
)

scale_weight = (y_train == 0).sum() / (y_train == 1).sum()
print(f"✓ Data prepared: {len(X_train)} train, {len(X_test)} test")
print(f"✓ Features: {len(X_eng.columns)}")
print(f"✓ Scale weight: {scale_weight:.4f}")
print()

# ============================================================================
# EVALUATION FUNCTION
# ============================================================================
def evaluate_with_constraints(y_true, y_prob, threshold, verbose=True):
    """
    Evaluate model at given threshold with business constraints
    """
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    # Core metrics
    recall = tp / (tp + fn)  # TPR - Catch defaults
    specificity = tn / (tn + fp)  # TNR - Approve non-defaults
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    
    # Business metrics
    profit = R_GOOD * tn - C_REJECT * fp - C_DEFAULT * fn + R_PREVENT * tp
    
    # Check constraints
    meets_recall = recall >= 0.80
    meets_specificity = specificity >= 0.80
    meets_both = meets_recall and meets_specificity
    
    result = {
        'threshold': threshold,
        'recall': recall,
        'specificity': specificity,
        'precision': precision,
        'f1': f1,
        'accuracy': accuracy,
        'tn': tn,
        'fp': fp,
        'fn': fn,
        'tp': tp,
        'profit': profit,
        'meets_recall': meets_recall,
        'meets_specificity': meets_specificity,
        'meets_both': meets_both
    }
    
    if verbose:
        status = "✅ MEETS CONSTRAINTS" if meets_both else "❌ FAILS CONSTRAINTS"
        print(f"τ={threshold:.3f} | Recall={recall:.1%} | Spec={specificity:.1%} | Profit=${profit:,.0f} | {status}")
    
    return result

# ============================================================================
# TEST MULTIPLE MODELS
# ============================================================================
print("="*80)
print("TESTING MULTIPLE MODEL CONFIGURATIONS")
print("="*80)
print()

models_to_test = []

# 1. XGBoost with different scale weights
print("1. Testing XGBoost with various scale_pos_weight values...")
for spw in [2.0, 2.5, 3.0, 3.52, 4.0, 4.5, 5.0]:
    model = XGBClassifier(
        scale_pos_weight=spw,
        n_estimators=150,
        max_depth=4,
        learning_rate=0.1,
        random_state=42,
        eval_metric='logloss'
    )
    model.fit(X_train, y_train)
    y_prob = model.predict_proba(X_test)[:, 1]
    models_to_test.append(('XGB', f'SPW={spw}', y_prob))
    print(f"  ✓ XGBoost SPW={spw} trained")

print()

# 2. XGBoost with SMOTE
print("2. Testing XGBoost with SMOTE...")
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
model = XGBClassifier(
    scale_pos_weight=1.0,
    n_estimators=150,
    max_depth=5,
    learning_rate=0.1,
    random_state=42,
    eval_metric='logloss'
)
model.fit(X_train_smote, y_train_smote)
y_prob = model.predict_proba(X_test)[:, 1]
models_to_test.append(('XGB+SMOTE', 'Balanced', y_prob))
print("  ✓ XGBoost + SMOTE trained")
print()

# 3. Gradient Boosting with different configurations
print("3. Testing Gradient Boosting...")
for lr in [0.05, 0.1]:
    model = GradientBoostingClassifier(
        n_estimators=150,
        max_depth=4,
        learning_rate=lr,
        random_state=42
    )
    # Use sample weights
    sample_weights = np.where(y_train == 1, scale_weight, 1.0)
    model.fit(X_train, y_train, sample_weight=sample_weights)
    y_prob = model.predict_proba(X_test)[:, 1]
    models_to_test.append(('GradBoost', f'LR={lr}', y_prob))
    print(f"  ✓ GradBoost LR={lr} trained")

print()

# 4. Ensemble approaches
print("4. Testing Ensemble (Weighted Average)...")
# Get predictions from best models
xgb_3_52 = models_to_test[3][2]  # XGB SPW=3.52
xgb_smote = models_to_test[7][2]  # XGB+SMOTE
gb = models_to_test[8][2]  # GradBoost

# Different weighting schemes
for w1, w2, w3 in [(0.5, 0.3, 0.2), (0.4, 0.4, 0.2), (0.6, 0.2, 0.2)]:
    y_prob = w1 * xgb_3_52 + w2 * xgb_smote + w3 * gb
    models_to_test.append(('Ensemble', f'W={w1}/{w2}/{w3}', y_prob))
    print(f"  ✓ Ensemble {w1}/{w2}/{w3} created")

print()
print(f"✓ Total models to evaluate: {len(models_to_test)}")
print()

# ============================================================================
# FIND OPTIMAL THRESHOLD FOR EACH MODEL
# ============================================================================
print("="*80)
print("SEARCHING FOR OPTIMAL THRESHOLDS (WITH CONSTRAINTS)")
print("="*80)
print()

all_results = []
thresholds_to_test = np.arange(0.10, 0.91, 0.01)

for model_name, config, y_prob in models_to_test:
    print(f"\n{model_name} ({config}):")
    print("-" * 80)
    
    model_results = []
    for threshold in thresholds_to_test:
        result = evaluate_with_constraints(y_test, y_prob, threshold, verbose=False)
        result['model'] = model_name
        result['config'] = config
        model_results.append(result)
        all_results.append(result)
    
    # Find best threshold that meets constraints
    valid_results = [r for r in model_results if r['meets_both']]
    
    if valid_results:
        best = max(valid_results, key=lambda x: x['profit'])
        print(f"✅ BEST VALID THRESHOLD: τ={best['threshold']:.3f}")
        print(f"   Recall: {best['recall']:.1%} | Specificity: {best['specificity']:.1%}")
        print(f"   Profit: ${best['profit']:,.0f} | F1: {best['f1']:.4f}")
        print(f"   Confusion: TN={best['tn']}, FP={best['fp']}, FN={best['fn']}, TP={best['tp']}")
    else:
        # Show best trade-off even if doesn't meet constraints
        best = max(model_results, key=lambda x: min(x['recall'], x['specificity']))
        print(f"❌ NO THRESHOLD MEETS BOTH CONSTRAINTS")
        print(f"   Best trade-off at τ={best['threshold']:.3f}:")
        print(f"   Recall: {best['recall']:.1%} | Specificity: {best['specificity']:.1%}")

# ============================================================================
# ANALYZE RESULTS
# ============================================================================
print("\n" + "="*80)
print("COMPREHENSIVE ANALYSIS")
print("="*80)
print()

results_df = pd.DataFrame(all_results)

# Filter to only valid solutions
valid_df = results_df[results_df['meets_both']]

print(f"Total configurations tested: {len(results_df)}")
print(f"Configurations meeting both constraints: {len(valid_df)}")
print()

if len(valid_df) > 0:
    # Find absolute best
    best_solution = valid_df.loc[valid_df['profit'].idxmax()]
    
    print("="*80)
    print("🏆 OPTIMAL SOLUTION FOUND!")
    print("="*80)
    print()
    print(f"Model: {best_solution['model']} ({best_solution['config']})")
    print(f"Threshold: τ = {best_solution['threshold']:.3f}")
    print()
    print("PERFORMANCE METRICS:")
    print("-" * 80)
    print(f"  Recall (TPR):        {best_solution['recall']:.2%} ✅ (catches {best_solution['recall']*100:.1f}% of defaults)")
    print(f"  Specificity (TNR):   {best_solution['specificity']:.2%} ✅ (approves {best_solution['specificity']*100:.1f}% of non-defaults)")
    print(f"  Precision:           {best_solution['precision']:.2%}")
    print(f"  F1-Score:            {best_solution['f1']:.4f}")
    print(f"  Accuracy:            {best_solution['accuracy']:.2%}")
    print()
    print("CONFUSION MATRIX:")
    print("-" * 80)
    print(f"  True Negatives:      {int(best_solution['tn']):,} (good customers approved)")
    print(f"  False Positives:     {int(best_solution['fp']):,} (good customers rejected)")
    print(f"  False Negatives:     {int(best_solution['fn']):,} (defaults missed)")
    print(f"  True Positives:      {int(best_solution['tp']):,} (defaults caught)")
    print()
    print("BUSINESS IMPACT:")
    print("-" * 80)
    print(f"  Approval Rate:       {(best_solution['tn']/(best_solution['tn']+best_solution['fp']))*100:.1f}%")
    print(f"  Default Catch Rate:  {(best_solution['tp']/(best_solution['tp']+best_solution['fn']))*100:.1f}%")
    print(f"  Business Profit:     ${best_solution['profit']:,.0f}")
    print()
    
    # Show top 5 solutions
    print("="*80)
    print("TOP 5 VALID SOLUTIONS")
    print("="*80)
    print()
    
    top5 = valid_df.nlargest(5, 'profit')[['model', 'config', 'threshold', 'recall', 'specificity', 'f1', 'profit']]
    print(top5.to_string(index=False))
    print()
    
else:
    # No solution meets both constraints - show best trade-offs
    print("="*80)
    print("⚠️  NO SOLUTION MEETS BOTH CONSTRAINTS")
    print("="*80)
    print()
    print("The 80/80 constraint is too strict for this dataset.")
    print("Showing best trade-offs by different criteria:")
    print()
    
    # Best by minimum of recall and specificity
    results_df['min_tpr_tnr'] = results_df[['recall', 'specificity']].min(axis=1)
    best_balanced = results_df.loc[results_df['min_tpr_tnr'].idxmax()]
    
    print("BEST BALANCED SOLUTION (max of min(Recall, Specificity)):")
    print("-" * 80)
    print(f"  Model: {best_balanced['model']} ({best_balanced['config']})")
    print(f"  Threshold: τ = {best_balanced['threshold']:.3f}")
    print(f"  Recall: {best_balanced['recall']:.2%}")
    print(f"  Specificity: {best_balanced['specificity']:.2%}")
    print(f"  Profit: ${best_balanced['profit']:,.0f}")
    print(f"  Min(Recall, Spec): {best_balanced['min_tpr_tnr']:.2%}")
    print()
    
    # Find solutions at different constraint levels
    for min_rate in [0.75, 0.70, 0.65]:
        relaxed = results_df[(results_df['recall'] >= min_rate) & (results_df['specificity'] >= min_rate)]
        if len(relaxed) > 0:
            best_relaxed = relaxed.loc[relaxed['profit'].idxmax()]
            print(f"BEST SOLUTION AT {min_rate*100:.0f}%/{min_rate*100:.0f}% CONSTRAINT:")
            print("-" * 80)
            print(f"  Model: {best_relaxed['model']} ({best_relaxed['config']})")
            print(f"  Threshold: τ = {best_relaxed['threshold']:.3f}")
            print(f"  Recall: {best_relaxed['recall']:.2%} | Specificity: {best_relaxed['specificity']:.2%}")
            print(f"  Profit: ${best_relaxed['profit']:,.0f}")
            print(f"  Missed defaults: {int(best_relaxed['fn'])}, Rejected good: {int(best_relaxed['fp'])}")
            print()
            break

# ============================================================================
# VISUALIZATION
# ============================================================================
print("="*80)
print("GENERATING VISUALIZATIONS")
print("="*80)
print()

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Plot 1: Recall vs Specificity trade-off
ax1 = axes[0, 0]
for model_name in results_df['model'].unique():
    model_data = results_df[results_df['model'] == model_name]
    ax1.scatter(model_data['specificity'], model_data['recall'], alpha=0.6, s=20, label=model_name)

ax1.axhline(y=0.80, color='red', linestyle='--', linewidth=2, label='80% target')
ax1.axvline(x=0.80, color='red', linestyle='--', linewidth=2)
ax1.fill_between([0.80, 1.0], 0.80, 1.0, alpha=0.2, color='green', label='Target zone')
ax1.set_xlabel('Specificity (TNR - Approve Non-Defaults)', fontsize=12)
ax1.set_ylabel('Recall (TPR - Catch Defaults)', fontsize=12)
ax1.set_title('Recall vs Specificity Trade-off', fontsize=14, fontweight='bold')
ax1.legend(fontsize=8)
ax1.grid(alpha=0.3)

# Plot 2: Profit landscape
ax2 = axes[0, 1]
scatter = ax2.scatter(results_df['specificity'], results_df['recall'], 
                      c=results_df['profit'], s=50, cmap='RdYlGn', alpha=0.6)
plt.colorbar(scatter, ax=ax2, label='Profit ($)')
ax2.axhline(y=0.80, color='red', linestyle='--', linewidth=2)
ax2.axvline(x=0.80, color='red', linestyle='--', linewidth=2)
if len(valid_df) > 0:
    ax2.scatter(best_solution['specificity'], best_solution['recall'], 
                color='red', s=300, marker='*', edgecolors='black', linewidths=2, 
                label='Optimal', zorder=5)
    ax2.legend()
ax2.set_xlabel('Specificity (TNR)', fontsize=12)
ax2.set_ylabel('Recall (TPR)', fontsize=12)
ax2.set_title('Profit Landscape', fontsize=14, fontweight='bold')
ax2.grid(alpha=0.3)

# Plot 3: F1 Score vs Profit
ax3 = axes[1, 0]
for model_name in results_df['model'].unique():
    model_data = results_df[results_df['model'] == model_name]
    ax3.scatter(model_data['f1'], model_data['profit'], alpha=0.6, s=20, label=model_name)
if len(valid_df) > 0:
    ax3.scatter(best_solution['f1'], best_solution['profit'], 
                color='red', s=300, marker='*', edgecolors='black', linewidths=2)
ax3.set_xlabel('F1-Score', fontsize=12)
ax3.set_ylabel('Business Profit ($)', fontsize=12)
ax3.set_title('F1-Score vs Business Profit', fontsize=14, fontweight='bold')
ax3.legend(fontsize=8)
ax3.grid(alpha=0.3)

# Plot 4: Constraint satisfaction by model
ax4 = axes[1, 1]
constraint_summary = results_df.groupby('model').agg({
    'meets_both': 'sum',
    'meets_recall': 'sum',
    'meets_specificity': 'sum'
}).reset_index()

x = np.arange(len(constraint_summary))
width = 0.25
ax4.bar(x - width, constraint_summary['meets_recall'], width, label='Recall ≥ 80%', alpha=0.8)
ax4.bar(x, constraint_summary['meets_specificity'], width, label='Specificity ≥ 80%', alpha=0.8)
ax4.bar(x + width, constraint_summary['meets_both'], width, label='Both ≥ 80%', alpha=0.8, color='green')
ax4.set_xlabel('Model', fontsize=12)
ax4.set_ylabel('# Thresholds Meeting Constraint', fontsize=12)
ax4.set_title('Constraint Satisfaction by Model', fontsize=14, fontweight='bold')
ax4.set_xticks(x)
ax4.set_xticklabels(constraint_summary['model'], rotation=45, ha='right')
ax4.legend()
ax4.grid(alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('balanced_optimal_solution_analysis.png', dpi=300, bbox_inches='tight')
print("✓ Visualization saved: balanced_optimal_solution_analysis.png")
print()

# Save results
results_df.to_csv('balanced_solution_search_results.csv', index=False)
print("✓ Results saved: balanced_solution_search_results.csv")
print()

print("="*80)
print("ANALYSIS COMPLETE!")
print("="*80)
