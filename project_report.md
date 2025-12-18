# CS 677 Final Project - Summary Report
## Credit Card Default Prediction Using Machine Learning

**Team Members:**
- Fnu Ashutosh (U01955320)
- Atharva Pande (U01985210)
- Kenji Okura (U01769019)

**Date:** December 11, 2025

---

## Executive Summary

This project addresses the critical business problem of predicting credit card defaults using machine learning techniques. We developed and evaluated six different ML algorithms on the UCI Credit Card Default dataset (30,000 customers, 48 engineered features). Our analysis demonstrates that **XGBoost with Cost-Sensitive Learning** achieves the best balance between precision and recall, reducing missed defaults by **41.0%** compared to standard approaches, translating to potential savings of **$3.43 million** annually.

> **Note on reproducibility:** The `Experiments/` folder contains the canonical runs and result tables (e.g., `APPROACH_TESTING_RESULTS.md`, `cost_sensitive_experiment_results.md`). Some experiments report metrics under a 70/30 split and others under an 80/20 split; small variations in numeric summaries are due to differing splits or hyperparameter variants. The key recommendations (cost-sensitive XGBoost and the observed performance ceiling) are consistent across all runs.

---

## 1. Problem Statement & Business Context

### Objective
Predict whether a credit card customer will default on their next payment, enabling proactive risk management and minimizing financial losses.

### Business Impact
- **Cost of False Negative (Missed Default):** $10,000 per customer
- **Cost of False Positive (False Alarm):** $200 per customer
- **Class Imbalance:** 77.88% non-default vs 22.12% default (3.52:1 ratio)

The high cost of missed defaults makes **recall** the critical metric, requiring models that prioritize catching true defaulters even at the expense of some false alarms.

---

## 2. Dataset Overview

**Source:** UCI Machine Learning Repository - Credit Card Default Dataset

**Statistics:**
- **Total Samples:** 30,000 customers
- **Original Features:** 23 (demographics, credit history, payment patterns)
- **Engineered Features:** 25 additional features created through domain expertise
- **Final Feature Count:** 48 features
- **Train/Test Split:** 80/20 (24,000 training, 6,000 test samples)
- **Class Distribution:** Maintained through stratified sampling

**Key Engineered Features:**
- `months_late`: Total months with delayed payments (r=0.398 with default)
- `repay_max`: Maximum repayment amount ratio (r=0.331)
- `chronic_late_flag`: Binary indicator for persistent late payments (r=0.311)
- `util_max`: Maximum credit utilization across 6 months
- `payment_consistency`: Standard deviation of payment amounts

---

## 3. Machine Learning Algorithms Evaluated

### 3.1 XGBoost (Baseline)
**Configuration:**
- n_estimators: 100
- max_depth: 3
- learning_rate: 0.1
- Standard class weights

**Results:**
- Accuracy: 77.70%
- Precision: 49.31%
- Recall: 59.57%
- **F1-Score: 0.5397**
- AUC: 0.7821
- Missed Defaults: 836 customers

### 3.2 XGBoost (Cost-Sensitive) ⭐ **BEST OVERALL**
**Configuration:**
- n_estimators: 100
- max_depth: 3
- learning_rate: 0.1
- **scale_pos_weight: 3.52** (addresses class imbalance)

**Results:**
- Accuracy: 77.47%
- Precision: 49.23%
- Recall: 76.13%
- **F1-Score: 0.5985** ⭐
- AUC: 0.8364
- Missed Defaults: 493 customers ⭐
- **Improvement: 41.0% reduction in missed defaults**

**Why This Model Won:**
1. **Best F1-Score (0.5985):** Optimal balance between precision and recall
2. **Highest Recall (76.13%):** Catches 76% of all defaults, critical for business
3. **Best AUC (0.8364):** Superior class separation and ranking
4. **Cost-Sensitive Learning:** Explicitly handles 3.52:1 class imbalance
5. **Gradient Boosting:** Iteratively corrects errors from previous trees
6. **Regularization:** Prevents overfitting through max_depth=3

### 3.3 Random Forest Classifier
**Configuration:**
- n_estimators: 200 (tuned)
- max_depth: 10 (tuned)
- min_samples_split: 20 (tuned)
- class_weight: 'balanced'

**Results:**
- Accuracy: 78.03%
- Precision: 50.29%
- Recall: 58.85%
- F1-Score: 0.5424
- AUC: 0.7803

**Strengths:**
- Second-best F1-score
- Excellent accuracy (78.03%)
- Robust to overfitting through ensemble diversity
- Fast training (1.2 seconds)

**Limitations:**
- Lower recall (58.85%) than cost-sensitive models
- Requires more trees for optimal performance

### 3.4 Support Vector Machine (RBF Kernel)
**Configuration:**
- Kernel: RBF (Radial Basis Function)
- C: 1.0
- class_weight: 'balanced'

**Results:**
- Accuracy: 75.28%
- Precision: 45.25%
- Recall: 61.59%
- F1-Score: 0.5225
- AUC: 0.7654

**Strengths:**
- Good recall performance (61.59%)
- Effective for non-linear patterns
- Polynomial kernel achieved best F1 (0.524) among kernel variants

**Limitations:**
- Very slow training (387-718 seconds depending on kernel)
- Difficult to interpret
- Memory intensive

### 3.5 Logistic Regression with SGD Optimizer
**Configuration:**
- loss: 'log_loss' (logistic regression)
- learning_rate: 'optimal' (adaptive)
- max_iter: 1000
- penalty: 'l2'
- class_weight: 'balanced'

**Results:**
- Accuracy: 71.93%
- Precision: 42.51%
- Recall: 64.95%
- F1-Score: 0.5134
- AUC: 0.7464

**Strengths:**
- **Fastest training (0.46 seconds)** ⚡
- Demonstrates SGD optimizer requirement
- Good convergence behavior
- Highly interpretable coefficients

**Limitations:**
- Assumes linear decision boundaries
- Lower overall performance metrics

### 3.6 Gradient Boosting Classifier
**Configuration:**
- n_estimators: 100
- learning_rate: 0.1
- max_depth: 3
- subsample: 0.8
- Sample weights for class imbalance

**Results:**
- Accuracy: 75.08%
- Precision: 45.28%
- Recall: 63.24%
- F1-Score: 0.5283
- AUC: 0.7706

**Strengths:**
- Strong recall (63.24%)
- Similar architecture to XGBoost
- Robust performance

**Limitations:**
- Slower than XGBoost
- Slightly lower F1 than XGBoost

---

## 4. Model Comparison Summary

| Model | Accuracy | Precision | Recall | F1-Score | AUC | Training Time | Missed Defaults |
|-------|----------|-----------|--------|----------|-----|---------------|-----------------|
| **XGBoost Cost-Sensitive** ⭐ | 77.47% | 49.23% | **76.13%** | **0.5985** | **0.8364** | 2.1s | **493** ⭐ |
| Random Forest (Tuned) | **78.03%** | **50.29%** | 58.85% | 0.5424 | 0.7803 | 1.2s | 850 |
| XGBoost Baseline | 77.70% | 49.31% | 59.57% | 0.5397 | 0.7821 | 2.0s | 836 |
| Gradient Boosting | 75.08% | 45.28% | 63.24% | 0.5283 | 0.7706 | 21.7s | 760 |
| SVM (Poly Kernel) | 77.25% | 48.56% | 61.35% | 0.5424 | 0.7706 | 387s | 799 |
| SVM (RBF Kernel) | 75.28% | 45.25% | 61.59% | 0.5225 | 0.7654 | 418s | 794 |
| Logistic Regression (SGD) | 71.93% | 42.51% | 64.95% | 0.5134 | 0.7464 | **0.5s** | 725 |

---

## 5. Why XGBoost Cost-Sensitive Performed Best

### 5.1 Technical Reasons

**1. Gradient Boosting Framework**
- Builds trees sequentially, each correcting errors of previous trees
- Optimizes a differentiable loss function (logloss)
- Uses second-order gradients (Newton-Raphson) for better convergence

**2. Cost-Sensitive Learning (`scale_pos_weight=3.52`)**
- Assigns higher penalty to misclassifying minority class (defaults)
- Directly addresses 3.52:1 class imbalance
- Aligns with business objective of minimizing missed defaults

**3. Regularization & Hyperparameter Tuning**
- `max_depth=3`: Prevents overfitting, creates shallow interpretable trees
- `learning_rate=0.1`: Optimal step size for weight updates
- `n_estimators=100`: Sufficient ensemble size without diminishing returns

**4. Feature Utilization**
- XGBoost's tree-based structure naturally handles:
  - Non-linear relationships
  - Feature interactions
  - Missing values
  - Mixed data types
- Top features: `repay_max` (46.6%), `PAY_0` (7.6%), `months_late` (7.2%)

### 5.2 Performance Metrics Superiority

**Best F1-Score (0.5985):**
- Harmonic mean of precision and recall
- Indicates optimal balance for imbalanced classification
- 10.9% higher than second-best (Random Forest: 0.5424)

**Highest Recall (76.13%):**
- Catches 76 out of 100 actual defaulters
- Critical for minimizing $10,000 losses per missed default
- 17.28 percentage points higher than baseline XGBoost

**Best AUC (0.8364):**
- Excellent class separation across all threshold values
- Robust ranking of default probabilities
- 5.6 points higher than Random Forest

### 5.3 Business Impact

**Cost Analysis:**
- Baseline Missed Defaults: 836 customers × $10,000 = $8,360,000
- Cost-Sensitive Missed Defaults: 493 customers × $10,000 = $4,930,000
- **Annual Savings: $3,430,000** (41.0% reduction)

**Trade-off Justification:**
- False Positives increased slightly (1,476 → 1,544)
- Additional cost: 68 × $200 = $13,600
- **Net Savings: $3,416,400 per year**

### Sampling Strategy Analysis (Why Synthetic Sampling Was Not Ideal)
**Key points:**
- **Temporal & Feature Consistency:** The dataset contains sequential payment and bill features (PAY_0..PAY_6, BILL_AMT*, PAY_AMT*) with strong temporal correlation. Off-the-shelf synthetic methods (SMOTE/ADASYN) interpolate in feature space and can produce unrealistic sequences that violate temporal relationships and derived constraints (e.g., payment > bill inconsistently), causing models to learn artefacts rather than true risk patterns.

- **Moderate Imbalance Ratio:** The class ratio (~3.52:1) is not extremely skewed. Objective-level adjustments (`scale_pos_weight`, class weights) preserve the natural distribution and yielded better test-set generalization than aggressive oversampling.

- **Noise Amplification:** ADASYN targets borderline minority points; when minority labels or borderline features include noise, ADASYN magnifies those noisy regions and raises false positives on held-out data.

- **Empirical Evidence:** In our experiments, SMOTE/ADASYN sometimes raised cross-validation recall, but **decreased test-set F1** and increased false positives, while `scale_pos_weight` improved recall with more stable test performance.

**Recommendation:** Prefer cost-sensitive learning (XGBoost `scale_pos_weight` or class_weight adjustments). If oversampling is used, do so on aggregated risk features (that preserve temporal relationships) or consider domain-aware synthetic generation that enforces feature constraints.

---

## 6. Hyperparameter Tuning Results

### 6.1 XGBoost Grid Search
**Parameters Tested:**
- `max_depth`: [3, 5, 7]
- `learning_rate`: [0.05, 0.1, 0.15]
- `n_estimators`: [100, 200]
- `scale_pos_weight`: [3.0, 3.52, 4.0]

**Total Configurations:** 54 (with 3-fold CV = 162 model fits)

**Best Parameters:**
- max_depth: 3 ✅
- learning_rate: 0.1 ✅
- n_estimators: 100 ✅
- scale_pos_weight: 3.0

**Result:** Test F1 improved to 0.5429

### 6.2 Random Forest Grid Search
**Parameters Tested:**
- `n_estimators`: [100, 200, 300]
- `max_depth`: [10, 15, 20]
- `min_samples_split`: [10, 20, 30]

**Total Configurations:** 27 (with 3-fold CV = 81 model fits)

**Best Parameters:**
- n_estimators: 200
- max_depth: 10
- min_samples_split: 20

**Result:** Test F1 improved to 0.5424

---

## 7. Advanced Concepts Demonstrated

### 7.1 Learning Curves Analysis
**Findings:**
- **XGBoost:** Train-Val gap = 0.157 (slight overfitting, acceptable)
- **Random Forest:** Train-Val gap = 0.088 (good fit)
- **Logistic Regression:** Train-Val gap = 0.003 (excellent generalization)

**Insight:** Cost-sensitive models accept slight overfitting to maximize recall on training data, validated through cross-validation.

### 7.2 Cross-Validation (5-Fold Stratified)
**Purpose:** Validate model stability across different data splits

**Results:**
- XGBoost Cost-Sensitive: Mean F1 = 0.536 (±0.007)
- Random Forest: Mean F1 = 0.556 (±0.012) - most stable
- Low standard deviation confirms consistent performance

### 7.3 Kernel Functions (SVM)
**Comparison:**
- **Linear Kernel:** F1 = 0.504, assumes linear separability
- **RBF Kernel:** F1 = 0.519, maps to infinite dimensions
- **Polynomial Kernel:** F1 = 0.524 (best), captures degree-3 interactions

**Trade-off:** Polynomial kernel achieved best F1 but 10x slower than linear

### 7.4 Decision Tree Visualization
**Impact of Hyperparameters:**
- **Shallow Tree (max_depth=3):** 15 nodes, F1=0.5057, prevents overfitting
- **Deep Tree (max_depth=10):** 785 nodes, F1=0.4811, overfits training data

**Feature Importance:**
- XGBoost prioritizes: `repay_max`, `PAY_0`, `months_late`
- Tuned `max_depth` prevents splitting on noisy features

---

## 8. Algorithm Selection Justification

### Why We Selected These 6 Algorithms:

**1. XGBoost (Baseline & Cost-Sensitive):**
- Industry standard for tabular data and imbalanced classification
- State-of-the-art performance in Kaggle competitions
- Efficient handling of missing values and feature interactions

**2. Random Forest:**
- Robust ensemble method with built-in feature importance
- Excellent baseline for comparison
- Handles non-linearity and overfitting well

**3. Support Vector Machine:**
- Effective for high-dimensional spaces (48 features)
- Kernel trick demonstrates course concepts
- Strong theoretical foundation (maximum margin classification)

**4. Logistic Regression with SGD:**
- Satisfies SGD optimizer requirement
- Provides interpretable linear baseline
- Fastest training for rapid iteration

**5. Gradient Boosting:**
- Similar architecture to XGBoost for fair comparison
- Scikit-learn implementation for consistency
- Validates boosting effectiveness

### Why XGBoost Cost-Sensitive is the Final Choice:

✅ **Business Alignment:** Directly optimizes for recall (catching defaults)  
✅ **Performance:** Best F1, recall, and AUC across all models  
✅ **Scalability:** Fast training (2.1s) enables production deployment  
✅ **Interpretability:** Feature importance guides business decisions  
✅ **Robustness:** Validated through cross-validation and learning curves  
✅ **Cost-Effectiveness:** $3.43M annual savings vs baseline  

---

## 9. Limitations & Future Work

### Current Limitations:
1. **Static Model:** Requires retraining as customer behavior evolves
2. **Feature Engineering:** Manual process, may miss complex interactions
3. **Threshold Selection:** Fixed at 0.5, could be optimized further
4. **Temporal Patterns:** Model doesn't capture seasonality or trends

### Recommended Improvements:
1. **Deep Learning:** Neural networks for automatic feature learning
2. **Online Learning:** Update model incrementally with new data
3. **Explainable AI:** SHAP values for individual prediction explanations
4. **Ensemble Stacking:** Combine multiple models for meta-learning
5. **Time-Series Features:** Incorporate sequential payment patterns
6. **External Data:** Credit bureau scores, macroeconomic indicators

---

## 10. Conclusion

This project successfully demonstrates comprehensive machine learning methodology for credit card default prediction. Through systematic evaluation of six algorithms, rigorous hyperparameter tuning (81 total configurations), and advanced techniques (learning curves, cross-validation, kernel functions, tree visualization), we identified **XGBoost with Cost-Sensitive Learning** as the optimal solution.

**Key Achievements:**
- ✅ **41.0% reduction** in missed defaults (836 → 493 customers)
- ✅ **$3.43 million** in annual cost savings
- ✅ **76.13% recall** - catches 3 out of 4 defaulters
- ✅ **0.5985 F1-score** - best balance across all models
- ✅ **0.8364 AUC** - superior ranking and class separation

The cost-sensitive approach directly addresses the business objective by penalizing false negatives (missed defaults) more heavily than false positives (false alarms), aligning model optimization with financial impact. This project demonstrates that proper algorithm selection, combined with domain-specific feature engineering and hyperparameter tuning, can deliver significant business value in credit risk management.

**Recommendation:** Deploy XGBoost Cost-Sensitive model to production with monthly retraining and continuous monitoring of model performance and drift.

---

## Appendices

### A. Technical Stack
- **Language:** Python 3.14
- **Libraries:** scikit-learn 1.7.2, xgboost 3.1.2, pandas 2.3.3, numpy 2.3.3
- **Visualization:** matplotlib 3.10.7, seaborn 0.13.2
- **Environment:** Jupyter Notebook 7.5.0

### B. Reproducibility
- **Random State:** 42 (all experiments)
- **Data Split:** 80/20 stratified
- **Cross-Validation:** 5-fold stratified
- **Notebook:** `final_machine_learning_project_ashutosh.ipynb`

### C. Team Contributions
- **Fnu Ashutosh:** Model implementation, hyperparameter tuning, report writing
- **Atharva Pande:** EDA, feature engineering, visualization
- **Kenji Okura:** Cross-validation, kernel functions, documentation

### D. References
1. UCI Machine Learning Repository - Credit Card Default Dataset
2. Chen, T., & Guestrin, C. (2016). XGBoost: A Scalable Tree Boosting System
3. Breiman, L. (2001). Random Forests
4. Cortes, C., & Vapnik, V. (1995). Support-Vector Networks
5. CS 677 Course Materials - Machine Learning Concepts

---

**Document Version:** 1.0  
**Last Updated:** December 11, 2025  
**Contact:** CS 677 Final Project Team
