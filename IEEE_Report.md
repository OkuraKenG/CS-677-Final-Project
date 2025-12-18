# IEEE-Style Report — Credit Card Default Prediction

**Authors:** Fnu Ashutosh, Atharva Pande, Kenji Okura  
**Affiliation:** CS 677 — Machine Learning, Fall 2025  
**Date:** December 2025

---

## Abstract
We present a reproducible study on predicting credit card defaults using the UCI Credit Card Default dataset (30,000 customers). We compare classical and ensemble classifiers, sampling strategies, and objective modifications to evaluate operational trade-offs between recall and false alarm costs. Our experiments (over 1,000 model configurations and sampler/objective variants) show that **cost-sensitive XGBoost** (tuned `scale_pos_weight`) yields the most favorable balance for the business objective of minimizing missed defaults while controlling false alarms. We quantify the expected annual savings under conservative cost assumptions and provide an analysis of why common sampling techniques (e.g., SMOTE/ADASYN) produced suboptimal generalization for this dataset.

**Keywords:** credit default prediction, class imbalance, cost-sensitive learning, XGBoost, SMOTE, feature engineering, business impact

---

## 1. Introduction
Predicting credit card defaults enables financial institutions to proactively mitigate loss and allocate collection resources. This study evaluates modeling strategies on the UCI Credit Card Default dataset with a practical focus: minimize high-cost false negatives (missed defaults) while maintaining reasonable false alarm rates. We emphasize reproducibility and business alignment: all experiments, hyperparameters, and results are available in the `Experiments/` folder.

## 2. Related Work
Classic literature (Yeh & Lien, 2009) demonstrates that tree-based ensembles and logistic baselines are effective for credit default problems. More recent work focuses on imbalance handling, cost-sensitive learning, and synthetic oversampling. Our contribution is a practical comparison that explicitly quantifies business cost trade-offs and presents why certain sampling strategies were not ideal for the payment-history dataset.

## 3. Data and Exploratory Analysis (EDA)
Dataset and key properties (summarized from EDA):
- Total samples: 30,000
- Original features: 23; engineered features: 25+ → final ~48 features
- No missing values; high data quality
- Class balance: 22.12% defaults (ratio ≈ 3.52:1)
- Strong predictors: payment status (`PAY_0..PAY_6`), utilization, `repay_max`
- High temporal correlation among `BILL_AMT*` and `PAY_AMT*` features (lagged dependencies)

Critical EDA observations used in modeling:
- Demographics are weak predictors compared to payment behavior
- Payment/ bill sequences are temporally dependent and highly correlated
- Outliers are legitimate (large bills/payments) and capped rather than removed

## 4. Sampling Strategy Analysis (Why Sampling Techniques Were Not Always Appropriate)
We tested multiple sampling methods (RandomUnderSampling, SMOTE family, ADASYN, and combinations) alongside objective modifications (class weights, `scale_pos_weight`, focal loss). Key observations and empirical evidence from experiments are summarized below.

### 4.1 Structural Reasons Sampling Underperformed
- Temporal and sequential structure: The dataset contains payment and bill sequences across months with high inter-feature correlation. SMOTE/ADASYN produce synthetic examples by interpolation in feature space, which can break sequential consistency (e.g., creating a synthetic record with inconsistent `PAY_*` trajectories). Such inconsistent sequences are unrealistic and harm generalization.

- Feature interdependence: Many features are functionally related (payment ratio = pay_amt / bill_amt). Interpolating on raw features can produce synthetic samples that violate these relationships (negative payments, payment > bill inconsistently), inducing noise.

- Moderate imbalance ratio: The ratio (~3.5:1) is not extreme. Heavy reliance on oversampling may introduce more noise than signal; instead, objective-level adjustments (class weights) preserve the real distribution while emphasizing minority-class errors.

- Boundary and label noise amplification: ADASYN focuses on borderline minority samples; when labels or features have noise, amplification of these points increases false positives on held-out data.

### 4.2 Empirical Evidence
- Cross-validation vs test performance: SMOTE and ADASYN increased recall on cross-validation splits but produced **lower test F1** and **higher false positive rates** on the held-out test set, indicating poorer generalization.

- Cost-sensitive objective (`scale_pos_weight`) consistently produced better test F1 and recall trade-offs than oversampling variants while keeping false positives lower.

- Random undersampling reduced training set size, increased variance, and sometimes decreased recall on the test set despite improving balanced accuracy in CV.

### 4.3 Practical Recommendation
- For this dataset, prefer **cost-sensitive learning (scale_pos_weight)** or calibrated class weights and threshold optimization over blind synthetic oversampling.
- If oversampling is necessary, apply it carefully on **engineered features that preserve temporal/functional relations** (e.g., on aggregated risk scores rather than raw payment time-series), and validate with out-of-sample tests.

## 5. Methods
We evaluate the following strategies:
- Classifiers: XGBoost (baseline and cost-sensitive), Random Forest, GradientBoosting, SVM (RBF/Poly), LogisticRegression (SGD)
- Sampling: None, SMOTE, ADASYN, RandomUnderSampler
- Objectives: `scale_pos_weight`, custom cost matrices, focal loss, threshold tuning
- Feature engineering: `months_late`, `repay_max`, `util_max`, rolling payment trends, `payment_consistency` (std. dev.)

Evaluation metrics included: accuracy, precision, recall, F1-score, AUC, and business metrics (missed defaults and false alarm cost based on user-provided costs).

## 6. Experiments and Results (Summary)
All candidate models were evaluated under stratified 80/20 train-test splits and 5-fold stratified cross-validation for hyperparameter selection. Representative results: 

- **XGBoost (cost-sensitive)** — Accuracy: 77.47% | Precision: 49.23% | Recall: 76.13% | F1: **0.5985** | AUC: 0.8364  
- **Random Forest (tuned)** — Accuracy: 78.03% | Precision: 50.29% | Recall: 58.85% | F1: 0.5424 | AUC: 0.7803  
- **XGBoost (baseline)** — Accuracy: 77.70% | Precision: 49.31% | Recall: 59.57% | F1: 0.5397 | AUC: 0.7821  
- **Logistic Regression (SGD)** — Accuracy: 71.93% | Precision: 42.51% | Recall: 64.95% | F1: 0.5134 | AUC: 0.7464

**Business impact (conservative):** Using cost-sensitive XGBoost reduces missed defaults from 836 → 493 (41% reduction), translating to ~**$3.43M** annual savings under the assumption of $10,000 cost per missed default and $200 cost per false alarm.

**Sampling experiment takeaway:** Synthetic oversampling raised cross-validation recall but produced worse generalization on the test split and worse business cost in several sampler/classifier combinations.

## 7. Observations During Project Work
- **Data Quality:** Dataset from UCI is clean with no missing values — focus rightly placed on feature engineering.
- **Feature Importance:** Payment-history features dominate; engineered features like `repay_max` are highly predictive.
- **Model Behavior:** SVM variants are slow and memory-hungry (not practical for production). Logistic regression is fast and interpretable but lower in predictive power. XGBoost balances performance and speed.
- **Split Variation:** Small metric differences between 70/30 and 80/20 splits highlight the need for consistent experimental design; we used stratified 80/20 as canonical.
- **Model Ceiling:** Learning curves and cross-validation indicate a performance ceiling with available features; additional external attributes (credit bureau scores, income, employment history) are likely to yield larger gains than more complex modeling or aggressive oversampling.
- **Reproducibility Gaps:** The repository had a small set of missing deliverables (final summary, PPT, video). These have been documented in `Experiments/MISSING_INFORMATION_REPORT.md` and are now action items.

## 8. Discussion
- **What worked:** Cost-sensitive XGBoost produced the best operational trade-off for recall and business cost, with consistent cross-validation and test performance. Engineering features that capture payment patterns provided the largest gains.

- **What didn't work / Why:** Off-the-shelf synthetic oversampling (SMOTE/ADASYN) frequently created unrealistic minority samples that violated temporal/functional relationships and reduced test-set generalization.

- **Trade-offs:** Maximizing recall increases false positives; the business cost implication must be assessed. With our cost assumptions, the false-positive cost increase was minor compared to savings from recovered defaults.

## 9. Limitations and Future Work
- **Data limitations:** No external credit bureau variables; payment sequence limited to 6 months.
- **Sampling caveat:** Synthetic oversampling needs domain-aware implementations; naive SMOTE is unsuitable for time-dependent features.
- **Future steps:** Incorporate external features, explore sequence models (RNNs/Temporal models), use SHAP for per-customer explanations, implement production monitoring and monthly retraining.

## 10. Conclusion and Recommendations
- Deploy **XGBoost with cost-sensitive training** and apply threshold optimization aligned to business costs.  
- Avoid blind synthetic oversampling on raw time-series features; if oversampling is required, apply it at aggregated risk-score level or use validated generative approaches.  
- Add external data sources and build explainability (SHAP) for stakeholder acceptance.

---

## Acknowledgements
We thank CS 677 instructors and TAs for guidance and dataset access.

## References
1. Yeh, I.-C., & Lien, C.-H. (2009). The comparisons of data mining techniques for predicting default of credit card clients. Expert Systems with Applications.  
2. Chen, T., & Guestrin, C. (2016). XGBoost: A Scalable Tree Boosting System. KDD '16.  
3. Lemaître, G., et al. (2017). Imbalanced-learn: A Python Toolbox to Tackle the Curse of Imbalanced Datasets in Machine Learning.  

---

## Appendix / Reproducibility
- Experiments and notebooks: `Experiments/` (contains `ML_Final_Project.ipynb`, `cost_sensitive_experiment.ipynb`, `APPROACH_TESTING_RESULTS.md`)  
- Scripts: `test_all_approaches.py`, `comprehensive_verification.py`, `balanced_optimal_solution.py`  
- For presentation and submission deliverables see `Experiments/MISSING_INFORMATION_REPORT.md`

---

