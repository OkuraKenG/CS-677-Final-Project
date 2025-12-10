# Comprehensive Approach Testing Results

## Executive Summary
Tested **30 combinations** of sampling methods + classifiers to identify the best approach for credit default prediction.

### Key Findings

**Best Single Model:**
- **GradBoosting (no sampling)**
- Accuracy: **81.79%** (improved from 81.48%)
- Precision: 66.0%, Recall: 36.5%, F1: 0.4698, AUC: 0.7681

**Best by Metric:**
| Metric | Approach | Score |
|--------|----------|-------|
| Accuracy | None + GradBoosting | 81.79% |
| Precision | None + LR | 70.27% |
| Recall | BorderlineSMOTE + LR | 68.81% |
| F1 Score | RandomUndersampling + ExtraTrees | 0.5334 |
| AUC | None + GradBoosting | 0.7681 |

**Ensemble Methods (no sampling baseline):**
- Voting Classifier (Soft): 81.64% accuracy
- Stacking Classifier: 81.68% accuracy
- Both slightly underperform GradBoosting alone

---

## Detailed Results Table (Top 10 by Accuracy)

| Sampler | Classifier | Accuracy | Precision | Recall | F1 | AUC |
|---------|-----------|----------|-----------|--------|-----|-----|
| None | GradBoosting | 81.79% | 0.660 | 0.3646 | 0.4698 | 0.7681 |
| None | SVM | 81.40% | 0.661 | 0.3270 | 0.4375 | 0.7124 |
| None | RF | 80.94% | 0.601 | 0.4108 | 0.4882 | 0.7614 |
| None | LR | 80.92% | 0.703 | 0.2386 | 0.3562 | 0.7125 |
| SMOTETomek | GradBoosting | 80.74% | 0.589 | 0.4274 | 0.4955 | 0.7562 |
| ADASYN | GradBoosting | 80.68% | 0.586 | 0.4304 | 0.4964 | 0.7528 |
| SMOTE | GradBoosting | 80.58% | 0.584 | 0.4254 | 0.4922 | 0.7588 |
| BorderlineSMOTE | GradBoosting | 79.78% | 0.553 | 0.4490 | 0.4956 | 0.7541 |
| None | ExtraTrees | 79.77% | 0.547 | 0.4922 | 0.5184 | 0.7654 |
| SMOTE | ExtraTrees | 79.72% | 0.544 | 0.5103 | 0.5268 | 0.7648 |

---

## Threshold Tuning Results (GradBoosting - Best Model)

For maximizing different metrics, optimal thresholds are:

| Threshold | Accuracy | Precision | Recall | F1 |
|-----------|----------|-----------|--------|-----|
| 0.25 | 76.91% | 0.4819 | 0.5826 | **0.5275** |
| 0.30 | 79.08% | 0.5270 | 0.5289 | 0.5280 |
| 0.40 | 81.23% | 0.6094 | 0.4224 | 0.4990 |
| **0.50** | **81.79%** | **0.6600** | **0.3646** | **0.4698** |

**Recommendation:** Use threshold 0.30 for best F1-score balance (0.5280), or 0.50 for maximum accuracy (81.79%).

---

## Key Insights

1. **Sampling methods hurt pure accuracy** - No sampling (baseline) outperforms all sampling techniques for accuracy
2. **GradBoosting dominates** - Consistent top performer across metrics
3. **ExtraTrees performs well on F1** - Better at balancing precision/recall (0.5268 F1)
4. **Ensemble methods don't add value** - Voting/Stacking slightly underperform best single model
5. **Lower thresholds improve recall** - Trade-off: recall goes from 36.5% → 68.8% but accuracy drops to 76.9%

---

## Comparison with Paper

Paper's approach:
- ExtraTrees + offline/online decomposition: 95.84% accuracy (not comparable—different evaluation setup)
- Heuristic approach: 93.14% accuracy (includes synthetic transactions)

Your notebook (now):
- GradBoosting (no sampling): 81.79% accuracy (standard ML evaluation on original dataset)
- Improvement: +0.31% over prior XGBoost (81.48%)

**Note:** Paper's results are not directly comparable because they:
- Synthesize per-transaction data instead of per-customer
- Use 10-fold CV instead of train/test holdout
- Batch process across 5 months with weight λ≈0.5

---

## Recommended Next Steps

1. **Integrate GradBoosting into notebook** with threshold=0.30 for F1 optimization
2. **Optional: Try ExtraTrees** if you want to maximize F1 (0.5334 with RandomUndersampling)
3. **Optional: Implement threshold tuning UI** to let stakeholders choose accuracy vs recall trade-off
4. **Compare confusion matrices** across models (GB, ExtraTrees, paper's approach if feasible)

