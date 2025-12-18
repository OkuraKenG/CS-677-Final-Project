# CS-677 Final Project — Credit Card Default Prediction

A comprehensive final project repository for CS-677 (Fall 2025) that predicts credit card defaults using classical and advanced machine learning techniques. This repository contains raw data references, notebooks, scripts, and reproducible experiments demonstrating that cost-sensitive learning (XGBoost with `scale_pos_weight`) provides the best practical trade-off between catching defaults and minimizing business loss.

---

## Quick summary ✅
- Dataset: UCI Credit Card Default (30,000 customers, 23 original features)
- Problem: Binary classification - predict next-month default
- Main result: **Cost-sensitive XGBoost with `scale_pos_weight=3.52`** yields the best practical balance (recommended for deployment)
- Key metric emphasis: **Recall / F1 / Business cost (missed defaults)**

---

## What is in this repo 🔧
- `Final_Project/` — root project folder
  - `Experiments/` — all experiment notebooks, scripts, and analysis (primary source of truth)
  - `ML_Final_Project.ipynb` — main team notebook (multi-section)
  - `final_machine_learning_project_ashutosh.ipynb` — polished narrative notebook
  - `project_report.md` — final project report (paper-style summary)
  - `README.md` — this file
  - `requirements.txt` — pinned dependencies for reproducibility

---

## Key findings (high level) 📊
- Class imbalance is significant (default: 22.12%, ratio ≈ 3.52:1)
- PAYMENT STATUS features (PAY_0..PAY_6) and credit utilization are the strongest predictors
- Cost-sensitive learning (XGBoost with `scale_pos_weight`) reduces missed defaults by ~40% with acceptable trade-offs in false alarms
- Exhaustive verification shows a ~70% recall/specificity ceiling with current features — improving beyond this requires additional external data (credit bureau, income, employment)

---

## How to reproduce 🔁
1. Clone the repo and create a virtual environment

```bash
git clone <repo-url>
cd Final_Project
python -m venv venv
# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate
pip install -r requirements.txt
```

2. Run the primary notebooks:
- `final_machine_learning_project_ashutosh.ipynb` — narrative flow and business-oriented analysis
- `ML_Final_Project.ipynb` — full technical experiments
- `Experiments/cost_sensitive_experiment.ipynb` — cost-sensitive model exploration

3. Recommended single command (example):
```bash
jupyter nbconvert --ExecutePreprocessor.timeout=600 --to notebook --execute ML_Final_Project.ipynb --output executed_notebook.ipynb
```

---

## Recommended model & deployment notes 🚀
- **Model:** `xgboost.XGBClassifier(scale_pos_weight=3.5210, objective='binary:logistic', n_estimators=100, max_depth=4, learning_rate=0.1, random_state=42)`
- **Decision:** favor recall (catch defaults) — set threshold based on business trade-offs (0.25–0.35 recommended for F1/Recall balance)
- **Monitoring:** track recall and false alarm rates; retrain monthly or when drift is detected

---

## Files of interest (short map) 🗺️
- `Experiments/APPROACH_TESTING_RESULTS.md` — consolidated comparison (42 model combinations)
- `Experiments/cost_sensitive_experiment_results.md` — cost-sensitive analysis and business impact
- `Experiments/EXHAUSTIVE_VERIFICATION_REPORT.md` — exhaustive search summary and 70/70 ceiling rationale
- `Experiments/NOTEBOOK_COMPARISON_ANALYSIS.md` — notebook review and recommendation
- `Experiments/advanced_optimization_experiment.py` — feature engineering and optimization scripts
- `Experiments/comprehensive_verification.py` — verification and large grid search results

---

## A short note about variability 🔬
Different experiment runs (different train/test splits or hyperparameter settings) may report slightly different metrics. The most robust conclusions (cost-sensitive XGBoost as the recommended approach and the ~70% ceiling) are consistent across all experiments in the `Experiments/` folder. See `APPROACH_TESTING_RESULTS.md` and `EXHAUSTIVE_VERIFICATION_REPORT.md` for full reproducibility details.

---

## Contact & Citation ✉️
If you use this work in coursework or a report, please cite the project and include the team members listed in `project_report.md`.

---

Thank you for reviewing the project — open an issue or email the team for any clarifications.
