================================================================================
                    EXPLORATORY DATA ANALYSIS - SUMMARY
================================================================================

📊 DATASET CHARACTERISTICS
--------------------------------------------------------------------------------
  • Total Samples: 30,000
  • Total Features: 23
  • Missing Values: 0 (100% complete dataset)
  • Memory Usage: 5.26 MB
  • Data Quality: High (clean UCI repository dataset)

🎯 TARGET VARIABLE (Default Payment)
--------------------------------------------------------------------------------
  • Default Rate: 22.12% (6,636 defaults)
  • Non-Default Rate: 77.88% (23,364 no defaults)
  • Imbalance Ratio: 3.52:1
  • Implication: Requires SMOTE or class weighting in modeling

👥 DEMOGRAPHIC INSIGHTS
--------------------------------------------------------------------------------
  • Gender: 60.4% Female, 39.6% Male
  • Age: Mean 35.5 years, Range 21-79 years
  • Education: 46.8% University, 35.3% Graduate, 16.4% High School
  • Marital Status: 53.2% Single, 45.5% Married
  • Default Variance: Demographics show 19-27% default rates (modest variation)
  • Key Finding: Demographics are weak predictors compared to payment behavior

💳 CREDIT LIMIT & UTILIZATION
--------------------------------------------------------------------------------
  • Average Credit Limit: NT$ 167,484
  • Credit Limit Range: NT$ 10,000 - NT$ 1,000,000
  • Distribution: Right-skewed (skewness = 0.993)
  • Most Common: NT$ 50K-100K and NT$ 100K-200K brackets
  • Average Utilization: 37.2%
  • High Utilization (>70%): 7,232 customers (24.1%)
  • Key Finding: High utilization (>70%) strongly correlates with default risk

📅 PAYMENT STATUS (PAY_0 to PAY_6)
--------------------------------------------------------------------------------
  • Average On-Time Payments: 32.8% of observations
  • Average Delayed Payments: 13.9% of observations
  • Payment Status Range: -2 (fully paid) to 9 (9+ months delay)
  • Key Finding: PAY features show STRONGEST correlation with default (0.2-0.3)

💰 BILL AMOUNTS (BILL_AMT1 to BILL_AMT6)
--------------------------------------------------------------------------------
  • Average Bill Amount: NT$ 44,977
  • Bill Range: NT$ -339,603 to NT$ 1,664,089
  • Negative Bills: Valid (represent credits/refunds)
  • Outliers: ~8-9% per month (high spending)
  • Key Finding: Sequential bills highly correlated (>0.9) - temporal dependency

💵 PAYMENT AMOUNTS (PAY_AMT1 to PAY_AMT6)
--------------------------------------------------------------------------------
  • Average Payment: NT$ 5,275
  • Zero Payments: 20.5% of observations (no payment made)
  • Payment Ratio: Average 33.9% of bill paid
  • Full Payments: 25.4% pay ≥100% of bill
  • Key Finding: Low payment ratios (<25%) indicate financial distress

🔗 CORRELATION & MULTICOLLINEARITY
--------------------------------------------------------------------------------
  • Top Predictors: PAY_0 (0.324), PAY_2 (0.264), PAY_3 (0.234)
  • Weak Predictors: Demographics (correlation <0.1)
  • Multicollinearity: High within feature groups (PAY, BILL, PAY_AMT)
  • Sequential Features: Expected high correlation (temporal series)
  • Key Finding: Feature engineering needed to capture temporal patterns

⚠️ OUTLIERS DETECTED
--------------------------------------------------------------------------------
  • Credit Limits: 167 outliers (0.56%) - Premium cards >NT$525K
  • Age: 272 outliers (0.91%) - Senior customers 61-79 years
  • Bill Amounts: ~2,400-2,700 outliers (8-9%) per month
  • Payment Amounts: ~2,600-3,000 outliers (9-10%) per month
  • Treatment: Keep demographics, cap bill/payment at 99th percentile

================================================================================
🎯 CRITICAL INSIGHTS FOR MODELING
================================================================================

1. CLASS IMBALANCE (3.52:1 ratio)
   → Use SMOTE oversampling or class_weight='balanced' in models
   → Focus on Recall and F1-Score, not just Accuracy
   
2. PAYMENT STATUS = PRIMARY PREDICTOR
   → PAY_0 to PAY_6 have strongest correlation with default (0.2-0.3)
   → Engineer features: payment deterioration, consistency, momentum
   
3. CREDIT UTILIZATION = KEY RISK INDICATOR
   → High utilization (>70%) correlates with elevated default risk
   → Create utilization ratio features and trend indicators
   
4. TEMPORAL DEPENDENCIES
   → Sequential months show high correlation (>0.9)
   → Create lag features, rolling averages, trend indicators
   
5. FEATURE ENGINEERING PRIORITY
   → Payment behavior patterns (deterioration, skipped payments)
   → Utilization ratios and trends
   → Payment-to-bill ratios
   → Demographic interactions with financial behavior
   
6. DEMOGRAPHICS = WEAK PREDICTORS
   → Age, Gender, Education show <0.1 correlation with default
   → Use as supplementary features, not primary predictors
   → May be useful for interaction terms with financial features

7. DATA QUALITY = EXCELLENT
   → No missing values
   → Outliers are legitimate (not data errors)
   → Ready for modeling after feature engineering

================================================================================
                         END OF EDA SUMMARY
==================================================================