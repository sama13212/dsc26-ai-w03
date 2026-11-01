# Results

## Decision

Use the Logistic Regression model as the baseline for predicting failed COD deliveries. It outperformed the DummyClassifier prior baseline on PR-AUC and Brier Score across all three tested seeds.

## Metric

PR-AUC is the primary metric because failed deliveries are the minority class. The required improvement over the dummy baseline is at least 0.05 PR-AUC. Brier Score is also checked to assess probability quality, while ROC-AUC is monitored as a leakage guard.

## Baseline Results

All experiments used 4,000 generated parcels, a chronological 80/20 train-test split, and 800 test rows.

| Seed | Model ROC-AUC | Dummy ROC-AUC | Model PR-AUC | Dummy PR-AUC | PR-AUC Lift | Model Brier | Dummy Brier |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 11 | 0.6445 | 0.5000 | 0.3303 | 0.2250 | 0.1053 | 0.1701 | 0.1744 |
| 42 | 0.6349 | 0.5000 | 0.3200 | 0.2138 | 0.1062 | 0.1650 | 0.1681 |
| 123 | 0.6598 | 0.5000 | 0.4024 | 0.2400 | 0.1624 | 0.1718 | 0.1825 |

The model achieved at least 0.05 PR-AUC lift and a lower Brier Score than the dummy baseline in all three experiments. All model ROC-AUC values remained below the 0.9 leakage guard.

## Classical ML or LLM

This task uses classical machine learning: Logistic Regression with preprocessing and one-hot encoding. A DummyClassifier provides the prior-probability baseline. No LLM is used for prediction.

## Message to Sherif

The Logistic Regression baseline outperformed the DummyClassifier across all three tested seeds. PR-AUC lift ranged from 0.1053 to 0.1624, exceeding the required 0.05 threshold, while Brier Score was lower in every run. ROC-AUC remained between 0.6349 and 0.6598, below the 0.9 leakage guard.