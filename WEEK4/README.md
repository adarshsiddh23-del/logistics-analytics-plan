# Week 4: Predictive Modeling and Optimization in Logistics

Predicts shipment delivery time (hours) at dispatch with machine learning, then uses the forecasts for two optimisation
decisions: (A) data-driven delivery promises, (B) cost-minimising transport mode selection via MILP. Plus (C) a warehouse capacity what-if.

Data are simulated (25,000 shipments, 2024-2025, fixed seed). The simulator keeps each shipment's outcome under every mode,
so optimisation policies can be scored against ground truth.

## Run
    pip install -r requirements.txt
    python predictive_optimization.py      # about 3-4 minutes on one CPU core

(Create empty `figures/` and `data/` folders first if they do not exist.)
Outputs: `data/shipments_w4.csv`, 8 charts in `figures/`, `results_w4.json` (all numbers used in the report).

## Pipeline
1. Simulate data and potential outcomes  2. Leak-free features, time-based train/test split
3. Baseline, linear regression, decision tree, random forest, gradient boosting (TimeSeriesSplit CV, randomised search)
4. Metrics: MAE, RMSE, R2, within +/-6 h; residual and importance analysis
5. Quantile gradient boosting for 80% prediction intervals
6. Optimisation A (promise policy), B (MILP with scipy.optimize.milp), C (capacity what-if)

Report: `Week4_Predictive_Modeling_Optimization_Report.docx`
