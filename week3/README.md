# Week 3: Advanced Data Analysis and Visualization in Logistics

Exploratory data analysis (EDA) of a hypothetical logistics dataset: 15,000 shipments, 3 distribution centres, 5 regions,
3 transport modes, 4 carriers, 12 months. The data is simulated (fixed random seed) with built-in patterns so the analysis
can be checked against a known structure.

## Run
    pip install -r requirements.txt
    python logistics_eda.py

Creates `data/shipments.csv`, 10 charts in `figures/`, and `results.json` with all numbers used in the report.
(Create empty `figures/` and `data/` folders first if they do not exist.)

## What it does
1. Simulates shipment data  2. Descriptive statistics and group comparisons  3. Spearman correlations
4. Kruskal-Wallis and chi-square tests  5. Cost-driver regression  6. Ten charts (histograms, boxplots, heatmaps, scatter, stacked bars, line, bubble)
7. What-if scenarios for warehouse dispatch delay

Report: `Week3_Logistics_Analysis_Visualization_Report.docx`
