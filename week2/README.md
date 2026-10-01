# Week 2: Data Collection, Cleaning and Preprocessing for Logistics

Python pipeline that cleans and prepares shipment/order data. Reference dataset: DataCo Smart Supply Chain (Kaggle / Mendeley Data).
The script generates a synthetic dataset with the same kind of fields and injected quality problems (duplicates, missing values,
mixed date formats, inconsistent labels, impossible dates, outliers), so each step can be verified.

## Run
    pip install -r requirements.txt
    python preprocessing_pipeline.py

Outputs: cleaned data in `data/processed/`, charts in `figures/`, summary numbers in `metrics.json`.

## Pipeline steps
1. Audit raw data  2. Fix types, labels, dates  3. Remove duplicates / invalid records  4. Impute missing values
5. Treat outliers (IQR x 3, winsorize)  6. Feature engineering  7. Log transform, one-hot encode, scale (fit on train only)  8. Validate

To use real data, replace the `make_raw()` call in the main block with `pd.read_csv("your_file.csv")`.
Report: `Week2_Data_Cleaning_Preprocessing_Report.docx`
