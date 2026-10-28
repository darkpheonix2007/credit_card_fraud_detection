# Credit Card Fraud Detection

Student: S Aditya Anirudh  
Registration number: 245890012

This folder contains the complete fraud-detection project. It uses LightGBM
for classification and isotonic regression for probability calibration.

## Files

- `credit_card_fraud_model.py` — model training, evaluation, SHAP analysis and plots.
- `fraud_detection_ui.py` — interactive Streamlit user interface.
- `credit_card_fraud_project.ipynb` — Google Colab notebook with saved results.
- `requirements.txt` — required Python packages for the UI.
- `run_fraud_ui.bat` — Windows launcher.

The Kaggle dataset is not included. Place `creditcard.csv.zip` in this folder
when running the command-line model, or upload it through the UI.

## Run the user interface

Double-click `run_fraud_ui.bat`. When the browser opens, upload
`creditcard.csv.zip` using the left sidebar.

Alternatively, open PowerShell in this folder and run:

```powershell
py -m pip install -r requirements.txt
py -m streamlit run fraud_detection_ui.py
```

## Run the model without the UI

```powershell
py -m pip install -r requirements.txt
py credit_card_fraud_model.py --data creditcard.csv.zip
```

The model creates ROC, calibration and SHAP plots inside an `outputs` folder.

## Google Colab

Upload `credit_card_fraud_project.ipynb` to Google Colab. Run the cells only
when you want to reproduce the saved results. When requested, upload
`creditcard.csv.zip`.
