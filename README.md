# House Price Prediction App

A Streamlit web application that predicts housing prices based on property features using a trained machine learning model. Users can make predictions either manually or by uploading a CSV file. The app also provides visualizations to help interpret the model and its inputs.

---
#### Live Demo
Try the app online: https://csi-week-7-xd57ujvvmbnv6cdrbltr5i.streamlit.app

---

## Features

-  Manual input of house details for instant price prediction
- Upload CSV files (e.g., test data) for **batch predictions**
- Visualizations:
  - Feature importance
  - Distribution of predicted vs. historical sale prices
  - Summary of input features
-  Download predictions as CSV
-  Deployed live with Streamlit Cloud

---


##  Model Features

The model was trained using a mix of **raw** and **engineered** features:

### Raw Features
- `OverallQual`
- `GrLivArea`
- `GarageCars`
- `TotalBsmtSF`
- `1stFlrSF`
- `FullBath`
- `TotRmsAbvGrd`
- `YearBuilt`
- `YearRemodAdd`

### Engineered Features
- `OverallQual_GrLivArea`
- `Qual_Age_Interaction`
- `TotalSF`
- `TotalBath`
- `TotalRooms`
- `HouseAge`
- `YearsSinceRemodel`

> These features are automatically computed during CSV upload or manual input.

---

##  Installation & Running Locally

### 1. Clone the repository

```bash
git clone https://github.com/Nitesh-Yadavv/Week-7-Streamlit-Project.git
```

### 2.Install dependencies
```bash
pip install -r requirements.txt 
```

### 3.Run the app

```bash
streamlit run app.py
```
---

### CSV Format for Batch Prediction
- Uploaded CSV must contain the following raw columns only (like train.csv):
```bash
Id, OverallQual, GrLivArea, GarageCars, TotalBsmtSF, 1stFlrSF,
FullBath, HalfBath, BsmtFullBath, BsmtHalfBath, TotRmsAbvGrd,
YearBuilt, YearRemodAdd
```
 Do not include engineered features — they are created by the app automatically

---

