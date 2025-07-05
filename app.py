import streamlit as st
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load trained model
model = joblib.load('house_price_model.pkl')

try:
    df_train = pd.read_csv('data/train.csv')
    show_hist = True
except:
    df_train = None
    show_hist = False

selected_features = [
    'OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF',
    '1stFlrSF', 'FullBath', 'TotRmsAbvGrd', 'YearBuilt', 'YearRemodAdd',
    'OverallQual_GrLivArea', 'Qual_Age_Interaction', 'TotalSF',
    'TotalBath', 'TotalRooms', 'HouseAge', 'YearsSinceRemodel'
]

# ------------------ Streamlit UI ------------------

st.set_page_config(page_title="House Price Predictor", layout="centered")
st.title("House Price Prediction App")
st.markdown("Make predictions manually or upload a CSV file.")

st.header("Manual Entry for a Single Prediction")

# Manual input
OverallQual = st.slider("Overall Quality (1-10)", 1, 10, 5)
GrLivArea = st.number_input("Above Ground Living Area (sq ft)", 300, 5000, 1500)
GarageCars = st.selectbox("Garage Capacity (cars)", [0, 1, 2, 3, 4])
TotalBsmtSF = st.number_input("Total Basement Area (sq ft)", 0, 3000, 800)
FirstFlrSF = st.number_input("1st Floor Area (sq ft)", 0, 3000, 1000)
FullBath = st.slider("Number of Full Bathrooms", 0, 4, 2)
TotRmsAbvGrd = st.slider("Total Rooms Above Ground", 1, 15, 6)
YearBuilt = st.number_input("Year Built", 1800, 2025, 2000)
YearRemodAdd = st.number_input("Year Remodeled", 1900, 2025, 2010)

# Engineered features
OverallQual_GrLivArea = OverallQual * GrLivArea
Qual_Age_Interaction = OverallQual * (2025 - YearBuilt)
TotalSF = TotalBsmtSF + FirstFlrSF + GrLivArea
TotalBath = FullBath
TotalRooms = TotRmsAbvGrd + TotalBath
HouseAge = 2025 - YearBuilt
YearsSinceRemodel = 2025 - YearRemodAdd


features = np.array([[OverallQual, GrLivArea, GarageCars, TotalBsmtSF,
                      FirstFlrSF, FullBath, TotRmsAbvGrd, YearBuilt, YearRemodAdd,
                      OverallQual_GrLivArea, Qual_Age_Interaction, TotalSF,
                      TotalBath, TotalRooms, HouseAge, YearsSinceRemodel]])

# Prediction
if st.button(" Predict Price"):
    prediction = model.predict(features)[0]
    st.success(f" Estimated House Price: ${prediction:,.2f}")

    st.subheader(" Your Input Summary")
    fig1, ax1 = plt.subplots(figsize=(10, 5))
    ax1.barh(selected_features, features[0])
    ax1.set_xlabel("Feature Value")
    st.pyplot(fig1)

    if hasattr(model, "feature_importances_"):
        st.subheader(" Model Feature Importances")
        fig2, ax2 = plt.subplots(figsize=(10, 5))
        ax2.barh(selected_features, model.feature_importances_)
        st.pyplot(fig2)

    if show_hist:
        st.subheader(" Sale Price Distribution (Train Data)")
        fig3 = plt.figure(figsize=(10, 5))
        sns.histplot(df_train["SalePrice"], kde=True, bins=40)
        plt.axvline(prediction, color='red', linestyle='--', label='Prediction')
        plt.legend()
        st.pyplot(fig3)

# ----------------- CSV Upload Section -----------------

st.markdown("---")
st.header(" Upload CSV for Batch Prediction")

def preprocess_input(df):
    df = df.copy()

    # Engineer features
    df['OverallQual_GrLivArea'] = df['OverallQual'] * df['GrLivArea']
    df['Qual_Age_Interaction'] = df['OverallQual'] * (2025 - df['YearBuilt'])
    df['TotalSF'] = df['TotalBsmtSF'] + df['1stFlrSF'] + df['GrLivArea']
    df['TotalBath'] = df['FullBath'] + (0.5 * df.get('HalfBath', 0)) + df.get('BsmtFullBath', 0) + (0.5 * df.get('BsmtHalfBath', 0))
    df['TotalRooms'] = df['TotRmsAbvGrd'] + df['TotalBath']
    df['HouseAge'] = 2025 - df['YearBuilt']
    df['YearsSinceRemodel'] = 2025 - df['YearRemodAdd']

    # Ensure correct order
    df = df[selected_features]
    return df

uploaded_file = st.file_uploader("Upload raw input CSV (like train.csv)", type=["csv"])

if uploaded_file is not None:
    try:
        df_raw = pd.read_csv(uploaded_file)
        df_processed = preprocess_input(df_raw)

        st.write("Processed Preview")
        st.dataframe(df_processed.head())

        preds = model.predict(df_processed)
        df_raw['PredictedPrice'] = preds

        st.success("Batch prediction complete.")
        st.dataframe(df_raw[['Id', 'PredictedPrice']] if 'Id' in df_raw else df_raw)

        csv = df_raw.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Predictions",
            data=csv,
            file_name="predicted_prices.csv",
            mime="text/csv"
        )

    except Exception as e:
        st.error(f"Error during processing: {e}")
