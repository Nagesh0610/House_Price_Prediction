import streamlit as st
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load model
model = pickle.load(open('model.pkl', 'rb'))

# Load dataset
data = pd.read_csv('data.csv')   # same file

st.set_page_config(page_title="House Price Predictor", layout="wide")

st.title("🏠 House Price Prediction Dashboard")

# Sidebar input
st.sidebar.header("Enter House Details")

sqft_living = st.sidebar.slider("Living Area (sq ft)", 500, 5000, 1000)
bedrooms = st.sidebar.slider("Bedrooms", 1, 10, 2)
bathrooms = st.sidebar.slider("Bathrooms", 1, 5, 1)
floors = st.sidebar.slider("Floors", 1, 3, 1)
sqft_lot = st.sidebar.number_input("Lot Area", 1000, 100000, 5000)
condition = st.sidebar.slider("Condition", 1, 5, 3)
sqft_above = st.sidebar.number_input("Sqft Above", 500, 5000, 1000)
sqft_basement = st.sidebar.number_input("Sqft Basement", 0, 2000, 500)

# Prediction
input_data = [[sqft_living, bedrooms, bathrooms, floors,
               sqft_lot, condition, sqft_above, sqft_basement]]
prediction = model.predict(input_data)

st.subheader("💰 Predicted Price")
st.success(f"₹ {prediction[0]:,.2f}")

# Graphs
col1, col2 = st.columns(2)

with col1:
    st.write("### Living Area vs Price")
    fig1, ax1 = plt.subplots()
    sns.scatterplot(x=data['sqft_living'], y=data['price'], ax=ax1)
    st.pyplot(fig1)

with col2:
    st.write("### Price Distribution")
    fig2, ax2 = plt.subplots()
    sns.histplot(data['price'], kde=True, ax=ax2)
    st.pyplot(fig2)

# Heatmap
st.write("### Correlation Heatmap")
fig3, ax3 = plt.subplots()
sns.heatmap(data[['sqft_living','bedrooms','bathrooms','floors','price']].corr(), annot=True, cmap='coolwarm', ax=ax3)
st.pyplot(fig3)