# Import Libraries

import streamlit as st
import pandas as pd
import joblib

# Load Dataset

df = pd.read_csv("data/superstore.csv", encoding="latin1")

# Load Saved Files

model = joblib.load("model/random_forest_model.pkl")
encoder = joblib.load("model/encoder.pkl")
feature_columns = joblib.load("model/feature_columns.pkl")

# Configure Page

st.set_page_config(
    page_title="Sales Prediction System",
    page_icon="📈",
    layout="wide"
)

# Title

st.title("📈 Sales Prediction System")

st.markdown("""
Predict future sales using the trained Random Forest Regression Model.

Select the customer and product details from the sidebar, then click **Predict Sales**.
""")

st.markdown("---")

# Sidebar

st.sidebar.title("Input Features")

# Customer Information

ship_mode = st.sidebar.selectbox(
    "Ship Mode",
    sorted(df["Ship Mode"].unique())
)

segment = st.sidebar.selectbox(
    "Customer Segment",
    sorted(df["Segment"].unique())
)

country = st.sidebar.selectbox(
    "Country",
    sorted(df["Country"].unique())
)

city = st.sidebar.selectbox(
    "City",
    sorted(df["City"].unique())
)

state = st.sidebar.selectbox(
    "State",
    sorted(df["State"].unique())
)

postal_code = st.sidebar.number_input(
    "Postal Code",
    value=90001
)

region = st.sidebar.selectbox(
    "Region",
    sorted(df["Region"].unique())
)

category = st.sidebar.selectbox(
    "Category",
    sorted(df["Category"].unique())
)

sub_category = st.sidebar.selectbox(
    "Sub-Category",
    sorted(df["Sub-Category"].unique())
)

product_name = st.sidebar.selectbox(
    "Product Name",
    sorted(df["Product Name"].unique())
)

# Numerical Features

quantity = st.sidebar.number_input(
    "Quantity",
    min_value=1,
    max_value=20,
    value=2
)

discount = st.sidebar.slider(
    "Discount",
    min_value=0.0,
    max_value=1.0,
    value=0.10
)

profit = st.sidebar.number_input(
    "Profit",
    value=100.0
)

st.sidebar.markdown("---")

st.sidebar.subheader("Order Date")

order_year = st.sidebar.number_input(
    "Order Year",
    min_value=2014,
    max_value=2017,
    value=2017
)

order_month = st.sidebar.number_input(
    "Order Month",
    min_value=1,
    max_value=12,
    value=11
)

order_day = st.sidebar.number_input(
    "Order Day",
    min_value=1,
    max_value=31,
    value=15
)

st.sidebar.subheader("Shipping Date")

ship_year = st.sidebar.number_input(
    "Ship Year",
    min_value=2014,
    max_value=2018,
    value=2017
)

ship_month = st.sidebar.number_input(
    "Ship Month",
    min_value=1,
    max_value=12,
    value=11
)

ship_day = st.sidebar.number_input(
    "Ship Day",
    min_value=1,
    max_value=31,
    value=18
)

st.markdown("---")

predict = st.button("Predict Sales")

if predict:

    # Categorical Features

    categorical_df = pd.DataFrame({
        "Ship Mode": [ship_mode],
        "Segment": [segment],
        "Country": [country],
        "City": [city],
        "State": [state],
        "Region": [region],
        "Category": [category],
        "Sub-Category": [sub_category],
        "Product Name": [product_name]
    })

    # Encode Categorical Features

    encoded = encoder.transform(categorical_df)

    # Create Model Input

    input_data = pd.DataFrame({
        "Ship Mode": [encoded[0][0]],
        "Segment": [encoded[0][1]],
        "Country": [encoded[0][2]],
        "City": [encoded[0][3]],
        "State": [encoded[0][4]],
        "Postal Code": [postal_code],
        "Region": [encoded[0][5]],
        "Category": [encoded[0][6]],
        "Sub-Category": [encoded[0][7]],
        "Product Name": [encoded[0][8]],
        "Quantity": [quantity],
        "Discount": [discount],
        "Profit": [profit],
        "Order Year": [order_year],
        "Order Month": [order_month],
        "Order Day": [order_day],
        "Ship Year": [ship_year],
        "Ship Month": [ship_month],
        "Ship Day": [ship_day]
    })

    # Arrange Columns

    input_data = input_data[feature_columns]

    # Predict

    prediction = model.predict(input_data)

    predicted_sales = prediction[0]

    # Display Result

    st.success("Prediction Completed Successfully!")

    st.metric(
        label="Predicted Sales",
        value=f"${predicted_sales:,.2f}"
    )

    st.info(
        "The prediction is generated using the trained Random Forest Regression model."
    )