# Import Libraries

import streamlit as st
import pandas as pd
import joblib

# Load Dataset

df = pd.read_csv("data/superstore.csv", encoding="latin1")

# Load Saved Model Files

model = joblib.load("model/random_forest_model.pkl")
encoder = joblib.load("model/encoder.pkl")
feature_columns = joblib.load("model/feature_columns.pkl")

# Configure Page

st.set_page_config(
    page_title="Sales Prediction System",
    page_icon="📈",
    layout="wide"
)

# Main Heading

st.title("📈 Sales Prediction System")

st.caption("Machine Learning | Random Forest Regression | Streamlit Dashboard")

st.markdown("""
Welcome to the **Sales Prediction System**.

This web application predicts future sales using a Machine Learning model trained on the Superstore Sales Dataset.

Select the customer, product and order details from the sidebar and click **Predict Sales**.
""")

# Dashboard Metrics

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        label="Best Model",
        value="Random Forest"
    )

with col2:
    st.metric(
        label="R² Score",
        value="0.567"
    )

with col3:
    st.metric(
        label="Dataset",
        value="9,994 Records"
    )

st.markdown("---")

# Sidebar

st.sidebar.header("Customer & Product Information")

# Customer Details

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

st.sidebar.markdown("---")

# Numerical Inputs

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

# Order Date

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
    value=18
)

# Shipping Date

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

st.sidebar.markdown("---")

# Model Performance

st.sidebar.subheader("Model Performance")

st.sidebar.write("**Algorithm:** Random Forest")
st.sidebar.write("**R² Score:** 0.567")
st.sidebar.write("**MAE:** 86.39")
st.sidebar.write("**RMSE:** 505.72")

st.markdown("---")

predict = st.button("🚀 Predict Sales")
if predict:

    try:

        # Create DataFrame for Categorical Features

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

        # Arrange Features in Training Order

        input_data = input_data[feature_columns]

        # Generate Prediction

        prediction = model.predict(input_data)

        predicted_sales = prediction[0]

        # Success Message

        st.success("Prediction Completed Successfully!")

        # Display Prediction

        st.metric(
            label="Estimated Sales (USD)",
            value=f"${predicted_sales:,.2f}"
        )

        # Prediction Summary

        st.markdown("## 📊 Prediction Summary")

        st.write(f"""
The model estimates that the expected sales value for the selected order is **${predicted_sales:,.2f} USD**.

This prediction has been generated using the **Random Forest Regression Model**, which achieved the best performance during model evaluation.

### Factors Considered

- Customer Segment
- Shipping Method
- Product Category
- Product Name
- Quantity Ordered
- Discount
- Profit
- Order Date
- Shipping Date

The model learns historical sales patterns from the Superstore Sales Dataset and predicts the expected sales for similar future transactions.
""")

        st.info(
            "This prediction is intended for analytical and educational purposes."
        )

    except Exception as e:

        st.error("Prediction Failed!")

        st.exception(e)
# Footer

st.markdown("---")

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
### Project Information

**Project:** Sales Prediction Model

**Algorithm:** Random Forest Regression

**Dataset:** Superstore Sales Dataset

**Framework:** Streamlit

**Programming Language:** Python
""")

with col2:
    st.markdown("""
### Model Performance

**R² Score:** 0.567

**MAE:** 86.39

**RMSE:** 505.72

**Prediction Target:** Sales (USD)
""")

st.markdown("---")

st.caption(
    "Developed by Sarthak Vats | Data Science Intern | InCode Vision"
)