# 📈 Sales Prediction System

A complete Machine Learning project that predicts future product sales using historical Superstore sales data.

This project demonstrates an end-to-end Machine Learning workflow including data preprocessing, feature engineering, model training, model evaluation, model comparison, model deployment, and a Streamlit web application for real-time sales prediction.

---

# 🚀 Project Overview

The objective of this project is to build a Sales Prediction Model capable of estimating future sales using customer information, product details, shipping information, and order history.

Three Machine Learning algorithms were trained and compared:

- Linear Regression
- Decision Tree Regression
- Random Forest Regression

Among these, **Random Forest Regression** achieved the highest prediction accuracy and was selected as the final model for deployment.

---

# ✨ Features

- Data Cleaning & Preprocessing
- Feature Engineering
- Categorical Encoding
- Train-Test Split
- Multiple Regression Models
- Model Comparison
- Performance Evaluation
- Model Serialization using Joblib
- Interactive Streamlit Web Application
- Real-Time Sales Prediction

---

# 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- Streamlit
- Joblib
- Jupyter Notebook

---

# 📂 Project Structure

```text
Sales_Prediction_Model/
│
├── data/
│   └── superstore.csv
│
├── model/
│   ├── random_forest_model.pkl
│   ├── encoder.pkl
│   └── feature_columns.pkl
│
├── screenshots/
│   ├── 1.png
│   ├── 2.png
│   ├── 3.png
│   ├── featureimportancegraph.png
│   ├── homepage.png
│   ├── modelprediction.png
│   └── prediction.png
│
├── Sales_Prediction.ipynb
├── app.py
├── requirements.txt
└── README.md
```

---

# ⚙️ Machine Learning Workflow

1. Import Required Libraries
2. Load Dataset
3. Data Exploration
4. Data Preprocessing
5. Feature Engineering
6. Encode Categorical Features
7. Split Training & Testing Data
8. Train Multiple Regression Models
9. Evaluate Model Performance
10. Compare Models
11. Save Best Model
12. Build Streamlit Web Application
13. Predict Sales in Real Time

---

# 📊 Model Performance

| Model | R² Score | MAE | RMSE |
|-------|---------:|---------:|---------:|
| Random Forest | **0.567** | **86.39** | **505.72** |
| Decision Tree | 0.524 | 102.82 | 530.42 |
| Linear Regression | -0.141 | 243.47 | 820.93 |

✅ **Best Model:** Random Forest Regression

---

# 🖥️ Application Preview

## Home Page

![Home Page](Screenshots/homepage.png)

---

## Prediction Result

![Prediction](Screenshots/prediction.png)

---

# 📓 Notebook Preview

## Linear Regression

![Linear Regression](Screenshots/1.png)

---

## Decision Tree Regression

![Decision Tree](Screenshots/2.png)

---

## Random Forest Regression

![Random Forest](Screenshots/3.png)

---

## Model Comparison

![Model Comparison](Screenshots/modelprediction.png)

---

## Feature Importance

![Feature Importance](Screenshots/featureimportancegraph.png)

---

# 📈 Model Evaluation Metrics

The models were evaluated using:

- R² Score
- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)

Random Forest Regression outperformed the other algorithms and was selected as the final deployment model.

---

# ▶️ How to Run the Project

## 1. Clone the Repository

```bash
git clone <repository-url>
```

## 2. Navigate to the Project Folder

```bash
cd Sales_Prediction_Model
```

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

## 4. Launch the Streamlit Application

```bash
streamlit run app.py
```

---

# 💡 Project Outcome

This project successfully demonstrates an end-to-end Machine Learning pipeline for sales prediction.

The final Random Forest Regression model was integrated into a Streamlit web application, allowing users to generate real-time sales predictions through an interactive interface.

The project showcases practical skills in:

- Data Preprocessing
- Feature Engineering
- Machine Learning
- Model Evaluation
- Model Deployment
- Interactive Dashboard Development

---

# 👨‍💻 Developed By

**Sarthak Vats**
