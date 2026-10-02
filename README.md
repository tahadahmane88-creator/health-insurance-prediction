# Health Insurance Cost Prediction 🏥

A Machine Learning regression project built with Python and Scikit-Learn to predict individual health insurance charges based on demographic and lifestyle features.

## 📊 Project Overview
This project uses Linear Regression to analyze patient data (`insurance.csv`) and accurately estimate annual health insurance costs.

## 🚀 Key Features & Pipeline
- **Exploratory Data Analysis (EDA):** Inspected data distributions and checked for missing values.
- **Data Preprocessing:** Handled categorical variables using One-Hot Encoding (`pd.get_dummies`).
- **Data Splitting:** Applied 80/20 Train-Test split.
- **Model Training:** Trained a `LinearRegression` model.
- **Evaluation:** Calculated performance metrics on unseen test data.
- **Inference:** Created a feature-aligned prediction pipeline for new client profiles.

## 📈 Model Performance
- **R² Score:** `0.7836` (Explains ~78.4% of price variance)
- **Mean Absolute Error (MAE):** `$4,181.19`

## 💡 Key Finding
The model identified **smoking status (`smoker_yes`)** as the single most influential feature affecting insurance premiums, adding approximately **$23,651** to annual charges.

## 🛠️ How to Run
1. Clone the repository:
   ```bash
   git clone [https://github.com/tahadahmane88-creator/health-insurance-prediction.git](https://github.com/tahadahmane88-creator/health-insurance-prediction.git)
