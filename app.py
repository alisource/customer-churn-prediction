import joblib
import pandas as pd
import pydantic
from fastapi import FastAPI

app = FastAPI()

# 1. Muat Model dan Scaler
model = joblib.load("models/model_churn_logistic_regression.pkl")
scaler = joblib.load("models/scaler.pkl")


# 2. Skema Data Input Pelanggan
class CustomerData(pydantic.BaseModel):
    customer_id: int
    age: int
    gender: str
    city: str
    tenure_months: int
    avg_order_value: int
    total_orders: int
    last_purchase_days_ago: int
    support_tickets: int
    subscription_type: str


# 3. Endpoint Prediksi
@app.post("/predict")
def predict_churn(data: CustomerData):
    # --- [PROSES 1] LOGIKA ONE-HOT ENCODING (10 FITUR KATEGORIKAL) ---
    city_Delhi = 1 if data.city.title() == "Delhi" else 0
    city_Hyderabad = 1 if data.city.title() == "Hyderabad" else 0
    city_Chennai = 1 if data.city.title() == "Chennai" else 0
    city_Kolkata = 1 if data.city.title() == "Kolkata" else 0
    city_Mumbai = 1 if data.city.title() == "Mumbai" else 0
    city_Pune = 1 if data.city.title() == "Pune" else 0

    subscription_type_Silver = (
        1 if data.subscription_type.title() == "Silver" else 0
    )
    subscription_type_Gold = (
        1 if data.subscription_type.title() == "Gold" else 0
    )
    subscription_type_Platinum = (
        1 if data.subscription_type.title() == "Platinum" else 0
    )

    gender_encoded = 1 if data.gender.title() == "Female" else 0

    # --- [PROSES 2] HITUNG FITUR TURUNAN ---
    tenure_years = data.tenure_months / 12
    total_spend_estimate = data.total_orders * data.avg_order_value
    value_per_order = data.avg_order_value / (data.total_orders + 1)
    support_ratio = data.support_tickets / (data.total_orders + 1)

    # --- [PROSES 3] TAHAP SCALING (HANYA 10 FITUR NUMERIK) ---
    # Sesuai dengan hasil cek_kolom.py yang terbaca oleh scaler-mu
    numeric_dict = {
        "age": [data.age],
        "tenure_months": [data.tenure_months],
        "avg_order_value": [data.avg_order_value],
        "total_orders": [data.total_orders],
        "last_purchase_days_ago": [data.last_purchase_days_ago],
        "support_tickets": [data.support_tickets],
        "value_per_order": [value_per_order],
        "tenure_years": [tenure_years],
        "support_ratio": [support_ratio],
        "total_spend_estimate": [total_spend_estimate],
    }

    numeric_df = pd.DataFrame(numeric_dict)
    # Lakukan transformasi skala hanya untuk 10 fitur numerik ini
    numeric_scaled = scaler.transform(numeric_df)

    # Mengembalikan hasil array scaling menjadi DataFrame agar bisa digabungkan
    numeric_scaled_df = pd.DataFrame(
        numeric_scaled, columns=numeric_df.columns
    )

    # --- [PROSES 4] GABUNGKAN DATA SCALED DENGAN DATA ENCODING (TOTAL 20 FITUR) ---
    # Kita buat 10 kolom kategorikal sisanya
    categorical_dict = {
        "gender": [gender_encoded],
        "city_Chennai": [city_Chennai],
        "city_Delhi": [city_Delhi],
        "city_Hyderabad": [city_Hyderabad],
        "city_Kolkata": [city_Kolkata],
        "city_Mumbai": [city_Mumbai],
        "city_Pune": [city_Pune],
        "subscription_type_Gold": [subscription_type_Gold],
        "subscription_type_Platinum": [subscription_type_Platinum],
        "subscription_type_Silver": [subscription_type_Silver],
    }
    categorical_df = pd.DataFrame(categorical_dict)

    # Gabungkan secara horizontal (axis=1) menjadi satu kesatuan 20 fitur
    final_input_df = pd.concat([numeric_scaled_df, categorical_df], axis=1)

    # --- [PROSES 5] AMANKAN URUTAN 20 KOLOM SESUAI MODEL NOTEBOOK ---
    # Daftar 20 kolom final yang diwajibkan oleh Logistic Regression milikmu
    kolom_final_model = [
        "age",
        "gender",
        "tenure_months",
        "avg_order_value",
        "total_orders",
        "last_purchase_days_ago",
        "support_tickets",
        "city_Chennai",  
        "city_Delhi",
        "city_Hyderabad",
        "city_Kolkata",
        "city_Mumbai",
        "city_Pune",
        "subscription_type_Gold",
        "subscription_type_Platinum",
        "subscription_type_Silver",
        "value_per_order",
        "tenure_years",
        "support_ratio",
        "total_spend_estimate",
    ]

    final_input_df = final_input_df.reindex(columns=kolom_final_model)

    # --- [PROSES 6] PREDIKSI DENGAN 20 FITUR LENGKAP ---
    prediction = model.predict(final_input_df)[0]
    probability = model.predict_proba(final_input_df)[0][1] * 100

    return {
        "customer_id": data.customer_id,
        "churn_prediction": int(prediction),
        "churn_probability_%": round(probability, 2),
        "status": "🚨 Bahaya Churn" if prediction == 1 else "✅ Aman (Loyal)",
    }