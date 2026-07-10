import streamlit as st
import requests
import joblib
import pandas as pd

# 1. Judul & Desain Halaman Dashboard
st.set_page_config(page_title="Churn Prediction Dashboard", layout="centered")
st.title("📊 Customer Churn Prediction Dashboard")
st.markdown("Masukkan data pelanggan di bawah ini untuk menganalisis risiko *churn* secara realtime.")

st.divider()

# Muat Model dan Scaler langsung di Streamlit (Bertindak sebagai Backend Mandiri)
@st.cache_resource
def load_ml_components():
    model = joblib.load("models/model_churn_logistic_regression.pkl")
    scaler = joblib.load("models/scaler.pkl")
    return model, scaler

try:
    model, scaler = load_ml_components()
except Exception as e:
    st.error(f"⚠️ Gagal memuat model/scaler dari folder 'models/'. Pastikan file ada di GitHub. Eror: {e}")

# 2. Membuat Form Input untuk Pengguna
with st.form("churn_form"):
    st.subheader("📋 Informasi Pelanggan")
    
    customer_id = st.number_input("Customer ID", min_value=1, value=1001, step=1)
    age = st.number_input("Usia (Age)", min_value=10, max_value=100, value=25)
    gender = st.selectbox("Jenis Kelamin (Gender)", ["Male", "Female"])
    city = st.selectbox("Kota (City)", ["Delhi", "Mumbai", "Hyderabad", "Chennai", "Kolkata", "Pune"])
    
    st.divider()
    st.subheader("🛒 Aktivitas Transaksi & Layanan")
    
    tenure_months = st.number_input("Tenure (Bulan)", min_value=0, value=12, step=1)
    avg_order_value = st.number_input("Rata-rata Nilai Order", min_value=0, value=50000)
    total_orders = st.number_input("Total Jumlah Order", min_value=0, value=10, step=1)
    last_purchase_days_ago = st.number_input("Hari Sejak Pembelian Terakhir", min_value=0, value=30, step=1)
    support_tickets = st.number_input("Jumlah Tiket Komplain (Support Tickets)", min_value=0, value=2, step=1)
    subscription_type = st.selectbox("Tipe Langganan", ["Basic", "Silver", "Gold", "Platinum"])

    submitted = st.form_submit_button("Analisis Risiko Churn 🚀")

# 3. Penggabungan Pipeline Preprocessing FastAPI Langsung di Sini
if submitted:
    try:
        with st.spinner("Sedang memproses pipeline data dan menghitung prediksi..."):
            # --- [PROSES 1] LOGIKA ONE-HOT ENCODING (10 FITUR KATEGORIKAL) ---
            city_title = city.title()
            city_Delhi = 1 if city_title == "Delhi" else 0
            city_Hyderabad = 1 if city_title == "Hyderabad" else 0
            city_Chennai = 1 if city_title == "Chennai" else 0
            city_Kolkata = 1 if city_title == "Kolkata" else 0
            city_Mumbai = 1 if city_title == "Mumbai" else 0
            city_Pune = 1 if city_title == "Pune" else 0

            sub_title = subscription_type.title()
            subscription_type_Silver = 1 if sub_title == "Silver" else 0
            subscription_type_Gold = 1 if sub_title == "Gold" else 0
            subscription_type_Platinum = 1 if sub_title == "Platinum" else 0

            gender_encoded = 1 if gender.title() == "Female" else 0

            # --- [PROSES 2] HITUNG FITUR TURUNAN ---
            tenure_years = tenure_months / 12
            total_spend_estimate = total_orders * avg_order_value
            value_per_order = avg_order_value / (total_orders + 1)
            support_ratio = support_tickets / (total_orders + 1)

            # --- [PROSES 3] TAHAP SCALING (HANYA 10 FITUR NUMERIK) ---
            numeric_dict = {
                "age": [age],
                "tenure_months": [tenure_months],
                "avg_order_value": [avg_order_value],
                "total_orders": [total_orders],
                "last_purchase_days_ago": [last_purchase_days_ago],
                "support_tickets": [support_tickets],
                "value_per_order": [value_per_order],
                "tenure_years": [tenure_years],
                "support_ratio": [support_ratio],
                "total_spend_estimate": [total_spend_estimate],
            }

            numeric_df = pd.DataFrame(numeric_dict)
            numeric_scaled = scaler.transform(numeric_df)
            numeric_scaled_df = pd.DataFrame(numeric_scaled, columns=numeric_df.columns)

            # --- [PROSES 4] GABUNGKAN DATA SCALED DENGAN DATA ENCODING (TOTAL 20 FITUR) ---
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
            final_input_df = pd.concat([numeric_scaled_df, categorical_df], axis=1)

            # --- [PROSES 5] AMANKAN URUTAN 20 KOLOM SESUAI MODEL ---
            kolom_final_model = [
                "age", "gender", "tenure_months", "avg_order_value", "total_orders",
                "last_purchase_days_ago", "support_tickets", "city_Chennai", "city_Delhi",
                "city_Hyderabad", "city_Kolkata", "city_Mumbai", "city_Pune",
                "subscription_type_Gold", "subscription_type_Platinum", "subscription_type_Silver",
                "value_per_order", "tenure_years", "support_ratio", "total_spend_estimate"
            ]
            final_input_df = final_input_df.reindex(columns=kolom_final_model)

            # --- [PROSES 6] PREDIKSI DENGAN 20 FITUR LENGKAP ---
            prediction = model.predict(final_input_df)[0]
            probability = model.predict_proba(final_input_df)[0][1] * 100

            # 4. Tampilkan Hasil Akhir ke Tampilan UI Dashboard
            st.divider()
            st.subheader("🔮 Hasil Analisis Model:")
            
            if prediction == 1:
                st.error(f"### 🚨 Bahaya Churn (Customer ID: {customer_id})")
                st.markdown("Pelanggan ini terdeteksi memiliki risiko tinggi untuk berhenti berlangganan.")
            else:
                st.success(f"### ✅ Aman (Loyal) (Customer ID: {customer_id})")
                st.markdown("Pelanggan ini terdeteksi loyal dan kemungkinan besar akan tetap berlangganan.")
                
            st.metric(label="Probabilitas Churn", value=f"{round(probability, 2)} %")

    except Exception as e:
        st.error(f"❌ Terjadi kesalahan saat memproses data di backend Streamlit: {e}")