import streamlit as st
import requests

# 1. Judul & Desain Halaman Dashboard
st.set_page_config(page_title="Churn Prediction Dashboard", layout="centered")
st.title("📊 Customer Churn Prediction Dashboard")
st.markdown("Masukkan data pelanggan di bawah ini untuk menganalisis risiko *churn*.")

st.divider()

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
    avg_order_value = st.number_input("Rata-rata Nilai Order (Rp/Mata Uang)", min_value=0, value=50000)
    total_orders = st.number_input("Total Jumlah Order", min_value=0, value=10, step=1)
    last_purchase_days_ago = st.number_input("Hari Sejak Pembelian Terakhir", min_value=0, value=30, step=1)
    support_tickets = st.number_input("Jumlah Tiket Komplain (Support Tickets)", min_value=0, value=2, step=1)
    subscription_type = st.selectbox("Tipe Langganan", ["Basic", "Silver", "Gold", "Platinum"])

    # Tombol Kirim Form
    submitted = st.form_submit_button("Analisis Risiko Churn 🚀")

# 3. Proses Penembakan ke FastAPI setelah Tombol Diklik
if submitted:
    # Bungkus semua input menjadi format JSON sesuai kebutuhan FastAPI
    payload = {
        "customer_id": int(customer_id),
        "age": int(age),
        "gender": gender,
        "city": city,
        "tenure_months": int(tenure_months),
        "avg_order_value": int(avg_order_value),
        "total_orders": int(total_orders),
        "last_purchase_days_ago": int(last_purchase_days_ago),
        "support_tickets": int(support_tickets),
        "subscription_type": subscription_type
    }
    
    # Alamat URL FastAPI lokal kamu
    api_url = "http://127.0.0.1:8000/predict"
    
    try:
        with st.spinner("Sedang menghitung prediksi menggunakan model..."):
            # Tembak FastAPI menggunakan POST request
            response = requests.post(api_url, json=payload)
            
            if response.status_code == 200:
                result = response.json()
                
                # Ambil hasil dari JSON keluaran FastAPI
                prediction = result["churn_prediction"]
                probability = result["churn_probability_%"]
                status = result["status"]
                
                st.divider()
                st.subheader("🔮 Hasil Analisis Model:")
                
                # Tampilkan visualisasi hasil berdasarkan status
                if prediction == 1:
                    st.error(f"### {status}")
                else:
                    st.success(f"### {status}")
                    
                st.metric(label="Probabilitas Churn", value=f"{probability} %")
                
            else:
                st.error(f"Gagal mendapatkan prediksi dari server. (Kode Eror: {response.status_code})")
                
    except requests.exceptions.ConnectionError:
        st.error("❌ Tidak bisa terhubung ke FastAPI! Pastikan server FastAPI kamu sudah dinyalakan lewat terminal terlebih dahulu.")