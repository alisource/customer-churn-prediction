# src/data/ingest.py
import os
import pandas as pd
from src.data.schema import ChurnDataSchema # Mengambil skema yang kamu buat
from pydantic import ValidationError

def ingest_and_validate():
    # 1. Baca data mentah
    raw_data_path = "data/raw/ecommerce_customer_churn_large.csv"
    processed_dir = "data/processed"
    processed_data_path = os.path.join(processed_dir, "churn_clean.csv")
    
    # Cek apakah file mentah memang ada di folder data/raw
    if not os.path.exists(raw_data_path):
        print(f">>> ERROR: File tidak ditemukan di {raw_data_path}!")
        print("Silakan pindahkan file ecommerce_customer_churn_large.csv ke folder data/raw/ terlebih dahulu.")
        return

    # --- LANGKAH MEMBACA DATA HARUS DI SINI (SEBELUM VALIDASI) ---
    print(">>> Membaca data mentah...")
    df = pd.read_csv(raw_data_path)  
    
    print(">>> Memulai validasi data...")
    
    # 2. Lakukan looping untuk mengecek baris demi baris data menggunakan skema
    errors = []
    for index, row in df.iterrows():
        try:
            # Mengubah baris pandas menjadi dictionary dan dicek oleh Pydantic
            ChurnDataSchema(**row.to_dict())
        except ValidationError as e:
            errors.append(f"Baris {index} error: {e}")
            
    # 3. Cetak hasil pemeriksaan ke terminal
    if len(errors) == 0:
        print(">>> Sempurna! Semua data lolos validasi sesuai skema.")
        # Simpan ke folder processed jika aman
        df.to_csv("data/processed/churn_clean.csv", index=False)
    else:
        print(f">>> Waduh! Ditemukan {len(errors)} baris data yang rusak:")
        for err in errors[:5]: # Tampilkan 5 error pertama saja agar terminal tidak penuh
            print(err)

if __name__ == "__main__":
    ingest_and_validate()