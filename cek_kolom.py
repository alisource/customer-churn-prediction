import joblib

# Muat model Logistic Regression milikmu
model = joblib.load("notebooks/model_churn_logistic_regression.pkl")

# Cetak urutan 20 kolom yang terekam di dalam model
print("--- URUTAN 20 KOLOM FINAL MODEL ---")
for i, col in enumerate(model.feature_names_in_, 1):
    print(f"'{col}',")