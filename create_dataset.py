import pandas as pd
import numpy as np

def create_customer_churn_dataset(num_samples=1000):
    np.random.seed(42)

    customer_ids = [f'CUST_{i:04d}' for i in range(num_samples)]

    # Yeni Özellikler Ekleyelim:
    partner = np.random.choice(['Evet', 'Hayır'], num_samples, p=[0.48, 0.52]) # Partneri olanların oranı
    dependents = np.random.choice(['Evet', 'Hayır'], num_samples, p=[0.3, 0.7]) # Bağımlıları olanların oranı
    phone_service = np.random.choice(['Evet', 'Hayır'], num_samples, p=[0.9, 0.1]) # Telefon hizmeti alanların oranı
    multiple_lines = np.random.choice(['Evet', 'Hayır'], num_samples, p=[0.4, 0.6]) # Çoklu hatları olanların oranı (telefon hizmeti olanlarda)

    # Yaş: 18-70 arası
    age = np.random.randint(18, 71, num_samples)
    # Cinsiyet: 0.50 Erkek, 0.50 Kadın
    gender = np.random.choice(['Erkek', 'Kadın'], num_samples, p=[0.5, 0.5])
    # Aylık Ücret: 20-120 arası
    monthly_charges = np.random.uniform(20, 120, num_samples).round(2)
    # Sözleşme Süresi: 1-73 ay
    tenure = np.random.randint(1, 73, num_samples)
    # Toplam Ücret: Aylık Ücret * Sözleşme Süresi
    total_charges = monthly_charges * tenure
    # Sözleşme Tipi: Aylık, Bir Yıl, İki Yıl
    contract = np.random.choice(['Aylık', 'Bir Yıl', 'İki Yıl'], num_samples)
    # İnternet Hizmeti: DSL, Fiber Optik, No
    internet_service = np.random.choice(['DSL', 'Fiber Optik', 'No'], num_samples)

    # Online Hizmetler: Evet, Hayır, No internet service
    OnlineSecurity = np.random.choice(['Evet', 'Hayır', 'No internet service'], num_samples)
    OnlineBackup = np.random.choice(['Evet', 'Hayır', 'No internet service'], num_samples)
    DeviceProtection = np.random.choice(['Evet', 'Hayır', 'No internet service'], num_samples)
    TechSupport = np.random.choice(['Evet', 'Hayır', 'No internet service'], num_samples)
    StreamingTV = np.random.choice(['Evet', 'Hayır', 'No internet service'], num_samples)
    StreamingMovies = np.random.choice(['Evet', 'Hayır', 'No internet service'], num_samples)

    # Churn: Evet, Hayır
    churn = np.random.choice(['Hayır', 'Evet'], num_samples, p=[0.7, 0.3])
    churn[ (contract == 'Aylık') & (np.random.rand(num_samples) < 0.2) ] = 'Evet' # Aylık sözleşmesi olanların %20'si ek olarak terk etsin
    churn[ (internet_service == 'Fiber Optik') & (np.random.rand(num_samples) < 0.15) ] = 'Evet' # Fiber kullananların %15'i ek olarak terk etsin

    # Tüm özellikleri bir DataFrame'de birleştirin
    data = pd.DataFrame({
        'CustomerID': customer_ids,
        'Partner': partner,
        'Dependents': dependents,
        'PhoneService': phone_service,
        'MultipleLines': multiple_lines,
        'Age': age,
        'Gender': gender,
        'MonthlyCharges': monthly_charges,
        'Tenure': tenure,
        'TotalCharges': total_charges,
        'Contract': contract,
        'InternetService': internet_service,
        'OnlineSecurity': OnlineSecurity,
        'OnlineBackup': OnlineBackup,
        'DeviceProtection': DeviceProtection,
        'TechSupport': TechSupport,
        'StreamingTV': StreamingTV,
        'StreamingMovies': StreamingMovies,
        'Churn': churn
    })

    return data

if __name__ == '__main__':
    df = create_customer_churn_dataset(num_samples=2000)
    df.to_csv('customer_churn_dataset.csv', index=False)
    print("customer_churn_dataset.csv dosyası başarıyla oluşturuldu.")
    print(df.head())
