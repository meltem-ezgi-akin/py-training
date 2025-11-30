import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

def train_churn_model(filepath='customer_churn_dataset.csv'):
    """
    Müşteri terk veri setini yükler, ön işler ve bir lojistik regresyon modeli eğitir.
    Model performansını değerlendirir.
    """
    try:
        df = pd.read_csv(filepath)
        print(f"Veri seti başarıyla yüklendi: {filepath}")

        # Özellikler (X) ve Hedef Değişken (y) Ayırma
        X = df.drop(['CustomerID', 'Churn'], axis=1)
        y = df['Churn']

        # Kategorik Özellikleri Sayısallaştırma (Label Encoding)
        for column in X.select_dtypes(include='object').columns:
            le = LabelEncoder()
            X[column] = le.fit_transform(X[column])

        # Sayısal Özellikleri Ölçeklendirme (StandardScaler)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        X = pd.DataFrame(X_scaled, columns=X.columns)

        # Eğitim ve Test Setlerine Ayırma
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        print("\nModel Eğitimi Başlıyor...")
        # Lojistik Regresyon Modeli Oluşturma ve Eğitme
        model = LogisticRegression(random_state=42, solver='liblinear')
        model.fit(X_train, y_train)
        print("Model Eğitimi Tamamlandı.")

        # Tahminler Yapma
        y_pred = model.predict(X_test)

        # Model Performansını Değerlendirme
        accuracy = accuracy_score(y_test, y_pred)
        print(f"\nModel Doğruluğu (Accuracy): {accuracy:.2f}")
        print("\nSınıflandırma Raporu:")
        print(classification_report(y_test, y_pred))

    except FileNotFoundError:
        print(f"Hata: {filepath} dosyası bulunamadı. Lütfen dosya yolunu kontrol edin.")
    except Exception as e:
        print(f"Model eğitilirken veya değerlendirilirken bir hata oluştu: {e}")

if __name__ == '__main__':
    train_churn_model()
