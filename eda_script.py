import pandas as pd

def load_and_explore_data(filepath='customer_churn_dataset.csv'):
    """
    Veri setini yükler ve temel keşifçi veri analizi (EDA) yapar.
    """
    try:
        df = pd.read_csv(filepath)
        print(f"Veri seti başarıyla yüklendi: {filepath}")
        print("\nİlk 5 satır:")
        print(df.head())

        print("\nVeri seti bilgisi:")
        df.info()

        print("\nSayısal sütunlar için temel istatistikler:")
        print(df.describe())

        print("\nKategorik sütunlar için benzersiz değerler ve frekanslar:")
        for column in df.select_dtypes(include='object').columns:
            print(f"\n--- {column} ---")
            print(df[column].value_counts())

        print("\nChurn dağılımı:")
        print(df['Churn'].value_counts(normalize=True))

    except FileNotFoundError:
        print(f"Hata: {filepath} dosyası bulunamadı. Lütfen dosya yolunu kontrol edin.")
    except Exception as e:
        print(f"Veri yüklenirken veya analiz edilirken bir hata oluştu: {e}")

if __name__ == '__main__':
    load_and_explore_data()
