import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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

        # Görselleştirmeler
        print("\n--- Veri Görselleştirmeleri ---")

        # 1. Churn Dağılımı
        plt.figure(figsize=(6, 4))
        sns.countplot(x='Churn', data=df, palette='viridis')
        plt.title('Müşteri Terk Dağılımı')
        plt.xlabel('Churn Durumu')
        plt.ylabel('Müşteri Sayısı')
        plt.savefig('churn_distribution.png') # Grafiği kaydet
        plt.show()

        # 2. Kategorik Özelliklerin Churn ile İlişkisi
        categorical_cols = df.select_dtypes(include='object').columns.drop(['CustomerID', 'Churn'])
        plt.figure(figsize=(15, 10))
        for i, col in enumerate(categorical_cols):
            plt.subplot(3, 4, i + 1)  # Düzen için 3 satır, 4 sütun
            sns.countplot(x=col, hue='Churn', data=df, palette='viridis')
            plt.title(f'{col} vs Churn')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
        plt.savefig('categorical_features_vs_churn.png') # Grafiği kaydet
        plt.show()

        # 3. Sayısal Özelliklerin Dağılımı (Histogram)
        numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns.drop(['Tenure', 'MonthlyCharges', 'TotalCharges'])
        plt.figure(figsize=(15, 10))
        for i, col in enumerate(numerical_cols):
            plt.subplot(2, 2, i + 1)
            sns.histplot(df[col], kde=True, bins=20)
            plt.title(f'{col} Dağılımı')
            plt.tight_layout()
        plt.savefig('numerical_features_histograms.png') # Grafiği kaydet
        plt.show()

        # 4. Sayısal Özelliklerin Churn ile İlişkisi (Boxplot)
        numerical_churn_cols = ['Age', 'MonthlyCharges', 'Tenure', 'TotalCharges'] # Churn ile ilişkili önemli sayısal sütunlar
        plt.figure(figsize=(15, 10))
        for i, col in enumerate(numerical_churn_cols):
            plt.subplot(2, 2, i + 1)
            sns.boxplot(x='Churn', y=col, data=df, palette='viridis')
            plt.title(f'{col} vs Churn')
            plt.tight_layout()
        plt.savefig('numerical_features_vs_churn_boxplots.png') # Grafiği kaydet
        plt.show()

        # 5. Sayısal Özellikler Arası Korelasyon Matrisi
        plt.figure(figsize=(10, 8))
        sns.heatmap(df[numerical_churn_cols].corr(), annot=True, cmap='coolwarm', fmt='.2f')
        plt.title('Sayısal Özellikler Arası Korelasyon Matrisi')
        plt.savefig('correlation_matrix.png') # Grafiği kaydet
        plt.show()

    except FileNotFoundError:
        print(f"Hata: {filepath} dosyası bulunamadı. Lütfen dosya yolunu kontrol edin.")
    except Exception as e:
        print(f"Veri yüklenirken veya analiz edilirken bir hata oluştu: {e}")

if __name__ == '__main__':
    load_and_explore_data()
