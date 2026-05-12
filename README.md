# 📚 Kitap Gurmesi & Logic Gate Explorer (MLP from Scratch)

Bu proje, bir Yazılım Mühendisliği bitirme/araştırma çalışması kapsamında geliştirilen, yüksek seviyeli derin öğrenme kütüphaneleri (TensorFlow, PyTorch vb.) kullanılmadan, sadece **NumPy** kütüphanesi ile sıfırdan (from scratch) inşa edilmiş bir **Çok Katmanlı Algılayıcı (Multi-Layer Perceptron - MLP)** uygulamasıdır.

Proje iki ana modülden oluşmaktadır:
1.  **Kitap Gurmesi:** Kitap özelliklerine (Yazar, Sayfa Sayısı, Yaş vb.) göre kitap puanı tahmini yapan bir regresyon modeli.
2.  **Logic Gate Explorer:** MLP mimarisinin AND, OR, NAND ve XOR gibi mantıksal kapıları nasıl öğrendiğini görselleştiren bir analiz aracı.

## 🧠 Teknik Özellikler

* **Mimari:** Esnek katman yapısına sahip (Hidden Layers) Çok Katmanlı Algılayıcı.
* **Aktivasyon Fonksiyonları:** Gizli katmanlarda **ReLU**, çıkış katmanında **Linear** aktivasyon.
* **Optimizasyon:** Manuel Backpropagation (Geri Yayılım) ve Gradyan İnişi (Gradient Descent).
* **Gradyan Kırpma (Gradient Clipping):** Patlayan gradyanlar (Exploding Gradients) sorununu çözmek için `clip_value` mekanizması.
* **Ağırlık Başlatma:** He Initialization (ReLU ile optimize edilmiş).

## 📂 Proje Yapısı

* `mlp_scratch.py`: Sinir ağının çekirdek motoru (Forward & Backward pass).
* `preprocessing.py`: `Books.csv`, `Users.csv` ve `Ratings.csv` verilerinin temizlenmesi, normalizasyonu ve eğitim verisi haline getirilmesi.
* `interface.py`: Hiperparametre ayarı yapmaya ve sonuçları karşılaştırmaya olanak sağlayan Tkinter tabanlı GUI.
* `logic_gates.py`: Mantıksal kapıların karar sınırlarını (Decision Boundaries) görselleştiren ek modül.

## 🚀 Başlatma

1.  **Gerekli Paketleri Kurun:**
    ```bash
    pip install -r requirements.txt
    ```

2.  **Uygulamayı Çalıştırın:**
    * Kitap Tahmin Sistemi için: `python interface.py`
    * Mantıksal Kapı Görselleştirmesi için: `python logic_gates.py`

## 🛠️ Geliştirme Süreci (WIP)

Bu proje aktif olarak geliştirilmeye devam etmektedir. Gelecek güncellemelerde planlanan özellikler:
- [ ] Farklı hata fonksiyonlarının (MSE vs. MAE) entegre edilmesi.
- [ ] Adam ve RMSProp gibi gelişmiş optimizasyon algoritmalarının sıfırdan yazılması.
- [ ] Dropout katmanları ile overfitting engelleme mekanizması.

---
*Developed by Idil Esen as a Software Engineering Project.*