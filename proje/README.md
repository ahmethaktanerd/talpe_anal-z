# Mağaza Günlük Talep Tahmini

Bu proje, kullanıcının Streamlit arayüzünde seçtiği ürün ve gelecek hedef tarih için:

1. O gün ürüne talep beklenip beklenmediğini,
2. Talep bekleniyorsa ürünün kayıtlı birimine göre tahmini günlük `ADT` veya `KG`
   miktarını; hedef günün hafta sonu, resmî/dinî tatil, Ramazan ve MEB okul
   dönemi ile Bursa iklim normali bağlamını

üretir.

## Tahmin sözleşmesi

- Son bilinen veri tarihi: **2026-07-21**
- İlk sürümün doğrulanmış seçim aralığı: **1–180 gün**
- Seçilebilecek son tarih: **2027-01-17**
- Tarih seçimi tek bir örnek güne sabit değildir; 1 ve 2 Ocak 2027 gibi
  doğrulanmış aralıktaki her gün ayrı ayrı seçilebilir.
- `KG` ile `ADT` ayrı modellerle tahmin edilir.
- Olasılık kalibre edilmiş güven skoru değildir.
- Tahmin, eldeki stok ve tedarik bilgisi olmadan sipariş emri değildir.

## Proje akışı

```text
data/data.csv
→ EDA
→ günlük ürün-talep paneli
→ MEB/Diyanet kaynaklı sürümlü Türkiye özel takvimi
→ Bursa ERA5 hava EDA'sı ve zaman-güvenli klimatoloji adayları
→ doğrudan çok-ufuklu günlük eğitim örnekleri
→ KG/ADT occurrence + quantity modelleri
→ sürümlü model bundle
→ Streamlit ürün ve tarih seçimi
→ tahmin logu ve gerçekleşen satış izleme
```

## Yeniden üretim

Proje kökünde:

```bash
python3 -m scripts.eda_analysis
python3 -m scripts.fetch_bursa_weather
python3 -m scripts.data_preparation
python3 -m scripts.train_demand_models
python3 -m scripts.update_bundle_checksums
python3 -m scripts.update_story_notebook_calendar
python3 -m scripts.update_story_notebook_weather
python3 -m scripts.execute_story_notebook
python3 -m pytest -q
python3 -m streamlit run app/app.py
```

## Streamlit kullanımı

1. Ürünü ürün adı/ID listesinden seçin.
2. Son veri gününden sonraki hedef tarihi seçin.
3. **Tahmini oluştur** düğmesine basın.
4. Sonuç ekranı talep kararını, modelin olasılığını ve ürün birimindeki günlük
   miktarı gösterir. Mağaza stoku ve yoldaki sevkiyat sıfırsa bu miktar başlangıç
   depo transferi olarak yorumlanabilir. Hedef günün tatil/Ramazan/okul dönemi
   bağlamı ve Bursa için zaman-güvenli iklim normali ayrıca görünür.

## Temel dosyalar

- EDA raporu: `reports/markdown/EDA_FINAL_REPORT.md`
- DataPrep handoff: `reports/markdown/DATA_PREP_HANDOFF.md`
- Model raporu: `reports/markdown/MODEL_EVALUATION_REPORT.md`
- Özel takvim raporu: `reports/markdown/CALENDAR_FEATURE_REPORT.md`
- Takvim kaynak manifesti: `data/reference/calendar_sources.json`
- Hava feature/ablation raporu: `reports/markdown/WEATHER_FEATURE_REPORT.md`
- Hava kaynak manifesti: `data/reference/weather_sources.json`
- Model handoff: `reports/markdown/MODEL_EXPERT_HANDOFF.md`
- Deployment/kullanım rehberi: `reports/markdown/DEPLOYMENT_GUIDE.md`
- Uçtan uca final raporu: `reports/markdown/PROJECT_FINAL_REPORT.md`
- Ortak veri hikâyesi: `notebooks/retail_demand_forecasting_story.ipynb`
- Uygulama: `app/app.py`
- Model bundle: `models/demand_forecasting_bundle/`

## Bilinen sınırlılıklar

Satış veri setinde fiyat, kampanya, ürünün stokta olmadığı günler, mağaza özel
kapanışı, hava veya tedarik bilgileri bulunmamaktadır. Türkiye takvimi harici,
sürümlü feature olarak eklenmiştir. Bursa/Osmangazi merkez için ERA5 hava verisi
haricî referans katmanına alınmış; hedef-gün gerçekleşen havası sızıntı yaratacağı
için estimator'a verilmemiştir. Zaman-güvenli klimatoloji adayları validation
ablation'da sağlam ve birimler arası tutarlı katkı göstermediğinden final 47 model
feature'ından çıkarılmış, EDA/model-ready/arayüz bağlamında korunmuştur. Tam mağaza
koordinatı, yerel okul/mağaza değişiklikleri ve geleceğin kesin hava durumu hâlâ
kapsanmaz.
