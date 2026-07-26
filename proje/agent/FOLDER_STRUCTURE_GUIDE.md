# Perakende Talep Tahmini — Klasör ve Agent Dosya Sözleşmesi

Bu rehber, kullanıcının seçtiği ürün ve gelecek hedef gün için talep oluşumunu ve günlük miktarı (`KG` veya `ADT`) tahmin eden projenin tek dosyalama sözleşmesidir.

Klasör yapısı yalnızca düzen için değildir. Agent'lar arasındaki veri güvenliği, zaman sızıntısının önlenmesi, modelin yeniden üretilebilirliği ve deployment'ta doğru modelin yüklenmesi bu sözleşmeye bağlıdır.

```text
EDA Expert → DataPrep Expert → Model Expert → Deployment Expert → Monitoring / Retraining
```

Her agent yalnızca kendi sorumluluğundaki çıktıyı üretir ve kendinden önceki agent'ın imzalı/raporlanmış çıktısını okur. Bir agent başka bir agent'ın çıktısını sessizce farklı adla yeniden üretmez veya üzerine yazmaz.

## 1. Proje kökü ve hedef yapı

Mevcut ham dosya `data/data.csv` konumundadır. Bu dosya korunur; taşınması ya da yeniden adlandırılması zorunlu değildir.

```text
proje/
├── data/
│   ├── data.csv                                  # Ham satış verisi — salt okunur
│   ├── reference/
│   │   ├── calendar_sources.json                 # MEB/Diyanet kaynak ve sürüm manifesti
│   │   ├── weather_sources.json                  # Bursa konum/kaynak/checksum manifesti
│   │   ├── bursa_weather_observed_era5.csv       # EDA + time-safe klimatoloji
│   │   └── bursa_weather_previous_runs.csv       # Opsiyonel forecast-as-of arşivi
│   ├── processed/                                # DataPrep çıktısı
│   │   ├── sales_cleaned.csv
│   │   ├── daily_product_demand.csv
│   │   ├── product_catalog.csv
│   │   └── turkey_calendar_daily.csv
│   └── model_ready/                              # DataPrep'in Model Expert'e teslimi
│       ├── demand_features_train.csv
│       ├── demand_features_validation.csv
│       ├── demand_features_test.csv
│       ├── demand_targets_train.csv
│       ├── demand_targets_validation.csv
│       ├── demand_targets_test.csv
│       └── split_metadata.json
│
├── scripts/                                      # Çalıştırılabilir, tekrarlanabilir işlemler
│   ├── eda_analysis.py
│   ├── data_preparation.py
│   ├── turkey_calendar.py
│   ├── weather_features.py
│   ├── fetch_bursa_weather.py
│   ├── train_demand_models.py
│   ├── update_bundle_checksums.py
│   ├── execute_story_notebook.py
│   ├── update_story_notebook_calendar.py
│   ├── update_story_notebook_weather.py
│   └── final_validation.py
│
├── notebooks/                                    # Keşif/deneme; üretim kaynağı değildir
│   └── retail_demand_forecasting_story.ipynb     # Ortak, çalıştırılmış veri hikâyesi
│
├── figures/                                      # Raporlardaki HTML/PNG grafikler
│   ├── eda_*.html / eda_*.png
│   ├── dataprep_*.html / dataprep_*.png
│   └── model_*.html / model_*.png
│
├── reports/
│   ├── csv/                                      # Makine-okunur analiz ve doğrulama sonuçları
│   └── markdown/                                 # Agent handoff ve karar raporları
│       ├── EDA_FINAL_REPORT.md
│       ├── DATA_PREP_HANDOFF.md
│       ├── MODEL_EVALUATION_REPORT.md
│       ├── MODEL_EXPERT_HANDOFF.md
│       ├── CALENDAR_FEATURE_REPORT.md
│       ├── WEATHER_FEATURE_REPORT.md
│       └── DEPLOYMENT_GUIDE.md
│
├── models/
│   └── demand_forecasting_bundle/                # Deployment'ın tek model kaynağı
│       ├── occurrence_model_kg.pkl               # Varsa: KG talep oluşumu
│       ├── occurrence_model_adt.pkl              # Varsa: ADT talep oluşumu
│       ├── quantity_model_kg.pkl                 # KG talep miktarı
│       ├── quantity_model_adt.pkl                # ADT talep miktarı
│       ├── preprocessing_pipeline_kg.pkl         # Varsa; yalnızca train üzerinde fit edilmiş
│       ├── preprocessing_pipeline_adt.pkl        # Varsa; yalnızca train üzerinde fit edilmiş
│       ├── feature_builder.pkl veya feature_builder.py
│       ├── product_catalog.csv
│       ├── bursa_weather_observed_era5.csv
│       ├── bursa_weather_previous_runs.csv
│       ├── model_metadata.json
│       └── checksums.json                        # Önerilen bütünlük kaydı
│
├── app/                                          # Deployment Expert'in uygulaması
│   ├── app.py
│   └── services/
│       ├── bundle_loader.py
│       ├── forecast_service.py
│       ├── input_validator.py
│       └── monitoring_service.py
│
├── logs/                                         # Deployment/monitoring çıktıları; sürüm kontrolüne girmez
│   ├── forecast_log.csv veya güvenli_veritabani
│   ├── actuals_log.csv veya güvenli_veritabani
│   └── monitoring_summary.csv
│
├── tests/
│   ├── test_bundle_loader.py
│   ├── test_input_validator.py
│   ├── test_feature_time_safety.py
│   ├── test_weather_features.py
│   └── test_forecast_service.py
│
├── agent/                                        # Bu projedeki bağlı agent tanımları
│   ├── eda-expert-agent.md
│   ├── dataprep-expert-agent.md
│   ├── model-expert-agent.md
│   ├── deployment-expert-agent.md
│   └── FOLDER_STRUCTURE_GUIDE.md
│
├── requirements.txt
├── README.md
└── .gitignore
```

`notebooks/retail_demand_forecasting_story.ipynb` bu projenin zorunlu, çalıştırılmış veri hikâyesi çıktısıdır. Sırası sabittir: **1) İş problemi ve tahmin sözleşmesi → 2) EDA → 3) DataPrep ve zaman güvenliği → 4) Modelleme/rolling backtest → 5) Değerlendirme/model kararı → 6) Deployment simülasyonu/izleme**. Her alt adımda Markdown amaç sorusu, çalıştırılmış kod-görsel ve kanıta dayalı Türkçe karar/handoff yorumu bulunur. Notebook raporların veya üretim scriptlerinin yerine geçmez; aynı çıktı ve sürümleri anlaşılır biçimde birleştirir.

## 2. Temel sahiplik ve erişim kuralları

| Konum | Yazma sahibi | Okuyan agent(lar) | Kural |
|---|---|---|---|
| `data/data.csv` | İş sahibi / kaynak sistem | EDA, DataPrep | Salt okunur; asla üzerine yazılmaz |
| `data/processed/` | DataPrep Expert | Model, Deployment (gerekirse) | Temizlenmiş kayıt ve günlük talep paneli |
| `data/model_ready/` | DataPrep Expert | Model Expert | Hedef/feature/split sözleşmesinin tek kaynağı |
| `reports/csv/eda_*` | EDA Expert | DataPrep, Model | Ölçülmüş EDA kanıtları |
| `reports/csv/data_prep_*` | DataPrep Expert | Model | Hazırlama/doğrulama kanıtları |
| `reports/csv/model_*` | Model Expert | Deployment, Monitoring | Backtest, model seçimi ve performans |
| `reports/markdown/` | İlgili agent | Zincirdeki sonraki agent | Handoff raporu; geçmiş kararların kaydı |
| `models/demand_forecasting_bundle/` | Model Expert | Deployment Expert | Tek sürümlenmiş inference kaynağı |
| `app/` | Deployment Expert | Kullanıcı/operasyon | Bundle tüketir; eğitim mantığını kopyalamaz |
| `logs/` | Deployment/Monitoring | Model, operasyon | Hassas olabilir; korunur ve Git'e eklenmez |

EDA Expert ham veriyi inceler ama `processed/` veya `model_ready/` verisi yazmaz. DataPrep ham veriyi değiştirmez. Model Expert ham/işlenmiş veriyi yenilemez. Deployment Expert feature'ları elle yeniden hesaplamaz; bundle'daki onaylı feature builder'ı kullanır.

## 3. Veri katmanları ve dosya içerik sözleşmesi

### 3.1 Ham veri — `data/data.csv`

Kaynak satış kayıtlarıdır. Beklenen alanlar:

```text
satıs_tarıhı; urun_ıd; urun_ad; satılan_mıktar
```

`satılan_mıktar`, Türkçe ondalık/binlik biçimi ve birimi birlikte içerebilir. Ham dosyada düzeltme yapılmaz; sorunlar yalnızca EDA/DataPrep raporlarına kaydedilir.

### 3.2 İşlenmiş veri — `data/processed/`

| Dosya | Minimum amaç/içerik |
|---|---|
| `sales_cleaned.csv` | Ayrıştırılmış `date`, `product_id`, `product_name`, `quantity`, `unit`, `parse_status`; ham satır kaynağı izlenebilir olmalı |
| `daily_product_demand.csv` | Ürün × gün talep paneli, `daily_demand`, veri/gözlem durumu, aktiflik/geçmiş bayrakları |
| `product_catalog.csv` | `product_id`, ürün adı, birim, ilk/son satış, geçmiş özeti ve ürün durumları |

Bu katmanda `KG` ve `ADT` birleştirilmez. Satışsız günlerin `0` mı yoksa `missing_or_unobserved` mı olduğu `DATA_PREP_HANDOFF.md` içinde açıkça belirtilir.

### 3.3 Model-ready veri — `data/model_ready/`

Feature ve target dosyalarında en az şu kimlik alanları korunmalıdır:

```text
forecast_origin, target_date, product_id, product_name, unit, product_segment, lead_days
```

Hedef dosyaları ek olarak şunları içerir:

```text
demand_occurs, target_demand
```

Model seçili tek hedef günün günlük talebini tahmin eder. `lead_days = target_date - forecast_origin` feature olarak korunur; doğrulanan maksimum lead aralığı `split_metadata.json` ve model metadata'da yazılır.

`split_metadata.json` en az train/validation/test başlangıç-bitiş tarihlerini, purge/gap uzunluğunu, feature sürümünü, sıfır-talep politikasını ve satır/ürün sayılarını içerir.

## 4. Agent bazında zorunlu çıktılar

### EDA Expert

EDA yalnızca kanıt, rapor ve görsel üretir.

```text
reports/csv/
├── eda_data_profile.csv
├── eda_data_quality_issues.csv
├── eda_product_history_summary.csv
├── eda_daily_demand_summary.csv
├── eda_demand_segments.csv
├── eda_temporal_coverage.csv
└── data_prep_recommendations.csv

reports/markdown/EDA_FINAL_REPORT.md
figures/eda_*.html ve figures/eda_*.png
```

`data_prep_recommendations.csv` en az `Sorun`, `Kanıt`, `Öneri`, `Öncelik`, `Sorumlu` alanlarını içerir. DataPrep Expert, her öneriye kendi handoff raporunda uygulandı/reddedildi/ertelendi kararı verir.

### DataPrep Expert

DataPrep, EDA bulgularını doğrular ve veri/feature/split katmanını üretir.

```text
data/processed/*
data/model_ready/*
reports/csv/
├── data_prep_summary.csv
├── data_quality_issues.csv
└── product_history_summary.csv
reports/markdown/DATA_PREP_HANDOFF.md
models/feature_specification.json
figures/dataprep_*.html ve figures/dataprep_*.png
```

`feature_specification.json`, her feature'ın açıklamasını, dtype'ını, oluşturma kuralını, kaynak maksimum zamanını ve feature builder sürümünü içerir. Bu dosya, Model Expert ve Deployment Expert'in aynı feature anlamını kullanmasını sağlar.

### Model Expert

Model Expert, yalnızca `model_ready` katmanını ve handoff'ları kullanarak backtest/model seçimi yapar.

```text
reports/csv/
├── model_comparison_results.csv
├── backtest_results.csv
└── segment_metrics.csv
reports/markdown/
├── MODEL_EVALUATION_REPORT.md
└── MODEL_EXPERT_HANDOFF.md
figures/model_*.html ve figures/model_*.png
models/demand_forecasting_bundle/*
```

Bundle'daki model, pipeline, feature builder, ürün kataloğu ve `model_metadata.json` aynı sürümden olmalıdır. Sadece `final_model.pkl` dosyası deployment için yeterli kabul edilmez.

Varsayılan bundle düzeni birim bazındadır: `KG` ve `ADT` için ayrı model/pipeline dosyaları bulunur. Dosya adı bu düzenden farklıysa `model_metadata.json` içindeki zorunlu `unit_model_map`, her birim için occurrence modeli (varsa), quantity modeli ve pipeline yolunu açıkça göstermelidir. Deployment Expert bu harita dışında dosya adı tahmin etmez.

### Deployment Expert

Deployment, bundle'ı uygular, tahminleri güvenli loglar ve sonradan gerçekleşen satışlarla izlemeye hazırlar.

```text
app/*
tests/*
reports/csv/
├── deployment_validation_results.csv
└── deployment_feedback.csv
reports/markdown/DEPLOYMENT_GUIDE.md
logs/*
```

Deployment uygulamasının tahmin CSV çıktısı aşağıdaki alanları korur:

```text
forecast_id, forecast_origin, forecast_window_start, forecast_window_end,
product_id, product_name, unit, demand_probability, demand_prediction,
prediction_interval_lower, prediction_interval_upper, model_version,
feature_builder_version, data_freshness, status, warning_codes
```

## 5. Handoff sırası ve dosya kontrol listesi

```text
1. EDA Expert
   data/data.csv oku
   → reports/csv/eda_* + data_prep_recommendations.csv
   → reports/markdown/EDA_FINAL_REPORT.md

2. DataPrep Expert
   EDA raporlarını + data/data.csv oku
   → data/processed/*
   → data/reference/calendar_sources.json + data/processed/turkey_calendar_daily.csv
   → data/model_ready/* + split_metadata.json
   → DATA_PREP_HANDOFF.md + feature_specification.json

3. Model Expert
   DataPrep handoff + model_ready veri + feature specification oku
   → rolling backtest/model raporları
   → models/demand_forecasting_bundle/*
   → MODEL_EXPERT_HANDOFF.md

4. Deployment Expert
   Model bundle + tüm handoff'ları doğrula
   → app/* + tests/* + DEPLOYMENT_GUIDE.md
   → logs/forecast_log.*

5. Monitoring / Retraining
   forecast_log ile gerçekleşen satışları eşleştir
   → logs/actuals_log.* + monitoring_summary.csv
   → gerekiyorsa Model/DataPrep/EDA'ya geri bildirim
```

Bir sonraki aşama başlamadan önce aşağıdaki minimum dosyalar bulunmalıdır:

| Geçiş | Gerekli dosyalar |
|---|---|
| EDA → DataPrep | `EDA_FINAL_REPORT.md`, `data_prep_recommendations.csv` |
| DataPrep → Model | `DATA_PREP_HANDOFF.md`, model-ready feature/target dosyaları, `split_metadata.json`, `feature_specification.json` |
| Model → Deployment | `MODEL_EXPERT_HANDOFF.md`, geçerli model bundle, `model_metadata.json` |
| Deployment → Monitoring | tahmin logu, model sürümü, tahmin penceresi ve gerçekleşen satış eşleştirme anahtarı |

## 6. İsimlendirme, sürümleme ve zaman güvenliği

- Dosya/klasör adları küçük harf ve `snake_case` kullanır.
- Tarihler ISO biçimindedir: `YYYY-MM-DD`.
- Saat bilgisi gerekiyorsa ISO-8601 ve saat dilimi kullanılır.
- Tahmin tipi ve maksimum lead aralığı metadata'da görünür: `direct_multi_horizon_daily`, `max_forecast_lead_days`.
- Birim dosya/rapor kırılımında net olmalıdır: `kg`, `adt` veya `unit=KG/ADT` alanı.
- Ürün kimliği metin olarak korunur; baştaki sıfırlar veya büyük kimlikler kaybolmaz.
- Model bundle değiştiğinde `model_version`, `feature_builder_version` ve checksum birlikte güncellenir.
- Özel takvim değiştiğinde `calendar_version`, `calendar_features`, kaynak manifesti,
  model sürümü ve checksum birlikte güncellenir; train/inference farklı takvim
  sürümü kullanamaz.
- Hava katmanı değiştiğinde `weather_feature_version`, konum, kaynak manifesti,
  gözlem/forecast dosya hash'leri, `weather_candidate_features`,
  `weather_features`, ablation kararı, model sürümü ve bundle checksum'u birlikte
  güncellenir.
- `latest` adlı dosya ancak hangi sürüme işaret ettiği metadata'da izlenebiliyorsa kullanılır; sürüm numarasız model üzerine yazılmaz.

Zaman sızıntısını önlemek için:

- `forecast_origin` feature/target dosyalarında bulunur.
- Feature kaynak tarihi `forecast_origin` sonrasına geçemez.
- `split_metadata.json` içindeki purge/gap model/backtest boyunca aynen uygulanır.
- Test seti veya canlı gerçekleşen satışlar eğitim/feature üretim klasörüne geri yazılmaz.
- Gerçekleşen hedef-gün havası estimator feature'ı yapılmaz. Hava gözlemi yalnız
  EDA veya `weather_date < forecast_origin` filtresiyle klimatoloji üretiminde
  kullanılır; forecast arşivinde issue-time/lead eşleşmesi zorunludur.

## 7. Path kullanımı ve klasör oluşturma

Scriptlerin farklı çalışma dizinlerinden güvenle çalışması için proje kökünü kodla bul veya açık bir yapılandırma kullan. Körlemesine mevcut çalışma dizinine göre dosya yazma.

Örnek, `scripts/` içinden çalıştırılan işlem için:

```python
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA = PROJECT_ROOT / "data" / "data.csv"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODEL_READY_DIR = PROJECT_ROOT / "data" / "model_ready"
REPORT_CSV_DIR = PROJECT_ROOT / "reports" / "csv"
REPORT_MD_DIR = PROJECT_ROOT / "reports" / "markdown"
FIGURES_DIR = PROJECT_ROOT / "figures"
MODELS_DIR = PROJECT_ROOT / "models"

for directory in [PROCESSED_DIR, MODEL_READY_DIR, REPORT_CSV_DIR,
                  REPORT_MD_DIR, FIGURES_DIR, MODELS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)
```

Ham veri için `mkdir` veya yazma işlemi yapılmaz. Uygulama altında log yazılacaksa `logs/` ayrıca ve erişim politikasıyla oluşturulur.

## 8. Git, gizlilik ve büyük dosya politikası

`.gitignore` en az aşağıdakileri içermelidir:

```gitignore
__pycache__/
.ipynb_checkpoints/
.DS_Store
.env
logs/
data/processed/
data/model_ready/
models/demand_forecasting_bundle/*.pkl
models/demand_forecasting_bundle/checksums.json
*.html
```

Bu öneri, dosyaların yerelde üretilmesini engellemez; sadece hassas/büyük/tekrar üretilebilir çıktıları kaynak kontrolünden ayırır. Ham satış verisi, tahmin logları ve model dosyaları kişisel/işletme verisi içerebilir; erişim, yedekleme ve saklama süresi politikasına göre korunur.

## 9. Yasak ve kaçınılacak düzenler

```text
❌ data.csv üzerine temiz veri yazmak
❌ EDA'nın model-ready veri üretmesi
❌ DataPrep'in tahmin modeli kaydetmesi
❌ Deployment'ın notebook'taki geçici feature kodunu kopyalaması
❌ KG ve ADT model/metric sonuçlarını tek dosyada birimsiz toplam olarak vermek
❌ Test sonuçlarını eğitim verisine geri yazmak
❌ models/ altına sürümsüz rastgele .pkl dosyaları bırakmak
❌ reports/ köküne adı belirsiz CSV/MD dosyaları saçmak
❌ app/ içine ham satış verisi veya kişisel log kopyalamak
```

## 10. Son kontrol

Bir agent çalışmasını tamamlamadan önce şunları kontrol eder:

1. Çıktı kendi sahiplik alanına mı yazıldı?
2. Girdi dosyası ve sürümü rapora işlendi mi?
3. Tahmin ufku, birim ve tarih aralığı belirtilmiş mi?
4. Sonraki agent'ın zorunlu handoff dosyası var mı?
5. Veri/feature kaynak tarihi geleceğe taşmıyor mu?
6. Çıktı yeniden üretilebilir mi; script, sürüm ve yapılandırma izlenebilir mi?
7. Hassas veya büyük dosya yanlışlıkla kaynak kontrolüne eklenmiyor mu?

Bu yapı, agent'ların birbirinden kopuk dosyalar üretmesini önler; aynı talep tahmini kararının EDA bulgusundan canlı izlemeye kadar izlenebilir kalmasını sağlar.
