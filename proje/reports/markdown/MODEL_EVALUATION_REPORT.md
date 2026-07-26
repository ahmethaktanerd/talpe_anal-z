# Model Değerlendirme Raporu

## Problem

Seçilen ürün ve gelecek hedef gün için günlük talep oluşumu ile ürünün kendi birimindeki miktar doğrudan tahmin edilir. Model 1–180 günlük lead aralığı için eğitilmiştir.

Model `TR_CALENDAR_2023_2027_V1` takvimiyle resmî/dinî tatil, Ramazan, tatil önce/sonra pencereleri ve MEB okul dönemlerini kullanır.

## Seçilen modeller

### KG

- Occurrence: `extra_trees`
- Quantity: `hist_gradient_boosting_poisson`
- Validation threshold: `0.460`
- Test occurrence PR-AUC: `0.8307`
- Test occurrence F1: `0.7658`
- Test quantity MAE: `1.7100`
- Test quantity WAPE: `0.8777`
- Test bias: `0.0966`

### ADT

- Occurrence: `hist_gradient_boosting`
- Quantity: `hist_gradient_boosting_poisson`
- Validation threshold: `0.470`
- Test occurrence PR-AUC: `0.6658`
- Test occurrence F1: `0.6156`
- Test quantity MAE: `1.0935`
- Test quantity WAPE: `1.1323`
- Test bias: `0.2737`

## Değerlendirme ilkeleri

- Model seçimi validation sonuçlarıyla yapılmış, test final ölçüm için kullanılmıştır.
- KG ve ADT ayrı model ve metriklerle değerlendirilmiştir.
- Olasılıklar kalibre değildir; kullanıcıya güven skoru olarak sunulmaz.
- Tahmin stok/sipariş önerisi değildir.
- Özel takvim alanlarının katkısı validation ablation ile ölçülmüştür.
- Hava klimatolojisinin katkısı validation ablation ile ölçülmüştür.
- Takvim ablation sonucu: `reports/csv/calendar_feature_ablation.csv`.
- Hava ablation sonucu: `reports/csv/weather_feature_ablation.csv`.
- Özel tarih test segmentleri: `reports/csv/calendar_segment_metrics.csv`.

## Kalan riskler

- No price, promotion, stockout, or store-specific closure features.
- Weather is Bursa city-centre time-safe climatology, not the realised weather of the future target day.
- Exact store coordinates and a complete archived/live forecast contract are not yet available.
- School calendar is national; local closures and store opening hours are absent.
- Forecasts near 180 days are less certain than near-term forecasts.
- Probability is not calibrated and must not be labelled confidence.
- Demand forecast is not a replenishment order without inventory inputs.