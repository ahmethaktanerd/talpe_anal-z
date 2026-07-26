# Model Expert Handoff — Deployment

## Tahmin sözleşmesi

- Forecast type: `direct_multi_horizon_daily`
- Forecast origin: 2026-07-21
- Kullanıcı girdisi: ürün ve hedef tarih
- Doğrulanmış lead aralığı: 1–180 gün
- Hedef: seçili günde talep olasılığı ve günlük KG/ADT miktarı

## Bundle

- Sürüm: `2026.07.26-weather-v3`
- Layout: `per_unit`
- Feature builder: `3.0.0`
- Takvim: `TR_CALENDAR_2023_2027_V1`
- Hava: `BURSA_WEATHER_FEATURES_V1` — Bursa/Osmangazi time-safe klimatoloji
- Metadata: `models/demand_forecasting_bundle/model_metadata.json`
- Checksum: `models/demand_forecasting_bundle/checksums.json`

## Birim yönlendirmesi

{
  "KG": {
    "occurrence_model_path": "occurrence_model_kg.pkl",
    "quantity_model_path": "quantity_model_kg.pkl",
    "pipeline_path": null,
    "occurrence_model_name": "extra_trees",
    "quantity_model_name": "hist_gradient_boosting_poisson"
  },
  "ADT": {
    "occurrence_model_path": "occurrence_model_adt.pkl",
    "quantity_model_path": "quantity_model_adt.pkl",
    "pipeline_path": null,
    "occurrence_model_name": "hist_gradient_boosting",
    "quantity_model_name": "hist_gradient_boosting_poisson"
  }
}

## Güvenli sonuç sunumu

Olasılık kalibre değildir. Sonuç “modelin talep olasılığı” olarak gösterilmelidir.
Miktar ürün birimindedir. `ADT` sonuç ekranda tam sayıya yuvarlanabilir; ham tahmin
logda korunur. Modelin miktar tahmini sipariş önerisi değildir.

## İzleme

Tahmin kaydı ürün, forecast origin, target date, lead days, unit, model sürümü,
olasılık, miktar ve uyarı kodlarını içermelidir. Hedef gün geçtikten sonra gerçek
satışla ürün-tarih-birim anahtarında eşleştirilmelidir.
