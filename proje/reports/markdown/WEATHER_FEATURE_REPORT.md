# Bursa Hava Durumu Feature Raporu

## Güvenli veri sözleşmesi

- Hava feature sürümü: `BURSA_WEATHER_FEATURES_V1`
- Konum: Bursa/Osmangazi şehir merkezi (40.19559, 29.06013).
- Gerçekleşmiş hedef-gün havası model girdisi değildir; yalnız EDA'dadır.
- Her satırda ERA5 hava tarihleri `forecast_origin` gününden kesinlikle önce filtrelenir.
- Hedef mevsim için ±15 günlük dairesel pencere klimatolojisi kullanılır.
- Kaynak manifesti: `data/reference/weather_sources.json`.

## Validation ablation

| Birim | Aday havalı PR-AUC | Deploy PR-AUC | Δ PR-AUC | Aday havalı MAE | Deploy MAE | Δ MAE |
|---|---:|---:|---:|---:|---:|---:|
| KG | 0.8648 | 0.8648 | +0.0000 | 1.8103 | 1.8096 | +0.0007 |
| ADT | 0.6879 | 0.6887 | -0.0009 | 1.3671 | 1.3625 | +0.0046 |

Pozitif Δ PR-AUC ve negatif Δ MAE iyileşme demektir. Ablation aynı validation dönemi ve aynı seçilmiş model aileleriyle yapılmıştır.

## Deployment kararı

**Hava alanları final estimator feature listesinden çıkarıldı.** KG'de çok küçük occurrence farkı diğer metriklerde ve ADT'de doğrulanmadığı için sonuç sağlam bir genel kazanım sayılmadı. Aday alanlar model-ready tabloda, EDA'da ve kullanıcıya açıklayıcı hava bağlamında korunur.

## Betimsel gerçekleşen hava ilişkileri

| Birim | Koşul | Gün | Ortalama talep oranı (koşul / diğer günler) |
|---|---|---:|---:|
| KG | rainy_1mm_plus | 426 | 1.057 |
| KG | heavy_rain_10mm_plus | 110 | 1.104 |
| KG | snow_day | 68 | 1.176 |
| KG | hot_30c_plus | 245 | 0.882 |
| KG | cold_below_10c | 121 | 1.132 |
| KG | sunny_dry | 528 | 0.962 |
| ADT | rainy_1mm_plus | 426 | 1.080 |
| ADT | heavy_rain_10mm_plus | 110 | 1.107 |
| ADT | snow_day | 68 | 1.186 |
| ADT | hot_30c_plus | 245 | 0.893 |
| ADT | cold_below_10c | 121 | 1.150 |
| ADT | sunny_dry | 528 | 0.905 |

## Yorum sınırı

Gerçekleşen hava oranları nedensellik kanıtı değildir; mevsim, ürün portföyü, fiyat, kampanya ve stok durumu ile karışabilir. Modeldeki esas karar kanıtı zaman sıralı validation ablation'dır. Bu kanıt zayıf olduğu için hava feature grubu final estimator'lardan çıkarılmıştır.
