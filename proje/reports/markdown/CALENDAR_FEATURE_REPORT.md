# Özel Takvim Feature Raporu

## Kapsam

- Takvim sürümü: `TR_CALENDAR_2023_2027_V1`
- Feature builder: `3.0.0`
- Özel takvim feature sayısı: **18**
- Kaynak manifesti: `data/reference/calendar_sources.json`
- İçerik: resmî/dinî tatil, yarım gün, Ramazan, kandil, tatilden 1/3 gün önce-sonra, MEB okul dönemi ve tatilleri.

## Validation ablation

| Birim | Full PR-AUC | Takvimsiz PR-AUC | Δ PR-AUC | Full MAE | Takvimsiz MAE | Δ MAE |
|---|---:|---:|---:|---:|---:|---:|
| KG | 0.8648 | 0.8632 | +0.0016 | 1.8096 | 1.8010 | +0.0085 |
| ADT | 0.6887 | 0.6862 | +0.0025 | 1.3625 | 1.3356 | +0.0269 |

Pozitif Δ PR-AUC iyileşme; negatif Δ MAE iyileşme anlamına gelir. Bu karşılaştırma aynı model ailesi ve aynı validation dönemiyle yapılmıştır.

## Gözlenen takvim etkisi

| Birim | Bağlam | Pozitif talep oranı / normal gün | Ortalama miktar / normal gün |
|---|---|---:|---:|
| KG | public_holiday | 1.033 | 1.476 |
| KG | pre_holiday_3d | 1.085 | 1.334 |
| KG | post_holiday_3d | 0.980 | 1.028 |
| KG | ramadan_nonholiday | 1.157 | 1.584 |
| KG | school_midterm_break | 1.129 | 1.393 |
| KG | school_semester_break | 1.144 | 1.179 |
| KG | school_summer_break | 0.972 | 1.076 |
| KG | weekend_nonholiday | 1.220 | 1.409 |
| ADT | public_holiday | 1.070 | 1.087 |
| ADT | pre_holiday_3d | 1.096 | 1.139 |
| ADT | post_holiday_3d | 1.015 | 0.963 |
| ADT | ramadan_nonholiday | 1.122 | 1.421 |
| ADT | school_midterm_break | 1.092 | 1.214 |
| ADT | school_semester_break | 1.172 | 1.240 |
| ADT | school_summer_break | 1.071 | 1.035 |
| ADT | weekend_nonholiday | 1.242 | 1.429 |

## Yorum sınırı

Bu oranlar betimseldir; kampanya, fiyat, stokta yokluk ve ürün portföyü değişimini tek başına kontrol etmez. Model katkısı için esas kanıt validation ablation ve zaman sıralı test segmentleridir.