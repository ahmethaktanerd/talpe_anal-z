# Proje Final Doğrulama Raporu

## Teslim kararı

**30/30 kontrol geçti.**
Proje; kullanıcı tarafından seçilen ürün ve desteklenen herhangi bir hedef gün için
günlük talep oluşumu ile ürünün kendi birimindeki miktarı üretir.

## Tahmin sözleşmesi

- Son veri / forecast origin: **2026-07-21**
- Desteklenen hedef tarih aralığı: **2026-07-22–2027-01-17**
- Karar tanesi: **ürün + kullanıcının seçtiği tek hedef gün**
- Desteklenen birimler: **KG ve ADT**; ürünün katalog birimi değiştirilmez.
- 2 Ocak 2027, origin'den **165 gün** ileridedir ve desteklenir.

## 2 Ocak 2027 servis kanıtı

| Ürün | Birim | Hedef tarih | Lead | Talep bekleniyor | Olasılık | Tahmini günlük miktar |
|---|---:|---:|---:|---:|---:|---:|
| $KOFTE HAMBURGER KG | KG | 2027-01-02 | 165 | Hayır | 0.1144 | 0.0 KG |
| $ERSAN JAMBON DANA 100 G (HAZIR DILIM) | ADT | 2027-01-02 | 165 | Hayır | 0.4697 | 0.0 ADT |

Bu satırlar örnek ürünler içindir. Streamlit'te seçilen ürün değiştiğinde aynı servis
o ürüne özel sonucu üretir.

## Final test performansı

| Birim | Occurrence PR-AUC | Occurrence F1 | Günlük miktar MAE | WAPE | Bias |
|---|---:|---:|---:|---:|---:|
| KG | 0.8307 | 0.7658 | 1.7100 | 0.8777 | 0.0966 |
| ADT | 0.6658 | 0.6156 | 1.0935 | 1.1323 | 0.2737 |

## Özel takvim

- Takvim sürümü: **TR_CALENDAR_2023_2027_V1**
- Özel takvim feature'ı: **18**
- Toplam model feature'ı: **47**
- Kapsam: resmî/dinî tatil, yarım gün, Ramazan, kandil, tatilden önce/sonra
  pencereleri ve MEB okul/ara/yarıyıl/yaz tatili.
- Validation ablation: `reports/csv/calendar_feature_ablation.csv`
- Takvim segment testleri: `reports/csv/calendar_segment_metrics.csv`
- KG: validation PR-AUC farkı **+0.0016**, MAE farkı **+0.0085**.
- ADT: validation PR-AUC farkı **+0.0025**, MAE farkı **+0.0269**.

## Bursa hava katmanı

- Hava feature sürümü: **BURSA_WEATHER_FEATURES_V1**
- Konum: **Bursa/Osmangazi city centre (40.19559, 29.06013)**
- Gerçekleşen hedef-gün havası yalnız EDA'dadır; estimator'a verilmez.
- ERA5 geçmişinden origin öncesi ±15 günlük klimatolojiyle 7 aday alan üretildi.
- Validation'da sağlam/birimler arası tutarlı kazanım görülmediği için deploy
  edilen hava feature sayısı **0**;
  karar: `excluded_after_validation_ablation_no_robust_gain`.
- Adaylar model-ready tabloda ve açıklayıcı arayüz bağlamında korunur.
- Ablation: `reports/csv/weather_feature_ablation.csv`
- KG: aday hava PR-AUC farkı **+0.0000**, MAE farkı **+0.0007**.
- ADT: aday hava PR-AUC farkı **-0.0009**, MAE farkı **+0.0046**.

## Operasyonel yorum

Tahmini miktar, seçilen gündeki **brüt mağaza talebi** tahminidir. Elde mağaza stoku,
yoldaki sevkiyat ve emniyet stoku olmadığı için tek başına kesin net depo transfer
emri değildir. Stok sıfır kabul edilirse tahmini miktar depodan çekilecek başlangıç
miktarı olarak kullanılabilir.

## Bilinen riskler

- Fiyat, kampanya, stokta yokluk ve mağaza özel kapanışı verileri yoktur.
- Hava bağlamı Bursa merkez klimatolojisidir; geleceğin kesin hedef-gün havası değildir.
- Takvim Türkiye genelidir; yerel okul ve mağaza çalışma değişikliklerini içermez.
- 90–180 günlük tahminler yakın vadeden daha belirsizdir.
- Gösterilen olasılık kalibre edilmiş güven skoru değildir.
- ADT miktar WAPE değeri yüksek olduğundan adet tahminleri operasyonel izlemede
  özellikle takip edilmelidir.

Makine-okunur kontrol sonucu: `reports/csv/final_validation_results.csv`
