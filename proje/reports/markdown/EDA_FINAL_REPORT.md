# Perakende Talep Tahmini — EDA Final Raporu

## Yönetici özeti

Ham veri 183,184 satış kaydı ve 713 ürün içerir.
Gözlem dönemi 2023-01-01–2026-07-21 aralığıdır. İş hedefi, kullanıcının
seçtiği ürün ve gelecek hedef tarih için o gün talep oluşup oluşmayacağını ve oluşursa
ürünün kendi biriminde (`KG`/`ADT`) günlük miktarı tahmin etmektir.

## Veri sözleşmesi

- Ham dosya: `data/data.csv`
- SHA-256: `b674355d6d3783792d9c6a4cdfb3a0098db3702b8e0e725aca0ff35138e2cb75`
- Ayraç/kodlama: noktalı virgül / UTF-8 BOM
- Geçerli kayıt: 183,184
- Geçersiz kayıt: 0
- Ürün-tarih tekrarı: 0
- Genel eksik takvim günü: 1
- KG kayıt: 93,279
- ADT kayıt: 89,905

## Kritik veri kalitesi bulguları

Türkçe sayı biçimi uygulanmalıdır: virgül ondalık, nokta binlik ayıracıdır. Ham veri
değiştirilmemeli; orijinal miktar ve kaynak satır numarası temiz kayıtta korunmalıdır.
Olağandışı yüksek satışlar otomatik silinmemiştir.

## Zaman ve ürün yapısı

Ürünler sürekli, aralıklı, seyrek, kısa geçmişli ve muhtemel aktif olmayan segmentlere
ayrılmıştır. Bu segmentler model feature'ı değil, performans değerlendirme ve uyarı
bağlamıdır. Günlük talep ve hata metrikleri KG/ADT için ayrı tutulacaktır.

## Tahmin hedefi

Tahmin agregat bir `H` günlük toplam değildir. Kullanıcı ürün ve `target_date` seçer.
`forecast_origin`, verideki son bilinen gündür; `lead_days = target_date - forecast_origin`.
İlk sürüm 1–180 gün aralığında doğrudan günlük tahmin üretir.

## Özel takvim hipotezi

Hafta sonu, resmî/dinî tatil, Ramazan, bayram öncesi/sonrası ve MEB okul tatilleri
talep rejimini değiştirebilir. Ham satış dosyasında bu etiketler yoktur. DataPrep,
`TR_CALENDAR_2023_2027_V1` sürümlü resmî takvimi hedef tarihe ekler; betimsel oranlar
`calendar_demand_impact.csv`, model katkısı `calendar_feature_ablation.csv` ile
ölçülür. Kampanya/fiyat/stokta yokluk bulunmadığından takvim oranları nedensellik
olarak yorumlanmaz.

## DataPrep handoff

1. Gözlenen mağaza gününde ürünün aktif penceresindeki eksik kayıt `0` adayıdır.
2. Mağaza genelinde kayıtsız gün `missing_or_unobserved` kalır.
3. Ürün ilk satışından önce sıfır üretilmez.
4. Ürün son satışından sonra en fazla 90 günlük kuyruk aktiflik sinyali için tutulur.
5. Hedef günlük `demand_occurs` ve `target_demand` olarak hedef tarihte oluşturulur.
6. Snapshot feature'ları yalnız forecast origin ve öncesini kullanır.
7. Özel takvim feature'ları yalnız hedef tarihten ve sürümlü MEB/Diyanet
   kaynaklarından üretilir.

## Model readiness

**Koşullu hazır.** Türkçe parsing, günlük panel, doğrudan çok-ufuklu örnek üretimi ve
kronolojik target-date split tamamlandıktan sonra modelleme başlayabilir.
