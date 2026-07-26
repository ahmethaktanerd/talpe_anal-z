# Deployment Kılavuzu — Mağaza Günlük Talep Tahmini

## Amaç

Streamlit uygulaması kullanıcıdan ürün ve hedef tarih alır. Model, seçilen tek gün
için talep kararı ve ürünün kendi biriminde günlük miktar üretir.
Model hedef günün resmî/dinî tatil, Ramazan, tatil önce/sonra ve MEB okul dönemi
feature'larını tahmin anında otomatik üretir; kullanıcıdan bu alanları girmesi istenmez.
Servis ayrıca Bursa/Osmangazi merkez için tahmin-anı güvenli mevsimsel hava
normalini açıklar. Yedi hava alanı validation ablation'da sağlam kazanım
göstermediği için final estimator'ın 47 feature'ı içinde değildir.

## Çalıştırma

Proje kökünde:

```bash
python3 -m streamlit run app/app.py
```

## Tarih seçimi

`forecast_origin` model metadata'daki son bilinen veri günüdür. Tarih seçici
`forecast_origin + 1` ile `forecast_origin + max_forecast_lead_days` aralığını
kullanır. Mevcut bundle için aralık 2026-07-22–2027-01-17'dir.

## Ürün seçimi

Liste ürün adı, `KG`/`ADT` birimi ve ürün ID'sini birlikte gösterir. Aktif ürünler
önce sıralanır. 28 günlük snapshot üretilemeyen üründe tahmin yerine
`insufficient_history` durumu döner. Muhtemel aktif olmayan ürün tahminlerinde
uyarı görünür.

## Sonuç alanları

- Talep bekleniyor / beklenmiyor
- Modelin talep olasılığı
- Tahmini günlük miktar ve birim
- Hedef tarih ve ileri gün sayısı
- Takvim bağlamı: tatil adı, Ramazan/kandil ve okul dönemi/tatili
- Hava bağlamı: Bursa klimatoloji kaynağı, sıcaklık, yağış, bulut ve rüzgâr normali
- Son veri tarihi
- Model/feature/takvim sürümü
- Durum ve uyarı kodları

Olasılık kalibre edilmiş güven skoru değildir. Günlük talep tahmini stok, açık
sipariş ve tedarik süresi olmadan sipariş emri değildir.

## Toplu tahmin

CSV dosyası `product_id,target_date` kolonlarını içerir. Hatalı satırlar sonuçtan
silinmez; `error` statüsü ve mesajla döner.

## İzleme

Başarılı tekil tahminler `logs/forecast_log.csv` içine sürüm, tarih, olasılık,
miktar ve uyarılarla kaydedilir. Hedef tarih geçtikten sonra gerçekleşen satış
ürün-tarih-birim anahtarında eşleştirilmelidir.

## Güvenlik

Bundle yolları proje model klasörünün dışına çıkamaz. Dosya checksum'ları model
yüklenmeden doğrulanır. Kullanıcı dosyalarından model deserialization yapılmaz.
Hava feature kodu, kaynak manifesti, ERA5 referansı ve boş/opsiyonel previous-runs
dosyası ayrıca checksum ile doğrulanır. Canlı hava API'si estimator'a çağrı
sırasında bağlanmaz; böylece ağ kesintisi train/serving feature dağılımını değiştirmez.

## Test

```bash
python3 -m pytest -q
```

Bundle, tarih sınırı, birim, feature zamanı, KG/ADT tahmini, bilinmeyen ürün ve
kısa geçmiş senaryoları otomatik test edilir.
Takvim testleri ayrıca çakışan resmî/dinî tatili, bayram önce/sonra pencerelerini,
MEB ara tatilini ve 2 Ocak 2027 bağlamını doğrular.
Hava testleri origin sonrasındaki gözlemleri değiştirmenin üretilen klimatolojiyi
değiştirmediğini ve `weather_source_max_date < forecast_origin` koşulunu doğrular.
