# DataPrep Handoff — Seçili Tarih Günlük Talep Tahmini

## Veri durumu

- Ham SHA-256: `b674355d6d3783792d9c6a4cdfb3a0098db3702b8e0e725aca0ff35138e2cb75`
- Temiz kayıt: 183,184
- Günlük panel satırı: 492,927
- Model örneği: 432,063
- Aktif inference ürünü: 705
- Son bilinen tarih / forecast origin: 2026-07-21

## Tahmin sözleşmesi

Kullanıcı bir ürün ve gelecek `target_date` seçer. Model `lead_days`, hedef günün
takvim özellikleri ve son bilinen tarihteki ürün snapshot feature'larıyla doğrudan
o günün talep oluşumunu ve günlük miktarını tahmin eder. Doğrulanmış aralık
1–180 gündür.

## Sıfır ve aktiflik politikası

Gözlenen mağaza gününde ürünün aktif panel penceresinde satış satırı yoksa günlük
talep `0` kabul edilir. Mağaza genelinde hiç kayıt olmayan gün bilinmeyen kalır.
Panel ilk satıştan başlar ve son satıştan sonra en fazla 90 gün uzar.

## Feature sözleşmesi

Snapshot feature'ları yalnız `forecast_origin` ve öncesini kullanır. Hedef güne ait
yalnız önceden bilinen takvim alanları ve `lead_days` eklenir. Takvim sürümü
`TR_CALENDAR_2023_2027_V1`; resmî/dinî tatil, Ramazan, tatilden 1/3 gün önce-sonra,
MEB okul dönemi, ara/yarıyıl/yaz tatili ve olağanüstü okul kapanışı feature'larını
içerir. Kaynaklar `data/reference/calendar_sources.json`, model alanları
`models/feature_specification.json` dosyasındadır.

Hava katmanı `BURSA_WEATHER_FEATURES_V1` sürümündedir. Gerçekleşen hedef-gün havası
yalnız EDA'da kullanılır; aday alanlar her satır için yalnız `forecast_origin`
öncesindeki ERA5 geçmişinden hedef mevsime ait ±15 günlük iklim normalini hesaplar. Bursa/
Osmangazi merkez koordinatı kullanılmıştır. Kaynak ve checksum bilgisi
`data/reference/weather_sources.json` içindedir.

## Split

Split `target_date` temelindedir:

- Train: 2023-01-30 – 2026-01-20
- Validation: 2026-01-28 – 2026-04-21
- Test: 2026-04-29 – 2026-07-21

Aralarda 7 günlük tampon bulunur. Test model, threshold veya feature seçimi için
kullanılmayacaktır.

## Leakage

`source_max_date <= forecast_origin < target_date` kontrolü tüm örneklerde geçti.
KG ve ADT hedefleri ayrı modelleme için korunmuştur.

## Kalan riskler

- Kampanya, fiyat, stokta yokluk ve mağaza özel çalışma saatleri veri setinde yoktur.
- Aday hava alanları kesin hedef-gün gerçekleşmesi değil time-safe Bursa iklim normalidir;
  final kullanım kararı validation ablation'a bağlıdır.
- Takvim Türkiye geneli resmî kaynakları kullanır; yerel okul/mağaza kapanışları yoktur.
- 180 güne yaklaşan tahminler yakın tarihlerden daha belirsizdir.
- Son 90 günde satılmamış ürünler inference kataloğunda pasif kabul edilir.
