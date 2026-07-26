from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scripts.data_utils import clean_sales_records, file_sha256, read_raw_sales, valid_sales
from scripts.project_config import (
    FIGURES_DIR,
    MAX_FORECAST_LEAD_DAYS,
    RAW_DATA,
    REPORT_CSV_DIR,
    REPORT_MD_DIR,
    ensure_project_dirs,
)
from scripts.turkey_calendar import CALENDAR_VERSION


def save_figure(fig: plt.Figure, name: str) -> None:
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / f"{name}.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def classify_segment(row: pd.Series, data_end: pd.Timestamp) -> str:
    if row["history_days"] < 30:
        return "new_short_history"
    if (data_end - row["last_sale"]).days > 90:
        return "possibly_inactive"
    if row["active_ratio"] >= 0.50:
        return "continuous"
    if row["active_ratio"] >= 0.10:
        return "intermittent"
    return "sparse"


def main() -> None:
    ensure_project_dirs()
    raw = read_raw_sales()
    cleaned = clean_sales_records(raw)
    valid = valid_sales(cleaned)

    data_start = valid["date"].min()
    data_end = valid["date"].max()
    calendar = pd.DataFrame({"date": pd.date_range(data_start, data_end, freq="D")})
    daily_rows = valid.groupby("date").size().rename("row_count")
    calendar = calendar.join(daily_rows, on="date")
    calendar["store_observed"] = calendar["row_count"].notna().astype(int)
    calendar["row_count"] = calendar["row_count"].fillna(0).astype(int)

    exact_duplicates = int(raw.duplicated().sum())
    product_date_duplicates = int(valid.duplicated(["date", "product_id"]).sum())
    unit_counts = valid["unit"].value_counts()

    profile = pd.DataFrame(
        [
            ("raw_sha256", file_sha256(RAW_DATA)),
            ("raw_rows", len(raw)),
            ("valid_rows", len(valid)),
            ("invalid_rows", int(cleaned["record_status"].ne("valid").sum())),
            ("unique_products", valid["product_id"].nunique()),
            ("date_start", data_start.date().isoformat()),
            ("date_end", data_end.date().isoformat()),
            ("calendar_days", len(calendar)),
            ("observed_store_days", int(calendar["store_observed"].sum())),
            ("missing_store_days", int(calendar["store_observed"].eq(0).sum())),
            ("exact_duplicates", exact_duplicates),
            ("product_date_duplicates", product_date_duplicates),
            ("kg_rows", int(unit_counts.get("KG", 0))),
            ("adt_rows", int(unit_counts.get("ADT", 0))),
        ],
        columns=["metric", "value"],
    )
    profile.to_csv(REPORT_CSV_DIR / "eda_data_profile.csv", index=False)
    calendar.to_csv(REPORT_CSV_DIR / "eda_temporal_coverage.csv", index=False)

    quality_frames = []
    invalid = cleaned.loc[cleaned["record_status"].ne("valid")].copy()
    if not invalid.empty:
        invalid["issue_type"] = invalid["record_status"]
        quality_frames.append(invalid)

    name_counts = (
        valid.groupby("product_id")["product_name"].nunique().rename("name_count")
    )
    unit_per_product = valid.groupby("product_id")["unit"].nunique().rename("unit_count")
    inconsistent_ids = name_counts[name_counts.gt(1)].index.union(
        unit_per_product[unit_per_product.gt(1)].index
    )
    if len(inconsistent_ids):
        inconsistent = valid.loc[valid["product_id"].isin(inconsistent_ids)].copy()
        inconsistent["issue_type"] = "product_name_or_unit_inconsistency"
        quality_frames.append(inconsistent)

    if quality_frames:
        quality = pd.concat(quality_frames, ignore_index=True, sort=False)
    else:
        quality = pd.DataFrame(
            columns=[
                "source_row",
                "date",
                "product_id",
                "product_name",
                "raw_quantity",
                "quantity",
                "unit",
                "parse_status",
                "record_status",
                "issue_type",
            ]
        )
    quality.to_csv(REPORT_CSV_DIR / "eda_data_quality_issues.csv", index=False)

    product_history = (
        valid.groupby(["product_id", "product_name", "unit"], as_index=False)
        .agg(
            first_sale=("date", "min"),
            last_sale=("date", "max"),
            active_sales_days=("date", "nunique"),
            total_quantity=("quantity", "sum"),
            mean_positive_quantity=("quantity", "mean"),
            median_positive_quantity=("quantity", "median"),
            max_positive_quantity=("quantity", "max"),
        )
    )
    product_history["history_days"] = (
        product_history["last_sale"] - product_history["first_sale"]
    ).dt.days + 1
    product_history["active_ratio"] = (
        product_history["active_sales_days"] / product_history["history_days"]
    )
    product_history["days_since_last_sale"] = (
        data_end - product_history["last_sale"]
    ).dt.days
    product_history["segment"] = product_history.apply(
        classify_segment, axis=1, data_end=data_end
    )
    product_history.to_csv(
        REPORT_CSV_DIR / "eda_product_history_summary.csv", index=False
    )

    daily_unit = (
        valid.groupby(["date", "unit"], as_index=False)
        .agg(total_quantity=("quantity", "sum"), product_count=("product_id", "nunique"))
        .sort_values(["unit", "date"])
    )
    daily_unit["rolling_7d_quantity"] = daily_unit.groupby("unit")[
        "total_quantity"
    ].transform(lambda series: series.rolling(7, min_periods=1).mean())
    daily_unit.to_csv(REPORT_CSV_DIR / "eda_daily_demand_summary.csv", index=False)

    segment_summary = (
        product_history.groupby(["unit", "segment"], as_index=False)
        .agg(
            product_count=("product_id", "nunique"),
            total_quantity=("total_quantity", "sum"),
            median_history_days=("history_days", "median"),
            median_active_ratio=("active_ratio", "median"),
        )
    )
    segment_summary.to_csv(REPORT_CSV_DIR / "eda_demand_segments.csv", index=False)

    fig, axes = plt.subplots(2, 1, figsize=(15, 9), sharex=True)
    colors = {"KG": "#22d3ee", "ADT": "#fbbf24"}
    for axis, unit in zip(axes, ("KG", "ADT")):
        subset = daily_unit.loc[daily_unit["unit"].eq(unit)]
        axis.plot(
            subset["date"],
            subset["rolling_7d_quantity"],
            color=colors[unit],
            linewidth=1.8,
        )
        axis.set_title(f"{unit} — 7 Günlük Hareketli Ortalama Talep")
        axis.set_ylabel(f"Miktar ({unit})")
        axis.grid(alpha=0.2)
    axes[-1].set_xlabel("Tarih")
    save_figure(fig, "eda_daily_demand_by_unit")

    fig, axis = plt.subplots(figsize=(11, 6))
    axis.hist(product_history["history_days"], bins=35, color="#6366f1", alpha=0.85)
    axis.set_title("Ürün Geçmiş Uzunluğu Dağılımı")
    axis.set_xlabel("İlk ve Son Satış Arasındaki Gün")
    axis.set_ylabel("Ürün Sayısı")
    axis.grid(alpha=0.2)
    save_figure(fig, "eda_product_history_distribution")

    fig, axis = plt.subplots(figsize=(11, 6))
    pivot = segment_summary.pivot(
        index="segment", columns="unit", values="product_count"
    ).fillna(0)
    pivot.plot(kind="bar", ax=axis, color=[colors.get(c, "#6366f1") for c in pivot])
    axis.set_title("Birim ve Talep Segmentine Göre Ürün Sayısı")
    axis.set_xlabel("Talep Segmenti")
    axis.set_ylabel("Ürün Sayısı")
    axis.tick_params(axis="x", rotation=25)
    save_figure(fig, "eda_product_segments")

    top_products = (
        product_history.sort_values(["unit", "total_quantity"], ascending=[True, False])
        .groupby("unit")
        .head(10)
    )
    fig, axes = plt.subplots(1, 2, figsize=(17, 7))
    for axis, unit in zip(axes, ("KG", "ADT")):
        subset = top_products.loc[top_products["unit"].eq(unit)].sort_values(
            "total_quantity"
        )
        axis.barh(subset["product_name"], subset["total_quantity"], color=colors[unit])
        axis.set_title(f"Toplam Talebi En Yüksek 10 Ürün — {unit}")
        axis.set_xlabel(f"Toplam {unit}")
    save_figure(fig, "eda_top_products_by_unit")

    recommendations = pd.DataFrame(
        [
            {
                "Sorun": "Türkçe miktar biçimi",
                "Kanıt": "Nokta binlik, virgül ondalık ayıracı; karma biçimli kayıtlar mevcut.",
                "Öneri": "Miktarı Türkçe sayı sözleşmesiyle ayrıştır; ham değeri izlenebilir tut.",
                "Öncelik": "Yüksek",
                "Sorumlu": "DataPrep Expert",
            },
            {
                "Sorun": "Satışsız ürün-gün belirsizliği",
                "Kanıt": f"{int(calendar['store_observed'].eq(0).sum())} genel takvim gününde hiç kayıt yok.",
                "Öneri": "Gözlenen mağaza günlerinde aktif ürün penceresindeki yokluğu 0; genel eksik günü bilinmeyen kabul et.",
                "Öncelik": "Yüksek",
                "Sorumlu": "DataPrep Expert",
            },
            {
                "Sorun": "Uzak gelecek tarih tahmini",
                "Kanıt": f"İstenen tarih son gözlemden ileride olabilir; model en fazla {MAX_FORECAST_LEAD_DAYS} gün için doğrulanacak.",
                "Öneri": "Hedef tarihi ve lead_days değişkenini kullanan doğrudan çok-ufuklu günlük model kur.",
                "Öncelik": "Yüksek",
                "Sorumlu": "DataPrep + Model Expert",
            },
            {
                "Sorun": "Birim ayrımı",
                "Kanıt": f"KG={int(unit_counts.get('KG', 0))}, ADT={int(unit_counts.get('ADT', 0))} kayıt.",
                "Öneri": "KG ve ADT hedeflerini, modellerini ve metriklerini ayrı tut.",
                "Öncelik": "Yüksek",
                "Sorumlu": "DataPrep + Model Expert",
            },
            {
                "Sorun": "Kısa geçmiş / aktif olmayan ürün",
                "Kanıt": product_history["segment"].value_counts().to_dict(),
                "Öneri": "Cold-start ve muhtemel aktif olmayan ürünleri ayrı işaretle; uygulamada uyarı üret.",
                "Öncelik": "Orta",
                "Sorumlu": "DataPrep + Deployment Expert",
            },
            {
                "Sorun": "Özel tarihlerde talep rejimi",
                "Kanıt": (
                    "Bayram, Ramazan, hafta sonu ve okul tatili satış davranışını "
                    "değiştirebilir; ham satışta takvim etiketi yok."
                ),
                "Öneri": (
                    f"{CALENDAR_VERSION} resmî takvimini target_date üzerinden "
                    "ekle; normal gün oranları ve ablation ile doğrula."
                ),
                "Öncelik": "Yüksek",
                "Sorumlu": "DataPrep + Model Expert",
            },
        ]
    )
    recommendations.to_csv(
        REPORT_CSV_DIR / "data_prep_recommendations.csv", index=False
    )

    report = f"""# Perakende Talep Tahmini — EDA Final Raporu

## Yönetici özeti

Ham veri {len(raw):,} satış kaydı ve {valid['product_id'].nunique():,} ürün içerir.
Gözlem dönemi {data_start.date()}–{data_end.date()} aralığıdır. İş hedefi, kullanıcının
seçtiği ürün ve gelecek hedef tarih için o gün talep oluşup oluşmayacağını ve oluşursa
ürünün kendi biriminde (`KG`/`ADT`) günlük miktarı tahmin etmektir.

## Veri sözleşmesi

- Ham dosya: `data/data.csv`
- SHA-256: `{file_sha256(RAW_DATA)}`
- Ayraç/kodlama: noktalı virgül / UTF-8 BOM
- Geçerli kayıt: {len(valid):,}
- Geçersiz kayıt: {cleaned['record_status'].ne('valid').sum():,}
- Ürün-tarih tekrarı: {product_date_duplicates:,}
- Genel eksik takvim günü: {int(calendar['store_observed'].eq(0).sum()):,}
- KG kayıt: {int(unit_counts.get('KG', 0)):,}
- ADT kayıt: {int(unit_counts.get('ADT', 0)):,}

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
İlk sürüm 1–{MAX_FORECAST_LEAD_DAYS} gün aralığında doğrudan günlük tahmin üretir.

## Özel takvim hipotezi

Hafta sonu, resmî/dinî tatil, Ramazan, bayram öncesi/sonrası ve MEB okul tatilleri
talep rejimini değiştirebilir. Ham satış dosyasında bu etiketler yoktur. DataPrep,
`{CALENDAR_VERSION}` sürümlü resmî takvimi hedef tarihe ekler; betimsel oranlar
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
"""
    (REPORT_MD_DIR / "EDA_FINAL_REPORT.md").write_text(report, encoding="utf-8")

    print(profile.to_string(index=False))
    print("\nEDA tamamlandı.")


if __name__ == "__main__":
    main()
