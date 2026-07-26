"""Veri hikâyesi notebook'una Bursa hava analizi ve ablation hücreleri ekler."""

import nbformat

from scripts.project_config import PROJECT_ROOT


NOTEBOOK_PATH = (
    PROJECT_ROOT / "notebooks" / "retail_demand_forecasting_story.ipynb"
)
WEATHER_TAG = "weather-enhancement-v1"


def main() -> None:
    notebook = nbformat.read(NOTEBOOK_PATH, as_version=4)
    notebook.cells = [
        cell
        for cell in notebook.cells
        if WEATHER_TAG not in cell.get("metadata", {}).get("tags", [])
    ]

    calendar_code_index = next(
        index
        for index, cell in enumerate(notebook.cells)
        if "calendar_impact_path =" in cell.get("source", "")
    )
    weather_eda_markdown = nbformat.v4.new_markdown_cell(
        """### Bursa hava durumu ve talep ilişkisi

Gerçekleşen ERA5 hava değerleri bu bölümde **yalnız betimsel EDA** amacıyla
satışla eşlenir. Oranın `1` üzerinde olması ilgili hava koşulundaki toplam günlük
talebin diğer günlerden yüksek olduğunu gösterir; nedensellik göstermez. Mevsim,
ürün portföyü, fiyat, kampanya ve stok durumu bu ilişkiyi karıştırabilir.

Geleceğin gerçekleşmiş havası tahmin anında bilinmediği için aşağıdaki gerçekleşen
değerler model feature'ı değildir.""",
        metadata={"tags": [WEATHER_TAG]},
    )
    weather_eda_code = nbformat.v4.new_code_cell(
        """weather_impact = pd.read_csv(REPORTS_DIR / 'csv' / 'weather_demand_impact.csv')
display(weather_impact)

fig = px.bar(
    weather_impact,
    x='weather_condition',
    y='mean_total_demand_ratio',
    color='unit',
    barmode='group',
    text_auto='.3f',
    title='Bursa Gözlenen Hava Koşulu / Diğer Günler Talep Oranı',
)
fig.add_hline(y=1.0, line_dash='dash', line_color=PALETTE['rose'])
fig.update_xaxes(tickangle=30)
fig.update_layout(
    template='plotly_dark',
    paper_bgcolor='#111827',
    plot_bgcolor='#111827',
)
fig.show()""",
        metadata={"tags": [WEATHER_TAG]},
    )
    notebook.cells[calendar_code_index + 1 : calendar_code_index + 1] = [
        weather_eda_markdown,
        weather_eda_code,
    ]

    calendar_ablation_index = next(
        index
        for index, cell in enumerate(notebook.cells)
        if "calendar_feature_ablation.csv" in cell.get("source", "")
        and cell.get("cell_type") == "code"
    )
    weather_ablation_markdown = nbformat.v4.new_markdown_cell(
        """### Hava feature ablation ve deployment kararı

Yedi time-safe Bursa iklim normali adayı aynı model ailesi ve aynı validation
döneminde eklenip çıkarılmıştır. KG'deki çok küçük fark diğer metriklerde ve ADT'de
tekrarlanmadığı için hava alanları final estimator feature listesinden çıkarılmıştır.
Bu karar hedef-gün hava sızıntısını ve zayıf sinyale overfitting'i önler; aday
alanlar model-ready veride ve EDA'da izlenebilir kalır.""",
        metadata={"tags": [WEATHER_TAG]},
    )
    weather_ablation_code = nbformat.v4.new_code_cell(
        """weather_ablation = pd.read_csv(
    REPORTS_DIR / 'csv' / 'weather_feature_ablation.csv'
)
display(weather_ablation)

fig = px.bar(
    weather_ablation,
    x='feature_set',
    y='occurrence_pr_auc',
    color='unit',
    barmode='group',
    text_auto='.4f',
    title='Aday Hava Alanları Validation Ablation',
)
fig.update_layout(
    template='plotly_dark',
    paper_bgcolor='#111827',
    plot_bgcolor='#111827',
)
fig.show()""",
        metadata={"tags": [WEATHER_TAG]},
    )
    notebook.cells[
        calendar_ablation_index + 1 : calendar_ablation_index + 1
    ] = [weather_ablation_markdown, weather_ablation_code]

    for cell in notebook.cells:
        source = cell.get("source", "")
        if "keys = ['model_version'" in source:
            cell["source"] = source.replace(
                "'calendar_version', 'calendar_coverage', 'occurrence_threshold'",
                "'calendar_version', 'calendar_coverage', "
                "'weather_feature_version', 'weather_location', "
                "'weather_deployment_decision', 'occurrence_threshold'",
            )
        if "example_frame['calendar_context_label']" in source:
            cell["source"] = source.replace(
                "display(example_frame[['product_name','unit','target_date','lead_days',"
                "'calendar_context_label','demand_expected','demand_probability',"
                "'display_quantity','status']])",
                "example_frame['weather_source'] = example_frame['weather_context'].map("
                "lambda value: value.get('source'))\n"
                "example_frame['weather_used_by_model'] = example_frame["
                "'weather_context'].map(lambda value: value.get('used_by_model'))\n"
                "display(example_frame[['product_name','unit','target_date','lead_days',"
                "'calendar_context_label','weather_source','weather_used_by_model',"
                "'demand_expected',"
                "'demand_probability','display_quantity','status']])",
            )
            source = cell["source"]
        if (
            "example_frame['weather_source']" in source
            and "weather_used_by_model" not in source
        ):
            cell["source"] = source.replace(
                "example_frame['weather_source'] = example_frame['weather_context'].map("
                "lambda value: value.get('source'))",
                "example_frame['weather_source'] = example_frame['weather_context'].map("
                "lambda value: value.get('source'))\n"
                "example_frame['weather_used_by_model'] = example_frame["
                "'weather_context'].map(lambda value: value.get('used_by_model'))",
            ).replace(
                "'calendar_context_label','weather_source','demand_expected',",
                "'calendar_context_label','weather_source','weather_used_by_model',"
                "'demand_expected',",
            )

    nbformat.write(notebook, NOTEBOOK_PATH)
    print(f"Notebook Bursa hava hücreleriyle güncellendi: {NOTEBOOK_PATH}")


if __name__ == "__main__":
    main()
