"""Veri hikâyesi notebook'una özel takvim analiz hücrelerini ekler."""

import nbformat

from scripts.project_config import PROJECT_ROOT


NOTEBOOK_PATH = (
    PROJECT_ROOT / "notebooks" / "retail_demand_forecasting_story.ipynb"
)
CALENDAR_TAG = "calendar-enhancement-v1"


def main() -> None:
    notebook = nbformat.read(NOTEBOOK_PATH, as_version=4)
    notebook.cells = [
        cell
        for cell in notebook.cells
        if CALENDAR_TAG not in cell.get("metadata", {}).get("tags", [])
    ]

    eda_index = next(
        index
        for index, cell in enumerate(notebook.cells)
        if "daily_unit = eda.dropna" in cell.get("source", "")
    )
    calendar_markdown = nbformat.v4.new_markdown_cell(
        """### Özel tarihlerin talebe etkisi

Resmî/dinî tatiller, Ramazan, bayramdan önceki ve sonraki üç gün, hafta sonu ile
MEB okul/ara/yarıyıl/yaz tatilleri aynı sürümlü Türkiye takviminden gelir. Aşağıdaki
oranlarda `1`, normal gün seviyesidir; `1` üzeri artış, altı azalış gösterir.
Bu bölüm betimseldir; model katkısı ayrıca validation ablation ile ölçülür.""",
        metadata={"tags": [CALENDAR_TAG]},
    )
    calendar_code = nbformat.v4.new_code_cell(
        """calendar_impact_path = REPORTS_DIR / 'csv' / 'calendar_demand_impact.csv'
calendar_impact = pd.read_csv(calendar_impact_path)
display(calendar_impact)

plot_frame = calendar_impact[
    calendar_impact['calendar_context'].ne('ordinary_day')
].melt(
    id_vars=['unit', 'calendar_context'],
    value_vars=[
        'positive_rate_ratio_vs_ordinary',
        'mean_demand_ratio_vs_ordinary',
    ],
    var_name='metric',
    value_name='ratio',
)
fig = px.bar(
    plot_frame,
    x='calendar_context',
    y='ratio',
    color='metric',
    facet_col='unit',
    barmode='group',
    title='Normal Güne Göre Özel Takvim Talep Etkisi',
)
fig.add_hline(y=1.0, line_dash='dash', line_color=PALETTE['rose'])
fig.update_xaxes(tickangle=35)
fig.update_layout(
    template='plotly_dark',
    paper_bgcolor='#111827',
    plot_bgcolor='#111827',
)
fig.show()""",
        metadata={"tags": [CALENDAR_TAG]},
    )
    notebook.cells[eda_index + 1 : eda_index + 1] = [
        calendar_markdown,
        calendar_code,
    ]

    model_index = next(
        index
        for index, cell in enumerate(notebook.cells)
        if "comparison_path =" in cell.get("source", "")
    )
    ablation_markdown = nbformat.v4.new_markdown_cell(
        """### Özel takvim feature ablation

Aynı seçilmiş model ailesi, aynı train/validation döneminde iki kez ölçülür:
`full_calendar` 18 özel takvim alanını içerir; `special_calendar_removed` bunları
çıkarır ama standart gün/ay/hafta sonu alanlarını korur. Böylece takvimin gerçekten
genelleme performansına katkısı ayrı olarak görülür.""",
        metadata={"tags": [CALENDAR_TAG]},
    )
    ablation_code = nbformat.v4.new_code_cell(
        """ablation = pd.read_csv(REPORTS_DIR / 'csv' / 'calendar_feature_ablation.csv')
display(ablation)

fig = px.bar(
    ablation,
    x='feature_set',
    y='occurrence_pr_auc',
    color='unit',
    barmode='group',
    text_auto='.4f',
    title='Özel Takvimli / Takvimsiz Validation PR-AUC',
)
fig.update_layout(
    template='plotly_dark',
    paper_bgcolor='#111827',
    plot_bgcolor='#111827',
)
fig.show()""",
        metadata={"tags": [CALENDAR_TAG]},
    )
    notebook.cells[model_index + 1 : model_index + 1] = [
        ablation_markdown,
        ablation_code,
    ]

    for cell in notebook.cells:
        source = cell.get("source", "")
        if "if str(PROJECT_ROOT) not in sys.path:" in source:
            cell["source"] = source.replace(
                "if str(PROJECT_ROOT) not in sys.path:\n"
                "    sys.path.insert(0, str(PROJECT_ROOT))",
                "project_root_text = str(PROJECT_ROOT)\n"
                "sys.path[:] = [entry for entry in sys.path "
                "if entry != project_root_text]\n"
                "sys.path.insert(0, project_root_text)",
            )
            source = cell["source"]
        if "Bu bölümde aşağıdakiler çalıştırılmış kanıtla görünmelidir:" in source:
            cell["source"] = source.replace(
                "ve target-date tabanlı train-validation-test zaman çizelgesi.",
                "Türkiye resmî/dinî tatil ve MEB okul takvimi feature'ları ile "
                "target-date tabanlı train-validation-test zaman çizelgesi.",
            )
        if "Model Expert bu alana;" in source:
            cell["source"] = source.replace(
                "ve `KG`/`ADT` ile ürün segmenti kırılımını ekler.",
                "özel takvim ablation sonucunu ve `KG`/`ADT` ile ürün/takvim "
                "segmenti kırılımını ekler.",
            )
        if "keys = ['model_version'" in source:
            cell["source"] = source.replace(
                "'training_end_date', 'occurrence_threshold'",
                "'training_end_date', 'calendar_version', 'calendar_coverage', "
                "'occurrence_threshold'",
            )
        if "example_results = []" in source:
            cell["source"] = source.replace(
                "display(pd.DataFrame(example_results)[['product_name','unit',"
                "'target_date','lead_days','demand_expected','demand_probability',"
                "'display_quantity','status']])",
                "example_frame = pd.DataFrame(example_results)\n"
                "example_frame['calendar_context_label'] = example_frame["
                "'calendar_context'].map(lambda value: ' · '.join(filter(None, ["
                "value.get('public_holiday_name'), value.get('religious_special_name'), "
                "value.get('school_status_label'), "
                "'Ramazan' if value.get('is_ramadan') else None])))\n"
                "display(example_frame[['product_name','unit','target_date','lead_days',"
                "'calendar_context_label','demand_expected','demand_probability',"
                "'display_quantity','status']])",
            )

    nbformat.write(notebook, NOTEBOOK_PATH)
    print(f"Notebook özel takvim hücreleriyle güncellendi: {NOTEBOOK_PATH}")


if __name__ == "__main__":
    main()
