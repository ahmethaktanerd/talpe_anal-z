from pathlib import Path

from streamlit.testing.v1 import AppTest

from app.services.forecast_service import DemandForecastService


ROOT = Path(__file__).resolve().parents[1]
APP_PATH = ROOT / "app" / "app.py"


def test_corporate_sidebar_navigation_and_pages_render():
    app = AppTest.from_file(str(APP_PATH), default_timeout=60).run()
    assert not app.exception
    assert len(app.tabs) == 0
    assert len(app.radio) == 1
    assert len(app.radio[0].options) == 5
    assert len(app.selectbox) == 1
    assert app.selectbox[0].label == "Ürün"

    for page in (
        "▦  Toplu Planlama",
        "◫  Model Merkezi",
        "⌁  Tahmin İzleme",
        "?  Kullanım Rehberi",
    ):
        app.radio[0].set_value(page)
        app.run()
        assert not app.exception


def test_forecast_result_has_interactive_charts_and_no_legacy_white_panel():
    app = AppTest.from_file(str(APP_PATH), default_timeout=60).run()
    service = DemandForecastService(ROOT)
    selected_label = app.selectbox[0].value
    product = service.product_options().loc[
        lambda frame: frame["label"].eq(selected_label)
    ].iloc[0]
    result = service.forecast(str(product["product_id"]), app.date_input[0].value)

    app.session_state["last_forecast"] = result
    app.session_state["history_days"] = 180
    app.run()

    assert not app.exception
    assert len(app.get("plotly_chart")) == 2
    assert len(app.selectbox) == 2
    assert app.selectbox[1].label == "Grafikte gösterilecek geçmiş"
    assert list(app.selectbox[1].options) == [
        "Son 60 gün",
        "Son 90 gün",
        "Son 180 gün",
        "Son 1 yıl (365 gün)",
    ]
    assert any("Model karar özeti" in item.value for item in app.markdown)
    assert not any('class="result"' in item.value for item in app.markdown)

    probability_before = app.session_state["last_forecast"]["demand_probability"]
    app.selectbox[1].set_value(60)
    app.run()
    assert not app.exception
    assert app.session_state["last_forecast"]["demand_probability"] == probability_before
    assert len(app.get("plotly_chart")) == 2


def test_model_limitations_are_presented_in_plain_turkish():
    app = AppTest.from_file(str(APP_PATH), default_timeout=60).run()
    app.radio[0].set_value("◫  Model Merkezi")
    app.run()

    limitation_markup = "\n".join(
        item.value
        for item in app.markdown
        if 'class="limitation-item"' in item.value
    )
    assert not app.exception
    assert limitation_markup.count('class="limitation-item"') == 7
    assert "Talep olasılığı güven yüzdesi değildir" in limitation_markup
    assert "Tahmin doğrudan sevkiyat emri değildir" in limitation_markup
    assert "No price, promotion" not in limitation_markup


def test_help_page_explains_forecast_and_model_center_for_new_users():
    app = AppTest.from_file(str(APP_PATH), default_timeout=60).run()
    app.radio[0].set_value("?  Kullanım Rehberi")
    app.run()

    markup = "\n".join(item.value for item in app.markdown)
    assert not app.exception
    assert "Önce hangi ekrana gitmeliyim?" in markup
    assert "Talep Planlama ekranı: adım adım" in markup
    assert "Tahmin sonucunu nasıl okuyacağım?" in markup
    assert "Model Merkezi: sade anlatım" in markup
    assert "Basit bir net transfer örneği" in markup
    assert "Sık yapılan hatalar" in markup
