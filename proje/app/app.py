import html
import json
import sys
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[1]
project_root_text = str(PROJECT_ROOT)
sys.path[:] = [entry for entry in sys.path if entry != project_root_text]
sys.path.insert(0, project_root_text)

from app.services.forecast_service import DemandForecastService
from app.services.monitoring_service import log_forecast, read_forecast_log


ORANGE = "#F58220"
ORANGE_LIGHT = "#FFB36B"
INK = "#0A0A0B"
PANEL = "#17181B"
PANEL_SOFT = "#202226"
TEXT = "#F7F7F5"
MUTED = "#A6A8AD"
GREEN = "#36C98F"
RED = "#F26B5E"
GRID = "#34363C"

KNOWN_LIMITATION_TRANSLATIONS = {
    "No price, promotion, stockout, or store-specific closure features.": (
        "Ticari ve operasyonel değişkenler sınırlı",
        (
            "Modelde fiyat, kampanya, ürünün rafta bulunmaması ve mağazaya özel "
            "kapanış bilgileri yer almıyor. Bu nedenle bu koşulların yaşandığı "
            "günlerde tahmin ayrıca operasyon ekibi tarafından değerlendirilmelidir."
        ),
    ),
    (
        "Weather is Bursa city-centre time-safe climatology, not the realised "
        "weather of the future target day."
    ): (
        "Hava bilgisi gerçekleşmiş gelecek havası değil",
        (
            "Gösterilen hava bağlamı Bursa şehir merkezi iklim normalidir. Hedef "
            "günün ileride gerçekten nasıl olacağını bildiren bir hava tahmini "
            "değildir ve aktif model kararına girdi olarak kullanılmaz."
        ),
    ),
    (
        "Exact store coordinates and a complete archived/live forecast contract "
        "are not yet available."
    ): (
        "Mağaza konumu ve canlı hava arşivi eksik",
        (
            "Mağazanın kesin koordinatları ile geçmiş tarihlerde gerçekten "
            "erişilebilir olan hava tahminlerini içeren eksiksiz bir arşiv henüz "
            "bulunmuyor."
        ),
    ),
    (
        "School calendar is national; local closures and store opening hours are "
        "absent."
    ): (
        "Yerel takvim ayrıntıları eksik",
        (
            "Okul takvimi ülke genelindeki tarihlere dayanır. Bursa'ya özel yerel "
            "kapanışlar ve mağazanın değişen çalışma saatleri modelde bulunmuyor."
        ),
    ),
    "Forecasts near 180 days are less certain than near-term forecasts.": (
        "Uzak tarihlerin belirsizliği daha yüksek",
        (
            "180 güne yaklaşan tahminler, yakın tarihler için üretilen tahminlere "
            "göre daha az kesindir. Uzak tarih kararları hedef gün yaklaştığında "
            "yeniden çalıştırılmalıdır."
        ),
    ),
    "Probability is not calibrated and must not be labelled confidence.": (
        "Talep olasılığı güven yüzdesi değildir",
        (
            "Ekrandaki olasılık modelin talep puanıdır; kalibre edilmiş bir güven "
            "oranı değildir. Karar, ürün birimine ait eşikle birlikte okunmalıdır."
        ),
    ),
    "Demand forecast is not a replenishment order without inventory inputs.": (
        "Tahmin doğrudan sevkiyat emri değildir",
        (
            "Tahmin edilen miktar brüt mağaza ihtiyacıdır. Depodan gönderilecek "
            "net miktar için mağaza stoku ve yoldaki sevkiyat düşülmeli, emniyet "
            "stoku ayrıca eklenmelidir."
        ),
    ),
}


st.set_page_config(
    page_title="Özdilek | Talep Planlama",
    page_icon="◉",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    f"""
    <style>
    :root {{
      --oz-orange:{ORANGE};
      --oz-orange-light:{ORANGE_LIGHT};
      --oz-ink:{INK};
      --oz-panel:{PANEL};
      --oz-panel-soft:{PANEL_SOFT};
      --oz-text:{TEXT};
      --oz-muted:{MUTED};
      --oz-grid:{GRID};
    }}

    html, body, [class*="css"] {{
      font-family: Inter, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }}
    .stApp {{
      background:
        radial-gradient(circle at 92% 2%, rgba(245,130,32,.08), transparent 24rem),
        #0E0F11;
      color:var(--oz-text);
    }}
    [data-testid="stHeader"] {{
      background:rgba(14,15,17,.82);
      backdrop-filter:blur(12px);
    }}
    [data-testid="stToolbar"] {{right:1rem;}}
    [data-testid="stMainBlockContainer"] {{
      max-width:1480px;
      padding:3.25rem 2.4rem 4rem;
    }}
    [data-testid="stSidebar"] {{
      background:#09090A;
      border-right:1px solid #28292D;
    }}
    [data-testid="stSidebarContent"] {{
      padding:1.1rem .85rem 1.5rem;
    }}
    [data-testid="stSidebar"] hr {{
      border-color:#2A2B2F;
      margin:1rem 0;
    }}
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
    [data-testid="stSidebar"] label {{
      color:#D8D8D5 !important;
    }}
    [data-testid="stSidebar"] div[role="radiogroup"] {{
      gap:.35rem;
    }}
    [data-testid="stSidebar"] div[role="radiogroup"] label {{
      background:transparent;
      border:1px solid transparent;
      border-radius:10px;
      padding:.72rem .75rem;
      transition:all .18s ease;
    }}
    [data-testid="stSidebar"] div[role="radiogroup"] label:hover {{
      background:#18191C;
      border-color:#2E3035;
    }}
    [data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked) {{
      background:linear-gradient(90deg,rgba(245,130,32,.18),rgba(245,130,32,.04));
      border-color:rgba(245,130,32,.45);
      color:white !important;
    }}
    [data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked) p {{
      color:white !important;
      font-weight:700;
    }}
    [data-testid="stSidebar"] div[role="radiogroup"] input {{
      display:none;
    }}

    h1,h2,h3,h4,p,span,label {{
      color:var(--oz-text);
    }}
    .page-kicker {{
      color:var(--oz-orange);
      font-size:.72rem;
      letter-spacing:.15em;
      font-weight:800;
      text-transform:uppercase;
      display:inline-flex;
      align-items:center;
      min-height:28px;
      padding:.32rem .58rem;
      margin:.15rem 0 .65rem;
      border:1px solid rgba(245,130,32,.32);
      border-radius:999px;
      background:rgba(245,130,32,.08);
    }}
    .page-title {{
      color:#FFFFFF;
      font-size:2rem;
      line-height:1.12;
      letter-spacing:-.035em;
      font-weight:760;
      margin:0;
    }}
    .page-subtitle {{
      color:var(--oz-muted);
      font-size:.96rem;
      line-height:1.55;
      margin:.65rem 0 0;
      max-width:860px;
    }}
    .page-header {{
      padding:.6rem 0 1.35rem;
      border-bottom:1px solid #292A2F;
      margin-bottom:1.4rem;
      overflow:visible;
    }}
    .section-title {{
      display:flex;
      align-items:center;
      gap:.65rem;
      margin:1.4rem 0 .8rem;
      color:#FFFFFF;
      font-size:1.05rem;
      font-weight:750;
    }}
    .section-title:before {{
      content:"";
      width:4px;
      height:18px;
      border-radius:6px;
      background:var(--oz-orange);
    }}

    .brand {{
      display:flex;
      align-items:center;
      gap:.75rem;
      padding:.3rem .25rem .9rem;
    }}
    .brand-mark {{
      width:42px;
      height:42px;
      display:grid;
      place-items:center;
      border-radius:10px;
      background:var(--oz-orange);
      color:#0A0A0B;
      font-weight:900;
      font-size:1.25rem;
      box-shadow:0 8px 24px rgba(245,130,32,.18);
    }}
    .brand-name {{
      color:#FFFFFF;
      font-weight:850;
      letter-spacing:.06em;
      font-size:1rem;
    }}
    .brand-product {{
      color:#8D8F94;
      font-size:.68rem;
      letter-spacing:.06em;
      margin-top:.15rem;
    }}
    .sidebar-label {{
      color:#6F7177;
      font-size:.65rem;
      letter-spacing:.14em;
      font-weight:800;
      margin:.35rem .45rem .45rem;
    }}
    .system-card {{
      background:#141517;
      border:1px solid #292A2E;
      border-radius:12px;
      padding:.85rem;
      margin:.2rem .15rem;
    }}
    .system-card .label {{
      color:#888A90;
      font-size:.68rem;
      letter-spacing:.08em;
      text-transform:uppercase;
    }}
    .system-card .value {{
      color:#F4F4F2;
      font-size:.82rem;
      font-weight:680;
      margin-top:.28rem;
    }}
    .online-dot {{
      display:inline-block;
      width:7px;height:7px;border-radius:50%;
      background:{GREEN};
      box-shadow:0 0 0 4px rgba(54,201,143,.1);
      margin-right:.45rem;
    }}

    .input-shell {{
      background:linear-gradient(145deg,#191A1E,#141517);
      border:1px solid #2C2D32;
      border-radius:16px;
      padding:1.1rem 1.15rem .3rem;
      box-shadow:0 14px 40px rgba(0,0,0,.16);
    }}
    [data-testid="stVerticalBlockBorderWrapper"] {{
      background:linear-gradient(145deg,#191A1E,#141517);
      border-color:#2C2D32 !important;
      border-radius:16px !important;
      box-shadow:0 14px 40px rgba(0,0,0,.12);
    }}
    [data-testid="stSelectbox"] label,
    [data-testid="stDateInput"] label,
    [data-testid="stFileUploader"] label,
    [data-testid="stMultiSelect"] label {{
      color:#E9E9E7 !important;
      font-size:.82rem;
      font-weight:680;
    }}
    [data-baseweb="select"] > div,
    [data-testid="stDateInput"] input,
    [data-testid="stTextInput"] input {{
      background:#222329 !important;
      border-color:#373940 !important;
      color:white !important;
      border-radius:10px !important;
    }}
    [data-testid="stDateInput"] input {{
      -webkit-text-fill-color:white !important;
    }}
    .stButton > button,
    .stDownloadButton > button {{
      border-radius:10px;
      min-height:44px;
      font-weight:760;
      border:1px solid #3A3B40;
      background:#202126;
      color:#F7F7F5;
      transition:all .18s ease;
    }}
    .stButton > button:hover,
    .stDownloadButton > button:hover {{
      border-color:var(--oz-orange);
      color:white;
      transform:translateY(-1px);
    }}
    .stButton > button[kind="primary"] {{
      background:var(--oz-orange);
      border-color:var(--oz-orange);
      color:#0A0A0B;
      box-shadow:0 8px 22px rgba(245,130,32,.18);
    }}
    .stButton > button[kind="primary"]:hover {{
      background:#FF963D;
      border-color:#FF963D;
      color:#050505;
    }}
    [data-testid="stAlert"] {{
      border-radius:12px;
      border:1px solid #35373D;
      background:#1B1C20;
      color:#F4F4F2 !important;
    }}
    [data-testid="stAlert"] * {{
      color:#F4F4F2 !important;
    }}

    .decision-panel {{
      display:flex;
      justify-content:space-between;
      align-items:center;
      gap:1.2rem;
      padding:1.25rem 1.35rem;
      border-radius:15px;
      border:1px solid #303138;
      background:#18191D;
      margin:.8rem 0 1rem;
      overflow:hidden;
      position:relative;
    }}
    .decision-panel:before {{
      content:"";
      position:absolute;
      left:0;top:0;bottom:0;width:5px;
      background:var(--decision-color);
    }}
    .decision-eyebrow {{
      color:#94969C;
      font-size:.7rem;
      letter-spacing:.12em;
      text-transform:uppercase;
      font-weight:800;
    }}
    .decision-title {{
      color:#FFFFFF;
      font-size:1.35rem;
      font-weight:780;
      margin-top:.25rem;
    }}
    .decision-copy {{
      color:#B5B6BB;
      font-size:.85rem;
      margin-top:.35rem;
    }}
    .decision-chip {{
      color:var(--decision-color);
      border:1px solid color-mix(in srgb,var(--decision-color) 50%,transparent);
      background:color-mix(in srgb,var(--decision-color) 10%,transparent);
      border-radius:999px;
      padding:.45rem .75rem;
      font-size:.75rem;
      font-weight:800;
      white-space:nowrap;
    }}
    .kpi-card {{
      min-height:116px;
      background:linear-gradient(145deg,#1B1C20,#151619);
      border:1px solid #2D2F34;
      border-radius:14px;
      padding:1rem 1.05rem;
      position:relative;
      overflow:hidden;
    }}
    .kpi-card:after {{
      content:"";
      position:absolute;right:-25px;top:-25px;
      width:74px;height:74px;border-radius:50%;
      background:rgba(245,130,32,.06);
    }}
    .kpi-label {{
      color:#8F9197;
      font-size:.7rem;
      letter-spacing:.08em;
      text-transform:uppercase;
      font-weight:750;
    }}
    .kpi-value {{
      color:#FFFFFF;
      font-size:1.55rem;
      letter-spacing:-.03em;
      line-height:1.15;
      font-weight:780;
      margin-top:.55rem;
    }}
    .kpi-detail {{
      color:#92949A;
      font-size:.72rem;
      margin-top:.35rem;
    }}
    .context-card {{
      background:#17181B;
      border:1px solid #2D2F34;
      border-radius:14px;
      padding:1rem 1.05rem;
      min-height:132px;
    }}
    .context-icon {{
      width:32px;height:32px;display:grid;place-items:center;
      border-radius:9px;background:rgba(245,130,32,.12);
      color:var(--oz-orange);font-weight:900;margin-bottom:.65rem;
    }}
    .context-title {{
      color:#FFFFFF;font-size:.9rem;font-weight:760;
    }}
    .context-copy {{
      color:#A8AAB0;font-size:.78rem;line-height:1.55;margin-top:.4rem;
    }}
    .notice {{
      background:#151619;
      border:1px solid #2C2E33;
      border-left:4px solid var(--oz-orange);
      border-radius:12px;
      padding:.9rem 1rem;
      color:#BFC0C4;
      font-size:.78rem;
      line-height:1.55;
      margin-top:.75rem;
    }}
    .limitation-item {{
      display:grid;
      grid-template-columns:30px 1fr;
      gap:.75rem;
      padding:.85rem 0;
      border-bottom:1px solid #2A2C31;
    }}
    .limitation-item:last-child {{border-bottom:none;}}
    .limitation-number {{
      width:28px;height:28px;display:grid;place-items:center;
      border-radius:8px;background:rgba(245,130,32,.12);
      color:var(--oz-orange);font-size:.72rem;font-weight:850;
    }}
    .limitation-title {{
      color:#FFFFFF;font-size:.86rem;font-weight:760;
    }}
    .limitation-copy {{
      color:#A8AAB0;font-size:.78rem;line-height:1.55;margin-top:.25rem;
    }}
    .guide-module {{
      background:linear-gradient(145deg,#191A1E,#141517);
      border:1px solid #2D2F34;
      border-radius:14px;
      padding:1rem 1.05rem;
      min-height:154px;
    }}
    .guide-module .tag {{
      color:var(--oz-orange);font-size:.66rem;letter-spacing:.1em;
      text-transform:uppercase;font-weight:820;
    }}
    .guide-module .title {{
      color:#FFFFFF;font-size:.92rem;font-weight:770;margin-top:.4rem;
    }}
    .guide-module .copy {{
      color:#A8AAB0;font-size:.77rem;line-height:1.55;margin-top:.4rem;
    }}
    .guide-callout {{
      background:linear-gradient(135deg,rgba(245,130,32,.13),#17181B 58%);
      border:1px solid rgba(245,130,32,.28);
      border-radius:14px;
      padding:1rem 1.1rem;
      color:#CACBCF;
      font-size:.8rem;
      line-height:1.65;
    }}
    .guide-callout strong {{color:#FFFFFF;}}
    .status-row {{
      display:flex;gap:.5rem;flex-wrap:wrap;margin:.7rem 0 0;
    }}
    .status-badge {{
      border-radius:999px;
      padding:.35rem .62rem;
      background:#1B1C20;
      border:1px solid #303238;
      color:#BFC1C6;
      font-size:.7rem;
      font-weight:700;
    }}
    .status-badge strong {{color:#FFFFFF;}}
    .orange {{color:var(--oz-orange) !important;}}

    [data-testid="stPlotlyChart"] {{
      background:#151619;
      border:1px solid #2B2D32;
      border-radius:15px;
      overflow:hidden;
    }}
    [data-testid="stDataFrame"] {{
      border:1px solid #2C2E33;
      border-radius:12px;
      overflow:hidden;
    }}
    [data-testid="stExpander"] {{
      background:#16171A;
      border-color:#2D2F34;
      border-radius:12px;
    }}
    [data-testid="stExpander"] summary p {{
      color:#F2F2F0 !important;
      font-weight:680;
    }}
    code {{
      color:#FFB36B !important;
      background:#202126 !important;
    }}
    a {{color:var(--oz-orange-light) !important;}}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def get_service() -> DemandForecastService:
    return DemandForecastService(PROJECT_ROOT)


@st.cache_data(show_spinner=False)
def read_report_csv(path_text: str, modified_at: float) -> pd.DataFrame:
    del modified_at
    return pd.read_csv(path_text)


def report_csv(relative_path: str) -> pd.DataFrame:
    path = PROJECT_ROOT / relative_path
    return read_report_csv(str(path), path.stat().st_mtime)


def page_header(kicker: str, title: str, subtitle: str) -> None:
    st.markdown(
        f"""
        <div class="page-header">
          <div class="page-kicker">{html.escape(kicker)}</div>
          <h1 class="page-title">{html.escape(title)}</h1>
          <p class="page-subtitle">{html.escape(subtitle)}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def section_title(title: str) -> None:
    st.markdown(
        f'<div class="section-title">{html.escape(title)}</div>',
        unsafe_allow_html=True,
    )


def kpi_card(label: str, value: str, detail: str) -> None:
    st.markdown(
        f"""
        <div class="kpi-card">
          <div class="kpi-label">{html.escape(label)}</div>
          <div class="kpi-value">{html.escape(value)}</div>
          <div class="kpi-detail">{html.escape(detail)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def plot_layout(height: int = 330) -> dict:
    return {
        "height": height,
        "paper_bgcolor": "#151619",
        "plot_bgcolor": "#151619",
        "font": {"color": TEXT, "family": "Inter, Arial, sans-serif"},
        "margin": {"l": 45, "r": 25, "t": 55, "b": 45},
        "hoverlabel": {
            "bgcolor": "#0E0F11",
            "font": {"color": "#FFFFFF"},
            "bordercolor": ORANGE,
        },
    }


def probability_gauge(result: dict) -> go.Figure:
    probability = result["demand_probability"] * 100
    threshold = result["decision_threshold"] * 100
    bar_color = ORANGE if result["demand_expected"] else "#7B7E86"
    figure = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=probability,
            number={"suffix": "%", "font": {"size": 40, "color": "#FFFFFF"}},
            title={
                "text": (
                    "Talep olasılığı"
                    f"<br><span style='font-size:12px;color:#989AA0'>"
                    f"Karar eşiği %{threshold:.0f}</span>"
                ),
                "font": {"size": 17, "color": "#FFFFFF"},
            },
            gauge={
                "axis": {
                    "range": [0, 100],
                    "tickwidth": 1,
                    "tickcolor": "#50525A",
                    "tickfont": {"color": "#8E9096"},
                },
                "bar": {"color": bar_color, "thickness": 0.32},
                "bgcolor": "#222328",
                "borderwidth": 0,
                "steps": [
                    {"range": [0, threshold], "color": "#222328"},
                    {"range": [threshold, 100], "color": "rgba(245,130,32,.10)"},
                ],
                "threshold": {
                    "line": {"color": ORANGE, "width": 4},
                    "thickness": 0.72,
                    "value": threshold,
                },
            },
        )
    )
    figure.update_layout(**plot_layout(315))
    return figure


def history_figure(
    history: pd.DataFrame,
    result: dict,
    product_name: str,
) -> go.Figure:
    working = history.copy()
    working["rolling_mean_7"] = (
        working["daily_demand"].rolling(7, min_periods=1).mean()
    )
    unit = result["unit"]
    figure = go.Figure()
    figure.add_trace(
        go.Bar(
            x=working["date"],
            y=working["daily_demand"],
            name=f"Günlük talep ({unit})",
            marker_color="rgba(245,130,32,.46)",
            hovertemplate=f"%{{x|%d.%m.%Y}}<br>%{{y:.3f}} {unit}<extra></extra>",
        )
    )
    figure.add_trace(
        go.Scatter(
            x=working["date"],
            y=working["rolling_mean_7"],
            name="7 günlük hareketli ortalama",
            mode="lines",
            line={"color": "#F7F7F5", "width": 2},
            hovertemplate=f"%{{x|%d.%m.%Y}}<br>%{{y:.3f}} {unit}<extra></extra>",
        )
    )
    target_value = float(result["demand_prediction"])
    figure.add_trace(
        go.Scatter(
            x=[pd.Timestamp(result["target_date"])],
            y=[target_value],
            name="Seçili gün tahmini",
            mode="markers",
            marker={
                "color": ORANGE if result["demand_expected"] else RED,
                "size": 15,
                "symbol": "diamond",
                "line": {"color": "#FFFFFF", "width": 1},
            },
            hovertemplate=(
                f"Hedef gün<br>%{{x|%d.%m.%Y}}<br>%{{y:.3f}} {unit}<extra></extra>"
            ),
        )
    )
    origin = pd.Timestamp(result["forecast_origin"])
    target = pd.Timestamp(result["target_date"])
    figure.add_vline(
        x=origin.timestamp() * 1000,
        line_color="#8C8E94",
        line_dash="dot",
        annotation_text="Forecast origin",
        annotation_font_color="#A9ABB0",
    )
    if target > origin:
        figure.add_vrect(
            x0=origin,
            x1=target,
            fillcolor=ORANGE,
            opacity=0.025,
            line_width=0,
        )
    figure.update_layout(
        **plot_layout(350),
        title={
            "text": (
                f"{html.escape(product_name[:54])}"
                "<br><span style='font-size:12px;color:#93959B'>"
                "Geçmiş talep ve seçili hedef gün</span>"
            ),
            "x": 0.04,
            "font": {"size": 16},
        },
        barmode="overlay",
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.02,
            "xanchor": "right",
            "x": 1,
            "font": {"size": 10},
        },
        xaxis={
            "showgrid": False,
            "rangeslider": {"visible": True, "thickness": 0.08},
            "tickfont": {"color": "#8E9096"},
        },
        yaxis={
            "title": unit,
            "gridcolor": GRID,
            "zerolinecolor": GRID,
            "tickfont": {"color": "#8E9096"},
        },
    )
    return figure


def context_cards(result: dict) -> None:
    calendar_context = result["calendar_context"]
    calendar_labels = [calendar_context["school_status_label"]]
    if calendar_context["public_holiday_name"]:
        calendar_labels.append(calendar_context["public_holiday_name"])
    if calendar_context["religious_special_name"]:
        calendar_labels.append(calendar_context["religious_special_name"])
    if calendar_context["is_ramadan"]:
        calendar_labels.append("Ramazan dönemi")
    if pd.Timestamp(result["target_date"]).dayofweek >= 5:
        calendar_labels.append("Hafta sonu")

    weather = result["weather_context"]
    weather_use = (
        "Model girdisi"
        if weather["used_by_model"]
        else "Ablation sonrası yalnız açıklayıcı bağlam"
    )
    left, right = st.columns(2)
    with left:
        st.markdown(
            f"""
            <div class="context-card">
              <div class="context-icon">T</div>
              <div class="context-title">Takvim bağlamı</div>
              <div class="context-copy">
                {html.escape(" · ".join(calendar_labels))}<br>
                Sonraki resmî tatile {calendar_context['days_to_public_holiday']} gün
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with right:
        st.markdown(
            f"""
            <div class="context-card">
              <div class="context-icon">H</div>
              <div class="context-title">Bursa hava bağlamı</div>
              <div class="context-copy">
                Ort. {weather['temperature_mean_c']} °C · Yağış
                {weather['precipitation_mm']} mm · Bulut %{weather['cloud_cover_pct']}
                · Rüzgâr {weather['wind_max_kmh']} km/sa<br>
                {html.escape(weather_use)}
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_forecast_result(service: DemandForecastService, result: dict) -> None:
    if result["status"] == "insufficient_history":
        st.warning(result["message"])
        return

    decision_color = ORANGE if result["demand_expected"] else "#8B8E95"
    decision_title = (
        f"Talep bekleniyor · {result['display_quantity']} {result['unit']}"
        if result["demand_expected"]
        else "Bu hedef gün için talep beklenmiyor"
    )
    decision_copy = (
        "Mağazanın tahmini brüt ihtiyacı için sevkiyat planı değerlendirilmeli."
        if result["demand_expected"]
        else "Model olasılığı birim karar eşiğinin altında kaldı."
    )
    st.markdown(
        f"""
        <div class="decision-panel" style="--decision-color:{decision_color}">
          <div>
            <div class="decision-eyebrow">Model karar özeti</div>
            <div class="decision-title">{html.escape(decision_title)}</div>
            <div class="decision-copy">{html.escape(decision_copy)}</div>
          </div>
          <div class="decision-chip">
            {result['demand_probability'] * 100:.1f}% olasılık
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    cols = st.columns(4)
    values = [
        (
            "Talep kararı",
            "Bekleniyor" if result["demand_expected"] else "Beklenmiyor",
            f"Eşik %{result['decision_threshold'] * 100:.0f}",
        ),
        (
            "Brüt mağaza ihtiyacı",
            f"{result['display_quantity']} {result['unit']}",
            "Stok ve yoldaki sevkiyat hariç",
        ),
        (
            "Hedef gün",
            pd.Timestamp(result["target_date"]).strftime("%d.%m.%Y"),
            f"{result['lead_days']} gün ileri",
        ),
        (
            "Ürün birimi",
            result["unit"],
            "Katalog birimi değiştirilmez",
        ),
    ]
    for column, (label, value, detail) in zip(cols, values):
        with column:
            kpi_card(label, value, detail)

    chart_heading, chart_period = st.columns([3.2, 1])
    with chart_heading:
        section_title("Talep görünümü")
        st.caption(
            "Ürünün geçmiş satış hareketi ile hedef gün kararını birlikte inceleyin."
        )
    with chart_period:
        if "history_days" not in st.session_state:
            st.session_state["history_days"] = 180
        history_days = st.selectbox(
            "Grafikte gösterilecek geçmiş",
            [60, 90, 180, 365],
            key="history_days",
            format_func=lambda value: (
                "Son 1 yıl (365 gün)" if value == 365 else f"Son {value} gün"
            ),
            help=(
                "Yalnızca aşağıdaki geçmiş satış grafiğinin görüntü aralığını "
                "değiştirir. Talep kararını, olasılığı veya miktar tahminini "
                "değiştirmez."
            ),
        )
        st.caption("Yalnızca grafik görünümünü değiştirir.")

    chart_left, chart_right = st.columns([1.55, 1])
    history = service.product_history(result["product_id"], days=history_days)
    with chart_left:
        st.plotly_chart(
            history_figure(history, result, result["product_name"]),
            use_container_width=True,
            config={"displaylogo": False, "scrollZoom": True},
        )
    with chart_right:
        st.plotly_chart(
            probability_gauge(result),
            use_container_width=True,
            config={"displayModeBar": False},
        )

    section_title("Kararı etkileyen bağlam")
    context_cards(result)
    warning_text = ", ".join(result["warning_codes"]) or "Yok"
    st.markdown(
        f"""
        <div class="notice">
          <strong style="color:#FFFFFF">Operasyon notu:</strong>
          Tahmin brüt mağaza talebidir. Net depo transferi için mağaza stoku ve
          yoldaki sevkiyat düşülmeli, emniyet stoku eklenmelidir. Olasılık kalibre
          edilmiş güven skoru değildir.<br>
          <span style="color:#8E9096">Model {html.escape(result['model_version'])}
          · Veri {html.escape(result['data_freshness'])}
          · Uyarılar: {html.escape(warning_text)}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_forecast_page(
    service: DemandForecastService,
    metadata: dict,
    origin,
    max_date,
) -> None:
    page_header(
        "Perakende analitik / ürün-gün kararı",
        "Talep Planlama",
        (
            "Ürün ve hedef günü seçin; talep oluşumu ile günlük KG/ADT ihtiyacını "
            "tek karar ekranında değerlendirin."
        ),
    )
    options = service.product_options()
    label_map = dict(zip(options["label"], options["product_id"].astype(str)))

    with st.container(border=True):
        left, right = st.columns([2.2, 1])
        with left:
            selected_label = st.selectbox(
                "Ürün",
                list(label_map),
                help="Ürün adı, katalog birimi ve ürün ID'si birlikte gösterilir.",
            )
        with right:
            target_date = st.date_input(
                "Hedef tarih",
                value=origin + timedelta(days=7),
                min_value=origin + timedelta(days=1),
                max_value=max_date,
                help=f"Doğrulanmış aralık: {origin + timedelta(days=1)}–{max_date}",
            )
        selected_id = label_map[selected_label]
        selected_info = options.loc[
            options["product_id"].astype(str).eq(selected_id)
        ].iloc[0]
        st.markdown(
            f"""
            <div class="status-row">
              <span class="status-badge">Birim <strong>{selected_info['unit']}</strong></span>
              <span class="status-badge">Durum <strong>{selected_info['current_status']}</strong></span>
              <span class="status-badge">Geçmiş <strong>{int(selected_info['history_days'])} gün</strong></span>
              <span class="status-badge">Veri kesimi <strong>{origin}</strong></span>
            </div>
            """,
            unsafe_allow_html=True,
        )
        create_forecast = st.button(
            "Tahmini oluştur",
            type="primary",
            width="stretch",
        )

    if create_forecast:
        try:
            with st.spinner("Ürün ve hedef gün için tahmin hesaplanıyor..."):
                result = service.forecast(selected_id, target_date)
            st.session_state["last_forecast"] = result
            if result["status"] != "insufficient_history":
                log_forecast(PROJECT_ROOT, result)
        except Exception as error:
            st.error(str(error))

    result = st.session_state.get("last_forecast")
    if result:
        same_selection = (
            str(result.get("product_id")) == str(selected_id)
            and result.get("target_date") == target_date.isoformat()
        )
        if same_selection:
            render_forecast_result(service, result)
        else:
            st.info(
                "Ürün veya tarih değiştirildi. Yeni seçim için “Tahmini oluştur” "
                "düğmesine basın."
            )


def render_batch_page(service: DemandForecastService) -> None:
    page_header(
        "Operasyon / çoklu ürün-tarih",
        "Toplu Talep Planlama",
        (
            "Birden fazla ürün ve hedef tarihi tek dosyada çalıştırın; satır bazlı "
            "hataları kaybetmeden sonuçları analiz edin ve dışarı aktarın."
        ),
    )
    template = pd.DataFrame(
        {
            "product_id": ["13393980", "10002243"],
            "target_date": ["2026-08-15", "2026-10-27"],
        }
    )
    top_left, top_right = st.columns([2, 1])
    with top_left:
        uploaded = st.file_uploader(
            "Talep dosyası",
            type=["csv"],
            help="Zorunlu kolonlar: product_id, target_date",
        )
    with top_right:
        st.download_button(
            "CSV şablonunu indir",
            template.to_csv(index=False).encode("utf-8-sig"),
            file_name="ozdilek_toplu_tahmin_sablonu.csv",
            mime="text/csv",
            width="stretch",
        )

    if uploaded is not None:
        requests = pd.read_csv(uploaded, dtype={"product_id": "string"})
        st.caption(f"Yüklenen satır sayısı: {len(requests):,}")
        if st.button(
            "Toplu tahmini çalıştır",
            type="primary",
            width="stretch",
        ):
            with st.spinner("Toplu tahminler üretiliyor..."):
                st.session_state["batch_results"] = service.batch_forecast(requests)

    results = st.session_state.get("batch_results")
    if results is None or results.empty:
        st.markdown(
            """
            <div class="notice">
              CSV dosyanızı yükleyin. Hatalı satırlar sonuçtan silinmez; açıklayıcı
              <code>error</code> statüsüyle birlikte döner.
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    ready = results["status"].astype(str).str.startswith("forecast_ready")
    demand = results.get("demand_expected", pd.Series(False, index=results.index))
    demand = demand.astype(str).str.lower().eq("true")
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        kpi_card("Toplam satır", f"{len(results):,}", "Dosyadaki tüm istekler")
    with k2:
        kpi_card("Başarılı", f"{int(ready.sum()):,}", "Tahmin üretilebilen")
    with k3:
        kpi_card("Talep beklenen", f"{int(demand.sum()):,}", "Karar eşiği üzeri")
    with k4:
        kpi_card("Hatalı", f"{int((~ready).sum()):,}", "İncelenmesi gereken")

    section_title("Toplu sonuç analizi")
    chart_left, chart_right = st.columns(2)
    chart_data = results.loc[ready].copy()
    with chart_left:
        if not chart_data.empty and "demand_probability" in chart_data:
            figure = go.Figure()
            for unit, subset in chart_data.groupby("unit"):
                figure.add_trace(
                    go.Histogram(
                        x=pd.to_numeric(
                            subset["demand_probability"], errors="coerce"
                        ),
                        name=unit,
                        opacity=.72,
                        nbinsx=15,
                    )
                )
            figure.update_layout(
                **plot_layout(320),
                barmode="overlay",
                title="Talep olasılığı dağılımı",
                xaxis_title="Olasılık",
                yaxis_title="Tahmin sayısı",
                colorway=[ORANGE, "#A9ABB0"],
                xaxis={"gridcolor": GRID},
                yaxis={"gridcolor": GRID},
            )
            st.plotly_chart(figure, use_container_width=True)
    with chart_right:
        if not chart_data.empty:
            unit_summary = (
                chart_data.assign(demand_expected=demand.loc[chart_data.index])
                .groupby("unit", as_index=False)
                .agg(
                    forecasts=("product_id", "size"),
                    demand_expected=("demand_expected", "sum"),
                )
            )
            figure = go.Figure(
                [
                    go.Bar(
                        x=unit_summary["unit"],
                        y=unit_summary["forecasts"],
                        name="Toplam",
                        marker_color="#5E6169",
                    ),
                    go.Bar(
                        x=unit_summary["unit"],
                        y=unit_summary["demand_expected"],
                        name="Talep beklenen",
                        marker_color=ORANGE,
                    ),
                ]
            )
            figure.update_layout(
                **plot_layout(320),
                barmode="group",
                title="Birim bazında karar dağılımı",
                xaxis={"gridcolor": GRID},
                yaxis={"gridcolor": GRID},
            )
            st.plotly_chart(figure, use_container_width=True)

    display_columns = [
        column
        for column in [
            "product_id",
            "product_name",
            "target_date",
            "unit",
            "lead_days",
            "demand_expected",
            "demand_probability",
            "display_quantity",
            "status",
            "message",
        ]
        if column in results.columns
    ]
    st.dataframe(
        results[display_columns],
        width="stretch",
        hide_index=True,
    )
    st.download_button(
        "Sonuçları indir",
        results.to_csv(index=False).encode("utf-8-sig"),
        file_name="ozdilek_talep_tahminleri.csv",
        mime="text/csv",
        width="stretch",
    )


def render_model_page(metadata: dict) -> None:
    page_header(
        "Model denetimi / performans ve veri",
        "Model Merkezi",
        (
            "Yayındaki modelin sürümünü, test sonuçlarını, kullandığı değişkenleri "
            "ve bilinen sınırlılıklarını tek denetim ekranında inceleyin."
        ),
    )
    metrics = metadata["test_metrics"]
    metric_rows = []
    for unit in ("KG", "ADT"):
        metric_rows.append(
            {
                "Birim": unit,
                "PR-AUC": metrics[unit]["occurrence"]["pr_auc"],
                "F1": metrics[unit]["occurrence"]["f1"],
                "Precision": metrics[unit]["occurrence"]["precision"],
                "Recall": metrics[unit]["occurrence"]["recall"],
                "Miktar MAE": metrics[unit]["quantity_end_to_end"]["mae"],
                "WAPE": metrics[unit]["quantity_end_to_end"]["wape"],
                "Bias": metrics[unit]["quantity_end_to_end"]["bias"],
            }
        )
    metric_frame = pd.DataFrame(metric_rows)
    top = st.columns(4)
    with top[0]:
        kpi_card("Model sürümü", metadata["model_version"], "Yayındaki model paketi")
    with top[1]:
        kpi_card(
            "Kullanılan değişken",
            str(len(metadata["feature_columns"])),
            "Tahmine giren veri alanı",
        )
    with top[2]:
        kpi_card(
            "Ürün evreni",
            "713",
            f"{', '.join(metadata['units'])} birimleri",
        )
    with top[3]:
        kpi_card(
            "Tahmin ufku",
            f"{metadata['max_forecast_lead_days']} gün",
            "Doğrulanmış tarih aralığı",
        )

    section_title("Son test dönemi performansı")
    left, right = st.columns([1.35, 1])
    with left:
        figure = go.Figure()
        for metric_name, color in (("PR-AUC", ORANGE), ("F1", "#D2D3D6")):
            figure.add_trace(
                go.Bar(
                    x=metric_frame["Birim"],
                    y=metric_frame[metric_name],
                    name=metric_name,
                    marker_color=color,
                    text=metric_frame[metric_name].map(lambda value: f"{value:.3f}"),
                    textposition="outside",
                )
            )
        figure.update_layout(
            **plot_layout(335),
            barmode="group",
            title="Talep oluşumu performansı",
            yaxis={"range": [0, 1], "gridcolor": GRID},
            xaxis={"gridcolor": GRID},
        )
        st.plotly_chart(figure, use_container_width=True)
    with right:
        figure = go.Figure(
            go.Bar(
                x=metric_frame["Birim"],
                y=metric_frame["Miktar MAE"],
                marker_color=[ORANGE, "#B4B6BB"],
                text=metric_frame["Miktar MAE"].map(lambda value: f"{value:.3f}"),
                textposition="outside",
            )
        )
        figure.update_layout(
            **plot_layout(335),
            title="Günlük miktar MAE",
            yaxis={"gridcolor": GRID},
            xaxis={"gridcolor": GRID},
        )
        st.plotly_chart(figure, use_container_width=True)

    st.dataframe(
        metric_frame.style.format(
            {
                column: "{:.4f}"
                for column in metric_frame.columns
                if column != "Birim"
            }
        ),
        width="stretch",
        hide_index=True,
    )

    section_title("Değişken yönetişimi")
    governance_left, governance_right = st.columns([1, 1.35])
    with governance_left:
        weather_candidates = len(metadata["weather_candidate_features"])
        figure = go.Figure(
            go.Pie(
                labels=[
                    "Tahminde kullanılan değişken",
                    "Kullanılmayan hava değişkeni adayı",
                ],
                values=[len(metadata["feature_columns"]), weather_candidates],
                hole=.72,
                marker={"colors": [ORANGE, "#3A3C42"]},
                textinfo="label+value",
                hovertemplate="%{label}: %{value}<extra></extra>",
            )
        )
        figure.update_layout(
            **plot_layout(320),
            title="Değişken kullanım kararı",
            showlegend=False,
            annotations=[
                {
                    "text": f"{len(metadata['feature_columns'])}<br>kullanılıyor",
                    "showarrow": False,
                    "font": {"size": 18, "color": "#FFFFFF"},
                }
            ],
        )
        st.plotly_chart(figure, use_container_width=True)
    with governance_right:
        weather_decision = {
            "excluded_after_validation_ablation_no_robust_gain": (
                "Doğrulama testlerinde kalıcı fayda sağlamadığı için modelden çıkarıldı"
            )
        }.get(
            metadata["weather_deployment_decision"],
            "Teknik model kararında tanımlandı",
        )
        kg_calibration = (
            "Evet" if metadata["probability_calibrated"]["KG"] else "Hayır"
        )
        adt_calibration = (
            "Evet" if metadata["probability_calibrated"]["ADT"] else "Hayır"
        )
        st.markdown(
            f"""
            <div class="context-card" style="min-height:320px">
              <div class="context-icon">G</div>
              <div class="context-title">Model değişkenleri özeti</div>
              <div class="context-copy" style="font-size:.84rem;line-height:1.75">
                <strong style="color:#FFFFFF">Takvim:</strong>
                {html.escape(metadata['calendar_version'])} ·
                {len(metadata['calendar_features'])} özel alan<br>
                <strong style="color:#FFFFFF">Hava:</strong>
                {len(metadata['weather_candidate_features'])} aday alan,
                {len(metadata['weather_features'])} kullanılan alan<br>
                <strong style="color:#FFFFFF">Hava kararı:</strong>
                {html.escape(weather_decision)}<br>
                <strong style="color:#FFFFFF">Son veri tarihi:</strong>
                {html.escape(metadata['forecast_origin'])}<br>
                <strong style="color:#FFFFFF">Olasılık kalibrasyonu:</strong>
                KG {kg_calibration} · ADT {adt_calibration}
              </div>
              <div class="notice">
                Hava değişkenleri yalnızca geçmişte bilinebilecek verilerle
                hazırlanmıştır. Doğrulama testlerinde KG ve ADT ürünlerde tutarlı
                bir iyileşme sağlamadığı için aktif tahmin modelinden çıkarılmıştır.
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with st.expander("Teknik model bilgilerini görüntüle"):
        st.json(metadata)
    with st.expander("Bilinen sınırlılıklar", expanded=True):
        st.caption(
            "Bu maddeler modelin hangi durumlarda tek başına karar vermek için "
            "yeterli olmadığını açıklar."
        )
        for index, limitation in enumerate(metadata["known_limitations"], start=1):
            title, explanation = KNOWN_LIMITATION_TRANSLATIONS.get(
                limitation,
                (
                    "Ek teknik sınırlılık",
                    (
                        "Model paketinde bu sürüme ait ek bir teknik sınırlılık "
                        "tanımlanmıştır. Ayrıntı için veri bilimi ekibine başvurun."
                    ),
                ),
            )
            st.markdown(
                f"""
                <div class="limitation-item">
                  <div class="limitation-number">{index:02d}</div>
                  <div>
                    <div class="limitation-title">{html.escape(title)}</div>
                    <div class="limitation-copy">{html.escape(explanation)}</div>
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def render_monitoring_page() -> None:
    page_header(
        "Operasyon / tahmin kayıtları",
        "Tahmin İzleme",
        (
            "Üretilen tahminleri zaman, birim, ufuk ve karar dağılımında izleyin; "
            "model davranışındaki değişimleri görünür kılın."
        ),
    )
    log = read_forecast_log(PROJECT_ROOT)
    if log.empty:
        st.info("Henüz kayıtlı tahmin bulunmuyor.")
        return

    log["created_at_dt"] = pd.to_datetime(
        log["created_at"], errors="coerce", utc=True
    ).dt.tz_convert("Europe/Istanbul")
    log["demand_probability"] = pd.to_numeric(
        log["demand_probability"], errors="coerce"
    )
    log["lead_days"] = pd.to_numeric(log["lead_days"], errors="coerce")
    log["demand_expected_bool"] = (
        log["demand_expected"].astype(str).str.lower().eq("true")
    )

    available_units = sorted(log["unit"].dropna().astype(str).unique())
    selected_units = st.multiselect(
        "Birim filtresi",
        available_units,
        default=available_units,
    )
    filtered = log.loc[log["unit"].astype(str).isin(selected_units)].copy()
    if filtered.empty:
        st.warning("Seçilen filtrede tahmin bulunmuyor.")
        return

    top = st.columns(4)
    with top[0]:
        kpi_card("Toplam tahmin", f"{len(filtered):,}", "Filtrelenen kayıt")
    with top[1]:
        kpi_card(
            "Talep beklenen",
            f"%{filtered['demand_expected_bool'].mean() * 100:.1f}",
            "Karar oranı",
        )
    with top[2]:
        kpi_card(
            "Ortalama olasılık",
            f"%{filtered['demand_probability'].mean() * 100:.1f}",
            "Kalibre güven skoru değildir",
        )
    with top[3]:
        kpi_card(
            "Medyan ufuk",
            f"{filtered['lead_days'].median():.0f} gün",
            "Hedef tarihe uzaklık",
        )

    section_title("Etkileşimli izleme grafikleri")
    left, right = st.columns(2)
    with left:
        daily = (
            filtered.dropna(subset=["created_at_dt"])
            .assign(day=lambda frame: frame["created_at_dt"].dt.date)
            .groupby(["day", "unit"], as_index=False)
            .size()
        )
        figure = go.Figure()
        for unit, subset in daily.groupby("unit"):
            figure.add_trace(
                go.Scatter(
                    x=subset["day"],
                    y=subset["size"],
                    name=unit,
                    mode="lines+markers",
                    line={"width": 2},
                )
            )
        figure.update_layout(
            **plot_layout(330),
            title="Günlük tahmin hacmi",
            colorway=[ORANGE, "#C6C7CA"],
            xaxis={"gridcolor": GRID},
            yaxis={"gridcolor": GRID, "title": "Tahmin"},
        )
        st.plotly_chart(figure, use_container_width=True)
    with right:
        figure = go.Figure()
        for unit, subset in filtered.groupby("unit"):
            figure.add_trace(
                go.Histogram(
                    x=subset["demand_probability"],
                    name=unit,
                    opacity=.72,
                    nbinsx=15,
                )
            )
        figure.update_layout(
            **plot_layout(330),
            title="Olasılık dağılımı",
            barmode="overlay",
            colorway=[ORANGE, "#C6C7CA"],
            xaxis={"title": "Talep olasılığı", "gridcolor": GRID},
            yaxis={"title": "Tahmin", "gridcolor": GRID},
        )
        st.plotly_chart(figure, use_container_width=True)

    figure = go.Figure()
    for unit, subset in filtered.groupby("unit"):
        figure.add_trace(
            go.Scatter(
                x=subset["lead_days"],
                y=subset["demand_probability"],
                name=unit,
                mode="markers",
                marker={
                    "size": 10,
                    "opacity": .72,
                },
                customdata=np.stack(
                    [
                        subset["product_name"].astype(str),
                        subset["target_date"].astype(str),
                    ],
                    axis=-1,
                ),
                hovertemplate=(
                    "%{customdata[0]}<br>Hedef: %{customdata[1]}"
                    "<br>Ufuk: %{x} gün<br>Olasılık: %{y:.1%}<extra></extra>"
                ),
            )
        )
    figure.update_layout(
        **plot_layout(355),
        title="Tahmin ufku ve talep olasılığı",
        colorway=[ORANGE, "#C6C7CA"],
        xaxis={"title": "İleri gün sayısı", "gridcolor": GRID},
        yaxis={
            "title": "Talep olasılığı",
            "tickformat": ".0%",
            "gridcolor": GRID,
        },
    )
    st.plotly_chart(figure, use_container_width=True)

    section_title("Son tahmin kayıtları")
    display_columns = [
        "created_at",
        "product_name",
        "unit",
        "target_date",
        "lead_days",
        "demand_expected",
        "demand_probability",
        "display_quantity",
        "model_version",
        "status",
    ]
    st.dataframe(
        filtered[display_columns]
        .sort_values("created_at", ascending=False)
        .head(250),
        width="stretch",
        hide_index=True,
    )


def render_help_page(origin, max_date) -> None:
    page_header(
        "Destek / başlangıç ve karar standardı",
        "Kullanım Rehberi",
        (
            "Sistemi ilk kez kullanan bir personelin ürün seçmesinden model "
            "sonucunu yorumlamasına kadar ihtiyaç duyacağı açıklamaları inceleyin."
        ),
    )

    section_title("Önce hangi ekrana gitmeliyim?")
    modules = [
        (
            "TEK ÜRÜN · TEK TARİH",
            "Talep Planlama",
            (
                "Belirli bir ürün için seçtiğiniz günde talep olup olmayacağını "
                "ve tahmini brüt KG/ADT ihtiyacını öğrenmek için kullanılır."
            ),
        ),
        (
            "ÇOKLU İŞLEM",
            "Toplu Planlama",
            (
                "Bir CSV dosyasındaki birden fazla ürün-tarih isteğini aynı anda "
                "hesaplamak ve sonuçları dosya olarak indirmek için kullanılır."
            ),
        ),
        (
            "MODEL DENETİMİ",
            "Model Merkezi",
            (
                "Tahmin üretmez. Yayındaki modelin test başarısını, kullandığı "
                "verileri ve hangi konularda sınırlı olduğunu açıklar."
            ),
        ),
        (
            "KAYIT KONTROLÜ",
            "Tahmin İzleme",
            (
                "Daha önce oluşturulan tahminlerin tarih, ürün, birim ve olasılık "
                "dağılımlarını izlemek için kullanılır."
            ),
        ),
    ]
    module_columns = st.columns(4)
    for column, (tag, title, copy) in zip(module_columns, modules):
        with column:
            st.markdown(
                f"""
                <div class="guide-module">
                  <div class="tag">{html.escape(tag)}</div>
                  <div class="title">{html.escape(title)}</div>
                  <div class="copy">{html.escape(copy)}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    section_title("Talep Planlama ekranı: adım adım")
    steps = [
        (
            "01",
            "Ürünü seçin",
            (
                "Açılır listeden ürünü seçin. Aynı adlı ürünleri ayırabilmek için "
                "ürün ID'sini kontrol edin. Birim KG ise sonuç kilogram, ADT ise "
                "adet olarak hesaplanır."
            ),
        ),
        (
            "02",
            "Hedef günü belirleyin",
            (
                f"{origin + timedelta(days=1)}–{max_date} aralığındaki herhangi "
                "bir günü seçin. Tarih uzaklaştıkça tahmin belirsizliğinin "
                "artabileceğini unutmayın."
            ),
        ),
        (
            "03",
            "Tahmini oluşturun",
            (
                "Turuncu “Tahmini oluştur” düğmesine basın. Ürün veya tarihi "
                "sonradan değiştirirseniz yeni seçim için düğmeye tekrar basmanız "
                "gerekir."
            ),
        ),
        (
            "04",
            "Sonucu birlikte okuyun",
            (
                "Talep kararını, olasılığı, karar eşiğini, brüt miktarı ve geçmiş "
                "satış grafiğini tek başına değil birlikte değerlendirin."
            ),
        ),
        (
            "05",
            "Net transferi hesaplayın",
            (
                "Brüt ihtiyaçtan mağaza stokunu ve yoldaki sevkiyatı düşüp "
                "emniyet stokunu ekleyin. Ekrandaki miktarı doğrudan sipariş "
                "miktarı olarak kullanmayın."
            ),
        ),
    ]
    columns = st.columns(5)
    for column, (number, title, copy) in zip(columns, steps):
        with column:
            st.markdown(
                f"""
                <div class="context-card" style="min-height:245px">
                  <div class="context-icon">{number}</div>
                  <div class="context-title">{html.escape(title)}</div>
                  <div class="context-copy">{html.escape(copy)}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    section_title("Tahmin sonucunu nasıl okuyacağım?")
    st.markdown(
        """
        <div class="guide-callout">
          <strong>En önemli kural:</strong> “Talep bekleniyor” kararı, modelin
          olasılık puanının ürünün KG veya ADT karar eşiğini geçmesiyle oluşur.
          “Talep beklenmiyor” sonucu satışın kesinlikle sıfır olacağı anlamına
          gelmez; model yalnızca eşik altında bir risk görmüştür.
        </div>
        """,
        unsafe_allow_html=True,
    )
    result_guide = pd.DataFrame(
        [
            {
                "Ekrandaki alan": "Talep kararı",
                "Ne anlatır?": (
                    "Seçilen ürün ve hedef gün için talep beklenip beklenmediğini "
                    "gösteren model kararıdır."
                ),
                "Nasıl kullanılmalı?": (
                    "Olasılık, eşik ve geçmiş satış grafiğiyle birlikte okunmalıdır."
                ),
            },
            {
                "Ekrandaki alan": "Talep olasılığı",
                "Ne anlatır?": (
                    "Modelin hedef günde satış oluşmasına verdiği puandır; kesinlik "
                    "veya güven yüzdesi değildir."
                ),
                "Nasıl kullanılmalı?": "Karar eşiğiyle karşılaştırılmalıdır.",
            },
            {
                "Ekrandaki alan": "Karar eşiği",
                "Ne anlatır?": (
                    "Model puanının talep var kararına dönüşmesi için geçmesi "
                    "gereken KG/ADT birimine özel sınırdır."
                ),
                "Nasıl kullanılmalı?": (
                    "Olasılık eşik üzerindeyse sistem talep bekliyor kararı verir."
                ),
            },
            {
                "Ekrandaki alan": "Brüt mağaza ihtiyacı",
                "Ne anlatır?": (
                    "Stok ve sevkiyat bilgileri uygulanmadan önce hedef gün için "
                    "beklenen KG veya ADT miktarıdır."
                ),
                "Nasıl kullanılmalı?": (
                    "Net transfer hesabının başlangıç değeridir; doğrudan sevkiyat "
                    "emri değildir."
                ),
            },
            {
                "Ekrandaki alan": "Geçmiş talep grafiği",
                "Ne anlatır?": (
                    "Ürünün geçmiş günlük satışlarını ve yedi günlük hareketli "
                    "ortalamasını gösterir."
                ),
                "Nasıl kullanılmalı?": (
                    "Ani sıçrama, uzun sıfır dönemleri ve mevsimsel davranış "
                    "model sonucuyla birlikte kontrol edilmelidir."
                ),
            },
            {
                "Ekrandaki alan": "Grafikte gösterilecek geçmiş",
                "Ne anlatır?": (
                    "Geçmiş satış grafiğinde son 60, 90, 180 veya 365 günün "
                    "hangisinin gösterileceğini belirler."
                ),
                "Nasıl kullanılmalı?": (
                    "Yalnızca grafiğin görüntü aralığını değiştirir; modelin talep "
                    "kararını, olasılığını ve miktar tahminini değiştirmez."
                ),
            },
            {
                "Ekrandaki alan": "Takvim ve hava bağlamı",
                "Ne anlatır?": (
                    "Hafta sonu, tatil, okul dönemi ve Bursa iklim normali gibi "
                    "hedef güne ait açıklayıcı bilgileri gösterir."
                ),
                "Nasıl kullanılmalı?": (
                    "Hava alanı aktif model girdisi değildir; operasyonel bağlam "
                    "olarak gösterilir."
                ),
            },
        ]
    )
    st.dataframe(result_guide, width="stretch", hide_index=True)

    section_title("Basit bir net transfer örneği")
    st.markdown(
        """
        <div class="guide-callout">
          Model seçilen ürün için <strong>18 ADT brüt ihtiyaç</strong> tahmin
          etmiş olsun. Mağazada <strong>6 ADT stok</strong>, yolda
          <strong>4 ADT sevkiyat</strong> ve korunmak istenen
          <strong>3 ADT emniyet stoku</strong> varsa:
          <br><br>
          <strong>Net transfer = 18 − 6 − 4 + 3 = 11 ADT</strong>
          <br>
          Gerçek sipariş kararı verilirken raf kapasitesi, minimum koli miktarı
          ve ürünün son kullanma süresi gibi işletme kuralları da kontrol edilmelidir.
        </div>
        """,
        unsafe_allow_html=True,
    )

    section_title("Model Merkezi: sade anlatım")
    model_intro_left, model_intro_right = st.columns(2)
    with model_intro_left:
        st.markdown(
            """
            <div class="context-card" style="min-height:155px">
              <div class="context-icon">M</div>
              <div class="context-title">Bu ekran ne işe yarar?</div>
              <div class="context-copy">
                Model Merkezi ürün tahmini üretmez. Yayında hangi modelin
                bulunduğunu, modelin test döneminde nasıl performans gösterdiğini,
                hangi veri alanlarını kullandığını ve sınırlarını denetlemek için
                kullanılır.
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with model_intro_right:
        st.markdown(
            """
            <div class="context-card" style="min-height:155px">
              <div class="context-icon">i</div>
              <div class="context-title">Kim, ne zaman bakmalı?</div>
              <div class="context-copy">
                Mağaza yöneticisi model sonucunun neden temkinli kullanılması
                gerektiğini anlamak için; veri bilimi ve operasyon ekipleri ise
                sürüm, performans ve değişken kararlarını kontrol etmek için bu
                ekranı kullanmalıdır.
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    metric_guide = pd.DataFrame(
        [
            {
                "Model Merkezi alanı": "Model sürümü",
                "Sade açıklama": (
                    "Şu anda tahmin üretmekte olan model paketinin kimliğidir. "
                    "Tahmin kayıtları karşılaştırılırken aynı sürüm kontrol edilir."
                ),
            },
            {
                "Model Merkezi alanı": "PR-AUC",
                "Sade açıklama": (
                    "Talebin seyrek olduğu ürünlerde modelin talep olan günleri "
                    "talep olmayan günlerden ne kadar iyi ayırdığını özetler. "
                    "Yüksek değer daha iyidir."
                ),
            },
            {
                "Model Merkezi alanı": "Precision (kesinlik)",
                "Sade açıklama": (
                    "Modelin “talep var” dediği günlerin ne kadarında gerçekten "
                    "talep oluştuğunu gösterir. Yüksek olması gereksiz sevkiyat "
                    "riskinin daha düşük olduğunu düşündürür."
                ),
            },
            {
                "Model Merkezi alanı": "Recall (yakalama oranı)",
                "Sade açıklama": (
                    "Gerçekte talep olan günlerin ne kadarını modelin yakaladığını "
                    "gösterir. Yüksek olması talebi kaçırma riskinin daha düşük "
                    "olduğunu düşündürür."
                ),
            },
            {
                "Model Merkezi alanı": "F1",
                "Sade açıklama": (
                    "Kesinlik ile yakalama oranını tek değerde dengeler. Tek başına "
                    "değil, diğer metriklerle birlikte değerlendirilir."
                ),
            },
            {
                "Model Merkezi alanı": "Miktar MAE",
                "Sade açıklama": (
                    "Tahmin edilen miktarın gerçek miktardan ortalama kaç KG/ADT "
                    "saptığını gösterir. Düşük değer daha iyidir."
                ),
            },
            {
                "Model Merkezi alanı": "WAPE",
                "Sade açıklama": (
                    "Toplam miktar hatasını toplam gerçek talebe oranlar. Farklı "
                    "ölçeklerdeki ürün gruplarını karşılaştırmaya yardımcı olur; "
                    "düşük değer daha iyidir."
                ),
            },
            {
                "Model Merkezi alanı": "Bias (yönlü hata)",
                "Sade açıklama": (
                    "Modelin miktarı sürekli fazla mı yoksa eksik mi tahmin etmeye "
                    "eğilimli olduğunu gösterir. Sıfıra yakın değer tercih edilir."
                ),
            },
            {
                "Model Merkezi alanı": "Bilinen sınırlılıklar",
                "Sade açıklama": (
                    "Modelin kullanmadığı bilgiler ile tahminin hangi durumlarda "
                    "operasyon kontrolüne daha fazla ihtiyaç duyduğunu açıklar."
                ),
            },
        ]
    )
    st.dataframe(metric_guide, width="stretch", hide_index=True)

    section_title("Sık yapılan hatalar")
    mistakes = [
        (
            "Olasılığı kesinlik sanmak",
            "Talep olasılığı kalibre edilmiş bir güven yüzdesi değildir.",
        ),
        (
            "Brüt ihtiyacı doğrudan sipariş etmek",
            "Stok, yoldaki sevkiyat ve emniyet stoku uygulanmadan sipariş verilmemelidir.",
        ),
        (
            "KG ile ADT'yi birbirine çevirmek",
            "Sistem katalog birimini korur; KG sonucu kilogram, ADT sonucu adettir.",
        ),
        (
            "Uzak tahmini güncellememek",
            "Hedef gün yaklaştığında tahmin yeniden oluşturulmalı ve son stok görülmelidir.",
        ),
    ]
    mistake_columns = st.columns(4)
    for column, (title, copy) in zip(mistake_columns, mistakes):
        with column:
            st.markdown(
                f"""
                <div class="context-card" style="min-height:145px">
                  <div class="context-title">{html.escape(title)}</div>
                  <div class="context-copy">{html.escape(copy)}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown(
        """
        <div class="notice">
          <strong style="color:#FFFFFF">Önemli:</strong> Bu sistem karar destek
          aracıdır. Nihai depo transferi; mevcut mağaza stoku, açık sipariş,
          sevkiyat süresi, raf kapasitesi ve emniyet stoku birlikte değerlendirilerek
          verilmelidir.
        </div>
        """,
        unsafe_allow_html=True,
    )


try:
    service = get_service()
except Exception as error:
    st.error(f"Model paketi yüklenemedi: {error}")
    st.stop()

metadata = service.metadata
origin = pd.Timestamp(metadata["forecast_origin"]).date()
max_date = origin + timedelta(days=int(metadata["max_forecast_lead_days"]))

with st.sidebar:
    st.markdown(
        """
        <div class="brand">
          <div class="brand-mark">Ö</div>
          <div>
            <div class="brand-name">ÖZDİLEK</div>
            <div class="brand-product">TALEP PLANLAMA PLATFORMU</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown('<div class="sidebar-label">MODÜLLER</div>', unsafe_allow_html=True)
    navigation = st.radio(
        "Modüller",
        [
            "◉  Talep Planlama",
            "▦  Toplu Planlama",
            "◫  Model Merkezi",
            "⌁  Tahmin İzleme",
            "?  Kullanım Rehberi",
        ],
        label_visibility="collapsed",
    )
    st.divider()
    st.markdown(
        f"""
        <div class="system-card">
          <div class="label">Sistem durumu</div>
          <div class="value"><span class="online-dot"></span>Model servisi aktif</div>
        </div>
        <div class="system-card">
          <div class="label">Aktif model</div>
          <div class="value">{html.escape(metadata['model_version'])}</div>
        </div>
        <div class="system-card">
          <div class="label">Veri kesim tarihi</div>
          <div class="value">{origin}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.caption("İç kullanım · Bursa mağaza operasyonları")

if navigation.startswith("◉"):
    render_forecast_page(service, metadata, origin, max_date)
elif navigation.startswith("▦"):
    render_batch_page(service)
elif navigation.startswith("◫"):
    render_model_page(metadata)
elif navigation.startswith("⌁"):
    render_monitoring_page()
else:
    render_help_page(origin, max_date)
