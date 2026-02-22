from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from branca.colormap import linear
from ipyleaflet import GeoJSON, Map, basemaps
from shiny import App, reactive, render, ui
from shinywidgets import output_widget, render_widget
import ipywidgets as widgets
from ipyleaflet import WidgetControl


HERE = Path(__file__).parent
DATA_PATH = HERE / "NL_crim_df.csv"

if not DATA_PATH.exists():
    raise FileNotFoundError(
        f"Could not find {DATA_PATH}. Put NL_crim_df.csv in the same folder as app.py."
    )

raw = pd.read_csv(DATA_PATH)

raw = raw.rename(columns={"geo\\TIME_PERIOD": "geo"})
raw = raw.drop(columns=[c for c in ["unit", "freq"] if c in raw.columns], errors="ignore")


year_cols = [c for c in raw.columns if re.fullmatch(r"\d{4}", str(c))]
years = sorted(int(y) for y in year_cols)

# Long format
long = raw.melt(
    id_vars=["geo", "code_indic", "code_name"],
    value_vars=[str(y) for y in years],
    var_name="year",
    value_name="value",
)
long["year"] = long["year"].astype(int)
long["value"] = pd.to_numeric(long["value"], errors="coerce")


wide = (
    long.pivot_table(
        index=["geo", "year"],
        columns="code_indic",
        values="value",
        aggfunc="first",
    )
    .reset_index()
)

wide["occupancy_pct"] = (wide["PRIS_ACT_CAP"] / wide["PRIS_OFF_CAP"]) * 100

wide["personnel_total"] = wide[["PRISA", "PRISJ"]].sum(axis=1, min_count=1)

wide["personnel_per_inmate"] = wide["personnel_total"] / wide["PRIS_ACT_CAP"]

metrics = wide.set_index(["geo", "year"]).sort_index()

occ_by_year: dict[int, dict[str, float]] = {}
for y in years:
    sub = wide[wide["year"] == y][["geo", "occupancy_pct"]].copy()
    occ_by_year[y] = dict(zip(sub["geo"], sub["occupancy_pct"]))

occ_all = wide["occupancy_pct"].dropna()
if len(occ_all) == 0:
    occ_min, occ_max = 0.0, 100.0
else:
    occ_min = float(occ_all.min())
    occ_max = float(occ_all.max())
    pad = max(1.0, 0.05 * (occ_max - occ_min))
    occ_min, occ_max = max(0.0, occ_min - pad), occ_max + pad

cmap = linear.YlOrRd_09.scale(occ_min, occ_max)
cmap.caption = "Prison occupancy (%) = (no. of inmates) / (official capacity) × 100"

COUNTRIES = ["NL", "DE", "FR", "IT"]


EUROPE_GEOJSON_URL = (
    "https://raw.githubusercontent.com/jupyter-widgets/ipyleaflet/master/examples/europe_110.geo.json"
)

_geojson_cache = None


def get_europe_geojson() -> dict:
    global _geojson_cache
    if _geojson_cache is None:
        resp = requests.get(EUROPE_GEOJSON_URL, timeout=10)
        resp.raise_for_status()
        _geojson_cache = resp.json()
    return _geojson_cache


def filter_geojson_to_countries(geojson: dict, keep_iso2: set[str]) -> tuple[dict, dict[str, str]]:
    """Return (filtered_geojson, iso2_to_name)."""
    feats = geojson.get("features", [])
    kept = []
    iso2_to_name: dict[str, str] = {}
    for f in feats:
        props = f.get("properties", {})
        iso2 = props.get("iso_a2")
        if iso2 in keep_iso2:
            kept.append(f)
            iso2_to_name[iso2] = props.get("name", iso2)
    return {"type": "FeatureCollection", "features": kept}, iso2_to_name



# Shiny UI
app_ui = ui.page_sidebar(
    ui.sidebar(
        ui.h3("Controls"),
        ui.input_slider(
            "year",
            "Year",
            min(years),
            max(years),
            value=max(years),
            step=1,
            sep="",
        ),
        ui.hr(),
        ui.h4("Prison Occupancy (%)"),
        ui.output_ui("hover_info"),
        ui.hr(),
        ui.output_ui("details"),
        ui.hr(),


    ),
    ui.card(
    ui.card_header(
        ui.tags.div(
            "Europe (NL, DE, FR, IT) - Comparison of prison occupancy and personnel",
            ui.tags.br(),
            ui.tags.small("(click a country)")
        )
    ),
        output_widget("map", height="650px"),
    ),
)


# -----------------------
# Shiny server
# -----------------------
def server(input, output, session):
    selected_geo = reactive.Value(None)  
    geo_error = reactive.Value(None)     
    iso2_to_name = {"NL": "Netherlands", "DE": "Germany", "FR": "France", "IT": "Italy"}

    m = Map(basemap=basemaps.CartoDB.PositronNoLabels, center=(48.8, 10.0), zoom=4)
    def colormap_svg_percent(cm):
        width = 320          
        bar_y1, bar_y2 = 14, 22
        font = 9             
        caption_font = 11
        svg_h = 40
        nb_ticks = 7
        x_ticks = [int(i * width / (nb_ticks - 1)) for i in range(nb_ticks)]
        val_ticks = [cm.vmin + i * (cm.vmax - cm.vmin) / (nb_ticks - 1) for i in range(nb_ticks)]

        svg = [f'<svg height="{svg_h}" width="{width}">']
        svg.append("".join(
            f'<line x1="{i}" y1="{bar_y1}" x2="{i}" y2="{bar_y2}" style="stroke:{cm.rgba_hex_str(cm.vmin + (cm.vmax-cm.vmin)*i/(width-1))};stroke-width:2;" />'
            for i in range(width)
        ))
        for i, (x, v) in enumerate(zip(x_ticks, val_ticks)):
            anchor = "start" if i == 0 else ("end" if i == nb_ticks - 1 else "middle")
            svg.append(
                f'<text x="{x}" y="{svg_h-2}" style="text-anchor:{anchor}; font-size:{font}px; font:Arial; fill:{cm.text_color}">{v:.0f}%</text>'
            )
        svg.append(
            f'<text x="0" y="10" style="font-size:{caption_font}px; font:Arial; fill:{cm.text_color}">{cm.caption}</text>'
        )
        svg.append("</svg>")
        return "".join(svg)

    legend = widgets.HTML(value=colormap_svg_percent(cmap))
    m.add_control(WidgetControl(widget=legend, position="topright"))

    geo_layer = GeoJSON(
        data={"type": "FeatureCollection", "features": []},
        style={"opacity": 1, "weight": 1, "fillOpacity": 0.8},
        hover_style={"weight": 2, "fillOpacity": 0.9},
    )
    m.add_layer(geo_layer)


    def handle_click(feature, **kwargs):
        props = feature.get("properties", {})
        iso2 = props.get("iso_a2")
        if iso2 in COUNTRIES:
            selected_geo.set(iso2)

    geo_layer.on_click(handle_click)
    hover_text = reactive.Value("Hover over a country to see occupancy %.")

    def handle_hover(feature, **kwargs):
        props = feature.get("properties", {})
        iso2 = props.get("iso_a2")
        val = props.get("occ_pct")  

        if iso2 is None:
            return

        if val is None:
            hover_text.set(f"{iso2}: No data")
        else:
            hover_text.set(f"{iso2}: {val:.1f}% occupancy")

    geo_layer.on_hover(handle_hover)

    @render_widget
    def map():
        return m

    # Load GeoJSON 
    filtered_geojson = {"type": "FeatureCollection", "features": []}

    @reactive.Effect
    def _load_geojson_once():
        nonlocal filtered_geojson, iso2_to_name
        if geo_layer.data.get("features"):
            return  
        try:
            gj = get_europe_geojson()
            filtered_geojson, iso2_to_name = filter_geojson_to_countries(gj, set(COUNTRIES))
            geo_layer.data = filtered_geojson
            geo_error.set(None)
        except Exception as e:
            geo_error.set(str(e))

    # Update styles when year or selected country changes
    @reactive.Effect
    def _update_styles():
        year = int(input.year())
        sel = selected_geo.get()

        year_vals = occ_by_year.get(year, {})
        data = geo_layer.data
        for f in data.get("features", []):
            iso2 = f.get("properties", {}).get("iso_a2")
            v = year_vals.get(iso2, np.nan)
            f["properties"]["occ_pct"] = None if pd.isna(v) else float(v)
        geo_layer.data = data  # triggers redraw with updated properties

        def style_callback(feature):
            props = feature.get("properties", {})
            iso2 = props.get("iso_a2")

            val = feature.get("properties", {}).get("occ_pct")
            has_data = val is not None
            fill = cmap(val) if has_data else "#d9d9d9"
            has_data = (val is not None) and (not (isinstance(val, float) and np.isnan(val)))

            fill = cmap(val) if has_data else "#d9d9d9"  
            weight = 3 if (sel is not None and iso2 == sel) else 1

            return {
                "color": "black",
                "weight": weight,
                "fillColor": fill,
                "fillOpacity": 0.8,
            }

        geo_layer.style_callback = style_callback

        # Re-assigning data triggers redraw without changing the geometry.
        if geo_layer.data.get("features"):
            geo_layer.data = geo_layer.data

    # Details panel
    def fmt_num(x, decimals=0):
        if x is None or (isinstance(x, float) and np.isnan(x)):
            return "No data"
        if decimals == 0:
            return f"{int(round(float(x))):,}"
        return f"{float(x):,.{decimals}f}"
    @render.ui
    def hover_info():
        return ui.p(hover_text.get())

    @render.ui
    def details():
        err = geo_error.get()
        if err:
            return ui.div(
                ui.p("GeoJSON failed to load (network issue)."),
                ui.tags.code(err),
                ui.p("You can still run the app, but the map layer won’t display until the URL is reachable."),
            )

        sel = selected_geo.get()
        year = int(input.year())

        if sel is None:
            return ui.p("Click on the map to see prison personnel to inmate ratio and bribery details.")

        name = iso2_to_name.get(sel, sel)

        if (sel, year) not in metrics.index:
            return ui.div(
                ui.h4(f"{name} ({sel})"),
                ui.p(f"No row found for year {year}."),
            )

        row = metrics.loc[(sel, year)]

        inmates = row.get("PRIS_ACT_CAP", np.nan)
        cap = row.get("PRIS_OFF_CAP", np.nan)
        occ = row.get("occupancy_pct", np.nan)

        staff_total = row.get("personnel_total", np.nan)
        staff_per_inmate = row.get("personnel_per_inmate", np.nan)

        convicted = row.get("PER_CNV", np.nan)

        return ui.div(
            ui.h4(f"{name} {year}"),
            ui.tags.ul(
                ui.tags.li(ui.tags.strong("Inmates: "), fmt_num(inmates)),
                ui.tags.li(ui.tags.strong("Official capacity: "), fmt_num(cap)),
                ui.tags.li(ui.tags.strong("Occupancy: "), "No data" if np.isnan(occ) else f"{occ:.1f}%"),
            ),
            ui.hr(),
            ui.tags.ul(
                ui.tags.li(ui.tags.strong("Total prison employees: "), fmt_num(staff_total)),
                ui.tags.li(
                    ui.tags.strong("No. of employees per inmate: "),
                    fmt_num(staff_per_inmate, decimals=3),
                ),
            ),
            ui.hr(),
            ui.tags.ul(
                ui.tags.li(ui.tags.strong("Convicted bribery cases: "), fmt_num(convicted)),
            ),
        )

    return


app = App(app_ui, server)