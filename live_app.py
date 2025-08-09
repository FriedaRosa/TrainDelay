# app.py
from __future__ import annotations

import os
import time
import datetime
import pandas as pd

from shiny import App, ui, render, reactive
from dotenv import load_dotenv
from deutsche_bahn_api import StationHelper, TimetableHelper, ApiAuthentication

# Plotly + shinywidgets
from shinywidgets import render_plotly, output_widget
import plotly.express as px
import plotly.graph_objects as go

# ---------- auth ----------
load_dotenv()
CLIENT_ID = os.getenv("DB_CLIENT_ID")
CLIENT_SECRET = os.getenv("DB_CLIENT_SECRET")
if not CLIENT_ID or not CLIENT_SECRET:
    raise RuntimeError("Missing DB_CLIENT_ID / DB_CLIENT_SECRET in .env")

AUTH = ApiAuthentication(CLIENT_ID, CLIENT_SECRET)

# ---------- helpers ----------
def parse_db_time(val):
    if val is None or (isinstance(val, str) and not val.strip()):
        return pd.NaT
    if isinstance(val, (datetime.datetime, pd.Timestamp)):
        return pd.to_datetime(val)
    s = str(val)
    try:
        return pd.to_datetime(datetime.datetime.strptime(s, "%y%m%d%H%M"))
    except Exception:
        return pd.to_datetime(s, errors="coerce")

def compute_delay(planned, current):
    p = parse_db_time(planned)
    c = parse_db_time(current)
    if pd.isna(p) or pd.isna(c):
        return 0, 0
    diff_min = (c - p).total_seconds() / 60.0
    delay_minutes = int(round(diff_min))
    return (1 if delay_minutes > 0 else 0), delay_minutes

def fetch_station_df(station_name: str) -> pd.DataFrame:
    try:
        sh = StationHelper()
        matches = sh.find_stations_by_name(station_name)
        if not matches:
            return pd.DataFrame()

        th = TimetableHelper(matches[0], AUTH)
        timetable = th.get_timetable()
        trains = th.get_timetable_changes(timetable)

        rows = []
        request_time = datetime.datetime.now()
        for t in trains:
            line = f"{t.train_type}{getattr(t, 'train_line', '')}"
            train_id = t.train_number
            first_station = getattr(t, "passed_stations", None)
            first_station = (first_station or t.stations).split("|")[0]
            last_station = t.stations.split("|")[-1]
            planned_departure = t.departure
            current_departure = getattr(t.train_changes, "departure", None)
            track = t.platform

            msg_parts = []
            for m in t.train_changes.messages:
                msg = getattr(m, "message", None)
                if msg:
                    msg_parts.append(str(msg))
            message = " ".join(msg_parts) if msg_parts else "No message"

            is_delayed, delayed_minutes = compute_delay(planned_departure, current_departure)

            rows.append(
                {
                    "request_time": request_time,
                    "line": line,
                    "train_id": train_id,
                    "first_station": first_station,
                    "last_station": last_station,
                    "planned_departure": parse_db_time(planned_departure),
                    "current_departure": parse_db_time(current_departure),
                    "track": track,
                    "message": message,
                    "train_station": matches[0][3],
                    "is_delayed": is_delayed,
                    "delayed_minutes": delayed_minutes,
                }
            )
        return pd.DataFrame(rows)
    except Exception as e:
        return pd.DataFrame({"error": [str(e)]})

def empty_fig(title: str, message: str = "No data for selected filters") -> go.Figure:
    fig = go.Figure()
    fig.update_layout(
        title=title,
        xaxis_visible=False,
        yaxis_visible=False,
        annotations=[dict(text=message, x=0.5, y=0.5, xref="paper", yref="paper", showarrow=False)],
        margin=dict(l=10, r=10, t=60, b=10),
    )
    return fig

# ---------- UI ----------
app_ui = ui.page_fluid(
    ui.h3("Live Deutsche Bahn delays (Shiny for Python)"),
    ui.row(
        ui.column(
            4,
            ui.input_text("station", "Station", value="Köln Hbf", placeholder="Type a station"),
            ui.input_select("line", "Line", choices=["All"], selected="All"),  # <-- Line dropdown
            ui.input_switch("auto", "Auto-refresh", value=True),
            ui.input_slider("secs", "Refresh every (seconds)", min=100, max=2000, value=300, step=5),
            ui.input_action_button("refresh", "Refresh now"),
        ),
        ui.column(
            8,
            ui.card(
                ui.card_header("Last updated"),
                ui.output_text("updated"),
            ),
            ui.card(
                ui.card_header("Live trains (current hour)"),
                ui.output_data_frame("table"),
                style="height:40vh; overflow:auto;",
            ),
        ),
    ),
    ui.hr(),
    ui.row(
        ui.column(4, ui.card(ui.card_header("On-time vs Delayed"), output_widget("delay_plot"))),
        ui.column(4, ui.card(ui.card_header("Delayed by Line"), output_widget("line_delay_plot"))),
        ui.column(4, ui.card(ui.card_header("Delay Minutes per Hour"), output_widget("delay_timeline_plot"))),
    ),
    ui.hr(),
    ui.p(
        "Tip: Reduce refresh frequency to respect API rate limits. "
        "This demo pulls the current hour only; continuous refresh builds a transient 'live' view."
    ),
)

# ---------- Server ----------
def server(input, output, session):
    # reactive ticker
    @reactive.calc
    def tick():
        if input.auto():
            reactive.invalidate_later(int(input.secs()) * 1000)
        _ = input.refresh()
        return time.time()

    @reactive.calc
    def data():
        _ = tick()
        st = input.station().strip()
        if not st:
            return pd.DataFrame()
        return fetch_station_df(st)

    # update line choices whenever fresh data arrives
    @reactive.effect
    def _update_line_choices():
        df = data()
        if df.empty or "line" not in df.columns or "error" in df.columns:
            ui.update_select("line", choices=["All"], selected="All")
            return
        lines = sorted([x for x in df["line"].dropna().unique().tolist() if str(x).strip()])
        choices = ["All"] + lines
        # keep current selection if still valid, else "All"
        sel = input.line()
        selected = sel if sel in choices else "All"
        ui.update_select("line", choices=choices, selected=selected)

    # filtered view by selected line
    @reactive.calc
    def filtered():
        df = data()
        sel = input.line()
        if df.empty or "error" in df.columns:
            return df
        if sel and sel != "All" and "line" in df.columns:
            return df[df["line"] == sel]
        return df

    @render.text
    def updated():
        df = data()
        if df.empty:
            return "No data (yet)."
        if "error" in df.columns:
            return f"Error: {df.loc[0,'error']}"
        ts = df["request_time"].max()
        return f"{ts.strftime('%Y-%m-%d %H:%M:%S')}"

    @render.data_frame
    def table():
        df = filtered()
        if "error" in df.columns:
            return df
        cols = [
            "request_time",
            "line",
            "train_id",
            "first_station",
            "last_station",
            "planned_departure",
            "current_departure",
            "track",
            "message",
            "train_station",
            "is_delayed",
            "delayed_minutes",
        ]
        existing = [c for c in cols if c in df.columns]
        if not existing:
            return pd.DataFrame({"Message": ["No columns to display."]})
        return df[existing].sort_values(["planned_departure", "line", "train_id"])

    # ---- Plotly charts (use filtered() so they respect the Line dropdown) ----
    @render_plotly
    def delay_plot():
        df = filtered()
        if df.empty or "is_delayed" not in df.columns:
            return empty_fig("Trains Delayed vs On Time")
        d = df[["is_delayed"]].copy()
        d["status"] = d["is_delayed"].map({0: "On Time", 1: "Delayed"}).fillna("Unknown")
        counts = d["status"].value_counts().reindex(["On Time", "Delayed", "Unknown"], fill_value=0).reset_index()
        counts.columns = ["Status", "Count"]
        fig = px.bar(counts, x="Status", y="Count", title="Trains Delayed vs On Time")
        fig.update_layout(yaxis_title="Number of Trains", xaxis_title="", margin=dict(l=10, r=10, t=60, b=10))
        return fig

    @render_plotly
    def line_delay_plot():
        # For this chart we want all lines (ignoring the line filter), but for the current station/hour
        df = data()
        if df.empty or "is_delayed" not in df.columns:
            return empty_fig("Delayed Trains by Line")
        d = df[df["is_delayed"] == 1].copy()
        if d.empty:
            return empty_fig("Delayed Trains by Line", "No delayed trains in view")
        counts = (
            d.groupby("line", dropna=False)
             .size()
             .reset_index(name="delay_count")
             .sort_values("delay_count", ascending=False)
        )
        fig = px.bar(counts, x="delay_count", y="line", orientation="h", title="Delayed Trains by Line")
        fig.update_layout(xaxis_title="Number of Delays", yaxis_title="", bargap=0.2, margin=dict(l=10, r=10, t=60, b=10))
        fig.update_yaxes(autorange="reversed")
        return fig

    @render_plotly
    def delay_timeline_plot():
        df = filtered()
        if df.empty:
            return empty_fig("Total Delay Minutes Per Hour", "No data in view")

        needed = {"planned_departure", "delayed_minutes", "is_delayed"}
        if not needed.issubset(df.columns):
            return empty_fig("Total Delay Minutes Per Hour", "Missing required columns")

        d = df.copy()
        d["planned_departure"] = pd.to_datetime(d["planned_departure"], errors="coerce")
        d["delayed_minutes"] = pd.to_numeric(d["delayed_minutes"], errors="coerce")
        d = d.dropna(subset=["planned_departure", "delayed_minutes"])
        d = d[d["is_delayed"] == 1]
        if d.empty:
            return empty_fig("Total Delay Minutes Per Hour", "No delayed trains in view")

        # Hourly rollup
        d["hour"] = d["planned_departure"].dt.floor("h")
        summary = d.groupby("hour", as_index=False)["delayed_minutes"].sum().sort_values("hour")

        # Convert datetime64[ns] -> datetime64[ms] to avoid huge ns numbers on x-axis
        try:
            # drop tz if present (won't error if naive with this guard)
            summary["hour"] = summary["hour"].dt.tz_localize(None)
        except Exception:
            pass
        summary["hour_ms"] = summary["hour"].astype("datetime64[ms]")

        fig = px.line(
            summary,
            x="hour_ms",
            y="delayed_minutes",
            markers=True,
            title="Total Delay Minutes Per Hour",
        )
        fig.update_xaxes(tickformat="%H:%M", hoverformat="%Y-%m-%d %H:%M")
        fig.update_layout(
            xaxis_title="Hour",
            yaxis_title="Total Delay Minutes",
            margin=dict(l=10, r=10, t=60, b=10),
            yaxis=dict(rangemode="tozero"),
        )
        return fig


app = App(app_ui, server)
