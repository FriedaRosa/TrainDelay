# app.py
from __future__ import annotations

import os
import time
import datetime
import pandas as pd

from shiny import App, ui, render, reactive
from dotenv import load_dotenv
from deutsche_bahn_api import StationHelper, TimetableHelper, ApiAuthentication

# ---------- auth ----------
load_dotenv()
CLIENT_ID = os.getenv("DB_CLIENT_ID")
CLIENT_SECRET = os.getenv("DB_CLIENT_SECRET")
if not CLIENT_ID or not CLIENT_SECRET:
    raise RuntimeError("Missing DB_CLIENT_ID / DB_CLIENT_SECRET in .env")

AUTH = ApiAuthentication(CLIENT_ID, CLIENT_SECRET)

# ---------- helpers ----------
def parse_db_time(val):
    """
    DB format often 'yymmddHHMM' (e.g., 2508081549), but be permissive.
    Returns pandas.Timestamp (NaT on failure).
    """
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
    """
    Positive delayed_minutes = late, negative = early, 0 = on time.
    Returns (is_delayed:int, delayed_minutes:int) or (0,0) if unknown.
    """
    p = parse_db_time(planned)
    c = parse_db_time(current)
    if pd.isna(p) or pd.isna(c):
        return 0, 0
    diff_min = (c - p).total_seconds() / 60.0
    delay_minutes = int(round(diff_min))
    return (1 if delay_minutes > 0 else 0), delay_minutes

def fetch_station_df(station_name: str) -> pd.DataFrame:
    """Fetch current-hour changes for a station and return a tidy DataFrame."""
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

            # message list -> joined string
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
                    "train_station": matches[0][3],  # station name at idx 3 in wrapper
                    "is_delayed": is_delayed,
                    "delayed_minutes": delayed_minutes,
                }
            )
        return pd.DataFrame(rows)
    except Exception as e:
        # Return a small DF with the error so the UI can show it
        return pd.DataFrame({"error": [str(e)]})

# ---------- UI ----------
app_ui = ui.page_fluid(
    ui.h3("Live Deutsche Bahn delays (Shiny for Python)"),
    ui.row(
        ui.column(
            4,
            ui.input_text("station", "Station", value="Köln Hbf", placeholder="Type a station"),
            ui.input_switch("auto", "Auto-refresh", value=True),
            ui.input_slider("secs", "Refresh every (seconds)", min=10, max=120, value=30, step=5),
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
                style="height:60vh; overflow:auto;",  # scrollable area
            ),
        ),
    ),
    ui.p(
        "Tip: Reduce refresh frequency to respect API rate limits. "
        "This demo pulls the current hour only; polling regularly builds a history "
        "client-side if you store it."
    ),
)


# ---------- Server ----------
def server(input, output, session):
    # A reactive "ticker" that invalidates on a timer and when the button is clicked
    @reactive.calc
    def tick():
        # timer part
        if input.auto():
            reactive.invalidate_later(int(input.secs()) * 1000)
        # button part: depend on its value (increments each click)
        _ = input.refresh()
        return time.time()

    @reactive.calc
    def data():
        _ = tick()  # depend on the ticker
        st = input.station().strip()
        if not st:
            return pd.DataFrame()
        return fetch_station_df(st)

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
        df = data()
        if "error" in df.columns:
            # show the error as a 1-row table
            return df
        # nice column order
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
        return df[existing].sort_values(["planned_departure", "line", "train_id"])

app = App(app_ui, server)
