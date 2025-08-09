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

from pathlib import Path
import io

EXPORT_DIR = Path(__file__).resolve().parent / "exports"
EXPORT_DIR.mkdir(parents=True, exist_ok=True)

def _slugify(text: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in text).strip("_").lower()

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
            line_last = f"{line} {last_station}"  # merged column

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
                    "line": line,                 # <-- keep line
                    "last_station": last_station, # <-- (optional) keep last_station
                    "line_last": line_last,       # <-- merged column
                    "train_id": train_id,
                    "first_station": first_station,
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
        print("[fetch_station_df] ERROR:", e)
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
    ui.h3("🚆 Live Deutsche Bahn delays"),
    ui.row(
        ui.column(
            4,
            ui.input_text("station", "Station", value="Köln Hbf", placeholder="Type a station"),
            ui.input_select("line", "Line", choices=["All"], selected="All"),
            ui.input_switch("auto", "Auto-refresh", value=True),
            # hard minimum visually = 180 seconds (3 minutes)
            ui.input_slider("secs", "Refresh every (seconds)", min=180, max=2000, value=300, step=10),
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
    ui.row(
        ui.column(
            12,
            ui.download_button("download_csv", "Save & Download CSV (append if exists)"),
        )
    ),
    ui.hr(),
    ui.row(
        ui.column(4, ui.card(ui.card_header("On-time vs Delayed"), output_widget("delay_plot"))),
        ui.column(4, ui.card(ui.card_header("Delayed by Line"), output_widget("line_delay_plot"))),
        ui.column(4, ui.card(ui.card_header("Delay Minutes per Hour"), output_widget("delay_timeline_plot"))),
    ),
    ui.hr(),
    ui.card(
        ui.card_header("Debug"),
        ui.row(
            ui.column(4, ui.output_text("dbg_fetch")),
            ui.column(4, ui.output_text("dbg_history")),
            ui.column(4, ui.output_text("dbg_error")),
        ),
    ),
)

# ---------- Server ----------
def server(input, output, session):
    # --- state ---
    history = reactive.Value(pd.DataFrame())
    _line_choices_cache = reactive.Value(("All",))
    last_error = reactive.Value("")
    fetching = reactive.Value(False)          # lock to prevent overlap
    last_fetch_monotonic = reactive.Value(0)  # seconds from time.monotonic()

    # --- tuning knobs ---
    WINDOW_HOURS = 12
    MAX_ROWS = 5000
    DEDUPE_KEYS = ["train_station", "train_id", "planned_departure"]
    MIN_REFRESH_SEC = 180  # hard floor = 3 minutes

    # Shared fetch routine (force=True bypasses the throttle for manual refresh)
    def _do_fetch_and_merge(station_name: str, *, force: bool = False):
        now = time.monotonic()

        # HARD THROTTLE: never fetch more often than MIN_REFRESH_SEC in AUTO mode
        if not force:
            elapsed = now - (last_fetch_monotonic() or 0)
            if elapsed < MIN_REFRESH_SEC:
                # Too soon since last fetch; skip silently (or log if you want)
                # print(f"[history] skip: throttled ({elapsed:.1f}s since last)")
                return

        if fetching():
            return  # another fetch is still running

        fetching.set(True)
        try:
            df_new = fetch_station_df(station_name)
            # record a fetch attempt time regardless of outcome (prevents hammering on errors)
            last_fetch_monotonic.set(now)

            if df_new.empty:
                last_error.set("Fetch returned 0 rows")
                print("[history] fetched 0 rows")
                return

            if "error" in df_new.columns:
                err = str(df_new.loc[0, "error"])
                last_error.set(err)
                print("[history] fetch error:", err)
                return

            # Coerce only new batch
            for col in ("planned_departure", "current_departure", "request_time"):
                if col in df_new.columns:
                    df_new[col] = pd.to_datetime(df_new[col], errors="coerce")
            if "delayed_minutes" in df_new.columns:
                df_new["delayed_minutes"] = pd.to_numeric(df_new["delayed_minutes"], errors="coerce")
            if "line" in df_new.columns:
                df_new["line"] = df_new["line"].astype("category")

            print(f"[history] new rows: {len(df_new)}")

            # Append to history
            df = pd.concat([history(), df_new], ignore_index=True, copy=False)

            # Time window cap (keep NaT rows too)
            if "planned_departure" in df.columns:
                cutoff = pd.Timestamp.now() - pd.Timedelta(hours=WINDOW_HOURS)
                mask = df["planned_departure"].notna() & (df["planned_departure"] >= cutoff)
                df = pd.concat([df[mask], df[df["planned_departure"].isna()]], ignore_index=True)

            # Row cap
            if "request_time" in df.columns and len(df) > MAX_ROWS:
                df = df.nlargest(MAX_ROWS, "request_time")

            # Fast dedupe: keep latest request per key
            if "request_time" in df.columns and all(k in df.columns for k in DEDUPE_KEYS):
                dedupe_keys = DEDUPE_KEYS
                idx = df.groupby(dedupe_keys)["request_time"].idxmax()
                sort_cols = [c for c in ["planned_departure", "line_last", "line", "train_id"] if c in df.columns]
                df = df.loc[idx].sort_values(sort_cols, kind="stable")

            history.set(df)
            last_error.set("")
            print(f"[history] total rows: {len(df)}")
        finally:
            fetching.set(False)

    # -------- AUTO mode: periodic fetch only when auto is ON (hard limited) --------
    @reactive.effect
    def _auto_fetch():
        if not input.auto():
            return
        # schedule next check; even if effect fires earlier, _do_fetch_and_merge will throttle
        interval_ms = max(int(input.secs()), MIN_REFRESH_SEC) * 1000
        reactive.invalidate_later(interval_ms)
        st = input.station().strip()
        if not st:
            return
        _do_fetch_and_merge(st, force=False)  # obey hard throttle

    # -------- MANUAL mode: fetch only when button is clicked (no throttle) --------
    @reactive.effect
    @reactive.event(input.refresh)
    def _manual_fetch():
        st = input.station().strip()
        if not st:
            return
        _do_fetch_and_merge(st, force=True)  # manual overrides throttle

    # Current view = history filtered to station
    @reactive.calc
    def data():
        df = history()
        st = input.station().strip()
        if df.empty or not st or "train_station" not in df.columns:
            return df
        return df[df["train_station"] == st]

    # Line-filtered view
    @reactive.calc
    def filtered():
        df = data()
        sel = input.line()
        if df.empty or "error" in df.columns:
            return df
        if sel and sel != "All" and "line" in df.columns:
            return df[df["line"] == sel]
        return df

    # Keep Line choices in sync with station view
    @reactive.effect
    def _update_line_choices():
        df = data()
        if df.empty or "line" not in df.columns or "error" in df.columns:
            new_choices = ("All",)
        else:
            lines = sorted([x for x in df["line"].dropna().unique().tolist() if str(x).strip()])
            new_choices = tuple(["All"] + lines)

        if new_choices != _line_choices_cache():
            _line_choices_cache.set(new_choices)
            current = input.line()
            selected = current if current in new_choices else "All"
            ui.update_select("line", choices=list(new_choices), selected=selected)

    # Auto-load existing CSV for the selected station into history, then fetch once
    last_loaded_station = reactive.Value("")

    @reactive.effect
    @reactive.event(input.station)
    def _autoload_history():
        st = input.station().strip()
        if not st or st == last_loaded_station():
            return

        station_slug = _slugify(st)
        path = EXPORT_DIR / f"{station_slug}.csv"

        if path.exists():
            try:
                df_old = pd.read_csv(path)
                for col in ("planned_departure", "current_departure", "request_time"):
                    if col in df_old.columns:
                        df_old[col] = pd.to_datetime(df_old[col], errors="coerce")
                if "delayed_minutes" in df_old.columns:
                    df_old["delayed_minutes"] = pd.to_numeric(df_old["delayed_minutes"], errors="coerce")
                # Backfill line_last for legacy files
                if "line_last" not in df_old.columns:
                    if "line" in df_old.columns and "last_station" in df_old.columns:
                        df_old["line_last"] = (df_old["line"].fillna("") + " " + df_old["last_station"].fillna("")).str.strip()
                    elif "line" in df_old.columns:
                        df_old["line_last"] = df_old["line"].astype(str)
                    else:
                        df_old["line_last"] = ""
            except Exception as e:
                print("[autoload] ERROR:", e)
                df_old = pd.DataFrame()
        else:
            df_old = pd.DataFrame()

        # Keep only selected station if column present
        if not df_old.empty and "train_station" in df_old.columns:
            df_old = df_old[df_old["train_station"] == st]

        # MERGE (don’t overwrite): combine CSV with any existing in-memory rows
        df_hist = history()
        merged = pd.concat([df_old, df_hist], ignore_index=True)

        # Dedupe: latest request per (station, train, planned)
        keys = [c for c in ["train_station", "train_id", "planned_departure"] if c in merged.columns]
        if keys and "request_time" in merged.columns:
            idx = merged.groupby(keys)["request_time"].idxmax()
            merged = merged.loc[idx]

        # Stable sort (works with line_last or line)
        sort_cols = [c for c in ["planned_departure", "line_last", "line", "train_id"] if c in merged.columns]
        if sort_cols:
            merged = merged.sort_values(sort_cols, kind="stable")

        history.set(merged)
        last_loaded_station.set(st)
        print(f"[autoload] loaded+merged {len(merged)} rows for station: {st}")

        # Trigger one immediate fetch to ensure “Last updated” reflects live data
        _do_fetch_and_merge(st, force=True)



    # ---- Outputs ----
    @render.text
    def updated():
        df = data()
        if "error" in df.columns and not df.empty:
            return df.loc[0, "error"]
        if df.empty:
            return "No data (yet)."
        ts = df["request_time"].max() if "request_time" in df.columns else None
        if pd.isna(ts):
            # fallback to now so it doesn’t look frozen
            ts = pd.Timestamp.now()
        return f"{pd.to_datetime(ts).strftime('%Y-%m-%d %H:%M:%S')}"


    @render.data_frame
    def table():
        df = filtered()
        if "error" in df.columns:
            return df
        cols = [
            "request_time", "line_last", "train_id", "first_station",
            "planned_departure", "current_departure", "track", "message",
            "train_station", "is_delayed", "delayed_minutes",
        ]
        existing = [c for c in cols if c in df.columns]
        if not existing:
            return pd.DataFrame({"Message": ["No columns to display."]})
        return df[existing].sort_values(["planned_departure", "line_last", "train_id"])


    @render.download(filename=lambda: f"{_slugify(input.station() or 'station')}.csv")
    def download_csv():
        # Use full station history (ignores the Line dropdown)
        df = data()

        if df.empty:
            buf = io.StringIO()
            pd.DataFrame(columns=[
                "request_time","line","train_id","first_station","last_station",
                "planned_departure","current_departure","track","message",
                "train_station","is_delayed","delayed_minutes"
            ]).to_csv(buf, index=False)
            yield buf.getvalue()
            return

        # Coerce types we rely on
        for col in ("planned_departure", "current_departure", "request_time"):
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors="coerce")
        if "delayed_minutes" in df.columns:
            df["delayed_minutes"] = pd.to_numeric(df["delayed_minutes"], errors="coerce")

        # Load existing CSV for this station (if any), then append
        station_slug = _slugify(input.station() or "station")
        existing_path = EXPORT_DIR / f"{station_slug}.csv"

        if existing_path.exists():
            try:
                df_old = pd.read_csv(existing_path)
                for col in ("planned_departure", "current_departure", "request_time"):
                    if col in df_old.columns:
                        df_old[col] = pd.to_datetime(df_old[col], errors="coerce")
                if "delayed_minutes" in df_old.columns:
                    df_old["delayed_minutes"] = pd.to_numeric(df_old["delayed_minutes"], errors="coerce")
            except Exception as e:
                print("[download] read existing ERROR:", e)
                df_old = pd.DataFrame()
        else:
            df_old = pd.DataFrame()

        merged = pd.concat([df_old, df], ignore_index=True)

        # Dedupe: keep latest by request_time per (station, train, planned)
        keys = [c for c in ["train_station", "train_id", "planned_departure"] if c in merged.columns]
        if keys and "request_time" in merged.columns:
            idx = merged.groupby(keys)["request_time"].idxmax()
            merged = merged.loc[idx]

        sort_cols = [c for c in ["planned_departure", "line", "train_id"] if c in merged.columns]
        if sort_cols:
            merged = merged.sort_values(sort_cols, kind="stable")

        # Save to server and stream to user
        merged.to_csv(existing_path, index=False, encoding="utf-8", date_format="%Y-%m-%d %H:%M:%S")

        buf = io.StringIO()
        merged.to_csv(buf, index=False)
        yield buf.getvalue()

    # ---- Plotly charts (run on history) ----
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
        # ignore line filter; show all lines for the selected station's history
        df = data()
        if df.empty or "is_delayed" not in df.columns:
            return empty_fig("Delayed Trains by Line")
        d = df[df["is_delayed"] == 1].copy()
        if d.empty:
            return empty_fig("Delayed Trains by Line", "No delayed trains in view")
        counts = (
            d.groupby("line", dropna=False, observed=False)
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

        d["hour"] = d["planned_departure"].dt.floor("15min")
        summary = d.groupby("hour", as_index=False)["delayed_minutes"].sum().sort_values("hour")

        # Normalize datetime to ms to avoid ns-label issue; strip tz if present
        try:
            summary["hour"] = summary["hour"].dt.tz_localize(None)
        except Exception:
            pass
        summary["hour_ms"] = summary["hour"].astype("datetime64[ms]")

        fig = px.line(summary, x="hour_ms", y="delayed_minutes", markers=True, title="Total Delay Minutes Per Hour")
        fig.update_xaxes(tickformat="%H:%M", hoverformat="%Y-%m-%d %H:%M")
        fig.update_layout(
            xaxis_title="Time (15-min bins)",
            yaxis_title="Total Delay Minutes",
            margin=dict(l=10, r=10, t=60, b=10),
            yaxis=dict(rangemode="tozero"),
        )
        return fig

    # ---- Debug panel ----
    @render.text
    def dbg_fetch():
        df = data()
        if df.empty:
            return "Station view: 0 rows"
        have_pd = df["planned_departure"].notna().sum() if "planned_departure" in df.columns else 0
        return f"Station view: {len(df)} rows (planned_departure non-null: {have_pd})"

    @render.text
    def dbg_history():
        df = history()
        if df.empty:
            return "History: 0 rows"
        have_pd = df["planned_departure"].notna().sum() if "planned_departure" in df.columns else 0
        return f"History: {len(df)} rows (planned_departure non-null: {have_pd})"

    @render.text
    def dbg_error():
        msg = last_error()
        return f"Last error: {msg or '(none)'}"

app = App(app_ui, server)
