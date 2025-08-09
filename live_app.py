# app.py
from __future__ import annotations

import os
import time
from datetime import datetime
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

# ======================= Paths =======================
EXPORT_DIR = Path(__file__).resolve().parent / "exports"
EXPORT_DIR.mkdir(parents=True, exist_ok=True)

def _slugify(text: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in text).strip("_").lower()

# ======================= Timezones =======================
# API returns *German local time* (Europe/Berlin) as naïve timestamps.
# We'll localize those to SOURCE_TZ and display/compute in TARGET_TZ.
TARGET_TZ = os.getenv("APP_TIMEZONE", "Europe/Berlin")        # display/logic TZ
SOURCE_TZ = os.getenv("SOURCE_TIMEZONE", "Europe/Berlin")     # API wall-clock TZ

def ensure_in_tz(x: pd.Series | pd.Timestamp | None, tz: str):
    """
    Make a Series or Timestamp tz-aware in `tz`.
    - If tz-naive: localize to `tz`.
    - If tz-aware: convert to `tz`.
    """
    if x is None:
        return x
    if isinstance(x, pd.Series):
        s = pd.to_datetime(x, errors="coerce", utc=False)
        if getattr(s.dt, "tz", None) is not None:
            return s.dt.tz_convert(tz)
        else:
            return s.dt.tz_localize(tz)
    t = pd.to_datetime(x, errors="coerce", utc=False)
    if getattr(t, "tzinfo", None) is None:
        return t.tz_localize(tz)
    return t.tz_convert(tz)

def parse_db_time_local(val):
    """
    Parse DB API time that is in *German local time* (Europe/Berlin) but naïve.
    Steps:
      1) Parse to naïve Timestamp.
      2) Localize to SOURCE_TZ (Berlin).
      3) Convert to TARGET_TZ (default Berlin, but can be changed via APP_TIMEZONE).
    Returns tz-aware Timestamp in TARGET_TZ (or NaT).
    """
    if val is None or (isinstance(val, str) and not val.strip()):
        return pd.NaT

    # Fast path for datetime-like
    if isinstance(val, (datetime, pd.Timestamp)):
        ts = pd.to_datetime(val, errors="coerce", utc=False)
    else:
        s = str(val)
        # Try exact DB format first: "yyMMddHHmm" (e.g. "2508091504")
        try:
            ts = pd.to_datetime(datetime.strptime(s, "%y%m%d%H%M"), errors="raise")
        except Exception:
            ts = pd.to_datetime(s, errors="coerce", utc=False)

    if pd.isna(ts):
        return pd.NaT

    # Localize naïve to SOURCE_TZ (Berlin) then convert to TARGET_TZ
    if isinstance(ts, pd.Timestamp) and ts.tzinfo is None:
        ts = ts.tz_localize(SOURCE_TZ)
    else:
        ts = ensure_in_tz(ts, SOURCE_TZ)
    return ts.tz_convert(TARGET_TZ)

def strip_tz_for_plot(x: pd.Series) -> pd.Series:
    """Drop timezone for plotting while keeping wall-clock time (ms resolution)."""
    x = pd.to_datetime(x, errors="coerce", utc=False)
    if getattr(x.dt, "tz", None) is not None:
        x = x.dt.tz_convert(TARGET_TZ)
    return x.dt.tz_localize(None).astype("datetime64[ms]")

# ======================= Auth =======================
load_dotenv()
CLIENT_ID = os.getenv("DB_CLIENT_ID")
CLIENT_SECRET = os.getenv("DB_CLIENT_SECRET")
if not CLIENT_ID or not CLIENT_SECRET:
    raise RuntimeError("Missing DB_CLIENT_ID / DB_CLIENT_SECRET in .env")

AUTH = ApiAuthentication(CLIENT_ID, CLIENT_SECRET)

# ======================= Delay helpers =======================
def compute_delay(planned, current):
    p = parse_db_time_local(planned)
    c = parse_db_time_local(current)
    if pd.isna(p) or pd.isna(c):
        return 0, 0
    diff_min = (c - p).total_seconds() / 60.0
    delay_minutes = int(round(diff_min))
    return (1 if delay_minutes > 0 else 0), delay_minutes

# ======================= Fetch =======================
def fetch_station_df(station_name: str) -> pd.DataFrame:
    try:
        sh = StationHelper()
        matches = sh.find_stations_by_name(station_name)
        if not matches:
            return pd.DataFrame()

        th = TimetableHelper(matches[0], AUTH)
        timetable = th.get_timetable()
        try:
            trains = th.get_timetable_changes(timetable) or []
        except Exception:
            # Fallback: if changes parsing fails (e.g., missing 'ts' inside), use base timetable
            try:
                trains = timetable or []
            except Exception:
                trains = []


        rows = []
        # request_time taken directly in TARGET_TZ (Berlin by default)
        request_time = pd.Timestamp.now(tz=TARGET_TZ)

        for t in trains:
            line = f"{t.train_type}{getattr(t, 'train_line', '')}"
            train_id = t.train_number
            first_station = getattr(t, "passed_stations", None)
            first_station = (first_station or t.stations).split("|")[0]
            last_station = t.stations.split("|")[-1]
            line_last = f"{line} {last_station}"

            planned_departure = parse_db_time_local(t.departure)
            current_departure = parse_db_time_local(getattr(t.train_changes, "departure", None))
            track = t.platform
            
            
            # --- Robust message extraction (handles dicts/objects/missing fields safely) ---
            msg_parts = []
            try:
                msgs = getattr(t.train_changes, "messages", []) or []
            except Exception:
                msgs = []
            for m in msgs:
                try:
                    if isinstance(m, dict):
                        # Prefer typical text fields; fall back to any printable content
                        val = m.get("message") or m.get("text") or m.get("msg") or ""
                    else:
                        val = getattr(m, "message", None) or getattr(m, "text", None) or ""
                    if val:
                        msg_parts.append(str(val))
                except Exception:
                    # Last resort: best-effort repr without throwing
                    try:
                        msg_parts.append(str(m))
                    except Exception:
                        pass
            message = " ".join(msg_parts) if msg_parts else "No message"


            is_delayed, delayed_minutes = compute_delay(t.departure, getattr(t.train_changes, "departure", None))

            rows.append(
                {
                    "request_time": request_time,   # tz-aware (TARGET_TZ)
                    "line": line,
                    "last_station": last_station,
                    "line_last": line_last,
                    "train_id": train_id,
                    "first_station": first_station,
                    "planned_departure": planned_departure,  # tz-aware (TARGET_TZ)
                    "current_departure": current_departure,  # tz-aware (TARGET_TZ)
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

# ======================= UI =======================
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

app_ui = ui.page_fluid(
    ui.h3("🚆 Live Deutsche Bahn delays"),
    ui.markdown(f"_API TZ: **{SOURCE_TZ}** → Display TZ: **{TARGET_TZ}**_"),
    ui.row(
        ui.column(
            4,
            ui.input_text("station", "Station", value="Köln Hbf", placeholder="Type a station"),
            ui.input_select("line", "Line", choices=["All"], selected="All"),
            ui.input_select("line_last", "Destination (line_last)", choices=["All"], selected="All"),
            ui.input_switch("auto", "Auto-refresh", value=True),
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
                ui.card_header("Live trains (upcoming)"),
                ui.output_data_frame("table"),
                style="height:40vh; overflow:auto;",
            ),
        ),
    ),
    ui.row(
        ui.column(12, ui.output_text("refresh_debug"))
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
        ui.column(4, ui.card(ui.card_header("Delay Minutes per 15 min"), output_widget("delay_timeline_plot"))),
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

# ======================= Server =======================
def server(input, output, session):
    # --- state ---
    history = reactive.Value(pd.DataFrame())
    _line_choices_cache = reactive.Value(("All",))
    last_error = reactive.Value("")
    fetching = reactive.Value(False)
    last_fetch_monotonic = reactive.Value(0)

    # Debug
    refresh_clicks = reactive.Value(0)
    last_manual_msg = reactive.Value("")

    # --- knobs ---
    WINDOW_HOURS = 12
    MAX_ROWS = 5000
    DEDUPE_KEYS = ["train_station", "train_id", "planned_departure"]
    MIN_REFRESH_SEC = 180

    # Shared fetch routine
    def _do_fetch_and_merge(station_name: str, *, force: bool = False):
        now = time.monotonic()
        if not force:
            elapsed = now - (last_fetch_monotonic() or 0)
            if elapsed < MIN_REFRESH_SEC:
                return
        if fetching():
            return

        fetching.set(True)
        try:
            df_new = fetch_station_df(station_name)
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

            # Normalize types (tz-aware in TARGET_TZ)
            for col in ("planned_departure", "current_departure", "request_time"):
                if col in df_new.columns:
                    df_new[col] = ensure_in_tz(df_new[col], TARGET_TZ)
            if "delayed_minutes" in df_new.columns:
                df_new["delayed_minutes"] = pd.to_numeric(df_new["delayed_minutes"], errors="coerce")
            if "line" in df_new.columns:
                df_new["line"] = df_new["line"].astype("category")

            print(f"[history] new rows: {len(df_new)}")

            # Append to history
            df = pd.concat([history(), df_new], ignore_index=True, copy=False)

            # Time window cap (keep NaT rows too)
            if "planned_departure" in df.columns:
                cutoff = pd.Timestamp.now(tz=TARGET_TZ) - pd.Timedelta(hours=WINDOW_HOURS)
                mask = df["planned_departure"].notna() & (df["planned_departure"] >= cutoff)
                df = pd.concat([df[mask], df[df["planned_departure"].isna()]], ignore_index=True)

            # Row cap
            if "request_time" in df.columns and len(df) > MAX_ROWS:
                df = df.nlargest(MAX_ROWS, "request_time")

            # Dedupe: keep latest request per key
            if "request_time" in df.columns and all(k in df.columns for k in DEDUPE_KEYS):
                idx = df.groupby(DEDUPE_KEYS)["request_time"].idxmax()
                sort_cols = [c for c in ["planned_departure", "line_last", "line", "train_id"] if c in df.columns]
                df = df.loc[idx].sort_values(sort_cols, kind="stable")

            history.set(df)
            last_error.set("")
            print(f"[history] total rows: {len(df)}")
        finally:
            fetching.set(False)

    # Auto mode
    @reactive.effect
    def _auto_fetch():
        if not input.auto():
            return
        interval_ms = max(int(input.secs()), MIN_REFRESH_SEC) * 1000
        reactive.invalidate_later(interval_ms)
        st = input.station().strip()
        if not st:
            return
        _do_fetch_and_merge(st, force=False)

    # Manual mode
    @reactive.effect
    @reactive.event(input.refresh)
    def _manual_fetch():
        refresh_clicks.set(refresh_clicks() + 1)
        st = input.station().strip()
        if not st:
            last_manual_msg.set("Manual refresh clicked, but station is empty.")
            return
        if 'last_fetch_monotonic' in locals():
            last_fetch_monotonic.set(0)
        print("[manual] refresh clicked; fetching now…")
        last_manual_msg.set(f"Manual refresh #{refresh_clicks()} at {pd.Timestamp.now(tz=TARGET_TZ).strftime('%H:%M:%S')}")
        _do_fetch_and_merge(st, force=True)

    # Current view = by station
    @reactive.calc
    def data():
        df = history()
        st = input.station().strip()
        if df.empty or not st or "train_station" not in df.columns:
            return df
        return df[df["train_station"] == st]

    # Line filter + upcoming only (in TARGET_TZ)
    @reactive.calc
    def filtered():
        df = data()
        sel_line = input.line()
        sel_line_last = input.line_last()

        if df.empty or "error" in df.columns:
            return df

        # First filter by "line" (unless All)
        if sel_line and sel_line != "All" and "line" in df.columns:
            df = df[df["line"] == sel_line]

        # Then optionally filter by "line_last" within that line (unless All)
        if sel_line_last and sel_line_last != "All" and "line_last" in df.columns:
            df = df[df["line_last"] == sel_line_last]

        # Keep only upcoming trains in your configured TZ
        if "planned_departure" in df.columns:
            now_local = pd.Timestamp.now(tz=TARGET_TZ)
            df = df[df["planned_departure"].notna() & (df["planned_departure"] >= now_local)]

        return df

    # Keep Line choices synced
    @reactive.effect
    def _update_line_choices():
        df = data()

        # --- Build choices for "line"
        if df.empty or "line" not in df.columns or "error" in df.columns:
            line_choices = ("All",)
        else:
            lines = sorted([x for x in df["line"].dropna().unique().tolist() if str(x).strip()])
            line_choices = tuple(["All"] + lines)

        # Preserve current selection if still valid
        current_line = input.line()
        selected_line = current_line if current_line in line_choices else "All"
        if line_choices != _line_choices_cache():
            _line_choices_cache.set(line_choices)
        ui.update_select("line", choices=list(line_choices), selected=selected_line)

        # --- Build choices for "line_last" (depends on selected line)
        if df.empty or "line_last" not in df.columns or "error" in df.columns:
            line_last_choices = ("All",)
        else:
            # Restrict line_last options to the selected "line" (unless "All")
            if selected_line != "All" and "line" in df.columns:
                df_for_line = df[df["line"] == selected_line]
            else:
                df_for_line = df

            line_lasts = sorted(
                [x for x in df_for_line["line_last"].dropna().unique().tolist() if str(x).strip()]
            )
            line_last_choices = tuple(["All"] + line_lasts)

        current_line_last = input.line_last()
        selected_line_last = current_line_last if current_line_last in line_last_choices else "All"
        ui.update_select("line_last", choices=list(line_last_choices), selected=selected_line_last)


    # Autoload CSV history for station
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
                        dt_col = pd.to_datetime(df_old[col], errors="coerce", utc=False)
                        # If tz-naive, assume they were saved as TARGET_TZ and localize; else convert
                        if getattr(dt_col.dt, "tz", None) is None:
                            dt_col = dt_col.dt.tz_localize(TARGET_TZ)
                        else:
                            dt_col = dt_col.dt.tz_convert(TARGET_TZ)
                        df_old[col] = dt_col
                if "delayed_minutes" in df_old.columns:
                    df_old["delayed_minutes"] = pd.to_numeric(df_old["delayed_minutes"], errors="coerce")

                # Backfill line_last
                if "line_last" not in df_old.columns:
                    if "line" in df_old.columns and "last_station" in df_old.columns:
                        df_old["line_last"] = (
                            df_old["line"].fillna("") + " " + df_old["last_station"].fillna("")
                        ).str.strip()
                    elif "line" in df_old.columns:
                        df_old["line_last"] = df_old["line"].astype(str)
                    else:
                        df_old["line_last"] = ""
            except Exception as e:
                print("[autoload] ERROR:", e)
                df_old = pd.DataFrame()
        else:
            df_old = pd.DataFrame()

        if not df_old.empty and "train_station" in df_old.columns:
            df_old = df_old[df_old["train_station"] == st]

        history.set(df_old)
        last_loaded_station.set(st)
        print(f"[autoload] loaded {len(df_old)} rows for station: {st}")

    # ---- Outputs ----
    @render.text
    def updated():
        df = data()
        if df.empty:
            return "No data (yet)."
        if "error" in df.columns:
            return f"Error: {df.loc[0,'error']}"
        ts = df["request_time"].max() if "request_time" in df.columns else pd.Timestamp.now(tz=TARGET_TZ)
        ts = ensure_in_tz(ts, TARGET_TZ)
        return ts.strftime(f"%Y-%m-%d %H:%M:%S {TARGET_TZ}")

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

        d = df[existing].copy()
        # Pretty formatting
        for c in ("request_time", "planned_departure", "current_departure"):
            if c in d.columns:
                d[c] = ensure_in_tz(d[c], TARGET_TZ)
                d[c] = pd.to_datetime(d[c], errors="coerce")
                d[c] = d[c].dt.tz_convert(TARGET_TZ).dt.strftime(f"%Y-%m-%d %H:%M:%S {TARGET_TZ}")

        return d.sort_values(["planned_departure", "line_last", "train_id"])

    @render.download(filename=lambda: f"{_slugify(input.station() or 'station')}.csv")
    def download_csv():
        # Use full station view history (ignores line filter)
        df = data()

        if df.empty:
            buf = io.StringIO()
            pd.DataFrame(columns=[
                "request_time","line","train_id","first_station","last_station",
                "planned_departure","current_departure","track","message",
                "train_station","is_delayed","delayed_minutes","line_last"
            ]).to_csv(buf, index=False)
            yield buf.getvalue()
            return

        # Ensure tz-aware TARGET_TZ before saving
        for col in ("planned_departure", "current_departure", "request_time"):
            if col in df.columns:
                df[col] = ensure_in_tz(df[col], TARGET_TZ)
        if "delayed_minutes" in df.columns:
            df["delayed_minutes"] = pd.to_numeric(df["delayed_minutes"], errors="coerce")

        # Load existing CSV and append
        station_slug = _slugify(input.station() or "station")
        existing_path = EXPORT_DIR / f"{station_slug}.csv"

        if existing_path.exists():
            try:
                df_old = pd.read_csv(existing_path)
                for col in ("planned_departure", "current_departure", "request_time"):
                    if col in df_old.columns:
                        dt_col = pd.to_datetime(df_old[col], errors="coerce", utc=False)
                        if getattr(dt_col.dt, "tz", None) is None:
                            dt_col = dt_col.dt.tz_localize(TARGET_TZ)
                        else:
                            dt_col = dt_col.dt.tz_convert(TARGET_TZ)
                        df_old[col] = dt_col
                if "delayed_minutes" in df_old.columns:
                    df_old["delayed_minutes"] = pd.to_numeric(df_old["delayed_minutes"], errors="coerce")
            except Exception as e:
                print("[download] read existing ERROR:", e)
                df_old = pd.DataFrame()
        else:
            df_old = pd.DataFrame()

        merged = pd.concat([df_old, df], ignore_index=True)

        # Dedupe by latest request_time per (station, train, planned)
        keys = [c for c in ["train_station", "train_id", "planned_departure"] if c in merged.columns]
        if keys and "request_time" in merged.columns:
            idx = merged.groupby(keys)["request_time"].idxmax()
            merged = merged.loc[idx]

        sort_cols = [c for c in ["planned_departure", "line", "train_id"] if c in merged.columns]
        if sort_cols:
            merged = merged.sort_values(sort_cols, kind="stable")

        # Save (ISO-8601 strings; tz offset preserved)
        merged.to_csv(existing_path, index=False)

        buf = io.StringIO()
        merged.to_csv(buf, index=False)
        yield buf.getvalue()

    @render.text
    def refresh_debug():
        msg = last_manual_msg()
        return msg or "Click 'Refresh now' to fetch immediately."

    # ---- Plots ----
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
        # Use the line + line_last filtered view
        df = filtered()
        if df.empty or "is_delayed" not in df.columns:
            return empty_fig("Delayed Trains by Destination (line_last)")

        d = df[df["is_delayed"] == 1].copy()
        if d.empty:
            return empty_fig("Delayed Trains by Destination (line_last)", "No delayed trains in view")

        group_key = "line_last" if "line_last" in d.columns else "line"
        counts = (
            d.groupby(group_key, dropna=False, observed=False)
             .size()
             .reset_index(name="delay_count")
             .sort_values("delay_count", ascending=False)
        )

        ylab = "Destination (line_last)" if group_key == "line_last" else "Line"
        title = "Delayed Trains by Destination (line_last)" if group_key == "line_last" else "Delayed Trains by Line"

        fig = px.bar(counts, x="delay_count", y=group_key, orientation="h", title=title)
        fig.update_layout(xaxis_title="Number of Delays", yaxis_title=ylab, bargap=0.2, margin=dict(l=10, r=10, t=60, b=10))
        fig.update_yaxes(autorange="reversed")
        return fig


    @render_plotly
    def delay_timeline_plot():
        df = filtered()
        if df.empty:
            return empty_fig("Total Delay Minutes per 15 min", "No data in view")

        needed = {"planned_departure", "delayed_minutes", "is_delayed"}
        if not needed.issubset(df.columns):
            return empty_fig("Total Delay Minutes per 15 min", "Missing required columns")

        d = df.copy()
        d["planned_departure"] = ensure_in_tz(d["planned_departure"], TARGET_TZ)
        d["delayed_minutes"] = pd.to_numeric(d["delayed_minutes"], errors="coerce")
        d = d.dropna(subset=["planned_departure", "delayed_minutes"])
        d = d[d["is_delayed"] == 1]
        if d.empty:
            return empty_fig("Total Delay Minutes per 15 min", "No delayed trains in view")

        d["bin"] = d["planned_departure"].dt.floor("15min")
        summary = d.groupby("bin", as_index=False)["delayed_minutes"].sum().sort_values("bin")

        # For plotting: drop tz (keep wall-clock)
        plot_df = pd.DataFrame({
            "bin": strip_tz_for_plot(summary["bin"]),
            "delayed_minutes": summary["delayed_minutes"]
        })

        fig = px.line(plot_df, x="bin", y="delayed_minutes", markers=True, title="Total Delay Minutes per 15 min")
        fig.update_xaxes(tickformat="%H:%M", hoverformat="%Y-%m-%d %H:%M")
        fig.update_layout(
            xaxis_title=f"Time ({TARGET_TZ})",
            yaxis_title="Total Delay Minutes",
            margin=dict(l=10, r=10, t=60, b=10),
            yaxis=dict(rangemode="tozero"),
        )
        return fig

    # Debug panel
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
