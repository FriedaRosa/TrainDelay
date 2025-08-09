# scripts/poll_delays.py

from __future__ import annotations
import os
import time
import datetime
import traceback
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

# local editable package
from train_delay.train_delay_tracker import TrainDelayTracker
from train_delay.auth_data import AuthData

# ---------------------- CONFIG ---------------------- #
POLL_MINUTES   = 5                      # how often to poll
STATIONS       = ["Köln Hbf", "Spich"]  # change to what you like
OUT_DIR        = Path(__file__).resolve().parents[1] / "src" / "train_delay" / "data"
# If your tracker was initialized with a custom out_csv_path, you can ignore OUT_DIR here.

# Try to widen the window via 'hour_offset' (if your helper supports it).
# Example: [0, 1, 2] = now + next 2 hours. If unsupported, it will just fetch the current hour.
HOUR_OFFSETS   = [0]  # keep simple; expand later to [0,1,2] if your helper supports it.
# ---------------------------------------------------- #


def _dedupe_daily_file(daily_csv: Path) -> None:
    """
    Drop duplicates so we keep the latest snapshot for each (station, train_id, planned_departure).
    If 'current_departure' is present, we include it in the key so genuine updates (changed delay) remain.
    """
    if not daily_csv.exists():
        return

    try:
        df = pd.read_csv(
            daily_csv,
            parse_dates=["request_time", "planned_departure", "current_departure"],
            dtype={"train_id": "string", "line": "string", "first_station": "string",
                   "last_station": "string", "track": "string", "message": "string",
                   "train_station": "string"}
        )

        if df.empty:
            return

        # Ensure request_time exists; if not parsed, coerce
        if "request_time" not in df.columns:
            df["request_time"] = pd.Timestamp.now()

        # Sort so "latest" == last (we keep last per key)
        df = df.sort_values(["request_time", "train_station", "train_id", "planned_departure"], kind="stable")

        # Choose your dedupe key:
        #  - If you want "same state" duplicates dropped (same current_departure), use current_departure too.
        #  - If you want only one row per train/planned time per day (latest state), omit current_departure.
        dedupe_keys = ["train_station", "train_id", "planned_departure", "current_departure"]

        # Drop exact duplicates on the key, keeping the latest by request_time
        df = df.drop_duplicates(subset=dedupe_keys, keep="last", ignore_index=True)

        # Optional: make Int dtypes nullable so NA round-trips nicely
        if "is_delayed" in df.columns:
            df["is_delayed"] = pd.array(df["is_delayed"], dtype="Int8")
        if "delayed_minutes" in df.columns:
            df["delayed_minutes"] = pd.array(df["delayed_minutes"], dtype="Int16")

        # Write back (overwrite today's file with the deduped version)
        df.to_csv(daily_csv, index=False, encoding="utf-8", date_format="%Y-%m-%d %H:%M:%S")

    except Exception:
        print(f"[WARN] Dedupe failed for {daily_csv.name}")
        traceback.print_exc()


def _track_for_station(tracker: TrainDelayTracker, station: str) -> None:
    """
    Fetches current-hour changes. If your TimetableHelper supports hour_offset,
    this will try to pull additional hours too.
    """
    # Base call (current hour)
    tracker.track_station(station)

    # Try to expand hours if helper supports it
    for offset in HOUR_OFFSETS:
        if offset == 0:
            continue
        try:
            # This block assumes your TrackDelayTracker can be extended to accept an offset,
            # or your TimetableHelper.get_timetable(offset=...) exists.
            # If that’s not the case, this silently does nothing extra.
            # You can later extend your tracker to accept hour offsets if needed.
            # Example (pseudo):
            # tracker.track_station_at_hour(station, hour_offset=offset)
            pass
        except Exception:
            # ignore if unsupported
            pass


def main():
    load_dotenv()
    client_id     = os.getenv("DB_CLIENT_ID")
    client_secret = os.getenv("DB_CLIENT_SECRET")

    # out_csv_path is a seed path; tracker will roll to daily files itself.
    seed_csv_path = OUT_DIR / "trains_log_seed.csv"

    tracker = TrainDelayTracker(
        auth_data=AuthData(client_id=client_id, client_secret=client_secret),
        out_csv_path=str(seed_csv_path),
    )

    print("[INFO] Polling started. Press Ctrl+C to stop.")
    while True:
        cycle_start = datetime.datetime.now()
        try:
            for station in STATIONS:
                _track_for_station(tracker, station)

            # Dedupe today's file after each cycle
            daily_path = Path(tracker.daily_csv_path())
            _dedupe_daily_file(daily_path)

            # Optional: print a tiny heartbeat
            print(f"[OK] {cycle_start.strftime('%Y-%m-%d %H:%M:%S')} – polled {len(STATIONS)} station(s); wrote {daily_path.name}")

        except KeyboardInterrupt:
            print("\n[STOP] Stopping by user request.")
            break
        except Exception:
            print("[ERROR] Unexpected error in polling cycle:")
            traceback.print_exc()

        # Sleep until next cycle
        time.sleep(POLL_MINUTES * 60)


if __name__ == "__main__":
    main()
