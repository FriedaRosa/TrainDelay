# app.py
from shiny import App, ui, render, reactive
import pandas as pd
import matplotlib.pyplot as plt
from datetime import date

# ---- Load CSV once ----
df_all = pd.read_csv("data/train_data.csv", parse_dates=["planned_departure"])

# ---- HELPERS ----
def get_lines():
    lines = df_all["line"].dropna().unique().tolist()
    return ["All"] + sorted(lines)

def get_stations():
    stations = df_all["train_station"].dropna().unique().tolist()
    return ["All"] + sorted(stations)

def get_filtered_data(selected_line, selected_station, start_date, end_date):
    df = df_all.copy()
    df = df[(df["planned_departure"].dt.date >= start_date) & (df["planned_departure"].dt.date <= end_date)]

    if selected_line != "All":
        df = df[df["line"] == selected_line]
    if selected_station != "All":
        df = df[df["train_station"] == selected_station]

    return df

def get_delay_by_line(start_date, end_date):
    df = df_all.copy()
    df = df[(df["planned_departure"].dt.date >= start_date) & (df["planned_departure"].dt.date <= end_date)]
    df = df[df["delayed"] == 1]

    count_df = df.groupby("line").size().reset_index(name="delay_count").sort_values("delay_count", ascending=False)
    return count_df

# ---- UI ----
app_ui = ui.page_fluid(
    ui.h2("🚆 Train Delay Dashboard"),
    ui.row(
        ui.column(4, ui.input_select("line", "Select Line", choices=[])),
        ui.column(4, ui.input_select("station", "Select Station", choices=[])),
        ui.column(4, ui.input_date_range("date_range", "Select Date Range",
                                         start=date(2025, 8, 1), end=date.today())),
    ),
    ui.hr(),
    ui.output_plot("delay_plot"),
    ui.output_plot("line_delay_plot"),
    ui.output_plot("delay_timeline_plot"),
    ui.hr(),
    ui.h4("Top 20 Longest Delays"),
    ui.output_table("top_delays_table")
)

# ---- SERVER ----
def server(input, output, session):
    @reactive.Effect
    def _():
        ui.update_select("line", choices=get_lines())
        ui.update_select("station", choices=get_stations())

    @output
    @render.plot
    def delay_plot():
        start, end = input.date_range()
        df = get_filtered_data(input.line(), input.station(), start, end)

        if df.empty:
            fig, ax = plt.subplots()
            ax.text(0.5, 0.5, "No data for selected filters", ha='center', va='center')
            ax.axis('off')
            return fig

        df["status"] = df["delayed"].map({0: "On Time", 1: "Delayed"})
        count_df = df["status"].value_counts().reset_index()
        count_df.columns = ["Status", "Count"]

        fig, ax = plt.subplots()
        ax.bar(count_df["Status"], count_df["Count"], color=["green", "red"])
        ax.set_title("Trains Delayed vs On Time")
        ax.set_ylabel("Number of Trains")
        return fig


    @output
    @render.plot
    def line_delay_plot():
        station = input.station()
        df = df_all.copy()

        # Always filter by delayed trains
        df = df[df["delayed"] == 1]

        # Apply station filter only if a specific station is selected
        if station != "All":
            df = df[df["train_station"] == station]

        # If no data, show message
        if df.empty:
            fig, ax = plt.subplots()
            msg = f"No delay data for station: {station}" if station != "All" else "No delay data available"
            ax.text(0.5, 0.5, msg, ha='center', va='center')
            ax.axis('off')
            return fig

        # Aggregate and plot
        count_df = (
            df.groupby("line")
            .size()
            .reset_index(name="delay_count")
            .sort_values("delay_count", ascending=False)
        )

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.barh(count_df["line"], count_df["delay_count"], color="orange")
        title = f"Delayed Trains by Line"
        if station != "All":
            title += f" (Station: {station})"
        ax.set_title(title)
        ax.set_xlabel("Number of Delays")
        ax.invert_yaxis()
        return fig


    @output
    @render.plot
    def delay_timeline_plot():
        start, end = input.date_range()
        df = get_filtered_data(input.line(), input.station(), start, end)

        if df.empty or "delay_minutes" not in df.columns:
            fig, ax = plt.subplots()
            ax.text(0.5, 0.5, "No delay data to show", ha='center', va='center')
            ax.axis('off')
            return fig

        df = df[df["delayed"] == 1]
        df["date"] = df["planned_departure"].dt.date
        summary = df.groupby("date")["delay_minutes"].sum().reset_index()

        fig, ax = plt.subplots()
        ax.plot(summary["date"], summary["delay_minutes"], marker='o')
        ax.set_title("Total Delay Minutes Over Time")
        ax.set_xlabel("Date")
        ax.set_ylabel("Total Delay Minutes")
        fig.autofmt_xdate()
        return fig

    @output
    @render.table
    def top_delays_table():
        start, end = input.date_range()
        df = get_filtered_data(input.line(), input.station(), start, end)
        if df.empty:
            return pd.DataFrame({"Message": ["No delay data found."]})

        df = df[df["delayed"] == 1].sort_values("delay_minutes", ascending=False)
        return df[["planned_departure", "line", "train_station", "delay_minutes"]].head(20)

# ---- APP ----
app = App(app_ui, server)
