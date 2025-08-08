# app.py
from shiny import App, ui, render, reactive
import pandas as pd
import matplotlib.pyplot as plt
from datetime import date

# ---- Load CSV once ----
df_all = pd.read_csv("data/train_data.csv", parse_dates=["planned_departure"])

# ---- HELPER ----
def get_lines():
    lines = df_all["line"].dropna().unique().tolist()
    return ["All"] + sorted(lines)

def get_filtered_data(selected_line, start_date, end_date):
    df = df_all.copy()
    df = df[(df["planned_departure"].dt.date >= start_date) & (df["planned_departure"].dt.date <= end_date)]

    if selected_line != "All":
        df = df[df["line"] == selected_line]

    return df

def get_delay_by_line(start_date, end_date):
    df = df_all.copy()
    df = df[(df["planned_departure"].dt.date >= start_date) & (df["planned_departure"].dt.date <= end_date)]
    df = df[df["delayed"] == 1]

    count_df = df.groupby("line").size().reset_index(name="delay_count").sort_values("delay_count", ascending=False)
    return count_df


# ---- UI ----
app_ui = ui.page_fluid(
    ui.h2("Train Delay Dashboard"),
    ui.row(
        ui.column(4, ui.input_select("line", "Select Line", choices=[])),
        ui.column(4, ui.input_date_range("date_range", "Select Date Range",
                                         start=date(2025, 8, 1), end=date.today())),
    ),
    ui.output_plot("delay_plot"),
    ui.output_plot("line_delay_plot")
)

# ---- SERVER ----
def server(input, output, session):
    @reactive.Effect
    def _():
        ui.update_select("line", choices=get_lines())

    @output
    @render.plot
    def delay_plot():
        start, end = input.date_range()
        df = get_filtered_data(input.line(), start, end)

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
        start, end = input.date_range()
        df = get_delay_by_line(start, end)

        if df.empty:
            fig, ax = plt.subplots()
            ax.text(0.5, 0.5, "No delay data for selected period", ha='center', va='center')
            ax.axis('off')
            return fig

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.barh(df["line"], df["delay_count"], color="orange")
        ax.set_title("Delayed Trains by Line")
        ax.set_xlabel("Number of Delays")
        ax.invert_yaxis()
        return fig

# ---- APP ----
app = App(app_ui, server)
