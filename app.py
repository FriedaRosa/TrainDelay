# app.py

from shiny import App, ui, render, reactive
import pandas as pd
import plotly.express as px
import mysql.connector
import tempfile
import matplotlib.pyplot as plt
from datetime import date
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
import os

load_dotenv()  # Load variables from .env

# Use the environment variables for config
db_config = {
    "host": os.getenv("DB_HOST"),
    "user": os.getenv("DB_USER"),
    "password": os.getenv("DB_PASSWORD"),
    "database": os.getenv("DB_NAME"),
}
engine = create_engine(
    f"mysql+pymysql://{db_config['user']}:{db_config['password']}@{db_config['host']}/{db_config['database']}"
)

# ---- HELPER ----
def get_lines():
    with engine.connect() as conn:
        result = conn.execute(text("SELECT DISTINCT line FROM train_delay_status ORDER BY line"))  # wrap with `text()`
        lines = [row[0] for row in result.fetchall()]
        return ["All"] + lines

def get_filtered_data(selected_line, start_date, end_date):
    query = "SELECT `delayed` FROM train_delay_status WHERE 1=1"
    filters = {}

    if selected_line != "All":
        query += " AND line = :line"
        filters["line"] = selected_line

    if start_date and end_date:
        query += " AND DATE(planned_departure) BETWEEN :start AND :end"
        filters["start"] = start_date
        filters["end"] = end_date

    with engine.connect() as conn:
        df = pd.read_sql(text(query), conn, params=filters)
    return df

# ---- HELPER ----
def get_delay_by_line(start_date, end_date):
    query = """
        SELECT line, COUNT(*) AS delay_count
        FROM train_delay_status
        WHERE `delayed` = 1 AND DATE(planned_departure) BETWEEN :start AND :end
        GROUP BY line
        ORDER BY delay_count DESC
    """
    filters = {
        "start": start_date,
        "end": end_date
    }

    with engine.connect() as conn:
        df = pd.read_sql(text(query), conn, params=filters)
    return df


# ---- UI ----
app_ui = ui.page_fluid(
    ui.h2("Train Delay Dashboard"),
    ui.row(
        ui.column(4, ui.input_select("line", "Select Line", choices=[])),
        ui.column(4, ui.input_date_range("date_range", "Select Date Range", start=date(2025, 8, 8), end=date.today())),
    ),
    ui.output_plot("delay_plot"),
    ui.output_plot("line_delay_plot")
)


# ---- SERVER ----
def server(input, output, session):
    # Dynamically populate line choices
    @reactive.Effect
    def _():
        ui.update_select("line", choices=get_lines())

    @output
    @render.plot
    def line_delay_plot():
        start, end = input.date_range()
        df = get_delay_by_line(str(start), str(end))

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
    
    @output
    @render.plot
    def delay_plot():
        start, end = input.date_range()
        df = get_filtered_data(input.line(), str(start), str(end))

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

# ---- APP ----
app = App(app_ui, server)
