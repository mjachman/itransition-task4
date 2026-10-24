from pathlib import Path

import pandas as pd
import streamlit as st

from pipeline import load_all_stores

DATA_DIR = Path(__file__).parent / "data"


@st.cache_data
def get_stores_data() -> dict[str, dict]:
    return load_all_stores(DATA_DIR)


def render_card(col: object, label: str, val: object):
    box = col.container(border=True, height=115)
    box.caption(label)
    box.markdown(f"##### {val}")


def render_metrics(m: dict):
    c1, c2, c3, c4 = st.columns([1, 1, 2.2, 1.5])
    render_card(c1, "Number of Unique Users", m["unique_users_count"])
    render_card(c2, "Unique Sets of Authors", m["unique_author_sets_count"])
    render_card(c3, "Most Popular Author(s)", m["most_popular_author"])
    render_card(c4, "Best Buyer (with aliases)", m["best_buyer_ids"])


def build_top_days_df(top_days: list[dict]) -> pd.DataFrame:
    if not top_days:
        return pd.DataFrame(columns=["Date", "Revenue (USD)"])
    df = pd.DataFrame(top_days)
    df["revenue"] = df["revenue"].map(lambda v: f"${v:,.2f}")
    return df.rename(columns={"date": "Date", "revenue": "Revenue (USD)"})


def render_line_chart(col: object, m: dict):
    df = pd.DataFrame.from_dict(
        m["daily_revenue"], orient="index", columns=["Revenue (USD)"]
    )
    col.subheader("Daily Revenue")
    col.line_chart(df)


def render_top_days_table(col: object, m: dict):
    col.subheader("Top 5 Days by Revenue")
    df = build_top_days_df(m["top_5_days"])
    col.dataframe(df, hide_index=True, width="stretch")


def render_charts(m: dict):
    col_chart, col_table = st.columns([2, 1])
    render_line_chart(col_chart, m)
    render_top_days_table(col_table, m)


def render_store_tabs(stores: dict[str, dict]):
    for name, tab in zip(stores.keys(), st.tabs(list(stores.keys()))):
        with tab:
            render_metrics(stores[name])
            render_charts(stores[name])


def main():
    st.set_page_config(page_title="Bookstore Analytics", layout="wide")
    st.title("Bookstore Analytics Dashboard")
    render_store_tabs(get_stores_data())


if __name__ == "__main__":
    main()
