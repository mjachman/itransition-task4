import os
import sqlite3
from pathlib import Path

import pandas as pd
import streamlit as st
from google import genai
from google.genai import errors
from pipeline import (
    combine_stats,
    count_author_sets,
    format_author_set,
    load_books,
    load_orders,
    map_user_ids,
    match_users,
    read_yaml,
)

DATA_DIR = Path(__file__).parent / "data"
NO_KEY_MSG = "Configure GEMINI_API_KEY in secrets to enable AI."
HTTP_OPTS = {"retry_options": {"attempts": 1}}
EMPTY_STATS = {
    "top_5_days": [],
    "daily_revenue": {},
    "best_buyer_ids": [],
    "most_popular_author": "N/A",
}
MODELS = (
    "gemini-3.1-flash-lite",
    "gemini-3.5-flash-lite",
    "gemini-3.8-flash",
    "gemini-3.5-flash",
)
SYS_PROMPT = (
    "You are a bookstore data assistant for stores STORE1 to STORE5. "
    "SQLite tables: orders(store, id, user_id, book_id, quantity, unit_price, "
    "paid_price, timestamp, shipping, date), "
    "users(store, id, main_user_id, aliases, name, address, phone, email), "
    "books(store, id, title, author, author_set, genre, publisher, year), "
    "book_authors(store, book_id, author) with individual exploded authors. "
    "To answer queries, call query_sql(sql). "
    "When asked to filter or control the dashboard, call "
    "apply_dashboard_filter(description, table, sql) with a SELECT * query. "
    "To reset filters, call clear_dashboard_filter()."
)


def parse_book_dict(store: str, b: dict) -> dict:
    clean = {"store": store, **{str(k).lstrip(":"): v for k, v in b.items()}}
    clean["author_set"] = format_author_set(str(clean.get("author", "")))
    return clean


def load_clean_books(s_dir: Path) -> pd.DataFrame:
    raw = read_yaml(s_dir / "books.yaml")
    return pd.DataFrame([parse_book_dict(s_dir.name, b) for b in raw])


def build_alias_map(u_map: dict[int, int]) -> dict[int, list[int]]:
    aliases: dict[int, list[int]] = {}
    for alias_id, main_id in u_map.items():
        aliases.setdefault(main_id, []).append(alias_id)
    return aliases


def load_clean_users(s_dir: Path, u_map: dict[int, int]) -> pd.DataFrame:
    df = pd.read_csv(s_dir / "users.csv")
    df.insert(0, "store", s_dir.name)
    df["main_user_id"] = df["id"].map(u_map).fillna(df["id"]).astype(int)
    df["aliases"] = df["id"].map(build_alias_map(u_map)).astype(str)
    return df


def load_clean_orders(s_dir: Path, u_map: dict[int, int]) -> pd.DataFrame:
    orders = load_orders(s_dir / "orders.parquet")
    orders["store"] = s_dir.name
    orders["user_id"] = map_user_ids(orders, u_map)
    return orders


@st.cache_data
def load_store_bundle(s_dir: Path):
    u_map = match_users(s_dir / "users.csv")
    orders = load_clean_orders(s_dir, u_map)
    users = load_clean_users(s_dir, u_map)
    authors, _ = load_books(s_dir / "books.yaml")
    return orders, users, load_clean_books(s_dir), u_map, authors


@st.cache_data
def get_all_raw_stores() -> dict[str, tuple]:
    dirs = sorted(DATA_DIR.glob("STORE*"))
    return {s.name: load_store_bundle(s) for s in dirs}


def explode_authors(books_df: pd.DataFrame) -> pd.DataFrame:
    df = books_df[["store", "id", "author"]].rename(columns={"id": "book_id"})
    df["author"] = df["author"].str.split(r",\s*")
    return df.explode("author")


def build_store_table(b: list, i: int) -> pd.DataFrame:
    return pd.concat([x[i] for x in b], ignore_index=True)


def concat_all_stores() -> dict[str, pd.DataFrame]:
    b = list(get_all_raw_stores().values())
    tabs = ("orders", "users", "books")
    d = {t: build_store_table(b, i) for i, t in enumerate(tabs)}
    d["book_authors"] = explode_authors(d["books"])
    return d


def init_sqlite_tables(conn: sqlite3.Connection):
    for name, df in concat_all_stores().items():
        df.to_sql(name, conn, index=False)
    conn.execute("PRAGMA query_only = ON")


@st.cache_resource
def get_sqlite_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:", check_same_thread=False)
    init_sqlite_tables(conn)
    return conn


def is_safe_select(sql: str) -> bool:
    clean = sql.strip().rstrip(";")
    return clean.upper().startswith("SELECT") and ";" not in clean


def run_safe_sql(sql: str, limit: int = 500) -> list[dict]:
    try:
        df = pd.read_sql_query(sql.strip().rstrip(";"), get_sqlite_conn())
        return df.head(limit).to_dict(orient="records")
    except sqlite3.Error as exc:
        return [{"error": str(exc)}]


def add_turn_log(text: str):
    st.session_state.setdefault("turn_logs", []).append(text)
    st.markdown(text)


def log_sql_call(sql: str, rows: list[dict]) -> list[dict]:
    code = f"**Executing SQL:**\n```sql\n{sql.strip()}\n```"
    add_turn_log(f"{code}\n*Rows returned: {len(rows)}* | `{rows[:3]}`")
    return rows


def query_sql(sql: str) -> list[dict]:
    """Run read-only SELECT query on tables: orders, users, books."""
    if not is_safe_select(sql):
        return [{"error": "Only single SELECT queries are allowed."}]
    return log_sql_call(sql, run_safe_sql(sql))


def set_filter_state(desc: str, table: str, sql: str, df: pd.DataFrame):
    log_sql_call(sql, df.head(5).to_dict(orient="records"))
    st.session_state.active_filter = {"desc": desc, "table": table, "df": df}
    st.session_state.needs_rerun = True


def save_dashboard_filter(desc: str, table: str, sql: str) -> str:
    rows = run_safe_sql(sql.strip(), limit=10000)
    if rows and "error" in rows[0]:
        return f"SQL Error: {rows[0]['error']}"
    set_filter_state(desc, table, sql, pd.DataFrame(rows))
    return f"Dashboard updated: '{desc}' ({len(rows)} rows)."


def apply_dashboard_filter(description: str, table: str, sql: str) -> str:
    """Filter dashboard metrics and tables using a SELECT * query."""
    if not is_safe_select(sql):
        return "Error: Only single SELECT queries are allowed."
    return save_dashboard_filter(description, table.lower().strip(), sql)


def clear_dashboard_filter() -> str:
    """Clear active dashboard filter and restore default store views."""
    st.session_state.pop("active_filter", None)
    st.session_state.needs_rerun = True
    return "Dashboard filter cleared."


TOOLS = [query_sql, apply_dashboard_filter, clear_dashboard_filter]
CHAT_CFG = {"tools": TOOLS, "system_instruction": SYS_PROMPT}


def filter_orders_by_key(
    orders: pd.DataFrame, key: str, ids: set
) -> pd.DataFrame:
    return orders[orders[key].isin(ids)]


def get_filter_key(table: str) -> str:
    if table == "books":
        return "book_id"
    return "user_id" if table == "users" else "id"


def filter_store_orders(orders: pd.DataFrame, flt: dict, s: str):
    df = flt["df"]
    sub = df[df["store"] == s] if "store" in df.columns else df
    ids = set(sub["id"].dropna().astype(int)) if "id" in sub else set()
    return filter_orders_by_key(orders, get_filter_key(flt["table"]), ids)


def build_zero_stats(s_name: str, n: int) -> dict:
    base = {"store_name": s_name, "unique_users_count": 0}
    return {**base, "unique_author_sets_count": n, **EMPTY_STATS}


def run_filtered_stats(s: str, df: pd.DataFrame, u: dict, a: dict) -> dict:
    n = count_author_sets(a)
    if df.empty:
        return build_zero_stats(s, n)
    return combine_stats(s, df, u, a, n)


def compute_filtered_store(s_name: str, flt: dict) -> dict:
    orders, _, _, u_map, authors = get_all_raw_stores()[s_name]
    sub = filter_store_orders(orders.copy(), flt, s_name)
    return run_filtered_stats(s_name, sub, u_map, authors)


def get_active_stores(base_stores: dict[str, dict]) -> dict[str, dict]:
    if not (flt := st.session_state.get("active_filter")):
        return base_stores
    return {s: compute_filtered_store(s, flt) for s in base_stores}


def get_secret_val() -> str | None:
    try:
        return st.secrets.get("GEMINI_API_KEY")
    except Exception:
        return None


def get_api_key() -> str | None:
    return os.environ.get("GEMINI_API_KEY") or get_secret_val()


def build_prompt_with_history() -> str:
    recent = st.session_state.messages[-8:]
    lines = [f"{m['role']}: {m['content']}" for m in recent]
    return "\n\n".join(lines)


def send_to_gemini(client: genai.Client, model: str, prompt: str) -> str | None:
    add_turn_log(f"*Connecting to Gemini (`{model}`)...*")
    chat = client.chats.create(model=model, config=CHAT_CFG)
    text = chat.send_message(prompt).text
    add_turn_log(f"*Response generated by `{model}`.*")
    return text


def try_model(client: genai.Client, model: str, full_prompt: str) -> str | None:
    try:
        return send_to_gemini(client, model, full_prompt)
    except (errors.APIError, Exception) as exc:
        add_turn_log(f"*`{model}` unavailable:* `{exc}`")
        return None


def call_gemini(api_key: str, full_prompt: str) -> str | None:
    client = genai.Client(api_key=api_key, http_options=HTTP_OPTS)
    for model in MODELS:
        if text := try_model(client, model, full_prompt):
            return text
    return "All Gemini models are temporarily unavailable. Try again shortly."


def answer_question(full_prompt: str) -> str:
    key = get_api_key()
    return call_gemini(key, full_prompt) if key else NO_KEY_MSG


def render_msg_logs(msg: dict):
    if logs := msg.get("logs"):
        with st.expander(f"Tool executions ({len(logs)})"):
            for log_entry in logs:
                st.markdown(log_entry)


def show_chat_msg(msg: dict):
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        render_msg_logs(msg)


def display_chat_history():
    for msg in st.session_state.get("messages", []):
        show_chat_msg(msg)


def ask_and_display() -> str:
    with st.chat_message("assistant"), st.spinner("Analyzing data..."):
        reply = answer_question(build_prompt_with_history())
        st.markdown(reply)
        return reply


def run_assistant_turn(prompt: str) -> tuple[str, list]:
    st.session_state.turn_logs, st.session_state.needs_rerun = [], False
    reply = ask_and_display()
    return reply, list(st.session_state.turn_logs)


def save_assistant_msg(reply: str, logs: list):
    st.session_state.messages.append(
        {"role": "assistant", "content": reply, "logs": logs}
    )
    if st.session_state.get("needs_rerun"):
        st.rerun()


def reply_to_prompt(prompt: str):
    reply, logs = run_assistant_turn(prompt)
    save_assistant_msg(reply, logs)


def append_and_show_user(prompt: str):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)


def handle_user_prompt(prompt: str):
    append_and_show_user(prompt)
    reply_to_prompt(prompt)


def check_chat_input():
    if prompt := st.chat_input("Ask a question about the bookstore data..."):
        handle_user_prompt(prompt)


def render_chat_header():
    st.divider()
    st.subheader("Data Assistant")
    txt = "Try: 'Which store has highest revenue?' or 'Filter Fantasy books'"
    st.caption(txt)


def render_chat_section():
    render_chat_header()
    st.session_state.setdefault("messages", [])
    display_chat_history()
    check_chat_input()
