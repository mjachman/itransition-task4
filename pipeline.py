import re
from pathlib import Path

import pandas as pd
import yaml
from dateutil import parser

MATCH_FIELDS = (
    ["name", "address", "phone"],
    ["name", "address", "email"],
    ["name", "phone", "email"],
    ["address", "phone", "email"],
)


def get_rate(price_str: str) -> float:
    return 1.2 if ("€" in price_str or "EUR" in price_str.upper()) else 1.0


def parse_cents(price_str: str) -> float:
    nums = re.findall(r"\d+", price_str)
    return float(nums[0]) + float(nums[1]) / 100.0


def parse_number(price_str: str) -> float:
    m = re.search(r"(\d+(?:\.\d+)?)", price_str)
    return float(m.group(1)) if m else 0.0


def parse_price(raw_price: object) -> float:
    text = str(raw_price).strip()
    raw = parse_cents(text) if "¢" in text else parse_number(text)
    return round(raw * get_rate(text), 2)


def clean_date(raw_date: object) -> str:
    cleaned = str(raw_date).strip().replace(";", " ").replace(",", " ")
    cleaned = cleaned.replace("A.M.", "AM").replace("P.M.", "PM")
    return re.sub(r"(\d+)\.(\d+)\.(\d{4})", r"\3-\2-\1", cleaned)


def parse_date(raw_date: object) -> str:
    cleaned = clean_date(raw_date)
    return parser.parse(cleaned).strftime("%Y-%m-%d")


def read_yaml(yaml_path: Path) -> list[dict]:
    with open(yaml_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_authors_map(books: list[dict]) -> dict[int, str]:
    return {
        b.get(":id", b.get("id")): str(b.get(":author", b.get("author", "")))
        for b in books
    }


def parse_author_set(authors_str: str) -> frozenset[str]:
    return frozenset(a.strip() for a in authors_str.split(","))


def count_author_sets(authors: dict[int, str]) -> int:
    return len({parse_author_set(s) for s in authors.values()})


def load_books(yaml_path: Path) -> tuple[dict[int, str], int]:
    authors = get_authors_map(read_yaml(yaml_path))
    return authors, count_author_sets(authors)


def format_author_set(authors_str: str) -> str:
    return ", ".join(sorted(parse_author_set(authors_str)))


def get_top_author_set(orders: pd.DataFrame, authors: dict[int, str]) -> str:
    series = orders["book_id"].map(authors).apply(format_author_set)
    return str(orders.groupby(series)["quantity"].sum().idxmax())


def get_paid_price(df: pd.DataFrame) -> pd.Series:
    return (df["quantity"] * df["unit_price"].apply(parse_price)).round(2)


def load_orders(parquet_path: Path) -> pd.DataFrame:
    df = pd.read_parquet(parquet_path)
    df["paid_price"] = get_paid_price(df)
    df["date"] = df["timestamp"].apply(parse_date)
    return df


def get_main_id(parent: dict[int, int], uid: int) -> int:
    if parent[uid] != uid:
        parent[uid] = get_main_id(parent, parent[uid])
    return parent[uid]


def link_ids(parent: dict[int, int], id1: int, id2: int) -> None:
    r1, r2 = get_main_id(parent, id1), get_main_id(parent, id2)
    if r1 != r2:
        parent[r1] = r2


def merge_group(parent: dict[int, int], ids: list[int]) -> None:
    for other in ids[1:]:
        link_ids(parent, int(ids[0]), int(other))


def group_by_fields(users: pd.DataFrame, parent: dict, cols: list[str]) -> None:
    for ids in users.groupby(cols)["id"].apply(list):
        merge_group(parent, ids)


def match_users(csv_path: Path) -> dict[int, int]:
    users = pd.read_csv(csv_path)
    parent = {int(i): int(i) for i in users["id"]}
    for cols in MATCH_FIELDS:
        group_by_fields(users, parent, cols)
    return {uid: get_main_id(parent, uid) for uid in parent}


def map_user_ids(orders: pd.DataFrame, user_map: dict[int, int]) -> pd.Series:
    return orders["user_id"].map(lambda u: user_map.get(int(u), int(u)))


def get_top_buyer_id(orders: pd.DataFrame, user_map: dict[int, int]) -> int:
    series = map_user_ids(orders, user_map)
    return int(orders.groupby(series)["paid_price"].sum().idxmax())


def get_best_buyer(orders: pd.DataFrame, user_map: dict[int, int]) -> list[int]:
    top_id = get_top_buyer_id(orders, user_map)
    return sorted([u for u, m_id in user_map.items() if m_id == top_id])


def get_top_days(rev: pd.Series) -> list[dict]:
    return [{"date": d, "revenue": r} for d, r in rev.nlargest(5).items()]


def get_daily_revenue(orders: pd.DataFrame) -> pd.Series:
    return orders.groupby("date")["paid_price"].sum().round(2)


def get_revenue_stats(orders: pd.DataFrame) -> dict:
    rev = get_daily_revenue(orders)
    return {
        "top_5_days": get_top_days(rev),
        "daily_revenue": rev.sort_index().to_dict(),
    }


def get_user_stats(orders: pd.DataFrame, user_map: dict[int, int]) -> dict:
    return {
        "unique_users_count": len(set(user_map.values())),
        "best_buyer_ids": get_best_buyer(orders, user_map),
    }


def get_book_stats(orders: pd.DataFrame, authors: dict, n_sets: int) -> dict:
    return {
        "unique_author_sets_count": n_sets,
        "most_popular_author": get_top_author_set(orders, authors),
    }


def combine_stats(s: str, df: pd.DataFrame, u: dict, a: dict, n: int) -> dict:
    rev, usr = get_revenue_stats(df), get_user_stats(df, u)
    return {"store_name": s, **rev, **usr, **get_book_stats(df, a, n)}


def analyze_store(store_path: Path) -> dict:
    orders = load_orders(store_path / "orders.parquet")
    user_map = match_users(store_path / "users.csv")
    authors, n_sets = load_books(store_path / "books.yaml")
    return combine_stats(store_path.name, orders, user_map, authors, n_sets)


def load_all_stores(data_path: Path) -> dict[str, dict]:
    stores = sorted(data_path.glob("STORE*"))
    return {s.name: analyze_store(s) for s in stores}
