import numpy as np
import pandas as pd

from src.config import DATA_DIR, N_RANK_TEXTS, N_VAL_TEXTS, RANK_FOLD_SIZE, SEARCH_COLS, SEED, WORK_DIR
from src.text import clean_query, lemmatize_docs

ITEM_COLS = {
    "item_id": "item_id",
    "item_title_raw": "title",
    "item_description_raw": "description",
    "item_infm_params_text": "params",
    "item_category_id": "category",
    "item_microcat_id": "microcat",
    "item_price": "price",
    "item_rating": "rating",
    "item_rating_reviews_count": "reviews",
    "item_location_id": "location",
    "item_latitude": "lat",
    "item_longitude": "lon",
    "item_is_phone_hidden": "phone_hidden",
    "item_is_message_forbidden": "message_forbidden",
}


def prepare_items(df):
    items = df[list(ITEM_COLS)].rename(columns=ITEM_COLS)
    # цена и координаты лежат строками вроде 3500.000000000000000
    for col in ["price", "lat", "lon"]:
        items[col] = pd.to_numeric(items[col], errors="coerce").astype("float32")
    for col in ["title", "description", "params"]:
        items[col] = items[col].fillna("")
    items["phone_hidden"] = items["phone_hidden"].astype("int8")
    items["message_forbidden"] = items["message_forbidden"].astype("int8")
    return items


def prepare_queries(df):
    queries = df.copy()
    queries["search_infm_params_text"] = queries["search_infm_params_text"].fillna("")
    queries["text"] = queries["search_query"].map(clean_query)
    return queries


def build_events(train):
    # событие это один поиск то есть уникальный набор признаков запроса
    # все выбранные в нем объявления считаем позитивами
    train = prepare_queries(train)
    events = (
        train.groupby(SEARCH_COLS + ["text"], sort=False)["item_id"]
        .agg(lambda ids: list(dict.fromkeys(ids)))
        .rename("positives")
        .reset_index()
    )
    events.insert(0, "event_id", np.arange(len(events)))
    return events


def split_events(events):
    # в бенчмарке каждый текст запроса встречается один раз
    # поэтому берем случайные тексты и по одному случайному поиску на каждый
    # остальные поиски с тем же текстом остаются в base как и в реальном train
    rng = np.random.default_rng(SEED)
    texts = events["text"].unique()
    texts = texts[rng.permutation(len(texts))]
    val_texts = set(texts[:N_VAL_TEXTS])
    rank_texts = set(texts[N_VAL_TEXTS:N_VAL_TEXTS + N_RANK_TEXTS])

    shuffled = events.sample(frac=1, random_state=SEED)
    val_ids = shuffled[shuffled["text"].isin(val_texts)].drop_duplicates("text")["event_id"]
    rank_ids = shuffled[shuffled["text"].isin(rank_texts)].drop_duplicates("text")["event_id"]

    events["split"] = "base"
    events.loc[events["event_id"].isin(val_ids), "split"] = "val"
    events.loc[events["event_id"].isin(rank_ids), "split"] = "rank"

    # фолд задает какие чужие позитивы видны запросу в корпусе
    # val это отдельный фолд а rank нарезан на куски одинакового размера
    events["fold"] = -1
    rank_mask = events["split"] == "rank"
    order = rng.permutation(rank_mask.sum())
    events.loc[rank_mask, "fold"] = order // RANK_FOLD_SIZE
    events.loc[events["split"] == "val", "fold"] = events["fold"].max() + 1
    return events


def mark_in_corpus(events, bench_ids):
    # поиски из base где выбранные объявления уже лежат в корпусе бенчмарка
    # их история и так не идет в статистики поэтому для base они бесполезны
    # зато это честные запросы для ранкера с настоящим окружением позитива
    base = events["split"] == "base"
    in_corpus = events["positives"].apply(lambda ids: all(i in bench_ids for i in ids))
    events.loc[base & in_corpus, "split"] = "extra"
    return events


def load_raw():
    train = pd.read_parquet(DATA_DIR / "train.parquet")
    bench_items = pd.read_parquet(DATA_DIR / "benchmark_items.parquet")
    bench_queries = pd.read_parquet(DATA_DIR / "benchmark_queries.parquet")
    return train, bench_items, bench_queries


def prepare_all():
    train, bench_items, bench_queries = load_raw()

    bench_ids = set(bench_items["item_id"])
    events = mark_in_corpus(split_events(build_events(train)), bench_ids)

    # в общий справочник кладем и корпус и объявления из train
    # объявления из train нужны для обучения и как позитивы валидации
    items = pd.concat([prepare_items(bench_items), prepare_items(train)])
    items = items.drop_duplicates("item_id").reset_index(drop=True)
    items["in_bench"] = items["item_id"].isin(bench_ids)

    bench_queries = prepare_queries(bench_queries)

    events.to_parquet(WORK_DIR / "events.parquet")
    items.to_parquet(WORK_DIR / "items.parquet")
    bench_queries.to_parquet(WORK_DIR / "bench_queries.parquet")
    lemmatize_items(items).to_parquet(WORK_DIR / "item_lemmas.parquet")
    return events, items, bench_queries


def lemmatize_items(items):
    # леммы храним строками через пробел
    lemmas = pd.DataFrame({"item_id": items["item_id"]})
    for col in ["title", "params", "description"]:
        lemmas[col] = lemmatize_docs(items[col].tolist())
    return lemmas


def load_prepared():
    events = pd.read_parquet(WORK_DIR / "events.parquet")
    items = pd.read_parquet(WORK_DIR / "items.parquet")
    bench_queries = pd.read_parquet(WORK_DIR / "bench_queries.parquet")
    return events, items, bench_queries


def load_lemmas():
    return pd.read_parquet(WORK_DIR / "item_lemmas.parquet").set_index("item_id")
