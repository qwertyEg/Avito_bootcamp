import lightgbm as lgb
import numpy as np
import pandas as pd

from src.params import search_filters

PARAMS = {
    "objective": "lambdarank",
    "metric": "ndcg",
    "eval_at": [50],
    # позитив один на запрос поэтому важен только верх списка
    "lambdarank_truncation_level": 60,
    "learning_rate": 0.05,
    "num_leaves": 63,
    "min_data_in_leaf": 100,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "seed": 42,
    # без фиксированного режима гистограмм результат плавает от запуска к запуску
    "deterministic": True,
    "force_col_wise": True,
    "num_threads": 6,
    "verbose": -1,
}
N_ROUNDS = 800

# колонки которые не являются признаками
SERVICE_COLS = ["qid", "item_id", "target", "search_location_id", "search_infm_params_text"]


def filter_match(candidates, items):
    # какая доля фильтров поиска нашлась в параметрах объявления
    # у 97 процентов позитивов вид услуги из фильтра совпадает
    params = dict(zip(items["item_id"], items["params"]))
    item_ids = candidates["item_id"].to_numpy()
    share = np.full(len(candidates), np.nan, dtype=np.float32)
    for text, idx in candidates.groupby("search_infm_params_text").indices.items():
        filters = search_filters(text)
        if filters:
            share[idx] = [sum(f in params[i] for f in filters) / len(filters) for i in item_ids[idx]]
    return share


def item_features(items):
    # признаки объявления считаем один раз и потом присоединяем к парам
    return pd.DataFrame({
        "item_id": items["item_id"],
        "log_reviews": np.log1p(items["reviews"].fillna(0)),
        "rating": items["rating"],
        "log_price": np.log1p(items["price"].clip(lower=0)),
        "phone_hidden": items["phone_hidden"],
        "message_forbidden": items["message_forbidden"],
        "desc_len": items["description"].str.len(),
        "title_len": items["title"].str.len(),
        "is_services": (items["category"] == 114).astype(np.int8),
    })


def add_features(candidates, items, bench_locations):
    candidates = candidates.merge(item_features(items), on="item_id", how="left")
    candidates["filter_match"] = filter_match(candidates, items)
    candidates["no_category"] = (candidates["search_category"] == 0).astype(np.int8)
    # поиск по региону у которого нет своих объявлений
    candidates["is_region"] = (~candidates["search_location_id"].isin(bench_locations)).astype(np.int8)

    # пар десятки миллионов поэтому держим признаки во float32
    candidates = candidates.drop(columns=["search_infm_params_text"])
    features = feature_columns(candidates)
    candidates[features] = candidates[features].astype(np.float32)
    return candidates


def add_target(candidates, events):
    pairs = events[["event_id", "positives"]].explode("positives")
    positive = set(zip(pairs["event_id"].astype(str), pairs["positives"]))
    candidates["target"] = [int((q, i) in positive) for q, i in zip(candidates["qid"], candidates["item_id"])]
    return candidates


def feature_columns(candidates):
    return [c for c in candidates.columns if c not in SERVICE_COLS]


def train(candidates, n_rounds=N_ROUNDS):
    # запросы где позитив не попал в кандидатов ничему не учат
    has_positive = candidates.groupby("qid")["target"].transform("max") > 0
    data = candidates[has_positive].sort_values("qid", kind="stable")
    groups = data.groupby("qid", sort=False).size().to_numpy()
    features = feature_columns(data)
    dataset = lgb.Dataset(data[features].to_numpy(np.float32), data["target"].to_numpy(), group=groups, feature_name=features)
    return lgb.train(PARAMS, dataset, num_boost_round=n_rounds)


def top_k(model, candidates, k):
    candidates = candidates.assign(score=model.predict(candidates[model.feature_name()].to_numpy(np.float32)))
    candidates = candidates.sort_values(["qid", "score"], ascending=[True, False], kind="stable")
    return candidates.groupby("qid", sort=False)["item_id"].agg(lambda ids: list(ids[:k]))
