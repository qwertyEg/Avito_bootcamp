# шаг 3 сигналы кандидаты и признаки для отложенных запросов и бенчмарка
import time

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from src.behaviour import Behaviour
from src.bm25 import BM25
from src.candidates import SOURCE_TOP, collect
from src.config import WORK_DIR
from src.corpus import BENCH_FOLD, Corpus
from src.data import load_lemmas, load_prepared
from src.locations import LocationModel
from src.metrics import recall_at_k
from src.microcat import MicrocatPrior
from src.params import search_filters
from src.signals import Signals
from src.text import lemmatize_docs

CHUNK = 256
# заголовок важнее описания поэтому повторяем его в документе
TITLE_WEIGHT = 5


def base_choices(events, exclude_ids):
    # пары поиск и выбранное объявление из base
    # объявления корпуса выкидываем чтобы валидация не видела историю своих позитивов
    base = events[events["split"] == "base"].copy()
    base["weight"] = 1.0 / base["positives"].str.len()
    choices = base.explode("positives").rename(columns={"positives": "item_id"})
    choices = choices[~choices["item_id"].isin(exclude_ids)]
    return choices[["text", "search_location_id", "search_infm_params_text", "item_id", "weight"]].reset_index(drop=True)


def query_frame(frame, qids, folds):
    queries = pd.DataFrame({
        "qid": np.asarray(qids, dtype=str),
        "fold": np.asarray(folds),
        "text": frame["text"].to_numpy(),
        "search_location_id": frame["search_location_id"].to_numpy(),
        "search_infm_params_text": frame["search_infm_params_text"].to_numpy(),
        "search_category": frame["search_category"].to_numpy(),
    })
    queries["lemmas"] = lemmatize_docs(queries["text"])
    queries["filter_lemmas"] = lemmatize_docs([" ".join(search_filters(p)) for p in queries["search_infm_params_text"]])
    return queries


def build_signals(events, items, corpus, lemmas):
    choices = base_choices(events, exclude_ids=set(corpus.items["item_id"]))
    print("base choices", len(choices), flush=True)

    locations = LocationModel().fit(choices, items, items[items["in_bench"]])
    behaviour = Behaviour().fit(choices, items)
    microcat_prior = MicrocatPrior(behaviour).fit(choices, items)

    universe = lemmas.loc[corpus.items["item_id"]]
    docs = ((universe["title"] + " ") * TITLE_WEIGHT + universe["params"] + " " + universe["description"]).tolist()
    bm25 = BM25().fit(docs)

    titles = TfidfVectorizer(analyzer=str.split, sublinear_tf=True, min_df=2).fit(lemmas["title"])
    item_emb = np.load(WORK_DIR / "item_emb.npy")
    return Signals(corpus, lemmas["title"], bm25, titles, behaviour, microcat_prior, locations, item_emb), behaviour


def run_split(name, queries, q_emb, signals, behaviour, corpus):
    started = time.time()
    parts = []
    for start in range(0, len(queries), CHUNK):
        chunk = queries.iloc[start:start + CHUNK].reset_index(drop=True)
        s = signals.compute(chunk, q_emb[start:start + CHUNK])
        parts.append(collect(s, chunk, corpus))
        print(name, start + len(chunk), "/", len(queries), round(time.time() - started), "s", flush=True)
    candidates = pd.concat(parts, ignore_index=True)

    # признаки самого запроса
    queries = queries.assign(
        n_words=queries["text"].str.split().str.len(),
        n_filters=[len(search_filters(p)) for p in queries["search_infm_params_text"]],
        nearest_base_sim=behaviour.nearest_similarity(queries["text"].tolist()),
    )
    query_cols = ["qid", "search_location_id", "search_infm_params_text", "search_category", "n_words", "n_filters", "nearest_base_sim"]
    candidates = candidates.merge(queries[query_cols], on="qid", how="left")
    candidates.to_parquet(WORK_DIR / f"candidates_{name}.parquet")
    return candidates


def report(candidates, events):
    # recall каждого источника на валидации и потолок объединения
    val = events[events["split"] == "val"]
    positives = dict(zip(val["event_id"].astype(str), val["positives"]))
    for source, k in SOURCE_TOP.items():
        top = candidates[candidates[f"src_{source}_pos"] <= 50].groupby("qid")["item_id"].agg(list)
        print(source, "recall@50", round(recall_at_k([top.get(q, []) for q in positives], positives.values()), 4))
    union = candidates.groupby("qid")["item_id"].agg(list)
    print("union size", round(union.str.len().mean()), "recall", round(recall_at_k([union.get(q, []) for q in positives], positives.values(), k=10**6), 4))


def main():
    events, items, bench_queries = load_prepared()
    lemmas = load_lemmas()
    corpus = Corpus(items, events)
    signals, behaviour = build_signals(events, items, corpus, lemmas)

    held_out = events[events["split"] != "base"]
    emb = np.load(WORK_DIR / "query_emb_events.npz")
    held_emb = pd.DataFrame(emb["emb"], index=emb["ids"]).loc[held_out["event_id"].astype(str)].to_numpy()

    for name in ["val", "rank", "extra"]:
        part = held_out["split"] == name
        queries = query_frame(held_out[part], held_out.loc[part, "event_id"], held_out.loc[part, "fold"])
        candidates = run_split(name, queries, held_emb[part.to_numpy()], signals, behaviour, corpus)
        if name == "val":
            report(candidates, events)

    emb = np.load(WORK_DIR / "query_emb_bench.npz")
    bench_emb = pd.DataFrame(emb["emb"], index=emb["ids"]).loc[bench_queries["query_id"]].to_numpy()
    queries = query_frame(bench_queries, bench_queries["query_id"], np.full(len(bench_queries), BENCH_FOLD))
    run_split("bench", queries, bench_emb, signals, behaviour, corpus)


if __name__ == "__main__":
    main()
