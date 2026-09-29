# шаг 4 обучение ранкера на rank и проверка на val
import pandas as pd

from src.config import TOP_K, WORK_DIR
from src.data import load_prepared
from src.metrics import recall_report
from src.ranker import add_features, add_target, feature_columns, top_k, train


def load_candidates(name, events, items):
    candidates = pd.read_parquet(WORK_DIR / f"candidates_{name}.parquet")
    bench_locations = set(items.loc[items["in_bench"], "location"])
    return add_target(add_features(candidates, items, bench_locations), events)


def main():
    events, items, _ = load_prepared()
    # rank повторяет распределение запросов бенчмарка
    # extra добавляет запросы где вокруг позитива настоящий корпус бенчмарка
    train_part = pd.concat([load_candidates("rank", events, items), load_candidates("extra", events, items)], ignore_index=True)
    val = load_candidates("val", events, items)
    print("train pairs", len(train_part), "val pairs", len(val), flush=True)

    model = train(train_part)
    model.save_model(str(WORK_DIR / "ranker.txt"))

    val_events = events[events["split"] == "val"].assign(qid=lambda d: d["event_id"].astype(str))
    predictions = top_k(model, val, 300)
    preds = [predictions.get(q, []) for q in val_events["qid"]]
    print("val recall", recall_report(preds, val_events["positives"], ks=(TOP_K, 100, 300)))

    # отдельно запросы с текстом из base и без него
    base_texts = set(events.loc[events["split"] == "base", "text"])
    seen = val_events["text"].isin(base_texts).to_numpy()
    # позитив из самого корпуса бенчмарка ближе всего к условиям бенчмарка
    bench_ids = set(items.loc[items["in_bench"], "item_id"])
    in_corpus = val_events["positives"].apply(lambda ids: all(i in bench_ids for i in ids)).to_numpy()
    for name, part in [("seen text", seen), ("new text", ~seen), ("positive in bench corpus", in_corpus)]:
        part_preds = [p for p, keep in zip(preds, part) if keep]
        print(name, part.sum(), recall_report(part_preds, val_events["positives"][part], ks=(TOP_K,)))

    importance = pd.Series(model.feature_importance("gain"), index=model.feature_name()).sort_values(ascending=False)
    print((importance / importance.sum()).round(3).head(25).to_string())
    print("features", len(feature_columns(val)))


if __name__ == "__main__":
    main()
