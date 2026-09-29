# шаг 5 ранжируем кандидатов бенчмарка и пишем answer.csv
import re

import lightgbm as lgb
import pandas as pd

from src.config import ROOT, TOP_K, WORK_DIR
from src.data import load_prepared
from src.ranker import add_features, top_k

ITEM_ID_RE = re.compile(r"^[0-9a-f]{16}$")


def check_answer(answer, bench_queries, items):
    # те же проверки что делает платформа плюс проверка самих item_id
    assert list(answer.columns) == ["query_id", "answer"]
    assert set(answer["query_id"]) == set(bench_queries["query_id"])
    assert answer["query_id"].is_unique
    corpus_ids = set(items.loc[items["in_bench"], "item_id"])
    for line in answer["answer"]:
        ids = line.split(" ")
        assert 0 < len(ids) <= TOP_K and len(set(ids)) == len(ids)
        assert all(ITEM_ID_RE.match(i) and i in corpus_ids for i in ids)


def main():
    events, items, bench_queries = load_prepared()
    candidates = pd.read_parquet(WORK_DIR / "candidates_bench.parquet")
    bench_locations = set(items.loc[items["in_bench"], "location"])
    candidates = add_features(candidates, items, bench_locations)

    model = lgb.Booster(model_file=str(WORK_DIR / "ranker.txt"))
    predictions = top_k(model, candidates, TOP_K)

    answer = pd.DataFrame({
        "query_id": bench_queries["query_id"],
        "answer": [" ".join(predictions[q]) for q in bench_queries["query_id"]],
    })
    check_answer(answer, bench_queries, items)
    answer.to_csv(ROOT / "answer.csv", index=False)
    print("saved", len(answer), "rows")


if __name__ == "__main__":
    main()
