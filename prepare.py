# шаг 1 подготовка таблиц и разбиение train
from src.data import prepare_all

if __name__ == "__main__":
    events, items, bench_queries = prepare_all()
    print(events["split"].value_counts().to_dict())
    print("items", len(items), "in bench", int(items["in_bench"].sum()))
