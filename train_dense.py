# шаг 2 дообучение энкодера и эмбеддинги корпуса и запросов
import numpy as np

from src.config import WORK_DIR
from src.corpus import Corpus
from src.data import load_prepared
from src.dense import DOC_PREFIX, QUERY_PREFIX, encode, item_text, load_encoder, query_text, train_encoder, training_pairs

MODEL_DIR = WORK_DIR / "dense_model"
EMBEDDING_FILES = [WORK_DIR / "item_emb.npy", WORK_DIR / "query_emb_events.npz", WORK_DIR / "query_emb_bench.npz"]


def main():
    # веса и эмбеддинги из архива с решением используются как есть
    # на другой видеокарте bf16 может дать крошечные отличия и сдвинуть ответ
    # чтобы пересчитать все с нуля достаточно удалить эти файлы
    if all(path.exists() for path in EMBEDDING_FILES):
        print("embeddings already exist")
        return

    events, items, bench_queries = load_prepared()
    corpus = Corpus(items, events)

    if not MODEL_DIR.exists():
        dataset = training_pairs(events, items, exclude_ids=set(corpus.items["item_id"]))
        print("training pairs", len(dataset), flush=True)
        train_encoder(dataset, MODEL_DIR)

    model = load_encoder(MODEL_DIR)

    docs = [item_text(t, p, d) for t, p, d in zip(corpus.items["title"], corpus.items["params"], corpus.items["description"])]
    np.save(WORK_DIR / "item_emb.npy", encode(model, docs, DOC_PREFIX))

    # запросы отложенных событий и бенчмарка
    held_out = events[events["split"] != "base"]
    for name, queries, ids in [
        ("events", held_out, held_out["event_id"].astype(str)),
        ("bench", bench_queries, bench_queries["query_id"]),
    ]:
        texts = [query_text(t, p) for t, p in zip(queries["text"], queries["search_infm_params_text"])]
        np.savez(WORK_DIR / f"query_emb_{name}.npz", ids=np.array(ids, dtype=str), emb=encode(model, texts, QUERY_PREFIX))
    print("done", flush=True)


if __name__ == "__main__":
    main()
