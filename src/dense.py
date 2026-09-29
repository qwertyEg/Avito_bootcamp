import numpy as np
import torch
from datasets import Dataset
from sentence_transformers import SentenceTransformer, SentenceTransformerTrainer, SentenceTransformerTrainingArguments
from sentence_transformers.base.sampler import BatchSamplers
from sentence_transformers.losses import CachedMultipleNegativesRankingLoss

from src.params import kind_text

# русскоязычный энкодер от deepvk
# на zero-shot сравнении он был лучшим из e5 bge-m3 frida qwen3-embedding и rubert-tiny2
BASE_MODEL = "deepvk/USER2-base"
QUERY_PREFIX = "search_query: "
DOC_PREFIX = "search_document: "
MAX_LEN = 128
DESC_CHARS = 300


def query_text(text, search_params):
    # фильтры поиска подсказывают модели в какой рубрике искать
    return f"{text} | {search_params}" if search_params else text


def item_text(title, params, description):
    parts = [title, kind_text(params), description[:DESC_CHARS]]
    return ". ".join(p for p in parts if p)


def training_pairs(events, items, exclude_ids, max_per_text=50):
    # пары запрос и выбранное объявление только из base
    # объявления из корпуса выкидываем иначе модель запомнит позитивы валидации
    pairs = events[events["split"] == "base"].explode("positives")
    pairs = pairs[~pairs["positives"].isin(exclude_ids)]
    pairs["query"] = [query_text(t, p) for t, p in zip(pairs["text"], pairs["search_infm_params_text"])]
    pairs = pairs.drop_duplicates(["query", "positives"]).sample(frac=1.0, random_state=42)
    # у популярных запросов тысячи пар и они забивают обучение
    pairs = pairs[pairs.groupby("text").cumcount() < max_per_text]

    info = items.set_index("item_id").loc[pairs["positives"], ["title", "params", "description"]]
    docs = [item_text(t, p, d) for t, p, d in zip(info["title"], info["params"], info["description"])]
    return Dataset.from_dict({
        "anchor": [QUERY_PREFIX + q for q in pairs["query"]],
        "positive": [DOC_PREFIX + d for d in docs],
    })


def train_encoder(dataset, out_dir, seed=42):
    model = SentenceTransformer(BASE_MODEL, device="cuda")
    model.max_seq_length = MAX_LEN

    # cached версия лосса дает большой батч 512 при небольшой памяти
    loss = CachedMultipleNegativesRankingLoss(model, mini_batch_size=128)
    args = SentenceTransformerTrainingArguments(
        output_dir=str(out_dir) + "_ckpt",
        num_train_epochs=1,
        per_device_train_batch_size=512,
        learning_rate=5e-5,
        warmup_steps=0.05,
        weight_decay=0.01,
        bf16=True,
        batch_sampler=BatchSamplers.NO_DUPLICATES,
        save_strategy="no",
        logging_steps=50,
        seed=seed,
        data_seed=seed,
        report_to="none",
    )
    trainer = SentenceTransformerTrainer(model=model, args=args, train_dataset=dataset, loss=loss)
    trainer.train()
    model.save(str(out_dir))
    return model


def load_encoder(path):
    model = SentenceTransformer(str(path), device="cuda")
    model.max_seq_length = MAX_LEN
    return model.to(torch.bfloat16).eval()


def encode(model, texts, prefix, batch_size=256):
    with torch.inference_mode():
        emb = model.encode(
            [prefix + t for t in texts],
            batch_size=batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
    return emb.astype(np.float16)
