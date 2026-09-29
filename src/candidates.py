import numpy as np
import pandas as pd
from scipy.stats import rankdata

# веса приора локации подобраны на валидации
# сетка вокруг них плоская так что точные значения не критичны
LOC_TRANSITION = 1.5
LOC_SAME = 1.0
LOC_DISTANCE = 1.0

# у косинусов другой масштаб чем у bm25
DENSE_SCALE = 50.0
PROFILE_SCALE = 20.0

# сколько кандидатов берем из каждого источника
SOURCE_TOP = {"lex": 300, "dense": 200, "profile": 200}

SIGNAL_FEATURES = ["text", "dense", "profile", "mc", "loc_logp", "same_loc", "dist_km", "title_cover", "doc_cover"]


def location_prior(s):
    return LOC_TRANSITION * s["loc_logp"] + LOC_SAME * s["same_loc"] - LOC_DISTANCE * np.log1p(s["dist_km"])


def source_scores(s):
    # три способа оценить совпадение по смыслу плюс общие приоры рубрики и локации
    prior = s["mc"] + location_prior(s)
    return {
        "lex": s["text"] + prior,
        "dense": DENSE_SCALE * s["dense"] + prior,
        "profile": PROFILE_SCALE * s["profile"] + prior,
    }


def top_rows(scores, k):
    k = min(k, int(np.isfinite(scores).sum()))
    if k == 0:
        return np.array([], dtype=np.int64)
    top = np.argpartition(-scores, k - 1)[:k]
    return top[np.argsort(-scores[top])]


def positions(rows, top, missing):
    # место объявления в топе источника или missing если его там нет
    result = np.full(len(rows), missing, dtype=np.float32)
    if len(top):
        order = np.argsort(top)
        idx = np.searchsorted(top[order], rows).clip(max=len(top) - 1)
        found = top[order][idx] == rows
        result[found] = order[idx][found] + 1
    return result


def collect(s, queries, corpus):
    """
    объединяем топы источников и собираем признаки для каждой пары
    все значения берем из уже посчитанных матриц сигналов
    """
    sources = source_scores(s)
    item_ids = corpus.items["item_id"].to_numpy()
    frames = []
    for i, qid in enumerate(queries["qid"]):
        mask = s["mask"][i]
        visible = {name: np.where(mask, scores[i], -np.inf) for name, scores in sources.items()}
        tops = {name: top_rows(visible[name], k) for name, k in SOURCE_TOP.items()}
        rows = np.unique(np.concatenate(list(tops.values())))
        if len(rows) == 0:
            continue

        frame = {"qid": qid, "item_id": item_ids[rows]}
        for name in SIGNAL_FEATURES:
            frame[name] = s[name][i, rows]
        for name in ["text", "dense", "profile"]:
            # отставание от лучшего видимого объявления и место внутри кандидатов
            frame[f"{name}_gap"] = s[name][i, rows] - np.where(mask, s[name][i], -np.inf).max()
            frame[f"{name}_rank"] = rankdata(-s[name][i, rows], method="min").astype(np.float32)
        for name in sources:
            frame[f"src_{name}"] = sources[name][i, rows]
            frame[f"src_{name}_gap"] = sources[name][i, rows] - visible[name].max()
            frame[f"src_{name}_pos"] = positions(rows, tops[name], SOURCE_TOP[name] + 1)

        # вероятность подкатегории и ее место среди подкатегорий запроса
        mc_prob = s["mc_prob"][i]
        item_mc = s["item_mc"][rows]
        frame["mc_prob"] = mc_prob[item_mc]
        frame["mc_rank"] = rankdata(-mc_prob, method="min")[item_mc].astype(np.float32)
        frame["mc_top_prob"] = np.float32(mc_prob.max())
        frames.append(pd.DataFrame(frame))
    return pd.concat(frames, ignore_index=True)
