import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize


def top_k_sparse(sims, k):
    # оставляем в каждой строке k самых больших значений
    sims = sims.tocsr()
    rows, cols, vals = [], [], []
    for i in range(sims.shape[0]):
        start, end = sims.indptr[i], sims.indptr[i + 1]
        if start == end:
            continue
        row_vals = sims.data[start:end]
        keep = np.argsort(-row_vals)[:k]
        rows.append(np.full(len(keep), i))
        cols.append(sims.indices[start:end][keep])
        vals.append(row_vals[keep])
    if not rows:
        return sp.csr_matrix(sims.shape, dtype=np.float32)
    return sp.csr_matrix(
        (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=sims.shape, dtype=np.float32
    )


class Behaviour:
    """
    что пользователи выбирали в base по похожим запросам

    для нового запроса ищем близкие по написанию тексты из base
    и смотрим какие рубрики и какие заголовки там выбирали
    объявления из корпуса в статистику не берем
    у 37 процентов позитивов валидации есть история в base а у объявлений бенчмарка только у 9
    с ними валидация была бы заметно оптимистичнее бенчмарка
    """

    def __init__(self, k=100, power=6):
        self.k = k
        self.power = power

    def fit(self, choices, items):
        # choices это пары text item_id weight из base
        info = items.set_index("item_id")[["microcat"]]
        choices = choices.join(info, on="item_id")

        self.texts = np.array(sorted(choices["text"].unique()))
        text_index = pd.Series(np.arange(len(self.texts)), index=self.texts)
        self.vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(1, 4), sublinear_tf=True, min_df=2)
        self.text_matrix = self.vectorizer.fit_transform(self.texts)

        self.microcats = np.array(sorted(items["microcat"].unique()))
        mc_index = pd.Series(np.arange(len(self.microcats)), index=self.microcats)
        rows = text_index[choices["text"]].to_numpy()
        self.text_microcat = normalize(
            sp.csr_matrix(
                (choices["weight"], (rows, mc_index[choices["microcat"]].to_numpy())),
                shape=(len(self.texts), len(self.microcats)),
            ),
            norm="l1",
        )

        self.chosen_ids = np.array(sorted(choices["item_id"].unique()))
        item_index = pd.Series(np.arange(len(self.chosen_ids)), index=self.chosen_ids)
        self.text_item = normalize(
            sp.csr_matrix(
                (choices["weight"], (rows, item_index[choices["item_id"]].to_numpy())),
                shape=(len(self.texts), len(self.chosen_ids)),
            ),
            norm="l1",
        )
        return self

    def neighbours(self, query_texts, k=None, power=None):
        # веса соседей sim^power сильно поднимают почти совпадающие тексты
        sims = self.vectorizer.transform(query_texts) @ self.text_matrix.T
        sims = top_k_sparse(sims, k or self.k)
        sims.data = sims.data ** (power or self.power)
        return normalize(sims, norm="l1")

    def nearest_similarity(self, query_texts):
        # насколько близкий текст вообще есть в base
        sims = self.vectorizer.transform(query_texts) @ self.text_matrix.T
        return np.asarray(sims.max(axis=1).todense()).ravel()

    def microcat_distribution(self, weights):
        return np.asarray((weights @ self.text_microcat).todense())

    def chosen_items(self, weights):
        # распределение по выбранным объявлениям соседних запросов
        return weights @ self.text_item
