import numpy as np
import pandas as pd

from src.params import KIND_KEYS, parse_params


def filter_key(search_params):
    # каноничная запись смысловых фильтров поиска
    pairs = sorted(f"{k}={v}" for k, v in parse_params(search_params) if k in KIND_KEYS)
    return "|".join(pairs)


def service_kind(search_params):
    for key, value in parse_params(search_params):
        if key == "Вид услуги":
            return value
    return ""


class MicrocatPrior:
    """
    распределение подкатегории объявления для запроса

    складываем три источника
    1 что выбирали по похожим текстам в base
    2 в каких подкатегориях лежат лучшие лексические совпадения
    3 фильтры поиска вроде "вид услуги красота здоровье" которые сильно сужают рубрику
    """

    def __init__(self, behaviour, params_power=0.6, smoothing=5.0):
        self.behaviour = behaviour
        self.microcats = behaviour.microcats
        self.params_power = params_power
        self.smoothing = smoothing

    def fit(self, choices, items):
        # choices это пары search_infm_params_text item_id weight из base
        mc_index = pd.Series(np.arange(len(self.microcats)), index=self.microcats)
        mc = choices["item_id"].map(items.set_index("item_id")["microcat"])
        frame = pd.DataFrame({
            "mc": mc_index[mc].to_numpy(),
            "weight": choices["weight"].to_numpy(),
            "filters": choices["search_infm_params_text"].map(filter_key).to_numpy(),
            "kind": choices["search_infm_params_text"].map(service_kind).to_numpy(),
        })

        self.global_dist = self._counts(frame)
        self.global_dist /= self.global_dist.sum()
        # таблицы частот по точному набору фильтров и отдельно по виду услуги
        self.by_filters = {k: self._counts(g) for k, g in frame[frame["filters"] != ""].groupby("filters")}
        self.by_kind = {k: self._counts(g) for k, g in frame[frame["kind"] != ""].groupby("kind")}
        return self

    def _counts(self, frame):
        return np.bincount(frame["mc"], weights=frame["weight"], minlength=len(self.microcats))

    def params_dist(self, search_params):
        # сглаживаем к более общему уровню если точных фильтров в base было мало
        dist = self.global_dist
        for table, key in [(self.by_kind, service_kind(search_params)), (self.by_filters, filter_key(search_params))]:
            counts = table.get(key)
            if counts is not None:
                dist = (counts + self.smoothing * dist) / (counts.sum() + self.smoothing)
        return dist

    def predict(self, query_texts, search_params, lexical_mc):
        # lexical_mc это распределение подкатегорий по топу лексического поиска
        knn = self.behaviour.microcat_distribution(self.behaviour.neighbours(query_texts))
        dist = knn + lexical_mc
        dist /= np.maximum(dist.sum(1, keepdims=True), 1e-9)

        # фильтры работают как множитель к общей частоте рубрики
        params = np.stack([self.params_dist(p) for p in search_params])
        dist *= ((params + 1e-4) / (self.global_dist + 1e-4)) ** self.params_power
        # запрос без соседей и без лексики получает общую частоту
        dist += 1e-6 * self.global_dist
        return dist / dist.sum(1, keepdims=True)
