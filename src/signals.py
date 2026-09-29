import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize

# слова из фильтров которые есть почти в каждом объявлении и ничего не дают
FILTER_STOPWORDS = {"вид", "услуга", "тип"}


def query_terms(query_lemmas, filter_lemmas, filter_weight=0.5):
    terms = {word: 1.0 for word in query_lemmas}
    for word in filter_lemmas:
        if word not in terms and word not in FILTER_STOPWORDS:
            terms[word] = filter_weight
    return terms


class Signals:
    """
    сигналы запрос x объявление сразу по всему корпусу

    text     bm25 по лемматизированным заголовку параметрам и описанию
    dense    косинус дообученного энкодера
    profile  похожесть заголовка на заголовки того что выбирали по похожим запросам
    mc       логарифм вероятности подкатегории объявления для запроса
    loc      приор локации из переходов base и расстояния до центра локации поиска
    """

    def __init__(self, corpus, title_lemmas, bm25, titles, behaviour, microcat_prior, locations, item_emb):
        # title_lemmas это леммы заголовков всех объявлений по item_id
        self.corpus = corpus
        self.bm25 = bm25
        self.titles = titles
        self.behaviour = behaviour
        self.microcat_prior = microcat_prior
        self.locations = locations
        self.item_emb = item_emb.astype(np.float32)

        items = corpus.items
        mc_index = pd.Series(np.arange(len(behaviour.microcats)), index=behaviour.microcats)
        self.item_mc = mc_index[items["microcat"]].to_numpy()
        self.item_loc = items["location"].to_numpy()
        self.item_lat = items["lat"].to_numpy()
        self.item_lon = items["lon"].to_numpy()
        self.unique_locs, self.item_loc_idx = np.unique(self.item_loc, return_inverse=True)

        self.item_titles = titles.transform(title_lemmas[items["item_id"]])
        # заголовки объявлений которые выбирали в base
        self.chosen_titles = normalize(titles.transform(title_lemmas[behaviour.chosen_ids]))

        # какие слова вообще есть в заголовке и в документе
        # нужно для доли слов запроса которые нашлись в объявлении
        title_terms = bm25.vectorizer.transform(title_lemmas[items["item_id"]])
        title_terms.data[:] = 1.0
        self.title_terms_t = title_terms.T.tocsr()
        self.doc_terms_t = bm25.weights_t.copy()
        self.doc_terms_t.data[:] = 1.0

    def lexical_microcats(self, text, masks, top=200):
        # в каких подкатегориях лежат лучшие совпадения по bm25
        dist = np.zeros((len(text), len(self.behaviour.microcats)), dtype=np.float32)
        for i, (row, mask) in enumerate(zip(text, masks)):
            scores = np.where(mask, row, -np.inf)
            best = np.argpartition(-scores, top)[:top]
            best = best[np.isfinite(scores[best]) & (scores[best] > 0)]
            if len(best) == 0:
                continue
            top_score = scores[best].max()
            weights = np.exp((scores[best] - top_score) / (top_score / 10))
            np.add.at(dist[i], self.item_mc[best], weights)
        return dist / np.maximum(dist.sum(1, keepdims=True), 1e-9)

    def location_scores(self, search_location):
        logp = self.locations.log_prob(search_location, self.unique_locs)[self.item_loc_idx]
        same = (self.item_loc == search_location).astype(np.float32)
        dist = self.locations.distance_km(search_location, self.item_lat, self.item_lon)
        dist = np.nan_to_num(dist, nan=np.nanmedian(dist) if np.isfinite(dist).any() else 0.0)
        return logp, same, dist

    def profile(self, query_texts, query_title_lemmas):
        weights = self.behaviour.neighbours(query_texts, k=50, power=4)
        prof = normalize(self.behaviour.chosen_items(weights) @ self.chosen_titles)
        # сам текст запроса тоже часть профиля
        prof = normalize(prof + self.titles.transform(query_title_lemmas))
        return (prof @ self.item_titles.T).toarray().astype(np.float32)

    def coverage(self, query_lemmas):
        # доля слов запроса в заголовке и во всем тексте объявления
        # bm25 сглаживает точные совпадения а тут они видны явно
        words = [dict.fromkeys(q.split(), 1.0) for q in query_lemmas]
        n_words = np.array([max(len(w), 1) for w in words], dtype=np.float32)[:, None]
        q = self.bm25.query_matrix(words)
        title = (q @ self.title_terms_t).toarray() / n_words
        doc = (q @ self.doc_terms_t).toarray() / n_words
        return title.astype(np.float32), doc.astype(np.float32)

    def compute(self, queries, q_emb):
        masks = np.stack([self.corpus.masks[f] for f in queries["fold"]])
        terms = [query_terms(q.split(), f.split()) for q, f in zip(queries["lemmas"], queries["filter_lemmas"])]

        text = self.bm25.score(terms).astype(np.float32)
        prior = self.microcat_prior.predict(
            queries["text"].tolist(),
            queries["search_infm_params_text"].tolist(),
            self.lexical_microcats(text, masks),
        )
        mc = np.log(prior[:, self.item_mc] + 1e-2).astype(np.float32)

        loc_parts = [self.location_scores(s) for s in queries["search_location_id"]]
        logp = np.stack([p[0] for p in loc_parts])
        same = np.stack([p[1] for p in loc_parts])
        dist = np.stack([p[2] for p in loc_parts])
        title_cover, doc_cover = self.coverage(queries["lemmas"])

        return {
            "title_cover": title_cover,
            "doc_cover": doc_cover,
            "mask": masks,
            "item_mc": self.item_mc,
            "text": text,
            "dense": (q_emb.astype(np.float32) @ self.item_emb.T).astype(np.float32),
            "profile": self.profile(queries["text"].tolist(), queries["lemmas"].tolist()),
            "mc": mc,
            "mc_prob": prior,
            "loc_logp": logp,
            "same_loc": same,
            "dist_km": dist,
        }
