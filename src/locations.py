import numpy as np
import pandas as pd


def haversine_km(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 6371.0 * 2 * np.arcsin(np.sqrt(a))


class LocationModel:
    """
    связь локации поиска с локациями выбранных объявлений

    поиск часто идет по региону а объявление привязано к городу
    например поиск по московской области чаще всего заканчивается объявлением из москвы
    поэтому по base считаем куда реально ходят из каждой локации поиска
    """

    def fit(self, choices, items, corpus_items):
        # choices это пары search_location_id и item_id из base
        # вес пары 1 / число позитивов в поиске чтобы большие поиски не перетягивали
        info = items.set_index("item_id")[["location", "lat", "lon"]]
        choices = choices.join(info, on="item_id")

        counts = choices.groupby(["search_location_id", "location"])["weight"].sum().rename("n").reset_index()
        totals = counts.groupby("search_location_id")["n"].sum()
        self.counts = {(s, l): n for s, l, n in zip(counts["search_location_id"], counts["location"], counts["n"])}
        self.totals = totals.to_dict()
        self.targets = counts.groupby("search_location_id")["location"].agg(set).to_dict()

        # центр локации считаем по объявлениям корпуса
        # для регионов своих объявлений нет и центр берем по выбранным из них объявлениям
        centers = corpus_items.groupby("location")[["lat", "lon"]].median()
        search_centers = choices.groupby("search_location_id")[["lat", "lon"]].median()
        missing = search_centers.index.difference(centers.index)
        self.centers = pd.concat([centers, search_centers.loc[missing]])
        self.own_locations = set(corpus_items["location"])
        return self

    def is_region(self, search_location):
        # у региона нет своих объявлений в корпусе
        return search_location not in self.own_locations

    def log_prob(self, search_location, item_locations):
        # сглаживание единицей на собственную локацию
        # иначе редкая локация без истории в base получает ноль даже для объявлений из нее самой
        own = 0.0 if self.is_region(search_location) else 1.0
        total = self.totals.get(search_location, 0.0) + own
        counts = np.array([self.counts.get((search_location, loc), 0.0) for loc in item_locations])
        counts = counts + own * (item_locations == search_location)
        with np.errstate(divide="ignore"):
            logp = np.log(counts / max(total, 1e-9) + 1e-3)
        return logp.astype(np.float32)

    def distance_km(self, search_location, lat, lon):
        if search_location not in self.centers.index:
            return np.full(len(lat), np.nan, dtype=np.float32)
        center = self.centers.loc[search_location]
        return haversine_km(center["lat"], center["lon"], lat, lon).astype(np.float32)

    def allowed_locations(self, search_location):
        return {search_location} | self.targets.get(search_location, set())
