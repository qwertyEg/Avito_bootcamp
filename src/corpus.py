import numpy as np

BENCH_FOLD = -1


class Corpus:
    """
    общий корпус для поиска

    в нем весь бенчмарк плюс позитивы отложенных запросов из train
    каждый запрос видит бенчмарк и только позитивы своего фолда
    так на валидации в корпус подмешано столько же чужих позитивов сколько в бенчмарке
    """

    def __init__(self, items, events):
        held_out = events[events["split"] != "base"]
        fold_of_item = held_out[["fold", "positives"]].explode("positives")

        keep = items["in_bench"] | items["item_id"].isin(set(fold_of_item["positives"]))
        self.items = items[keep].reset_index(drop=True)
        self.row_of = dict(zip(self.items["item_id"], self.items.index))

        in_bench = self.items["in_bench"].to_numpy()
        self.masks = {BENCH_FOLD: in_bench}
        for fold, ids in fold_of_item.groupby("fold")["positives"]:
            mask = in_bench.copy()
            mask[[self.row_of[i] for i in ids]] = True
            self.masks[fold] = mask

        self.location_rows = self.items.groupby("location").indices

    def __len__(self):
        return len(self.items)

    def pool(self, locations, fold):
        # строки корпуса из нужных локаций которые видны этому фолду
        rows = [self.location_rows[loc] for loc in locations if loc in self.location_rows]
        if not rows:
            return np.array([], dtype=np.int64)
        rows = np.concatenate(rows)
        return rows[self.masks[fold][rows]]
