import numpy as np


def recall_at_k(predictions, positives, k=50):
    # доля позитивов попавших в топ k и среднее по запросам как в условии
    scores = []
    for pred, pos in zip(predictions, positives):
        top = set(pred[:k])
        scores.append(sum(p in top for p in pos) / len(pos))
    return float(np.mean(scores))


def recall_report(predictions, positives, ks=(50, 100, 200, 500, 1000)):
    return {k: round(recall_at_k(predictions, positives, k), 4) for k in ks}
