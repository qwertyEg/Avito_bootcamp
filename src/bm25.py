import numpy as np
import scipy.sparse as sp
from sklearn.feature_extraction.text import CountVectorizer


class BM25:
    """
    обычный okapi bm25 на разреженных матрицах

    документы это строки лемм через пробел
    вес поля задаем повтором его текста в документе
    скоринг пачки запросов это одно умножение матриц
    """

    def __init__(self, k1=1.2, b=0.75):
        self.k1 = k1
        self.b = b

    def fit(self, docs):
        self.vectorizer = CountVectorizer(analyzer=str.split, dtype=np.float32)
        tf = self.vectorizer.fit_transform(docs).tocsr()
        self.vocab = self.vectorizer.vocabulary_

        n_docs = tf.shape[0]
        df = np.bincount(tf.indices, minlength=tf.shape[1])
        idf = np.log(1 + (n_docs - df + 0.5) / (df + 0.5)).astype(np.float32)

        doc_len = np.asarray(tf.sum(axis=1)).ravel()
        norm = self.k1 * (1 - self.b + self.b * doc_len / doc_len.mean())
        rows = np.repeat(np.arange(n_docs), np.diff(tf.indptr))
        tf.data = tf.data * (self.k1 + 1) / (tf.data + norm[rows]) * idf[tf.indices]

        # слово x документ для скоринга всего корпуса сразу
        self.weights_t = tf.T.tocsr()
        return self

    def query_matrix(self, weighted_queries):
        # каждый запрос это словарь слово -> вес
        rows, cols, vals = [], [], []
        for i, terms in enumerate(weighted_queries):
            for word, weight in terms.items():
                j = self.vocab.get(word)
                if j is not None:
                    rows.append(i)
                    cols.append(j)
                    vals.append(weight)
        return sp.csr_matrix((vals, (rows, cols)), shape=(len(weighted_queries), len(self.vocab)), dtype=np.float32)

    def score(self, weighted_queries):
        return (self.query_matrix(weighted_queries) @ self.weights_t).toarray()
