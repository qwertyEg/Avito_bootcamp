#!/bin/bash
# полный прогон от сырых parquet до answer.csv
# файлы train.parquet benchmark_items.parquet benchmark_queries.parquet должны лежать в data/
set -e
cd "$(dirname "$0")"

python prepare.py
python train_dense.py
python build_candidates.py
python train_ranker.py
python predict.py
