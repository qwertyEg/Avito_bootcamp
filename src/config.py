from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
WORK_DIR = ROOT / "work"
WORK_DIR.mkdir(exist_ok=True)

SEED = 42

# сколько кандидатов отдаем на выходе
TOP_K = 50

# валидация повторяет устройство бенчмарка
# один случайный поиск на уникальный текст запроса
N_VAL_TEXTS = 3000
# отдельные запросы для обучения ранкера
N_RANK_TEXTS = 24000
# ранкерные запросы режем на фолды размером примерно с бенчмарк
# чтобы в корпус не подмешивалось слишком много чужих позитивов сразу
RANK_FOLD_SIZE = 3000

SEARCH_COLS = [
    "search_query",
    "search_location_id",
    "search_is_delivery_search",
    "search_infm_params_text",
    "search_category",
]
