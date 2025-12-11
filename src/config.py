from pathlib import Path
#Building path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"

RAW_AMAZON_FILE = DATA_DIR / "Books_5.json.gz"

#Standardizing column names
USER_COL = "user_id"
ITEM_COL = "item_id"
RATING_COL = "rating"
TS_COL = "timestamp"
