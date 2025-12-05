import os
import pandas as pd
from typing import Tuple
def load_amazon_books_raw(data_dir: str, filename: str = "Books_5.json.gz") -> pd.DataFrame:
    """Load in raw amazon books data from direct data"""
    data_path = Path(data_dir) / filename
    df = pd.read_json(data_path, lines = True)

    return df

def build_interactions(df: pd.DataFrame) -> pd.DataFrame:
    """Builds a user-item interaction table, returning: 
    user_id, item_id, rating, timestamp"""
    #Only keep necessary columns
    df = df[
        ["reviewerID", "asin", "overall", "unixReviewTime"]
    ].dropna(subset = ["reviewerID", "asin"])

    interactions = df.rename(
        columns = {
            "reviewerID": "user_id",
            "asin": "item_id",
            "overall": "rating",
            "unixReviewTime": "timestamp",
        }
    ).copy()
    interactions["timestamp"] = pd.to_datetime(interactions["timestamp"], unit = "s")
    
    
    
    return interactions[["user_id", "item_id", "rating", "timestamp"]]

def load_interactions(data_dir: str) -> pd.DataFrame:
    "Convenience"
    raw = load_amazon_books_raw(data_dir)
    interactions = build_interactions(raw)
    return interactions
    
    