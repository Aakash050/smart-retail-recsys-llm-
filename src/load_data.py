import os
import pandas as pd
from typing import Tuple
from pathlib import Path
def load_amazon_books_raw(data_dir: str, 
                          filename: str = "Books_5.json.gz",
                         nrows: int | None = None,) -> pd.DataFrame:
    """Load in raw amazon books data from direct data"""
    data_path = Path(data_dir) / filename
    df = pd.read_json(data_path, lines = True, nrows = nrows)

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

def filter_min_activity(df, min_user_interactions = 3, min_item_interactions = 3): 
    """
    Helps clean dataset, filters out too few interactions
    """
    user_counts = df['user_id'].value_counts()
    good_users = user_counts[user_counts >= min_user_interactions].index
    df = df[df['user_id'].isin(good_users)]

    item_counts = df['item_id'].value_counts()
    good_items = item_counts[item_counts >= min_item_interactions].index
    df = df[df['item_id'].isin(good_items)]
    
    return df

def load_interactions(data_dir: str, nrows: int | None = None) -> pd.DataFrame:
    "Convenience"
    raw = load_amazon_books_raw(data_dir, nrows = nrows)
    interactions = build_interactions(raw)
    return interactions
    
    