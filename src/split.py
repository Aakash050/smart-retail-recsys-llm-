import pandas as pd
from .config import USER_COL, TS_COL, ITEM_COL
def temporal_train_val_split(
    df: pd.DataFrame, 
    val_fraction: float = 0.2
) -> tuple[pd.DataFrame, pd.DataFrame]: 
    """Split per user, by time, assumes timestamps are already converted. 
    Based on a cutoff value of 0.2, can be changed depending on user preference"""
    
    #Check for sanity, otherwise may run into silent errors: 
    if TS_COL not in df.columns: 
        raise ValueError("Split requires '{TS_COL}' column.")
    df = df.sort_values([USER_COL, TS_COL])
    
    def split_user(user_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]: 
        n = len(user_df)
        if n <= 1: 
            return user_df, user_df.iloc[0:0]     #In case of 1 or 0 interactions
        cutoff = int((1 - val_fraction) * n)
        if cutoff == 0: 
            cutoff = n - 1
        train_u = user_df.iloc[:cutoff]
        val_u = user_df.iloc[cutoff:]
        return train_u, val_u
    
    train_parts = []
    val_parts = []
    
    for _, u_df in df.groupby(USER_COL):
        tr, va = split_user(u_df)
        train_parts.append(tr)
        val_parts.append(va)
    train_df = pd.concat(train_parts).reset_index(drop = True)
    val_df = pd.concat(val_parts).reset_index(drop = True)
    
    return train_df, val_df

def get_val_ground_truth(val_df: pd.DataFrame) -> dict: 
    #Map user, getting the set of items in validation for metric computation
    user_to_items = {}
    for user, group in val_df.groupby(USER_COL):
        user_to_items[user] = set(group[ITEM_COL].tolist())
    return user_to_items