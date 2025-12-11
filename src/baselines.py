import pandas as pd
from collections import defaultdict
from .config import USER_COL, ITEM_COL, RATING_COL
class PopularityRecommender:
    """Global item popularity baseline"""
    def __init__(self, top_k: int = 1000):
        self.top_k = top_k
        self.item_ranking: list | None = None
    def fit(self, df: pd.DataFrame) -> "PopularityRecommender": 
        item_scores = (
            df.groupby(ITEM_COL)[RATING_COL]
            .sum()
            .sort_values(ascending = False)
        )
        self.item_ranking = item_scores.index.tolist()
        if self.top_k is not None: 
            self.item_ranking = self.item_ranking[: self.top_k]
        return self
    def recommend(self, user_id, k: int = 10) -> list: 
        if self.item_ranking is None: 
            raise RuntimeError("Model not fitted yet")
        return self.item_ranking[:k]
    
class UserHistoryFilterMixin: 
    """
    Class to remove items user has already interacted with in train from 
    reccomendation lists
    """
    def build_user_history(self, df: pd.DataFrame):
        self.user_history = defaultdict(set)
        for _,row in df[[USER_COL, ITEM_COL]].iterrows():
            self.user_history[row[USER_COL]].add(row[ITEM_COL])
            
from sklearn.neighbors import NearestNeighbors
import numpy as np
class ItemKNNRecommender(UserHistoryFilterMixin):
    "Item based KNN using cosine similarity on implicit feedback"
    def __init__(self, n_neighbors: int = 50): 
        self.n_neighbors = n_neighbors
        self.model: NearestNeighbors | None = None
        self.item_index: dict | None = None
        self.index_item: dict | None = None
        self.user_item_matrix: np.ndarray | None = None
    def fit(self, df: pd.DataFrame) -> "ItemKNNRecommender": 
        unique_items = df[ITEM_COL].unique()
        self.item_index = {item: i for i, item in enumerate(unique_items)}
        self.index_item = {i: item for item, i in self.item_index.items()}
        unique_users = df[USER_COL].unique()
        user_index = {u: i for i, u in enumerate(unique_users)}
        mat = np.zeros((len(unique_users), len(unique_items)), dtype = np.float32)
        
        for _, row in df.iterrows():
            u = row[USER_COL]
            it = row[ITEM_COL]
            qty = row.get(RATING_COL, 1)
            mat[user_index[u], self.item_index[it]] += float (qty)
        self.model = NearestNeighbors(
            n_neighbors = self.n_neighbors,
            metric = "cosine", #Cosine is used because we only care about direction, not magnitude
            algorithm = "brute", #Best for single latop use and with small dataset 
        )
        self.model.fit(mat.T)
        self.user_item_matrix = mat
        self.user_index = user_index
        self.build_user_history(df)
        return self
    def recommend(self, user_id, k: int = 10) -> list: 
        if self.model is None: 
            raise RuntimeError("Model not fitted")
        if user_id not in self.user_index:
            return []
        u_idx = self.user_index[user_id]
        user_vec = self.user_item_matrix[u_idx, :].reshape(1,-1)
        interacted_items = np.where(self.user_item_matrix[u_idx, :] >0)[0]
        if len(interacted_items) == 0: 
            return []
        candidate_items = set()
        for it_idx in interacted_items:
            distances, indices = self.model.kneighbors(
                self.user_item_matrix[:, it_idx].reshape(1,-1),
                n_neighbors = self.n_neighbors,
            )
            candidate_items.update(indices[0].tolist())
        already_seen = getattr(self, "user_history", {}).get(user_id, set())
        ranked = [
            self.index_item[i]
            for i in candidate_items
            if self.index_item[i] not in already_seen
            ]
        return ranked[:k]
    