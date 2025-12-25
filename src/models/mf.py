from implicit.als import AlternatingLeastSquares
from scipy.sparse import coo_matrix
class MFRecommender:
    def __init__(self, factors = 64, regularization = 0.01, iterations = 20):
        self.model = AlternatingLeastSquares(
            factors = factors, 
            regularization=regularization,
            iterations = iterations
        )
        self.user_map = None
        self.item_map = None
        self.matrix = None
    def fit(self, df):
        users = df['user_id'].astype("category")
        items = df['item_id'].astype("category")
        self.user_map = dict(enumerate(users.cat.categories))
        self.item_map = dict(enumerate(items.cat.categories))
        rows = users.cat.codes.to_numpy()
        cols = items.cat.codes.to_numpy()
        """While standard is to use 1 for all implicit ratings (all interactions are worth the same, 
        I want higher ratings to have higher confidence values in our interactions)"""
        vals = df["rating"].fillna(1).astype(float).to_numpy()
        self.matrix = coo_matrix((vals, (rows, cols))).tocsr()
        self.model.fit(self.matrix)
        return self
    def recommend(self, user_id, k = 10):
        if user_id not in set(self.user_map.values()):
            #Placeholder, we will eventually want some type of general population recommendation for new useres
            return[]
        inv_user_map = {v: k for k, v in self.user_map.items()}
        u_idx = inv_user_map[user_id]
        ids, scores = self.model.recommend(u_idx, self.matrix[u_idx], N = k)        
        mapped = [self.item_map[i] for i in ids]
        return mapped
        