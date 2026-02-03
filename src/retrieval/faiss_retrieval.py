import faiss
import numpy as np
class FaissRetriever:
    def __init__(self, item_vecs: np.ndarray, use_cosine: bool = False):
        """
            item_vecs: shape (num_items, dim)
            use_cosine: if true, normalize vectors so that IP becomes cosine similarity 
        """
        if item_vecs is None: 
            raise ValueError("item_vecs can't be none")
        
        self.item_vecs = item_vecs.astype("float32")
        self.use_cosine = use_cosine
        
        if use_cosine:
            faiss.normalize_L2(self.item_vecs)
        
        dim = self.item_vecs.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(self.item_vecs)
        
    def search(self, query_vec: np.ndarray, k: int = 10):
        """
            query_vec: shape (dim) or (1,dim)

        Returns:
            (indices, scores)
        """
        query_vec = query_vec.astype("float32").reshape(1,-1)
        
        if self.use_cosine:
            faiss.normalize_L2(query_vec)
        
        scores, idxs = self.index.search(query_vec, k)
        return idxs[0], scores[0]
    
    def save(self, filepath:str):
        faiss.write_index(self.index, filepath)
    
    @staticmethod
    def load(filepath: str, use_cosine: bool = False):
        """
        Loads a saved FAISS index (embeddings no longer needed)
        """
        retriever = object.__new__(FaissRetriever)
        retriever.index = faiss.read_index(filepath)
        retriever.use_cosine = use_cosine
        retriever.item_vecs = None
        return retriever 