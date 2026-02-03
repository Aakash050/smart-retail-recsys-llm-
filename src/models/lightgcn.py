import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from src.retrieval.faiss_retrieval import FaissRetriever
from collections import defaultdict

class BPRDataset(Dataset):
    """
    Dataset of users, positive_items, negative_items for BPR training
    """
    def __init__(self, user_indices, item_indices, num_items, user_seen):
        self.user_indices = user_indices
        self.item_indices = item_indices
        self.num_items = num_items
        self.user_seen = user_seen
    def __len__(self):
        return len(self.user_indices)
    def __getitem__(self, idx):
        u = self.user_indices[idx]
        i_pos = self.item_indices[idx]
        while True: 
            j_neg = np.random.randint(self.num_items)
            if j_neg not in self.user_seen[u] and j_neg != i_pos:
                break
        return u, i_pos, j_neg
class LightGCNRecommender:
    def __init__(
        self, 
        num_users,
        num_items,
        embedding_dim = 64,
        num_layers = 3,
        reg = 1e-4,
        lr = 1e-3,
        device = None,
    ):
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers
        self.reg = reg
        self.lr = lr
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.user_embeddings = nn.Embedding(num_users, embedding_dim)
        self.item_embeddings = nn.Embedding(num_items, embedding_dim)
        self.adj = None        #adjacency matrix
        self.optimizer = torch.optim.Adam(
            list(self.user_embeddings.parameters()) +
            list(self.item_embeddings.parameters()),
            lr = self.lr
        )
        self.user_id_to_index = {}
        self.item_id_to_index = {}
        self.index_to_user_id = {}
        self.index_to_item_id = {}
        self.user_embeddings.to(self.device)
        self.item_embeddings.to(self.device)
    def build_mappings(self, df):
        users = df["user_id"].astype("category")
        items = df["item_id"].astype("category")
        self.index_to_user_id = dict(enumerate(users.cat.categories))
        self.index_to_item_id = dict(enumerate(items.cat.categories))
        self.user_id_to_index = {v: k for k, v in self.index_to_user_id.items()}
        self.item_id_to_index = {v: k for k, v in self.index_to_item_id.items()}
        user_idx = users.cat.codes.to_numpy()
        item_idx = items.cat.codes.to_numpy()
        self.user_seen = defaultdict(set)
        for u, i in zip(user_idx, item_idx):
            self.user_seen[u].add(i)
        return user_idx, item_idx
    def build_adj_matrix(self, user_idx, item_idx):
        """
        Build normalized adj matrix for algorithim
        """
        num_nodes = self.num_users + self.num_items
        user_nodes = torch.tensor(user_idx, dtype = torch.long)
        item_nodes = torch.tensor(item_idx + self.num_users, dtype = torch.long)
        indices = torch.stack([
            torch.cat([user_nodes, item_nodes]),
            torch.cat([item_nodes, user_nodes])
        ])
        values = torch.ones(indices.size(1), dtype = torch.float32)
        adj = torch.sparse.FloatTensor(
            indices, 
            values, 
            torch.Size([num_nodes, num_nodes])
        )
        deg = torch.sparse.sum(adj, dim=1).to_dense()
        deg_inv_sqrt = torch.pow(deg + 1e-10, -0.5)
        d_mat_inv_sqrt = deg_inv_sqrt
        row, col = indices
        norm_values = d_mat_inv_sqrt[row] * values * d_mat_inv_sqrt[col]
        self.adj = torch.sparse.FloatTensor(
            indices, 
            norm_values,
            torch.Size([num_nodes, num_nodes])
        ).coalesce().to(self.device)
    def propogate_embeddings(self):
        """
        Perform k layers of propogation, 
        return final user/item embeddings
        """
        user_emb = self.user_embeddings.weight
        item_emb = self.item_embeddings.weight
        all_emb = torch.cat([user_emb, item_emb], dim = 0)
        embs_per_layer = [all_emb]
        for _ in range(self.num_layers):
            all_emb = torch.sparse.mm(self.adj, all_emb)
            embs_per_layer.append(all_emb)
        final_emb = torch.stack(embs_per_layer, dim = 0).mean(dim = 0)
        final_user_emb = final_emb[:self.num_users]
        final_item_emb = final_emb[self.num_users:]
        return final_user_emb, final_item_emb
    def bpr_loss(self, u_emb, pos_emb, neg_emb):
        """

        Bayesian Personalized Ranking Loss
        Positive scores > Negative Scores
        """
        pos_scores = (u_emb * pos_emb).sum(dim=1)
        neg_scores = (u_emb * neg_emb).sum(dim=1)
        loss = -torch.log(torch.sigmoid(pos_scores - neg_scores) + 1e-10).mean()
        reg_term = (
            u_emb.norm(2).pow(2) + 
            pos_emb.norm(2).pow(2) +
            neg_emb.norm(2).pow(2) 
        ) / u_emb.size(0)
        
        return loss + self.reg * reg_term
    def fit(self, df, epochs = 80, batch_size = 1024):
        """
        Train LightGCN on given DataFrame
        """
        user_idx, item_idx = self.build_mappings(df)
        self.build_adj_matrix(user_idx, item_idx)
        dataset = BPRDataset(user_idx, item_idx, self.num_items, self.user_seen)
        loader = DataLoader(dataset, batch_size = batch_size, shuffle = True)
        self.user_embeddings.train()
        self.item_embeddings.train()
        for epoch in range(epochs):
            for u, i_pos, j_neg in loader:
                u = u.to(self.device).long()
                i_pos = i_pos.to(self.device).long()
                j_neg = j_neg.to(self.device).long()
                user_emb_final, item_emb_final = self.propogate_embeddings()
                u_emb = user_emb_final[u]
                pos_emb = item_emb_final[i_pos]
                neg_emb = item_emb_final[j_neg]
                loss = self.bpr_loss(u_emb, pos_emb, neg_emb)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
        self.user_emb_final, self.item_emb_final = self.propogate_embeddings()
        self.build_faiss(use_cosine=True)
        self.user_embeddings.eval()
        self.item_embeddings.eval()
        return self
    def recommend(self, user_id, k = 10):
        """
        Recommend top - k items for given user ID
        """
        if user_id not in self.user_id_to_index:
            return[]
        if not hasattr(self, "faiss_retriever"):
            raise ValueError ("Faiss Index not built, call fit first")
        u_idx = self.user_id_to_index[user_id]
        u_vec = self.get_user_embedding_np(user_id)
        idxs, _ = self.faiss_retriever.search(u_vec, k*2)
        seen =  self.user_seen.get(u_idx, set())
        results = []
        for i in idxs:
            if i not in seen: 
                results.append(self.index_to_item_id[int(i)])    
            if len(results) == k: 
                break
        return results
    #Use for FAISS: 
    def get_item_embeddings_np(self):
        if not hasattr(self, "item_emb_final"):
            raise ValueError("No item_emb_final found. Train the model first using fit().")
        return self.item_emb_final.detach().cpu().numpy().astype("float32")
    def get_user_embedding_np(self, user_id):
        if user_id not in self.user_id_to_index:
            raise ValueError(f"Unknown user_id: {user_id}")
        if not hasattr(self, "user_emb_final"):
            raise ValueError("No user_emb_final found. Train the model first using fit().")

        u_idx = self.user_id_to_index[user_id]
        return self.user_emb_final[u_idx].detach().cpu().numpy()
    def build_faiss(self, use_cosine=True):
        item_vecs = self.get_item_embeddings_np()
        self.faiss_retriever = FaissRetriever(
        item_vecs,
        use_cosine=True
    )
        return self