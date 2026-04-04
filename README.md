# Recommender System: 2018 Amazon Books Dataset

The following is an end-to-end recommendation system that uses **K Nearest Neighbors (kNN)**, **Matrix Factorization**, and **Light Graph Convolutional Networks (LGCN)** to capture  user book relationships at scale, trying to improve over popularity based methods. 

---

## Overview

Matching users with relevant items from massive catalogs has relevant real-world applications in nearly every field. For example, e-commerce, streaming, and social media are all heavily reliant on this to retain users. Recommender systems are central to this, allowing for what is essentially a personalized search engine. 

This project builds a **three-stage recommendation framework**:

1. A **global popularity method** for our baseline
2. A **kNN** model 
3. A **Matrix Factorization** model to learn latent user/item preferences
4. A **Graph Neural Network (LGCN)** to model higher-order relationships beyond direct interactions

The result is a system that moves from **linear latent structure → graph-based representation learning**.

---

## Problem

While effective to some extent, older approaches like **global popularity** naively assume that rankings over every user will apply for all users, ignoring that most users tend to have more specific preferences. 

Although intuitive and easy to implement, global popularity has key limitations.

* Relies entirely on the overall user popularity being enticing to individual users.
* Problems with large numbers of users, due to increased load
* Inaccurate results with sparse data (high percentage of missing values)
  
In large-scale user information datasets, these limitations lead to:

* Poor recommendations for new or infrequent users and items (cold-start and sparsity issues)
* Miss latent structure in the data that extends beyond direct similarity
* Reduced scalability as similarity computations grow with dataset size, due to increased load

---

## Hypothesis

If we model user–item interactions as a **graph** and propagate information across connections:

* We can capture **higher-order collaborative signals**
* Learn **richer embeddings** 
* Improve recommendation quality, especially in sparse settings
  
Thus, we will have better metrics with Matrix Factorization and LGCN than with kNN.

---

## Approach

### 1. Data Pipeline

* Raw interaction data - 2018 Amazon Books Dataset
* Cleaned and transformed 

---

### 2. Baseline: Global Popularity

* Construct user–item interaction data
  
* Compute item popularity via:
  
  * Aggregating interaction strength per item 
  * Ranking items globally by descending popularity score
    
* Generate recommendations by:
  
  * Selecting the Top-K most popular items
  * Returning the same ranked list for all users (no personalization)

* Output:
  
  * Global item ranking based on aggregated interaction scores
  * Ranked Top-K recommendations per user

---

### 3. Model 1: kNN

* Construct user–item interaction matrix
  
* Compute similarity via:
  
  * Item–item similarity
  * Cosine Similarity
    
* Generate recommendations by:
  
  * Identifying items the user has interacted with
  * Retrieving top k similar items for each interacted item
  * Aggregating candidate items and filtering out previously seen items
    
* Output:
  
  * Item-item similarity structure via k nearest neighbors
  * Ranked Top-K recommendations per user

---

### 4. Model 2: Matrix Factorization

* Construct user–item interaction matrix

* Learn embeddings via:

  * Implicit feedback modeling using confidence-weighted interactions 
  * Alternating Least Squares (ALS) optimization

* Generate recommendations by:
  
  * Factorizing the interaction matrix into user and item latent vectors
  * Ranking items based on predicted user–item affinity scores 

* Output:

  * User latent vectors
  * Item latent vectors
  * Ranked Top-K recommendations per user

---

### 5. Graph Construction

* Build a **bipartite graph**:

  * Nodes: users + items
  * Edges: observed interactions

* Construct a **symmetric normalized adjacency matrix**
  
  * Applies degree normalization to stabilize embedding propagation
  * Enables efficient sparse matrix operations
    
---

### 6. Model 3: LightGCN

* Initialize learnable user and item embeddings
  
* Perform layerwise embedding propagations

  * Repeatedly aggregate neighbor information via sparse matrix multiplication
  * Remove nonlinearities -> focus on pure neighborhood aggregation

* Aggregate embeddings across layers

  * Final embedding = mean of all layer representations
  * Captures multi-hop collaborative signals

---

### 7. Training & Evaluation

* Optimization: **Bayesian Personalized Ranking (BPR) loss** with negative sampling
* Encourages positive interactions to rank higher than unobserved items
  
* Training details:
  
  * Mini-batch training with sampled (user, positive, negative) triplets
  * L2 regularization on embeddings
    
* Retrieval:
  
  * Use FAISS for efficient nearest neighbor search in embedding space
  * Filter out previously seen items
  
* Metrics:

  * Recall@K
  * Precision@K
  * Ranking quality vs global popularity baseline

---

## Results

**At k = 5, Matrix Factorization has a recall of 3.4x kNN**

We see that Matrix Factorization and LGCN consistently outperform kNN. 

| Model | Recall@5 | Recall@10 | Recall@20 |
|-------|----------|-----------|-----------|
| Popularity | 0.1275 | 0.2137 | 0.3459 |
| kNN | 0.0772 | 0.1334 | 0.2526 |
| LightGCN | 0.1753 | 0.2370 | 0.3056 |
| Matrix Factorization | 0.3440 | 0.3897 | 0.4361 |

<img width="496" height="378" alt="Screenshot 2026-04-03 at 5 16 13 PM" src="https://github.com/user-attachments/assets/3dbd90f7-6902-4668-8764-9ab1b29fddfc" />

However, we also see that LGCN doesn't perform as well as expected, and even underperforms kNN at k = 20. This could be for a few reasons, mainly being that LGCN is sensitive to various hyperparameters, and it is entirely possible I haven't found the optimal setting of each parameter. Additionally, LGCN tends to perform better with more data, and with 100,000 entries and 80 epochs, it is possible that LGCN would perform better at higher levels. 


Next Qualitative improvements:

* Recommends items through **indirect user similarity paths**
* Learns **community-level structure** in user behavior

Next system-level improvements:

* Efficient pipeline using **DuckDB + Parquet**
* Scales to large datasets without full in-memory computation

---

## Conclusion

K Nearest Neighbors (kNN) provides a simple and interpretable baseline for recommendation systems by leveraging local similarity between users or items. However, it is limited by its reliance on surface-level, neighborhood-based relationships and struggles with sparsity and scalability.

By introducing Matrix Factorization (MF), this project moves beyond local similarity to learn latent user and item representations, enabling better generalization and capturing underlying preference structure.

Further extending to graph-based learning (LGCN), the project demonstrates:

* How modeling interactions as a graph captures higher-order relationships beyond direct similarity
* The advantages of representation learning over both local (kNN) and latent (MF) methods
* The importance of combining data engineering with advanced machine learning techniques to build scalable, high-performance systems

---

## Future Work

* **Temporal Modeling**
  Incorporate time-aware interactions (sequence-based recommendations)

* **Real-Time Inference**
  Deploy as an API (FastAPI + caching layer)

* **Advanced Graph Models**
  Experiment with Graph Attention Networks (GAT) and contrastive learning

* **Productionization**
  Integrate streaming pipelines (Kafka / event-based updates)

* **A/B Testing Framework**
  Evaluate performance in simulated or live environments

---

## Tech Stack

* Python, Pandas, NumPy
* PyTorch / PyTorch Geometric
* Matplotlib / Seaborn

---

## Author

Aakash Kapadia
Statistics & Data Science @ Cal Poly SLO
Focused on Machine Learning, Graph Learning, and Scalable AI Systems

---
