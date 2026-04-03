# Recommender System: 2018 Amazon Books Dataset

The following is an end-to-end recommendation system that uses **K Nearest Neighbors (kNN)**, **Matrix Factorization**, and **Light Graph Convolutional Networks (LGCN)** to capture  user book relationships at scale, trying to improve over popularity based methods. 

---

## Overview

Matching users with relevant items from massive catalogs has relevant real-world applications in nearly every field. For example, e-commerce, streaming, and social media are all heavily reliant on this to retain users. Recommender systems are central to this, allowing for what is essentially a personalized search engine. 

This project builds a **three-stage recommendation framework**:

1. A **kNN** model for our baseline
2. A **Matrix Factorization** model to learn latent user/item preferences
3. A **Graph Neural Network (LGCN)** to model higher-order relationships beyond direct interactions

The result is a system that moves from **linear latent structure → graph-based representation learning**.

---

## Problem

While effective to some extent, older recommendation approaches like **k-Nearest Neighbors (kNN)** assume that user preferences can be inferred by identifying similar users or items based on past interactions.

Although intuitive and easy to implement, kNN has key limitations:

* Relies entirely on local similarity, limiting its ability to generalize beyond immediate neighbors
* Does not effectively capture higher-order relationships (e.g., connections through multiple users/items)
* Performance falls under data sparsity (high percentage of missing data), where similarity estimates become unreliable

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

### 2. Baseline: kNN

* Construct user–item interaction matrix
  
* Compute similarity via:
  
  * User–user similarity (collaborative filtering) or item–item similarity
  * Cosine Similarity
    
* Generate recommendations by:
  
  * Identifying top-k nearest neighbors
  * Aggregating neighbor preferences to score unseen items
    
* Output:
  
  * Similarity matrix (user–user or item–item)
  * Ranked Top-K recommendations per user

---

### 3. Model 1: Matrix Factorization

* Construct user–item interaction matrix

* Learn embeddings via:

  * Implicit feedback modeling
  * Pairwise ranking loss (BPR)

* Output:

  * User latent vectors
  * Item latent vectors
  * Baseline Top-K recommendations

---

### 4. Graph Construction

* Build a **bipartite graph**:

  * Nodes: users + items
  * Edges: interactions (purchases, views)

---

### 5. Model 2: LightGCN

* Remove nonlinearities → focus on **pure neighborhood aggregation**
* Propagate embeddings across graph layers:

  * Captures **multi-hop relationships**
  * Incorporates collaborative signals beyond direct interactions

---

### 5. Training & Evaluation

* Optimization: **Bayesian Personalized Ranking (BPR) loss**
* Metrics:

  * Recall@K
  * Precision@K
  * Ranking quality vs kNN baseline

---

## Results

**At k = 5, Matrix Factorization has a recall of 3.4x kNN**

Qualitative improvements:

* Recommends items through **indirect user similarity paths**
* Learns **community-level structure** in user behavior

System-level improvements:

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
