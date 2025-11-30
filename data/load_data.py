import os
import pandas as pd
from typing import Tuple
def load_instacart_raw(data_dir: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load in raw Instacart CV's from direct data, 
    return orders, orders_products, products"""
    orders_path = os.path.join(data_dir, "orders_csv")
    prior_path = os.path.join(data_dir, "orders_products_prior.csv")
    train_path = os.path.join(data_dir, "orders_products_train.csv")
    products_path = os.path.join(data_dir, "products.csv")

    orders = pd.read_csv(orders_path)
    prior = pd.read_csv(prior_path)
    train = pd.read_csv(train_path)
    products = pd.read_csv(products_path)

    orders_products = pd.concat([prior, train], ignore_index = True)

    return orders, orders_products, products

def build_interactions(
    orders: pd.DataFrame,
    orders_products: pd.DataFrame
) -> pd.DataFrame:
    """Builds a user-item interaction table, returning: 
    user_id, item_id, order_id, order_number, order_dow, order_hour_of_the_day"""
    #Only keep necessary columns
    orders_small = orders[
        ["orders_id", "user_id", "order_number", "order_dow", "order_hour_of_the_day"]
    ]
    merged = orders_products.merge(orders_small, on = "order_id", how = "inner")

    interactions = merged.rename(
        columns = {
            "user_id": "user_id",
            "product_id": "item_id"
        }
    )[
        [
            "user_id",
            "item_id",
            "order_id",
            "order_number",
            "order_dow",
            "order_hour_of_day"
        ]
    ]
    
    return interactions

def load_interactions(data_dir: str) -> pd.DataFrame:
    "Convenience"
    orders, orders_products, products = load_instacart_raw(data_dir)
    interactions = build_interactions(orders, orders_products)
    return interactions
    
    