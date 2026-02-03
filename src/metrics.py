from collections.abc import Sequence
from typing import Iterable
def recall_at_k(
    y_true: Iterable[set],
    y_pred: Iterable[Sequence],
    k: int = 10
) -> float:
    """ 
    Recall at k = 10 over multiple users
    y_true: set of iterable items
    y_pred: ranked item lists
    """
    recalls = []
    for true_items, pred_items in zip(y_true,y_pred):
        if not true_items:
            continue
        pred_topk = pred_items[:k]
        hit_count = len(true_items.intersection(pred_topk))
        recalls.append(hit_count / len(true_items))
    return float(sum(recalls) / len(recalls)) if recalls else 0.0
def precision_at_k(
    y_true: Iterable[set],
    y_pred: Iterable[Sequence],
    k: int = 10
) -> float: 
    """ 
    Precision at k = 10 over multiple users
    y_true: set of iterable items
    y_pred: ranked item lists
    """
    precisions = []
    for true_items, pred_items in zip(y_true,y_pred):
        if not true_items:
            continue
        if not pred_items: 
            continue
        pred_topk = pred_items[:k]
        if len(pred_topk) == 0:
            continue
        hit_count = len(true_items.intersection(pred_topk))
        precisions.append(hit_count / len(pred_topk))
    return float(sum(precisions) / len(precisions)) if precisions else 0.0

    
