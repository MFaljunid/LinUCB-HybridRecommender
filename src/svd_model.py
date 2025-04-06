import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import train_test_split
from config import SEED, SVD_COMPONENTS

def train_svd_model(train_data, n_components=SVD_COMPONENTS):
    """تدريب نموذج SVD"""
    user_item_matrix = train_data.pivot(
        index='user_id', 
        columns='item_id', 
        values='rating'
    ).fillna(0)
    
    svd = TruncatedSVD(n_components=n_components, random_state=SEED)
    svd.fit(user_item_matrix)
    
    user_factors = svd.transform(user_item_matrix)
    item_factors = svd.components_.T
    
    user_factors_dict = {
        user_id: user_factors[i] 
        for i, user_id in enumerate(user_item_matrix.index)
    }
    
    item_factors_dict = {
        item_id: item_factors[i] 
        for i, item_id in enumerate(user_item_matrix.columns)
    }
    
    return svd, user_factors_dict, item_factors_dict