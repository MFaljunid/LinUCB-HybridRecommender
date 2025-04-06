import numpy as np

def evaluate_recommender(recommender, test_data, item_features, k=5):
    """تقييم أداء النموذج الهجين"""
    hits = 0
    total = 0
    ndcg_scores = []
    
    all_items = test_data['item_id'].unique()
    
    for user_id in test_data['user_id'].unique():
        user_ratings = test_data[test_data['user_id'] == user_id]
        liked_items = set(user_ratings[user_ratings['rating'] > 0.7]['item_id'])
        
        if not liked_items:
            continue
            
        recommended_items = recommender.recommend(user_id, all_items, k=k)
        
        hits += len(set(recommended_items) & liked_items)
        total += len(liked_items)
        
        relevance = [1 if item in liked_items else 0 for item in recommended_items]
        dcg = sum([rel / np.log2(i + 2) for i, rel in enumerate(relevance)])
        ideal_relevance = [1] * min(k, len(liked_items)) + [0] * max(0, k - len(liked_items))
        idcg = sum([rel / np.log2(i + 2) for i, rel in enumerate(ideal_relevance)])
        ndcg = dcg / idcg if idcg > 0 else 0
        ndcg_scores.append(ndcg)
        
        for item_id in recommended_items:
            reward = 1.0 if item_id in liked_items else 0.0
            recommender.update_feedback(user_id, item_id, reward)
    
    hit_rate = hits / total if total > 0 else 0
    avg_ndcg = np.mean(ndcg_scores) if ndcg_scores else 0
    
    return hit_rate, avg_ndcg

def get_movie_title(item_id, item_features):
    """الحصول على عنوان الفيلم"""
    return item_features[item_features['item_id'] == item_id]['title'].values[0]