# import numpy as np
# def evaluate_recommender(recommender, test_data, item_features, k=10, max_users=None, rating_threshold=0.7):
#     """
#     تقييم شامل لنظام التوصية مع تحسينات
#     """
#     print(f"\n🔍 Evaluating Recommender System (k={k}, threshold={rating_threshold})...")
    
#     metrics = {
#         'avg_hits': 0,
#         'avg_ndcg': 0,
#         'avg_precision@k': 0,
#         'avg_recall@k': 0,
#         'user_results': []
#     }
    
#     unique_users = test_data['user_id'].unique()
#     if max_users is not None and max_users < len(unique_users):
#         unique_users = np.random.choice(unique_users, size=max_users, replace=False)
    
#     for user_id in unique_users:
#         try:
#             # الحصول على العناصر المعجبة للمستخدم
#             user_ratings = test_data[test_data['user_id'] == user_id]
#             liked_items = set(user_ratings[user_ratings['rating'] >= rating_threshold]['item_id'])
            
#             if not liked_items:
#                 continue
                
#             # توليد التوصيات
#             all_items = test_data['item_id'].unique()
#             recommended_items = recommender.recommend(user_id, all_items, k=k)
            
#             if not recommended_items:
#                 continue
                
#             # حساب مقاييس التقييم
#             hit_count = len(set(recommended_items) & liked_items)
            
#             # حساب NDCG باستخدام التقييمات الفعلية
#             relevance_scores = []
#             for item in recommended_items:
#                 user_rating = user_ratings[user_ratings['item_id'] == item]['rating'].values
#                 relevance_scores.append(user_rating[0] if len(user_rating) > 0 else 0)
            
#             dcg = sum([(2**rel - 1) / np.log2(idx + 2) for idx, rel in enumerate(relevance_scores)])
#             ideal_relevance = sorted([r for r in user_ratings['rating'] if r >= rating_threshold], reverse=True)[:k]
#             ideal_relevance = [2**r - 1 for r in ideal_relevance]
#             idcg = sum([rel / np.log2(idx + 2) for idx, rel in enumerate(ideal_relevance)])
#             ndcg = dcg / idcg if idcg > 0 else 0
            
#             # تخزين النتائج
#             user_result = {
#                 'user_id': user_id,
#                 'hits': hit_count,
#                 'total_liked': len(liked_items),
#                 'ndcg': ndcg,
#                 'recommended': recommended_items[:k],
#                 'liked': list(liked_items)[:k],
#                 'precision@k': hit_count / k,
#                 'recall@k': hit_count / len(liked_items) if len(liked_items) > 0 else 0
#             }
            
#             metrics['user_results'].append(user_result)
            
#         except Exception as e:
#             print(f"⚠️ Error evaluating user {user_id}: {str(e)}")
#             continue
    
#     # حساب المتوسطات
#     if metrics['user_results']:
#         metrics['avg_hits'] = sum(r['hits'] for r in metrics['user_results']) / sum(r['total_liked'] for r in metrics['user_results'])
#         metrics['avg_ndcg'] = np.mean([r['ndcg'] for r in metrics['user_results']])
#         metrics['avg_precision@k'] = np.mean([r['precision@k'] for r in metrics['user_results']])
#         metrics['avg_recall@k'] = np.mean([r['recall@k'] for r in metrics['user_results']])
    
#     print("✅ Evaluation completed successfully")
#     return metrics


# def get_movie_title(item_id, item_features):
#     """الحصول على عنوان الفيلم"""
#     return item_features[item_features['item_id'] == item_id]['title'].values[0]
