import os
import numpy as np
import pandas as pd

import tensorflow as tf
from src.data_loader import load_data
from src.svd_model import train_svd_model
from src.ncf_model import prepare_ncf_data, build_ncf_model
from src.hybrid import HybridRecommender
# from src.evaluation import evaluate_recommender, get_movie_title
from src.utils import split_data
from config import SEED, BATCH_SIZE, EPOCHS
from sklearn.decomposition import TruncatedSVD
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# from load_data import load_data
from src.linucb import LinUCB
# from hybrid import HybridRecommender
from config import SEED, EMBEDDING_DIM, SVD_COMPONENTS, TEST_SIZE, BATCH_SIZE, EPOCHS, ALPHA



# ضبط البذور العشوائية
np.random.seed(SEED)
tf.random.set_seed(SEED)

def train_svd_model(train_data):
    """تدريب نموذج SVD"""
    user_item_matrix = train_data.pivot(
        index='user_id', 
        columns='item_id', 
        values='rating'
    ).fillna(0)
    
    svd = TruncatedSVD(n_components=SVD_COMPONENTS, random_state=SEED)
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

def evaluate_recommender(recommender, test_data, item_features, k=10, max_users=10, rating_threshold=0.7):
    """دالة التقييم المعدلة مع تحليل مفصل"""
    hits = 0
    total = 0
    ndcg_scores = []
    user_results = []
    
    user_sample = test_data['user_id'].drop_duplicates().sample(min(max_users, len(test_data)), random_state=SEED)
    
    for user_id in user_sample:
        try:
            # [1] يمكن تعديل عتبة التقييم هنا (rating_threshold)
            user_ratings = test_data[test_data['user_id'] == user_id]
            liked_items = set(user_ratings[user_ratings['rating'] > rating_threshold]['item_id'])
            
            if not liked_items:
                continue
                
            all_items = test_data['item_id'].unique()
            recommended_items = recommender.recommend(user_id, all_items, k=k)
            
            hit_count = len(set(recommended_items) & liked_items)
            hits += hit_count
            total += len(liked_items)
            
            # [2] يمكن تعديل حساب NDCG هنا
            relevance = [1 if item in liked_items else 0 for item in recommended_items]
            dcg = sum([rel / np.log2(i+2) for i, rel in enumerate(relevance)])
            idcg = sum([1 / np.log2(i+2) for i in range(min(k, len(liked_items)))])
            ndcg = dcg / idcg if idcg > 0 else 0
            ndcg_scores.append(ndcg)
            
            user_results.append({
                'user_id': user_id,
                'hits': hit_count,
                'total_liked': len(liked_items),
                'ndcg': ndcg,
                'recommended': recommended_items[:5],
                'liked': list(liked_items)[:5]
            })
            
        except Exception as e:
            print(f"Error evaluating user {user_id}: {str(e)}")
            continue
    
    if not user_results:
        return 0.0, 0.0, []
    
    # [3] يمكن إضافة المزيد من التحليلات هنا
    print("\n🔍 Detailed Analysis:")
    print(f"Users evaluated: {len(user_results)}")
    print(f"Average hits per user: {hits/len(user_results):.2f}")
    print(f"Average NDCG per user: {np.mean([u['ndcg'] for u in user_results]):.4f}")
    
    return hits/total, np.mean(ndcg_scores), user_results

def get_movie_title(item_id, item_features):
    """الحصول على عنوان الفيلم"""
    return item_features[item_features['item_id'] == item_id]['title'].values[0]

def main():
    recommender = None


    # [4] تحميل البيانات - يمكن تغيير حجم dataset هنا
    print("Loading data...")
    ratings, user_features, item_features, user_to_index, item_to_index = load_data('100k')
    
    # تقسيم البيانات
    print("Splitting data...")
    train_data, test_data = train_test_split(
        ratings, 
        test_size=TEST_SIZE, 
        random_state=SEED
    )
    
    # [5] تدريب نموذج SVD - يمكن تعديل n_components في config.py
    print("Training SVD model...")
    svd_model, user_factors_svd, item_factors_svd = train_svd_model(train_data)
    
    # تحضير بيانات NCF
    print("Preparing NCF data...")
    train_user_indices, train_item_indices, train_context, train_ratings = prepare_ncf_data(
        train_data, item_features, user_to_index, item_to_index, user_features
    )
    
    test_user_indices, test_item_indices, test_context, test_ratings = prepare_ncf_data(
        test_data, item_features, user_to_index, item_to_index, user_features
    )
    
    # [6] بناء وتدريب نموذج NCF - يمكن تعديل البارامترات في config.py
    print("Building and training NCF model...")
    ncf_model = build_ncf_model(
        num_users=len(user_to_index),
        num_items=len(item_to_index),
        context_dim=train_context.shape[1]
    )
    
    ncf_model.fit(
        [train_user_indices, train_item_indices, train_context],
        train_ratings,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=([test_user_indices, test_item_indices, test_context], test_ratings),
        verbose=1
    )
    
    # [7] تهيئة النموذج الهجين - يمكن تعديل n_arms هنا
    print("Initializing hybrid recommender...")
    recommender = HybridRecommender(
        svd_model=svd_model,
        ncf_model=ncf_model,
        user_factors_svd=user_factors_svd,
        item_factors_svd=item_factors_svd,
        user_to_index=user_to_index,
        item_to_index=item_to_index,
        item_features=item_features,
        user_features=user_features,
        n_arms=10  # يمكن زيادتها لتحسين الأداء
    )
    
    # التحقق من النموذج
    sample_user = test_data['user_id'].iloc[0]
    sample_item = test_data['item_id'].iloc[0]
    sample_context = recommender.get_user_context(sample_user, sample_item)
    print(f"\nSample context vector length: {len(sample_context)}")
    print(f"Sample context vector: {sample_context[:5]}...")
    
    # [8] التقييم النهائي - يمكن تعديل max_users و rating_threshold
    print("\n🔍 Starting Comprehensive Evaluation...")
    hit_rate, ndcg, detailed_results = evaluate_recommender(
        recommender, 
        test_data, 
        item_features,
        k=10,
        max_users=1,
        rating_threshold=0.5
    )

    print("\n🎯 Final Metrics:")
    print(f"Hit Rate @10: {hit_rate:.4f}")
    print(f"NDCG @10: {ndcg:.4f}")

    # [9] حفظ النتائج للتحليل اللاحق
    results_df = pd.DataFrame(detailed_results)
    results_df.to_csv('evaluation_results.csv', index=False)
    
    # [10] عرض توصيات كمثال
    print("\nSample Recommendations:")
    user_id = test_data['user_id'].iloc[0]
    recs = recommender.recommend(user_id, test_data['item_id'].unique(), k=5)
    for i, item_id in enumerate(recs, 1):
        print(f"{i}. {get_movie_title(item_id, item_features)}")

if __name__ == "__main__":
    main()