import os
import numpy as np
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

def evaluate_recommender(recommender, test_data, item_features, k=10, max_users=5):
    """إصدارة معدلة مع تحسينات تتبع المشاكل"""
    hits = 0
    total = 0
    ndcg_scores = []
    evaluated_users = 0
    
    user_sample = test_data['user_id'].drop_duplicates().sample(min(max_users, len(test_data)), random_state=SEED)
    
    for user_id in user_sample:
        try:
            user_ratings = test_data[test_data['user_id'] == user_id]
            # خفض عتبة التقييم إلى 0.5 لزيادة فرص الحصول على بيانات
            liked_items = set(user_ratings[user_ratings['rating'] > 0.5]['item_id'])
            
            if not liked_items:
                print(f"\n⚠️ User {user_id} has no liked items (rating > 0.5)")
                continue
                
            all_items = test_data['item_id'].unique()
            recommended_items = recommender.recommend(user_id, all_items, k=k)
            
            # طباعة معلومات التصحيح
            print(f"\n📊 User {user_id} Evaluation:")
            print(f"Liked items count: {len(liked_items)}")
            print(f"Top recommended items: {recommended_items[:3]}...")
            
            hit_count = len(set(recommended_items) & liked_items)
            hits += hit_count
            total += len(liked_items)
            
            print(f"Hits: {hit_count}/{len(liked_items)}")
            
            # حساب NDCG مع التحقق من القسمة على صفر
            relevance = [1 if item in liked_items else 0 for item in recommended_items]
            dcg = sum([rel / np.log2(i+2) for i, rel in enumerate(relevance)])
            idcg = sum([1 / np.log2(i+2) for i in range(min(k, len(liked_items)))])
            current_ndcg = dcg / idcg if idcg > 0 else 0
            ndcg_scores.append(current_ndcg)
            
            print(f"NDCG: {current_ndcg:.4f}")
            evaluated_users += 1
            
        except Exception as e:
            print(f"\n❌ Error evaluating user {user_id}: {str(e)}")
            continue
    
    print(f"\n✅ Successfully evaluated {evaluated_users}/{max_users} users")
    
    if evaluated_users == 0:
        print("\n‼️ Critical Warning: No valid evaluations!")
        print("Possible solutions:")
        print("1. Decrease the rating threshold (current: >0.5)")
        print("2. Check data loading and preprocessing")
        print("3. Verify recommendation logic")
        return 0.0, 0.0
    
    final_hit_rate = hits / total
    final_ndcg = np.mean(ndcg_scores)
    
    print("\n📈 Evaluation Summary:")
    print(f"Total hits: {hits}")
    print(f"Total liked items: {total}")
    print(f"Users evaluated: {evaluated_users}")
    
    return final_hit_rate, final_ndcg

def get_movie_title(item_id, item_features):
    """الحصول على عنوان الفيلم"""
    return item_features[item_features['item_id'] == item_id]['title'].values[0]
def main():
    recommender = None  # تعريف مبدئي للمتغير
    
    try:
        # تحميل البيانات
        print("Loading data...")
        ratings, user_features, item_features, user_to_index, item_to_index = load_data('100k')
        
        # تقسيم البيانات
        print("Splitting data...")
        train_data, test_data = train_test_split(
            ratings, 
            test_size=TEST_SIZE, 
            random_state=SEED
        )
        
        # تدريب نموذج SVD
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
        
        # بناء وتدريب نموذج NCF
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
        
        # تهيئة النموذج الهجين
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
            n_arms=10
        )
        
        # التحقق من النموذج
        sample_user = test_data['user_id'].iloc[0]
        sample_item = test_data['item_id'].iloc[0]
        sample_context = recommender.get_user_context(sample_user, sample_item)
        print(f"\nSample context vector length: {len(sample_context)}")
        print(f"Sample context vector: {sample_context[:5]}...")  # عرض أول 5 قيم فقط
        
        
        # تقييم النموذج
        # print("\nEvaluating recommender...")
        # # hit_rate, ndcg = evaluate_recommender(recommender, test_data, item_features)
        # print("\nQuick Evaluation (50 users)...")
        # hit_rate, ndcg = evaluate_recommender(recommender, test_data, item_features, max_users=1)
        # print(f"Hit Rate @5: {hit_rate:.4f}")
        # print(f"NDCG @5: {ndcg:.4f}")
        
        # # عرض بعض التوصيات
        # print("\nSample recommendations:")
        # user_id = test_data['user_id'].iloc[0]
        # all_items = test_data['item_id'].unique()
        # recommended_items = recommender.recommend(user_id, all_items, k=5)
        
        # print(f"\nRecommendations for user {user_id}:")
        # for i, item_id in enumerate(recommended_items, 1):
        #     print(f"{i}. {get_movie_title(item_id, item_features)}")

        print("\n🔍 Starting Evaluation...")
        hit_rate, ndcg = evaluate_recommender(recommender, test_data, item_features, k=10, max_users=5)
        print("\n🎯 Final Results:")
        print(f"Hit Rate @10: {hit_rate:.4f}")
        print(f"NDCG @10: {ndcg:.4f}")

        # عرض المزيد من التفاصيل إذا كانت النتائج صفرية
        if hit_rate == 0 and ndcg == 0:
            print("\n🔴 Problem Detected: All metrics are zero")
            print("Debugging Steps:")
            print("1. Check if recommendations are being generated")
            print("2. Verify user-item interactions in test data")
            print("3. Inspect the first user's data:")
            
            sample_user = test_data['user_id'].iloc[0]
            user_ratings = test_data[test_data['user_id'] == sample_user]
            print(f"\nSample user {sample_user} ratings:")
            print(user_ratings[['item_id', 'rating']].head())
            
            # اختبار التوصيات يدوياً
            print("\nTesting recommendations for sample user:")
            recs = recommender.recommend(sample_user, test_data['item_id'].unique(), k=5)
            print(f"Recommended items: {recs}")
    except Exception as e:
        print(f"\nError during processing: {str(e)}")
        print("\nDebugging info:")
        if 'recommender' in locals():
            print(f"- Recommender initialized: {recommender is not None}")
        else:
            print("- Recommender not initialized")
        
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()