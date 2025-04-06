import os
import numpy as np
import tensorflow as tf
from src.data_loader import load_data
from src.svd_model import train_svd_model
from src.ncf_model import prepare_ncf_data, build_ncf_model
from src.hybrid import HybridRecommender
from src.evaluation import evaluate_recommender, get_movie_title
from src.utils import split_data
from config import SEED, BATCH_SIZE, EPOCHS

# ضبط البذور العشوائية
np.random.seed(SEED)
tf.random.set_seed(SEED)

def main():
    try:
        # تحميل البيانات
        data, user_features, item_features, user_to_index, item_to_index = load_data('100k')
        print("Data loaded successfully!")

        # التحقق من أنواع البيانات
        print("Item ID dtype:", item_features['item_id'].dtype)
        print("User ID dtype:", user_features['user_id'].dtype)
        print("Item Features dtypes:\n", item_features.dtypes)
        print("User Features dtypes:\n", user_features.dtypes)

        # تقسيم البيانات إلى تدريب واختبار
        train_data, test_data = split_data(data)

        # تدريب نموذج SVD
        svd_model, user_factors_svd, item_factors_svd = train_svd_model(train_data)

        # تحضير بيانات NCF للتدريب
        # train_user_indices, train_item_indices, train_context, train_ratings = prepare_ncf_data(
        #     train_data, item_features, user_to_index, item_to_index
        # )

        # # طباعة معلومات عن بيانات التدريب
        # print(f"Train user_indices dtype: {train_user_indices.dtype}, shape: {train_user_indices.shape}")
        # print(f"Train item_indices dtype: {train_item_indices.dtype}, shape: {train_item_indices.shape}")
        # print(f"Train context_features dtype: {train_context.dtype}, shape: {train_context.shape}")

        # تحضير بيانات NCF للاختبار
        train_user_indices, train_item_indices, train_context, train_ratings = prepare_ncf_data(
            train_data, item_features, user_to_index, item_to_index, user_features
        )

        test_user_indices, test_item_indices, test_context, test_ratings = prepare_ncf_data(
            test_data, item_features, user_to_index, item_to_index, user_features
        )
        print(f"Test user_indices shape: {test_user_indices.shape}")
        print(f"Test item_indices shape: {test_item_indices.shape}")
        print(f"Test context_features shape: {test_context.shape}")

        # بناء وتدريب نموذج NCF
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

        # تهيئة نموذج التوصية الهجين
        recommender = HybridRecommender(
            svd_model=svd_model,
            ncf_model=ncf_model,
            user_factors_svd=user_factors_svd,
            item_factors_svd=item_factors_svd,
            user_to_index=user_to_index,
            item_to_index=item_to_index,
            item_features=item_features,
            user_features=user_features
        )

        # تقييم النموذج
        hit_rate, ndcg = evaluate_recommender(recommender, test_data, item_features)
        print(f"Hit Rate @5: {hit_rate:.4f}")
        print(f"NDCG @5: {ndcg:.4f}")

        # عرض التوصيات لمستخدم معيّن
        user_id = test_data['user_id'].iloc[0]
        all_items = test_data['item_id'].unique()
        recommended_items = recommender.recommend(user_id, all_items, k=5)

        print(f"\nRecommendations for user {user_id}:")
        for i, item_id in enumerate(recommended_items, 1):
            print(f"{i}. {get_movie_title(item_id, item_features)}")

    except Exception as e:
        print(f"Error: {str(e)}")
        print("\nTroubleshooting tips:")
        print("1. Verify the data exists in the correct directory")
        print("2. Check file permissions")
        print("3. Make sure all required packages are installed")

if __name__ == "__main__":
    main()