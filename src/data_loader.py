import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from config import DATA_DIR, SEED


import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def load_data(dataset_size='100k'):
    """تحميل بيانات التقييمات والمستخدمين والأفلام"""
    print(f"Loading {dataset_size} dataset...")

    data_dir = os.path.join('data/', dataset_size)
    
    if dataset_size == '100k':
        ratings_path = os.path.join(data_dir, 'u.data')
        users_path = os.path.join(data_dir, 'u.user')
        movies_path = os.path.join(data_dir, 'u.item')
        sep_ratings = '\t'
        sep_users = '|'
        sep_movies = '|'
    elif dataset_size == '1m':
        ratings_path = os.path.join(data_dir, 'ratings.dat')
        users_path = os.path.join(data_dir, 'users.dat')
        movies_path = os.path.join(data_dir, 'movies.dat')
        sep_ratings = '::'
        sep_users = '::'
        sep_movies = '::'
    else:
        raise ValueError("Invalid dataset_size! Use '100k' or '1m'.")

    # تحميل بيانات التقييمات
    ratings = pd.read_csv(
        ratings_path, 
        sep=sep_ratings,
        names=['user_id', 'item_id', 'rating', 'timestamp'],
        encoding='latin-1'
    )

    # تحميل بيانات المستخدمين
    if dataset_size == '100k':
        user_features = pd.read_csv(
            users_path, 
            sep=sep_users,
            names=['user_id', 'age', 'gender', 'occupation', 'zip_code'],
            encoding='latin-1'
        )
    else:
        user_features = pd.read_csv(
            users_path,
            sep=sep_users,
            names=['user_id', 'gender', 'age', 'occupation', 'zip_code'],
            encoding='latin-1'
        )

    # تحميل بيانات الأفلام
    if dataset_size == '100k':
        item_features = pd.read_csv(
            movies_path, 
            sep=sep_movies,
            names=['item_id', 'title', 'release_date', 'video_release_date',
                   'IMDb_URL', 'unknown', 'Action', 'Adventure', 'Animation',
                   'Children', 'Comedy', 'Crime', 'Documentary', 'Drama',
                   'Fantasy', 'Film-Noir', 'Horror', 'Musical', 'Mystery',
                   'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western'],
            encoding='latin-1'
        )
    else:
        item_features = pd.read_csv(
            movies_path, 
            sep=sep_movies,
            names=['item_id', 'title', 'genres'],
            encoding='latin-1'
        )
        genres = item_features['genres'].str.get_dummies(sep='|')
        item_features = pd.concat([item_features, genres], axis=1)

    # معالجة البيانات
    ratings = ratings[['user_id', 'item_id', 'rating']]
    user_features = user_features[['user_id', 'age', 'gender', 'occupation']]
    
    # تحويل الجنس إلى قيم رقمية
    user_features['gender'] = user_features['gender'].map({'M': 0, 'F': 1}).fillna(0)
    
    # تحويل المهنة إلى رقمية
    user_features['occupation'] = user_features['occupation'].astype('category').cat.codes
    
    # تطبيع البيانات
    scaler = MinMaxScaler()
    user_features['age'] = scaler.fit_transform(user_features[['age']])
    ratings['rating'] = scaler.fit_transform(ratings[['rating']])

    # إنشاء فهارس للمستخدمين والعناصر
    user_to_index = {user_id: idx for idx, user_id in enumerate(ratings['user_id'].unique())}
    item_to_index = {item_id: idx for idx, item_id in enumerate(ratings['item_id'].unique())}

    return ratings, user_features, item_features, user_to_index, item_to_index