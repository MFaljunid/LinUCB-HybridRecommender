import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from config import DATA_DIR, SEED

def load_data(dataset_size='100k'):
    """تحميل بيانات MovieLens"""
    print(f"Loading {dataset_size} dataset...")

    data_dir = os.path.join('C:/Users/lenovo/Desktop/Explainable_CF_Project/data/', dataset_size)
    
    if dataset_size == '100k':
        ratings_path = os.path.join(data_dir, 'u.data')
        users_path = os.path.join(data_dir, 'u.user')
        movies_path = os.path.join(data_dir, 'u.item')
        sep_ratings = '\t'
        sep_movies = '|'
    elif dataset_size == '1m':
        ratings_path = os.path.join(data_dir, 'ratings.dat')
        users_path = os.path.join(data_dir, 'users.dat')
        movies_path = os.path.join(data_dir, 'movies.dat')
        sep_ratings = '::'
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
            sep='|',
            names=['user_id', 'age', 'gender', 'occupation', 'zip_code'],
            encoding='latin-1'
        )
    else:  # إنشاء بيانات وهمية لمجموعة 1m
        unique_users = ratings['user_id'].unique()
        user_features = pd.DataFrame({
            'user_id': unique_users,
            'age': np.random.randint(10, 70, size=len(unique_users)),
            'gender': np.random.choice(['M', 'F'], size=len(unique_users)),
            'occupation': np.random.choice(['student', 'engineer', 'teacher'], size=len(unique_users))
        })

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
    else:  # معالجة بيانات الأفلام لمجموعة 1m
        item_features = pd.read_csv(
            movies_path, 
            sep=sep_movies,
            names=['item_id', 'title', 'genres'],
            encoding='latin-1'
        )
        genres = item_features['genres'].str.get_dummies(sep='|')
        item_features = pd.concat([item_features, genres], axis=1)

    # دمج البيانات
    data = ratings.merge(user_features, on='user_id')
    data = data.merge(item_features, on='item_id')

    # تحويل السمات الفئوية إلى رقمية
    data['gender'] = data['gender'].map({'M': 0, 'F': 1})
    occupation_map = {occ: i for i, occ in enumerate(data['occupation'].unique())}
    data['occupation'] = data['occupation'].map(occupation_map)

    # تطبيع السمات الرقمية
    scaler = MinMaxScaler()
    if 'age' in data.columns:
        data['age'] = scaler.fit_transform(data[['age']])
    data['rating'] = scaler.fit_transform(data[['rating']])

    # إنشاء فهارس للمستخدمين والعناصر
    user_ids = data['user_id'].unique()
    item_ids = data['item_id'].unique()
    user_to_index = {user_id: idx for idx, user_id in enumerate(user_ids)}
    item_to_index = {item_id: idx for idx, item_id in enumerate(item_ids)}

    return data, user_features, item_features, user_to_index, item_to_index