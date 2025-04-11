import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from config import EMBEDDING_DIM, SEED, BATCH_SIZE, EPOCHS

tf.random.set_seed(SEED)

import numpy as np
import pandas as pd


def prepare_ncf_data(data, item_features, user_to_index, item_to_index, user_features):
    """تحضير بيانات NCF"""
    data = data.dropna(subset=['user_id', 'item_id', 'rating']).copy()
    
    # دمج ميزات المستخدم
    data = data.merge(user_features[['user_id', 'age', 'gender', 'occupation']], on='user_id', how='left')
    
    # معالجة الميزات
    data['gender'] = data['gender'].astype(np.float32)
    data['occupation'] = data['occupation'].astype(np.float32)
    data['age'] = data['age'].astype(np.float32)
    
    # تحضير المؤشرات
    user_indices = data['user_id'].map(user_to_index).values.astype(np.int64)
    item_indices = data['item_id'].map(item_to_index).values.astype(np.int64)
    ratings = data['rating'].values.astype(np.float32)
    
    # تحضير سياق المستخدم
    user_context = data[['age', 'gender', 'occupation']].values.astype(np.float32)
    
    # تحضير سياق العنصر (أنواع الأفلام)
    genre_columns = [col for col in item_features.columns 
                    if col not in ['item_id', 'title', 'release_date', 'video_release_date', 'IMDb_URL', 'genres']]
    item_data = data[['item_id']].merge(item_features[['item_id'] + genre_columns], on='item_id', how='left')
    item_data[genre_columns] = item_data[genre_columns].fillna(0)
    item_context = item_data[genre_columns].values.astype(np.float32)
    
    # دمج السياقات
    context_features = np.concatenate([user_context, item_context], axis=1).astype(np.float32)
    
    return user_indices, item_indices, context_features, ratings


def build_ncf_model(num_users, num_items, context_dim):
    """بناء نموذج NCF"""
    user_input = keras.Input(shape=(1,), name='user_input')
    item_input = keras.Input(shape=(1,), name='item_input')
    context_input = keras.Input(shape=(context_dim,), name='context_input')
    
    user_embedding = layers.Embedding(num_users, EMBEDDING_DIM)(user_input)
    item_embedding = layers.Embedding(num_items, EMBEDDING_DIM)(item_input)
    
    user_vec = layers.Flatten()(user_embedding)
    item_vec = layers.Flatten()(item_embedding)
    concat = layers.Concatenate()([user_vec, item_vec, context_input])
    
    dense = layers.Dense(128, activation='relu')(concat)
    dense = layers.Dropout(0.2)(dense)
    dense = layers.Dense(64, activation='relu')(dense)
    dense = layers.Dropout(0.2)(dense)
    dense = layers.Dense(32, activation='relu')(dense)
    
    output = layers.Dense(1, activation='sigmoid')(dense)
    
    model = keras.Model(
        inputs=[user_input, item_input, context_input], 
        outputs=output
    )
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model
