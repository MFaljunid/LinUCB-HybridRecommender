import numpy as np
from collections import defaultdict
from src.linucb import LinUCB
from src.utils import ensure_vector_length

class HybridRecommender:
    
    def __init__(self, svd_model, ncf_model, user_factors_svd, item_factors_svd, 
                 user_to_index, item_to_index, item_features, user_features, n_arms=10):
        self.svd_model = svd_model
        self.ncf_model = ncf_model
        self.user_factors_svd = user_factors_svd
        self.item_factors_svd = item_factors_svd
        self.user_to_index = user_to_index
        self.item_to_index = item_to_index
        self.item_features = item_features
        self.user_features = user_features
        
        # تهيئة user_history هنا قبل أي استخدام
        self.user_history = defaultdict(list)
        
        # حساب بعد السياق بشكل آمن
        try:
            sample_user = next(iter(user_to_index))
            sample_item = next(iter(item_to_index))
            sample_context = self.get_user_context(sample_user, sample_item)
            context_dim = len(sample_context)
        except Exception as e:
            print(f"Warning: Could not determine context dimension: {str(e)}")
            context_dim = 103  # القيمة الافتراضية
            
        self.bandit = LinUCB(n_arms=n_arms, context_dim=context_dim)
    
    def get_user_context(self, user_id, item_id):
        svd_user = self.user_factors_svd.get(user_id, np.zeros(50))
        svd_item = self.item_factors_svd.get(item_id, np.zeros(50))
        svd_features = np.concatenate([svd_user, svd_item])  # الطول الآن 100
        
        # تنبؤات NCF
        user_idx = self.user_to_index[user_id]
        item_idx = self.item_to_index[item_id]
        
        user_feats = self.user_features[self.user_features['user_id'] == user_id]
        item_feats = self.item_features[self.item_features['item_id'] == item_id]
        
        # سياق المستخدم (العمر، الجنس، المهنة)
        user_context = user_feats[['age', 'gender', 'occupation']].values[0]
        
        # سياق العنصر (أنواع الفيلم)
        genre_columns = [col for col in self.item_features.columns 
                        if col not in ['item_id', 'title', 'release_date', 
                                     'video_release_date', 'IMDb_URL', 'genres']]
        item_genres = item_feats[genre_columns].values[0]
        
        # دمج السياقات
        context_feats = np.concatenate([user_context.astype(np.float32), 
                                      item_genres.astype(np.float32)])
        
        # الحصول على تنبؤ NCF
        ncf_pred = self.ncf_model.predict([np.array([user_idx]), 
                                         np.array([item_idx]), 
                                         np.array([context_feats])])[0][0]
        
        # ميزات إضافية (شعبية العنصر وطول تاريخ المستخدم)
        popularity = len(self.item_features[self.item_features['item_id'] == item_id])
        history_len = len(self.user_history[user_id])  # الآن سيعمل بشكل صحيح
        
        # السياق النهائي
        context = np.concatenate([svd_features, [ncf_pred], [popularity], [history_len]])
        
        # التحقق من الطول
        expected_length = 103  # 50+50 + 1 + 1 + 1
        if len(context) != expected_length:
            context = np.pad(context, (0, expected_length - len(context)), 'constant')
        
        return context

    def recommend(self, user_id, all_items, k=5):
        """توليد التوصيات للمستخدم"""
        recommendations = []
        user_history = self.user_history[user_id]

        for item_id in all_items:
            if item_id not in user_history:  # تجنب العناصر التي شاهدها المستخدم مسبقًا
                try:
                    context = self.get_user_context(user_id, item_id)
                    score = np.dot(self.bandit.get_arm_weights(context), context)
                    recommendations.append((item_id, score))
                except Exception as e:
                    print(f"Error processing item {item_id} for user {user_id}: {str(e)}")
        
        # ترتيب العناصر حسب أعلى درجة
        recommendations.sort(key=lambda x: x[1], reverse=True)

        # إرجاع أفضل k عناصر
        return [item for item, _ in recommendations[:k]]
    
    def update_feedback(self, user_id, item_id, reward):
        """تحديث النموذج بناءً على تفاعل المستخدم"""
        context = self.get_user_context(user_id, item_id)
        arm = self.bandit.select_arm(context)
        self.bandit.update(arm, context, reward)
        self.user_history[user_id].append(item_id)