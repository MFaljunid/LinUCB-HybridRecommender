import numpy as np
from collections import defaultdict
from .linucb import LinUCB
class HybridRecommender:
    """النموذج الهجين للتوصية"""
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
        self.bandit = LinUCB(n_arms=n_arms, context_dim=50+1+2)
        self.user_history = defaultdict(list)

    def get_user_context(self, user_id, item_id):
        """الحصول على سياق المستخدم والعنصر"""
        svd_user = self.user_factors_svd.get(user_id, np.zeros(50))
        svd_item = self.item_factors_svd.get(item_id, np.zeros(50))
        svd_features = np.concatenate([svd_user, svd_item])

        user_idx = self.user_to_index[user_id]
        item_idx = self.item_to_index[item_id]

        item_feats = self.item_features[self.item_features['item_id'] == item_id]
        user_feats = self.user_features[self.user_features['user_id'] == user_id]

        user_context = user_feats[['age', 'gender', 'occupation']].values[0]
        item_genres = item_feats[self.item_features.columns[5:]].values[0]
        context_feats = np.concatenate([user_context, item_genres])

        ncf_pred = self.ncf_model.predict([np.array([user_idx]), np.array([item_idx]), np.array([context_feats])])[0][0]

        popularity = len(self.item_features[self.item_features['item_id'] == item_id])
        history_len = len(self.user_history[user_id])

        context = np.concatenate([svd_features, [ncf_pred], [popularity], [history_len]])
        return context

    def recommend(self, user_id, all_items, k=5):
        """التوصية بأعلى العناصر للمستخدم"""
        # تحديد العناصر الأعلى تصنيفًا باستخدام سياق المستخدم
        recommendations = []
        user_history = self.user_history[user_id]
        
        # الحصول على التوصيات لكل عنصر بناءً على السياق
        for item_id in all_items:
            if item_id not in user_history:  # إذا لم يتم التفاعل مع هذا العنصر من قبل
                context = self.get_user_context(user_id, item_id)
                score = np.dot(self.bandit.get_arm_weights(context), context)  # مثال على كيفية استخدام السياق
                recommendations.append((item_id, score))
        
        # ترتيب العناصر حسب التصنيف الأعلى
        recommendations.sort(key=lambda x: x[1], reverse=True)
        
        # إرجاع أعلى k عناصر
        return [item[0] for item in recommendations[:k]]
