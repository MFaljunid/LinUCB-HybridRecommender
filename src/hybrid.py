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

        ncf_pred = self.ncf_model.predict([
            np.array([user_idx]), 
            np.array([item_idx]), 
            np.array([context_feats])
        ])[0][0]

        popularity = len(self.item_features[self.item_features['item_id'] == item_id])
        history_len = len(self.user_history[user_id])

        context = np.concatenate([svd_features, [ncf_pred], [popularity], [history_len]])
        return context

    def recommend(self, user_id, candidate_items, k=5):
        """توليد توصيات للمستخدم"""
        svd_scores = []
        ncf_scores = []

        for item_id in candidate_items:
            user_factor = self.user_factors_svd.get(user_id, np.zeros(50))
            item_factor = self.item_factors_svd.get(item_id, np.zeros(50))
            svd_scores.append(np.dot(user_factor, item_factor))

            user_idx = self.user_to_index[user_id]
            item_idx = self.item_to_index[item_id]

            item_feats = self.item_features[self.item_features['item_id'] == item_id]
            user_feats = self.user_features[self.user_features['user_id'] == user_id]

            user_context = user_feats[['age', 'gender', 'occupation']].values[0]
            item_genres = item_feats[self.item_features.columns[5:]].values[0]
            context_feats = np.concatenate([user_context, item_genres])

            ncf_score = self.ncf_model.predict([
                np.array([user_idx]), 
                np.array([item_idx]), 
                np.array([context_feats])
            ])[0][0]
            ncf_scores.append(ncf_score)

        combined_scores = 0.6 * np.array(ncf_scores) + 0.4 * np.array(svd_scores)
        top_indices = np.argsort(combined_scores)[-2*k:]
        top_candidates = [candidate_items[i] for i in top_indices]

        recommendations = []
        for item_id in top_candidates:
            context = self.get_user_context(user_id, item_id)
            arm = self.bandit.select_arm(context)
            recommendations.append((item_id, arm, context))

        recommendations.sort(key=lambda x: x[1], reverse=True)
        final_recs = [rec[0] for rec in recommendations[:k]]
        return final_recs

    def update_feedback(self, user_id, item_id, reward):
        """تحديث النموذج بناءً على التغذية الراجعة"""
        context = self.get_user_context(user_id, item_id)
        arm = self.bandit.select_arm(context)
        self.bandit.update(arm, context, reward)
        self.user_history[user_id].append(item_id)