from sklearn.model_selection import train_test_split
from config import SEED, TEST_SIZE

def split_data(data):
    """تقسيم البيانات إلى تدريب واختبار"""
    return train_test_split(
        data, 
        test_size=TEST_SIZE, 
        random_state=SEED
    )
import numpy as np

def __init__(self, n_arms, context_dim, alpha=0.1):
    self.n_arms = n_arms
    self.context_dim = context_dim  # يجب أن يكون 103 في حالتنا
    self.alpha = alpha
    
    # مصفوفات A ومتجهات b لكل ذراع
    self.A = [np.identity(self.context_dim) for _ in range(self.n_arms)]
    self.b = [np.zeros(self.context_dim) for _ in range(self.n_arms)]
    self.theta = [np.zeros(self.context_dim) for _ in range(self.n_arms)]

def ensure_vector_length(vector, expected_length):
    """ضبط طول المتجه للقيمة المتوقعة"""
    if len(vector) > expected_length:
        return vector[:expected_length]
    elif len(vector) < expected_length:
        return np.concatenate([vector, np.zeros(expected_length - len(vector))])
    return vector