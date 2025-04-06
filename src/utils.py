from sklearn.model_selection import train_test_split
from config import SEED, TEST_SIZE

def split_data(data):
    """تقسيم البيانات إلى تدريب واختبار"""
    return train_test_split(
        data, 
        test_size=TEST_SIZE, 
        random_state=SEED
    )