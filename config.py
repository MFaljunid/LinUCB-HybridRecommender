import os

# إعدادات المسارات
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
MODELS_DIR = os.path.join(BASE_DIR, 'models')

# إعدادات النماذج
# إعدادات النظام الأساسية
SEED = 42
EMBEDDING_DIM = 64
SVD_COMPONENTS = 50
TEST_SIZE = 0.2
BATCH_SIZE = 64
EPOCHS = 10
ALPHA = 0.1  # معامل الاستكشاف في LinUCB