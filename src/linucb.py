import numpy as np

class LinUCB:
    """نموذج LinUCB Bandit لتحسين التوصيات"""
    
    def __init__(self, n_arms, context_dim, alpha=0.1):
        """
        المعلمات:
            n_arms: عدد العناصر الممكن التوصية بها
            context_dim: بعد متجه السياق
            alpha: معامل الاستكشاف
        """
        self.n_arms = n_arms
        self.context_dim = context_dim
        self.alpha = alpha
        
        # مصفوفات A ومتجهات b لكل ذراع
        self.A = [np.identity(self.context_dim) for _ in range(self.n_arms)]
        self.b = [np.zeros(self.context_dim) for _ in range(self.n_arms)]
        self.theta = [np.zeros(self.context_dim) for _ in range(self.n_arms)]
    
    def select_arm(self, context):
        """اختيار أفضل ذراع مع التحقق من الأبعاد"""
        if len(context) != self.context_dim:
            raise ValueError(f"Context dimension mismatch. Expected {self.context_dim}, got {len(context)}")
        
        ucbs = []
        for arm in range(self.n_arms):
            A_inv = np.linalg.inv(self.A[arm])
            theta = self.theta[arm]
            
            # التحقق من تطابق الأبعاد
            if len(theta) != len(context):
                theta = np.zeros(len(context))  # إعادة تهيئة إذا لزم الأمر
            
            score = np.dot(theta, context)
            uncertainty = self.alpha * np.sqrt(np.dot(context.T, np.dot(A_inv, context)))
            ucbs.append(score + uncertainty)
        
        return np.argmax(ucbs)
    def update(self, arm, context, reward):
        """تحديث النموذج بناءً على المكافأة"""
        self.A[arm] += np.outer(context, context)
        self.b[arm] += reward * context
        self.theta[arm] = np.linalg.inv(self.A[arm]).dot(self.b[arm])
    
    def get_arm_weights(self, context):
        """الحصول على أوزان الذراع المثلى للسياق المحدد"""
        return self.theta[self.select_arm(context)]