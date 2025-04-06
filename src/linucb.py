import numpy as np

class LinUCB:
    """نموذج LinUCB Bandit"""
    def __init__(self, n_arms, context_dim, alpha=0.1):
        self.n_arms = n_arms
        self.context_dim = context_dim
        self.alpha = alpha
        self.A = [np.identity(context_dim) for _ in range(n_arms)]
        self.b = [np.zeros(context_dim) for _ in range(n_arms)]
        self.theta = [np.zeros(context_dim) for _ in range(n_arms)]

    def select_arm(self, context):
        """اختيار أفضل ذراع بناءً على السياق"""
        ucbs = []
        for arm in range(self.n_arms):
            A_inv = np.linalg.inv(self.A[arm])
            theta = self.theta[arm]
            score = np.dot(theta, context)
            uncertainty = self.alpha * np.sqrt(np.dot(context.T, np.dot(A_inv, context)))
            ucbs.append(score + uncertainty)
        return np.argmax(ucbs)

    def update(self, arm, context, reward):
        """تحديث النموذج بناءً على المكافأة"""
        self.A[arm] += np.outer(context, context)
        self.b[arm] += reward * context
        self.theta[arm] = np.linalg.inv(self.A[arm]).dot(self.b[arm])