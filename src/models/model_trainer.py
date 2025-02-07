class ModelTrainer:
    """Base class for all model training implementations"""
    def __init__(self, model_params=None):
        self.model_params = model_params or {}
        self.model = None
        self.training_history = None

    def train(self, X_train, y_train):
        raise NotImplementedError("Subclasses must implement train method")

    def evaluate(self, X_test, y_test):
        raise NotImplementedError("Subclasses must implement evaluate method")

    def save_model(self, path):
        raise NotImplementedError("Subclasses must implement save_model method") 