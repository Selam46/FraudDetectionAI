from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from .model_trainer import ModelTrainer

class TraditionalModelTrainer(ModelTrainer):
    def __init__(self, model_type="random_forest", model_params=None):
        super().__init__(model_params)
        self.model_type = model_type
        self.model = self._initialize_model()

    def _initialize_model(self):
        models = {
            "random_forest": RandomForestClassifier,
            "gradient_boosting": GradientBoostingClassifier,
            "logistic_regression": LogisticRegression,
            "decision_tree": DecisionTreeClassifier,
            "mlp": MLPClassifier
        }
        return models[self.model_type](**self.model_params) 