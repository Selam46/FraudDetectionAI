import yaml
import mlflow
import pandas as pd
from src.models.traditional_models import TraditionalModelTrainer
from src.evaluation.metrics import ModelEvaluator
from src.model_utils import prepare_data


def train_and_evaluate():
    # Load and prepare data
    data = pd.read_csv('data/processed/train_data.csv')
    X_train, X_test, y_train, y_test, feature_columns = prepare_data(data, target_col='class')

    # Load configurations
    with open('configs/model_config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    # Train models
    for model_name, params in config['models'].items():
        with mlflow.start_run(run_name=f"training_{model_name}"):
            trainer = TraditionalModelTrainer(model_type=model_name, 
                                           model_params=params)
            trainer.train(X_train, y_train)
            
            # Evaluate
            results = ModelEvaluator.evaluate_model(
                trainer.model, X_test, y_test
            )
            
            # Log metrics
            mlflow.log_metrics({
                'roc_auc': results['roc_auc_score']
            })
            
            # Save model
            trainer.save_model(f"models/{model_name}") 