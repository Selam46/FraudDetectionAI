import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow
import mlflow.sklearn
import joblib

def train_and_evaluate_model(model, model_name, X_train, X_test, y_train, y_test):
    """
    Train and evaluate a model, log metrics with MLflow
    
    Parameters:
    -----------
    model : sklearn estimator
        The machine learning model to train
    model_name : str
        Name of the model for logging purposes
    X_train, X_test : numpy arrays
        Training and test features
    y_train, y_test : numpy arrays
        Training and test target variables
        
    Returns:
    --------
    model : trained model
    auc_score : float
        The ROC AUC score of the model
    """
    with mlflow.start_run(run_name=model_name):
        # Train the model
        print(f"\nTraining {model_name}...")
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        class_report = classification_report(y_test, y_pred)
        conf_matrix = confusion_matrix(y_test, y_pred)
        auc_score = roc_auc_score(y_test, y_pred_proba)
        
        # Print results
        print(f"\n{model_name} Results:")
        print("\nClassification Report:")
        print(class_report)
        print("\nConfusion Matrix:")
        print(conf_matrix)
        print(f"\nROC AUC Score: {auc_score:.4f}")
        
        # Log metrics with MLflow
        mlflow.log_metric("auc_score", auc_score)
        mlflow.log_params(model.get_params())
        mlflow.sklearn.log_model(model, model_name)
        
        # Plot confusion matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Confusion Matrix - {model_name}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(f'../models/confusion_matrix_{model_name}.png')
        mlflow.log_artifact(f'../models/confusion_matrix_{model_name}.png')
        plt.close()
        
        return model, auc_score

def get_traditional_models():
    """
    Initialize and return dictionary of traditional ML models
    
    Returns:
    --------
    dict : Dictionary of model name and initialized model object pairs
    """
    return {
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'Random Forest': RandomForestClassifier(random_state=42, n_estimators=100),
        'Gradient Boosting': GradientBoostingClassifier(random_state=42)
    }

def plot_model_comparison(results):
    """
    Plot comparison of model performances
    
    Parameters:
    -----------
    results : dict
        Dictionary containing model results with AUC scores
    """
    plt.figure(figsize=(10, 6))
    auc_scores = [result['auc_score'] for result in results.values()]
    model_names = list(results.keys())
    plt.bar(model_names, auc_scores)
    plt.title('Model Performance Comparison')
    plt.xlabel('Model')
    plt.ylabel('AUC Score')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('../models/model_comparison.png')
    plt.close()

def save_best_model(results):
    """
    Save the best performing model
    
    Parameters:
    -----------
    results : dict
        Dictionary containing model results with AUC scores
        
    Returns:
    --------
    str : Name of the best performing model
    float : Best AUC score
    """
    best_model_name = max(results.items(), key=lambda x: x[1]['auc_score'])[0]
    best_model = results[best_model_name]['model']
    best_auc = results[best_model_name]['auc_score']
    
    # Save the model
    joblib.dump(best_model, f'../models/best_model_{best_model_name}.joblib')
    
    return best_model_name, best_auc 