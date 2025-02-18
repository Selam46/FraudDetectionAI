import numpy as np
import matplotlib.pyplot as plt
import shap
import lime
import lime.lime_tabular
from typing import Any, Dict, List, Tuple

def generate_shap_explanations(
    model: Any,
    X: np.ndarray,
    feature_names: List[str],
    output_dir: str
) -> Dict[str, np.ndarray]:
    """
    Generate SHAP explanations for a model.
    
    Args:
        model: Trained model
        X: Feature matrix
        feature_names: List of feature names
        output_dir: Directory to save plots
    
    Returns:
        Dict containing SHAP values and feature importance
    """
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    
    # Generate and save summary plot
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X, feature_names=feature_names, show=False)
    plt.savefig(f'{output_dir}/shap_summary.png')
    plt.close()
    
    # Calculate feature importance
    feature_importance = np.abs(shap_values).mean(0)
    
    return {
        'shap_values': shap_values,
        'feature_importance': feature_importance
    }

def generate_lime_explanation(
    model: Any,
    X: np.ndarray,
    feature_names: List[str],
    instance_index: int,
    output_dir: str
) -> Tuple[Any, Dict[str, float]]:
    """
    Generate LIME explanation for a specific instance.
    
    Args:
        model: Trained model
        X: Feature matrix
        feature_names: List of feature names
        instance_index: Index of instance to explain
        output_dir: Directory to save plots
    
    Returns:
        Tuple of LIME explanation object and feature importance dict
    """
    explainer = lime.lime_tabular.LimeTabularExplainer(
        X,
        feature_names=feature_names,
        class_names=['Not Fraud', 'Fraud'],
        mode='classification'
    )
    
    exp = explainer.explain_instance(
        X[instance_index],
        model.predict_proba,
        num_features=10
    )
    
    # Save explanation plot
    plt.figure(figsize=(10, 6))
    exp.as_pyplot_figure()
    plt.savefig(f'{output_dir}/lime_explanation_{instance_index}.png')
    plt.close()
    
    return exp, dict(exp.as_list()) 