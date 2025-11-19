#!/usr/bin/env python3
"""
Train a single model for parallel workflow execution.

Usage:
    python scripts/train_single_model.py <model_code> <data_path> <output_dir>

Example:
    python scripts/train_single_model.py gb data/features.csv models/
"""

import sys
import json
import pickle
import time
from pathlib import Path
import pandas as pd

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import (
    roc_auc_score, f1_score, accuracy_score,
    precision_score, recall_score, confusion_matrix
)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier


MODEL_CONFIGS = {
    'lr': {
        'name': 'Logistic Regression',
        'class': LogisticRegression,
        'params': {'random_state': 42, 'max_iter': 2000, 'class_weight': 'balanced'}
    },
    'dt': {
        'name': 'Decision Tree',
        'class': DecisionTreeClassifier,
        'params': {'random_state': 42, 'max_depth': 10, 'class_weight': 'balanced'}
    },
    'rf': {
        'name': 'Random Forest',
        'class': RandomForestClassifier,
        'params': {
            'n_estimators': 100, 'max_depth': 10,
            'random_state': 42, 'class_weight': 'balanced', 'n_jobs': -1
        }
    },
    'gb': {
        'name': 'Gradient Boosting',
        'class': GradientBoostingClassifier,
        'params': {
            'n_estimators': 100, 'learning_rate': 0.1,
            'max_depth': 5, 'random_state': 42
        }
    },
    'xgb': {
        'name': 'XGBoost',
        'class': None,  # Loaded dynamically
        'params': {
            'n_estimators': 100, 'learning_rate': 0.1,
            'max_depth': 5, 'random_state': 42, 'eval_metric': 'logloss'
        }
    }
}


def load_xgboost():
    """Dynamically load XGBoost (optional dependency)."""
    try:
        import xgboost as xgb
        return xgb.XGBClassifier
    except ImportError:
        print("Warning: XGBoost not available")
        return None


def calculate_metrics(model, X_test, y_test):
    """
    Calculate all evaluation metrics for a trained model

    Args:
        model: Trained model
        X_test: Test features
        y_test: Test labels

    Returns:
        Dictionary of metrics
    """
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else y_pred

    return {
        'accuracy': float(accuracy_score(y_test, y_pred)),
        'precision': float(precision_score(y_test, y_pred)),
        'recall': float(recall_score(y_test, y_pred)),
        'f1': float(f1_score(y_test, y_pred)),
        'roc_auc': float(roc_auc_score(y_test, y_proba)),
        'confusion_matrix': confusion_matrix(y_test, y_pred).tolist()
    }


def extract_feature_importance(model, feature_names, top_n=20):
    """
    Extract feature importance from tree-based models

    Args:
        model: Trained model
        feature_names: List of feature names
        top_n: Number of top features to return

    Returns:
        Dictionary with feature names and importances or None
    """
    if not hasattr(model, 'feature_importances_'):
        return None

    # Create and sort feature importance pairs
    feature_importance_pairs = sorted(
        zip(feature_names, model.feature_importances_),
        key=lambda x: x[1],
        reverse=True
    )[:top_n]

    return {
        'feature_names': [f[0] for f in feature_importance_pairs],
        'importances': [float(f[1]) for f in feature_importance_pairs]
    }


def train_model(model_code, data_path, output_dir):
    """Train a single model and save results."""

    # Load configuration
    if model_code not in MODEL_CONFIGS:
        raise ValueError(f"Unknown model: {model_code}. Choose from {list(MODEL_CONFIGS.keys())}")

    config = MODEL_CONFIGS[model_code]
    model_name = config['name']

    print(f"Training {model_name} ({model_code})...")

    # Load data
    df = pd.read_csv(data_path)
    df = df[df['binary_outcome'].isin([0, 1])]

    # Separate features and labels
    label_columns = ['binary_outcome', 'outcome_label']
    X = df.drop(columns=[col for col in label_columns if col in df.columns])
    y = df['binary_outcome']

    print(f"Dataset: {len(df)} samples, {len(X.columns)} features")
    print(f"Class distribution: {y.value_counts().to_dict()}")

    # Check for at least 2 classes
    n_classes = len(y.unique())
    if n_classes < 2:
        raise ValueError(
            f"Dataset contains only {n_classes} class(es): {y.unique().tolist()}. "
            f"Classification requires at least 2 classes."
        )

    # Split data with stratification
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Initialize model
    ModelClass = config['class']
    if ModelClass is None:  # XGBoost
        ModelClass = load_xgboost()
        if ModelClass is None:
            raise ImportError("XGBoost not available")

    model = ModelClass(**config['params'])

    # Train with timing
    print("Training...")
    start_time = time.time()
    model.fit(X_train, y_train)
    training_time = time.time() - start_time
    print(f"Training completed in {training_time:.2f} seconds")

    # Evaluate
    print("Evaluating...")
    metrics = calculate_metrics(model, X_test, y_test)
    metrics.update({
        'model_name': model_name,
        'model_code': model_code,
        'training_time': float(training_time)
    })

    # Cross-validation
    print("Cross-validating...")
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='roc_auc', n_jobs=-1)

    metrics['cv_roc_auc_mean'] = float(cv_scores.mean())
    metrics['cv_roc_auc_std'] = float(cv_scores.std())

    # Extract feature importance
    feature_importance = extract_feature_importance(model, X.columns.tolist())
    metrics['feature_importance'] = feature_importance

    if feature_importance:
        print(f"Top 5 features: {feature_importance['feature_names'][:5]}")
    else:
        print("Feature importance not available for this model type")

    # Save model and metrics
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    model_file = output_path / f'{model_code}_model.pkl'
    with open(model_file, 'wb') as f:
        pickle.dump(model, f)
    print(f"Model saved: {model_file}")

    metrics_file = output_path / f'results_{model_code}.json'
    with open(metrics_file, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved: {metrics_file}")

    # Print summary
    print(f"\n{'='*60}")
    print(f"Model: {model_name}")
    print(f"ROC AUC: {metrics['roc_auc']:.4f}")
    print(f"F1 Score: {metrics['f1']:.4f}")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"CV ROC AUC: {metrics['cv_roc_auc_mean']:.4f} ± {metrics['cv_roc_auc_std']:.4f}")
    print(f"Training Time: {metrics['training_time']:.2f}s")
    print(f"{'='*60}\n")

    return metrics


if __name__ == '__main__':
    if len(sys.argv) != 4:
        print(__doc__)
        sys.exit(1)

    model_code = sys.argv[1]
    data_path = sys.argv[2]
    output_dir = sys.argv[3]

    train_model(model_code, data_path, output_dir)
