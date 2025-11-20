"""
XGBoost Classifier Model for Regulation State Prediction

This module implements a multi-class classifier using XGBoost for predicting
electricity market regulation states (UP, DOWN, BALANCED, UP_AND_DOWN).
"""

import logging
from typing import Dict, List, Optional, Tuple, Any
import pickle
from pathlib import Path

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

logger = logging.getLogger(__name__)


class RegulationStateXGBModel:
    """
    XGBoost classifier for multi-class regulation state prediction.

    This model uses XGBoost's gradient boosting framework to predict
    categorical regulation states in electricity markets.

    Attributes:
        model: The trained XGBoost classifier instance
        feature_columns: List of feature column names used for training
        class_labels: Dict mapping numeric labels to human-readable state names
        n_classes: Number of classification classes
    """

    def __init__(
        self,
        feature_columns: Optional[List[str]] = None,
        n_estimators: int = 100,
        max_depth: int = 6,
        learning_rate: float = 0.1,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        random_state: int = 42,
        **kwargs,
    ):
        """
        Initialize XGBoost classifier model.

        Args:
            feature_columns: List of feature column names. Required for training.
            n_estimators: Number of boosting rounds (trees)
            max_depth: Maximum tree depth
            learning_rate: Step size shrinkage to prevent overfitting
            subsample: Subsample ratio of training instances
            colsample_bytree: Subsample ratio of columns when constructing trees
            random_state: Random seed for reproducibility
            **kwargs: Additional XGBoost parameters
        """
        super().__init__()

        self.feature_columns = feature_columns
        self.model = None
        self.class_labels = {-1: "DOWN", 0: "BALANCED", 1: "UP", 2: "UP_AND_DOWN"}
        self.n_classes = len(self.class_labels)

        # XGBoost hyperparameters
        self.params = {
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "learning_rate": learning_rate,
            "subsample": subsample,
            "colsample_bytree": colsample_bytree,
            "random_state": random_state,
            "objective": "multi:softprob",  # Multi-class classification with probabilities
            "num_class": self.n_classes,
            "eval_metric": "mlogloss",
            "tree_method": "hist",  # Faster histogram-based algorithm
            "verbosity": 1,
        }
        self.params.update(kwargs)

        logger.info(
            f"Initialized RegulationStateXGBModel with {n_estimators} estimators, "
            f"max_depth={max_depth}, learning_rate={learning_rate}"
        )

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        eval_set: Optional[List[Tuple[pd.DataFrame, pd.Series]]] = None,
        early_stopping_rounds: Optional[int] = 10,
        verbose: bool = True,
    ) -> "RegulationStateXGBModel":
        """
        Train the XGBoost classifier on the provided data.

        Args:
            X: Training features
            y: Training target (regulation_state: -1, 0, 1, or 2)
            eval_set: Optional validation sets as list of (X_val, y_val) tuples
            early_stopping_rounds: Stop training if validation metric doesn't improve
            verbose: Whether to print training progress

        Returns:
            self: The trained model instance

        Raises:
            ValueError: If feature_columns is not set or X columns don't match
        """
        if self.feature_columns is None:
            self.feature_columns = X.columns.tolist()
            logger.info(f"Feature columns set to: {self.feature_columns}")

        if set(self.feature_columns) != set(X.columns):
            raise ValueError(
                f"Feature columns mismatch. Expected: {self.feature_columns}, "
                f"Got: {X.columns.tolist()}"
            )

        # Map regulation states to 0-indexed labels for XGBoost
        # Original: {-1: DOWN, 0: BALANCED, 1: UP, 2: UP_AND_DOWN}
        # XGBoost needs: {0, 1, 2, 3}
        state_to_idx = {-1: 0, 0: 1, 1: 2, 2: 3}
        idx_to_state = {v: k for k, v in state_to_idx.items()}
        self._state_to_idx = state_to_idx
        self._idx_to_state = idx_to_state

        y_encoded = y.map(state_to_idx)

        # Prepare evaluation sets if provided
        eval_list = None
        if eval_set is not None:
            eval_list = [(X_val, y_val.map(state_to_idx)) for X_val, y_val in eval_set]

        # Initialize and train XGBoost classifier
        self.model = xgb.XGBClassifier(**self.params)

        # Build kwargs for fit method (early_stopping_rounds deprecated in newer versions)
        fit_kwargs = {
            "X": X,
            "y": y_encoded,
        }

        if eval_list is not None:
            fit_kwargs["eval_set"] = eval_list
            # Only add early_stopping_rounds if we have validation set
            if early_stopping_rounds is not None:
                try:
                    # Try new parameter name first
                    self.model.set_params(early_stopping_rounds=early_stopping_rounds)
                except:
                    pass  # Ignore if parameter doesn't exist

        if verbose:
            fit_kwargs["verbose"] = verbose

        self.model.fit(**fit_kwargs)

        # Training metrics
        y_pred = self.predict(X)
        train_accuracy = accuracy_score(y, y_pred)

        logger.info(f"Training completed. Accuracy: {train_accuracy:.4f}")
        
        # Only log best_iteration if early stopping was used
        if eval_list is not None and early_stopping_rounds is not None:
            try:
                logger.info(f"Best iteration: {self.model.best_iteration}")
            except AttributeError:
                pass  # best_iteration not available without early stopping

        if verbose:
            print("\nTraining Classification Report:")
            print(
                classification_report(
                    y,
                    y_pred,
                    target_names=[self.class_labels[s] for s in sorted(self.class_labels.keys())],
                )
            )

        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict regulation state class labels.

        Args:
            X: Features for prediction

        Returns:
            Array of predicted regulation states (-1, 0, 1, or 2)

        Raises:
            ValueError: If model is not trained or feature columns don't match
        """
        if self.model is None:
            raise ValueError("Model not trained. Call fit() first.")

        if set(self.feature_columns) != set(X.columns):
            raise ValueError(
                f"Feature columns mismatch. Expected: {self.feature_columns}, "
                f"Got: {X.columns.tolist()}"
            )

        # Get XGBoost predictions (0-indexed)
        y_pred_encoded = self.model.predict(X)

        # Convert back to original regulation state labels
        y_pred = np.array([self._idx_to_state[idx] for idx in y_pred_encoded])

        return y_pred

    def predict_proba(self, X: pd.DataFrame) -> Dict[int, np.ndarray]:
        """
        Predict class probabilities for each regulation state.

        Args:
            X: Features for prediction

        Returns:
            Dictionary mapping regulation states to probability arrays
            Format: {-1: array([...]), 0: array([...]), 1: array([...]), 2: array([...])}

        Raises:
            ValueError: If model is not trained or feature columns don't match
        """
        if self.model is None:
            raise ValueError("Model not trained. Call fit() first.")

        if set(self.feature_columns) != set(X.columns):
            raise ValueError(
                f"Feature columns mismatch. Expected: {self.feature_columns}, "
                f"Got: {X.columns.tolist()}"
            )

        # Get probability predictions from XGBoost (shape: [n_samples, 4])
        proba_encoded = self.model.predict_proba(X)

        # Map probabilities back to original state labels
        proba_dict = {}
        for xgb_idx, state in self._idx_to_state.items():
            proba_dict[state] = proba_encoded[:, xgb_idx]

        return proba_dict

    def evaluate(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """
        Evaluate model performance on test data.

        Args:
            X: Test features
            y: True regulation states

        Returns:
            Dictionary containing evaluation metrics:
                - accuracy: Overall accuracy
                - confusion_matrix: Confusion matrix
                - classification_report: Detailed per-class metrics
        """
        y_pred = self.predict(X)

        accuracy = accuracy_score(y, y_pred)
        conf_matrix = confusion_matrix(y, y_pred)
        class_report = classification_report(
            y,
            y_pred,
            target_names=[self.class_labels[s] for s in sorted(self.class_labels.keys())],
            output_dict=True,
        )

        logger.info(f"Evaluation accuracy: {accuracy:.4f}")

        return {
            "accuracy": accuracy,
            "confusion_matrix": conf_matrix,
            "classification_report": class_report,
        }

    def get_feature_importance(self, importance_type: str = "weight") -> pd.Series:
        """
        Get feature importances from the trained model.

        Args:
            importance_type: Type of importance metric
                - "weight": Number of times feature is used
                - "gain": Average gain across splits using the feature
                - "cover": Average coverage across splits using the feature

        Returns:
            Series with feature names as index and importance scores as values

        Raises:
            ValueError: If model is not trained
        """
        if self.model is None:
            raise ValueError("Model not trained. Call fit() first.")

        importance = self.model.get_booster().get_score(importance_type=importance_type)

        # Create series with feature names
        importance_series = pd.Series(importance).sort_values(ascending=False)

        return importance_series

    def save(self, filepath: str) -> None:
        """
        Save the trained model to disk.

        Args:
            filepath: Path to save the model
        """
        if self.model is None:
            raise ValueError("Cannot save untrained model")

        Path(filepath).parent.mkdir(parents=True, exist_ok=True)

        model_data = {
            "model": self.model,
            "feature_columns": self.feature_columns,
            "class_labels": self.class_labels,
            "params": self.params,
            "state_to_idx": self._state_to_idx,
            "idx_to_state": self._idx_to_state,
        }

        with open(filepath, "wb") as f:
            pickle.dump(model_data, f)

        logger.info(f"Model saved to {filepath}")

    def load(self, filepath: str) -> None:
        """
        Load a trained model from disk.

        Args:
            filepath: Path to the saved model
        """
        with open(filepath, "rb") as f:
            model_data = pickle.load(f)

        self.model = model_data["model"]
        self.feature_columns = model_data["feature_columns"]
        self.class_labels = model_data["class_labels"]
        self.params = model_data["params"]
        self._state_to_idx = model_data["state_to_idx"]
        self._idx_to_state = model_data["idx_to_state"]

        logger.info(f"Model loaded from {filepath}")

    def get_params(self, deep: bool = True) -> dict:
        """Get model parameters (sklearn compatibility)."""
        return self.params.copy()

    def set_params(self, **params) -> "RegulationStateXGBModel":
        """Set model parameters (sklearn compatibility)."""
        self.params.update(params)
        # Update individual attributes for commonly used params
        if "n_estimators" in params:
            self.params["n_estimators"] = params["n_estimators"]
        if "max_depth" in params:
            self.params["max_depth"] = params["max_depth"]
        if "learning_rate" in params:
            self.params["learning_rate"] = params["learning_rate"]
        # Recreate model with new params
        self.model = xgb.XGBClassifier(**self.params)
        return self

    def score(self, X: pd.DataFrame, y: pd.Series) -> float:
        """
        Calculate accuracy score on the given data.

        Args:
            X: Feature matrix
            y: True target values

        Returns:
            Accuracy score
        """
        if self.model is None:
            raise ValueError("Model not trained. Call fit() first.")

        y_pred = self.predict(X)
        return accuracy_score(y, y_pred)
