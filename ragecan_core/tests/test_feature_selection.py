import os
import sys
import pandas as pd
import numpy as np
import argparse
import logging
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from feature_selection import (
    AnovaFeatureSelection,
    ChiSquareFeatureSelection,
    MutualInfoFeatureSelection,
    MRMRFeatureSelection,
    GiniIndexFeatureSelection,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def preprocess_data(df, target_column):
    """Quick preprocess for test"""
    X = df.drop(columns=[target_column])
    y = df[target_column].values

    # handling missing values
    imputer = SimpleImputer(strategy="mean")
    X_imputed = imputer.fit_transform(X)

    # Quick Standardize features for tests purpose
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imputed)

    return X_scaled, y, X.columns


def run_feature_selection_test(data_path, target_column, test_size=0.2, n_trials=5):
    logger.info(f"Loading dataset from {data_path}")
    df = pd.read_csv(data_path)

    logger.info(f"Dataset shape: {df.shape}, Target: {target_column}")
    if target_column not in df.columns:
        logger.error(f"Target column '{target_column}' not found in dataset")
        return

    X, y, feature_names = preprocess_data(df, target_column)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=42,
        stratify=y if len(np.unique(y)) > 1 else None,
    )
    logger.info(f"Train set: {X_train.shape}, Test set: {X_test.shape}")

    feature_selection_methods = [
        AnovaFeatureSelection("f1_macro", "rf"),
        ChiSquareFeatureSelection("f1_macro", "rf"),
        MutualInfoFeatureSelection("f1_macro", "rf"),
        MRMRFeatureSelection("f1_macro", "rf"),
        GiniIndexFeatureSelection("f1_macro", "rf"),
    ]

    min_features = min(5, X.shape[1])
    max_features = min(20, X.shape[1])
    desired_dimension = (min_features, max_features)

    results = []

    for fs_method in feature_selection_methods:
        logger.info(f"Testing {fs_method.name}...")
        try:
            result = fs_method.execute_selection(
                X_train,
                y_train,
                X_test,
                y_test,
                desired_dimension=desired_dimension,
                n_trials=n_trials,
            )
            logger.info(f"{fs_method.name} results:")
            logger.info(f"  Optimized parameters: {result.optimized_parameters}")
            logger.info(f"  Train metrics: {result.train_metrics}")
            logger.info(f"  Test metrics: {result.test_metrics}")
            selected_features = result.optimized_parameters.get("selected_features", [])
            if len(selected_features) > 0:
                selected_names = [feature_names[i] for i in selected_features]
                logger.info(f"  Selected features: {', '.join(selected_names)}")

            results.append(result)

        except Exception as e:
            logger.error(f"Error testing {fs_method.name}: {str(e)}")

    return results


def compare_results(results):
    """Compare results from different feature selection methods"""
    if not results or len(results) == 0:
        logger.warning("No results to compare")
        return

    logger.info("\n*** Tested Methods Performance Comparison ***")
    comparison_table = {
        "Method": [],
        "Features Selected": [],
        "F1 (Train)": [],
        "F1 (Test)": [],
        "MCC (Train)": [],
        "MCC (Test)": [],
    }

    for result in results:
        comparison_table["Method"].append(result.method_name)
        n_features = len(result.optimized_parameters.get("selected_features", []))
        comparison_table["Features Selected"].append(n_features)
        comparison_table["F1 (Train)"].append(
            result.train_metrics.get("f1_macro", "N/A")
        )
        comparison_table["F1 (Test)"].append(result.test_metrics.get("f1_macro", "N/A"))
        comparison_table["MCC (Train)"].append(result.train_metrics.get("mcc", "N/A"))
        comparison_table["MCC (Test)"].append(result.test_metrics.get("mcc", "N/A"))

    comparison_df = pd.DataFrame(comparison_table)
    logger.info("\n" + comparison_df.to_string(index=False))
    best_idx = comparison_df["F1 (Test)"].idxmax()
    logger.info(
        f"\nBest method based on test F1 score: {comparison_df.iloc[best_idx]['Method']}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test feature selection methods")
    parser.add_argument("--data", type=str, required=True, help="Path to CSV file")
    parser.add_argument("--target", type=str, required=True, help="Target column name")
    parser.add_argument(
        "--test_size", type=float, default=0.2, help="Test set size (default: 0.2)"
    )
    parser.add_argument(
        "--n_trials",
        type=int,
        default=5,
        help="Number of optimization trials (default: 5)",
    )

    args = parser.parse_args()

    logger.info("Starting feature selection unit tests")

    results = run_feature_selection_test(
        data_path=args.data,
        target_column=args.target,
        test_size=args.test_size,
        n_trials=args.n_trials,
    )

    if results:
        compare_results(results)
