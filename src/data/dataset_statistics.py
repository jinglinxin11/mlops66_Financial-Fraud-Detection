import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
from typing import List, Dict, Any
import math

from src.features.preprocessor import FraudPreprocessor
from src.config.settings import Config


class DatasetStatistics:
    def __init__(self, config: Config, verbose: bool = False):
        self.preprocessor = FraudPreprocessor(config=config, verbose=verbose)
        self.data = self.preprocessor.fit_transform()
        self.verbose = verbose
    
    def ClassDistribution(self):
        """ Plot class distribution in the dataset."""
        y_train = self.data["y_train"]
        y_test = self.data["y_test"]
        counter_train = Counter(y_train)
        counter_test = Counter(y_test)

        labels_train = list(counter_train.keys())
        sizes_train = list(counter_train.values())

        labels_test = list(counter_test.keys())
        sizes_test = list(counter_test.values())

        fig, ax = plt.subplots(1, 2, figsize=(12, 5))
        ax[0].pie(sizes_train, labels=labels_train, autopct='%1.1f%%', startangle=140, colors =['cornflowerblue', 'orange'])
        ax[0].set_title('Training Set Class Distribution')
        ax[1].pie(sizes_test, labels=labels_test, autopct='%1.1f%%', startangle=140, colors =['orange', 'cornflowerblue'])
        ax[1].set_title('Test Set Class Distribution')
        plt.tight_layout()
        plt.savefig('class_distribution.png')
    
    def FeatureDistributions(self, feature_names: List[str]):
        """ Plot distributions of specified features."""
        X_train = self.data["X_train"]
        feature_columns = self.data["feature_columns"]
        feature_indices = {name: idx for idx, name in enumerate(feature_columns)}
        num_features = len(feature_names)
        cols = 2
        rows = math.ceil(num_features / cols)
        fig, axes = plt.subplots(rows, cols, figsize=(12, 5 * rows))

        for i, feature_name in enumerate(feature_names):
            if feature_name not in feature_indices:
                print(f"Feature {feature_name} not found in dataset.")
                continue
            idx = feature_indices[feature_name]
            ax = axes[i // cols, i % cols] if rows > 1 else axes[i % cols]
            ax.hist(X_train[:, idx], bins=50, color='cornflowerblue', alpha=0.7, log = True)
            ax.set_title(f'Distribution of {feature_name}')
            ax.set_xlabel(feature_name)
            ax.set_ylabel('Frequency')
        plt.tight_layout()
        plt.savefig('feature_distributions.png')

    
    def CalculateStats(self):
        """ Calculate and print dataset statistics."""
        # Load data (requires data to be in project)
        X_train = self.data["X_train"]
        y_train = self.data["y_train"]
        print("====== TRAINING SET STATISTICS ======")
        print(f'Training samples: {X_train.shape[0]}')
        print(f'Features: {X_train.shape[1]}')
        print(f'Fraud cases: {np.sum(y_train)}')
        print(f'Non-fraud cases: {len(y_train) - np.sum(y_train)}')
        print('\n')

        X_test = self.data["X_test"]
        y_test = self.data["y_test"]
        print("====== TEST SET STATISTICS ======")
        print(f'Test samples: {X_test.shape[0]}')
        print(f'Fraud cases: {np.sum(y_test)}')
        print(f'Non-fraud cases: {len(y_test) - np.sum(y_test)}')
        print('\n')
        # Feature statistics

if __name__ == "__main__":
    config = Config()
    stats = DatasetStatistics(config)
    stats.CalculateStats()
    stats.ClassDistribution()
    stats.FeatureDistributions(['TransactionAmt', 'C1', 'C2', 'C3', 'C4', 'C5'])