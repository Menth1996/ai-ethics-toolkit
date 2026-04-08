import pandas as pd
from sklearn.metrics import accuracy_score
from aif360.datasets import BinaryLabelDataset
from aif360.metrics import BinaryLabelDatasetMetric, ClassificationMetric
from aif360.algorithms.preprocessing import Reweighing

class BiasDetector:
    """Detects and quantifies bias in AI models and datasets."""
    def __init__(self, model, dataset, sensitive_features: list):
        self.model = model
        self.dataset = dataset
        self.sensitive_features = sensitive_features
        self.privileged_groups = [{sf: 1} for sf in sensitive_features] # Example: assuming 1 is privileged
        self.unprivileged_groups = [{sf: 0} for sf in sensitive_features] # Example: assuming 0 is unprivileged

    def _create_aif_dataset(self, df: pd.DataFrame, label_names: list, protected_attribute_names: list):
        return BinaryLabelDataset(
            df=df,
            label_names=label_names,
            protected_attribute_names=protected_attribute_names
        )

    def generate_report(self):
        """Generates a bias report based on various fairness metrics."""
        print("
--- Bias Detection Report ---")
        
        # Assuming self.dataset is a pandas DataFrame with features and a 'label' column
        # And model.predict returns predictions for the dataset
        predictions = self.model.predict(self.dataset.drop(columns=['label']))
        
        # Convert to AIF360 format
        aif_dataset = self._create_aif_dataset(
            self.dataset.copy(), 
            label_names=['label'], 
            protected_attribute_names=self.sensitive_features
        )
        
        aif_predictions = aif_dataset.copy()
        aif_predictions.labels = predictions.reshape(-1, 1)

        metric = ClassificationMetric(
            aif_dataset,
            aif_predictions,
            unprivileged_groups=self.unprivileged_groups,
            privileged_groups=self.privileged_groups
        )

        report = {}
        report["Statistical Parity Difference"] = metric.statistical_parity_difference()
        report["Equal Opportunity Difference"] = metric.equal_opportunity_difference()
        report["Average Abs Odds Difference"] = metric.average_abs_odds_difference()
        
        print("Fairness Metrics:")
        for k, v in report.items():
            print(f"- {k}: {v:.4f}")
            
        return report

if __name__ == "__main__":
    # Mock data and model for demonstration
    data = {
        'feature1': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
        'gender': [0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
        'race': [0, 0, 1, 1, 0, 0, 1, 1, 0, 0],
        'label': [0, 1, 0, 1, 0, 1, 0, 1, 0, 1] # Mock labels
    }
    df = pd.DataFrame(data)

    class MockModel:
        def predict(self, X):
            # Simple mock prediction
            return (X['feature1'] > 50).astype(int).values

    model = MockModel()
    detector = BiasDetector(model, df, sensitive_features=['gender', 'race'])
    detector.generate_report()
