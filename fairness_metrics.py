
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

class FairnessMetrics:
    def __init__(self, sensitive_attribute, positive_label=1):
        self.sensitive_attribute = sensitive_attribute
        self.positive_label = positive_label

    def _get_group_data(self, df, group_value):
        return df[df[self.sensitive_attribute] == group_value]

    def demographic_parity(self, df, predictions, true_labels):
        """
        Calculates demographic parity (also known as statistical parity).
        Measures if the positive prediction rate is similar across different groups.
        """
        positive_rates = {}
        for group_value in df[self.sensitive_attribute].unique():
            group_df = self._get_group_data(df, group_value)
            group_predictions = predictions[group_df.index]
            positive_rates[group_value] = (group_predictions == self.positive_label).mean()
        return positive_rates

    def equal_opportunity(self, df, predictions, true_labels):
        """
        Calculates equal opportunity.
        Measures if the true positive rate (recall) is similar across different groups.
        """
        true_positive_rates = {}
        for group_value in df[self.sensitive_attribute].unique():
            group_df = self._get_group_data(df, group_value)
            group_predictions = predictions[group_df.index]
            group_true_labels = true_labels[group_df.index]
            
            # Filter for actual positive cases in the group
            actual_positives_indices = group_true_labels[group_true_labels == self.positive_label].index
            if not actual_positives_indices.empty:
                true_positive_rates[group_value] = recall_score(
                    group_true_labels[actual_positives_indices],
                    group_predictions[actual_positives_indices],
                    pos_label=self.positive_label,
                    zero_division=0
                )
            else:
                true_positive_rates[group_value] = 0.0 # No actual positives in this group
        return true_positive_rates

    def predictive_equality(self, df, predictions, true_labels):
        """
        Calculates predictive equality.
        Measures if the false positive rate is similar across different groups.
        """
        false_positive_rates = {}
        for group_value in df[self.sensitive_attribute].unique():
            group_df = self._get_group_data(df, group_value)
            group_predictions = predictions[group_df.index]
            group_true_labels = true_labels[group_df.index]
            
            # Filter for actual negative cases in the group
            actual_negatives_indices = group_true_labels[group_true_labels != self.positive_label].index
            if not actual_negatives_indices.empty:
                # FPR = FP / (FP + TN) = FP / N
                fp = ((group_predictions == self.positive_label) & (group_true_labels != self.positive_label)).sum()
                tn = ((group_predictions != self.positive_label) & (group_true_labels != self.positive_label)).sum()
                false_positive_rates[group_value] = fp / (fp + tn) if (fp + tn) > 0 else 0.0
            else:
                false_positive_rates[group_value] = 0.0 # No actual negatives in this group
        return false_positive_rates

    def calculate_all_metrics(self, df, predictions, true_labels):
        metrics = {
            "demographic_parity": self.demographic_parity(df, predictions, true_labels),
            "equal_opportunity": self.equal_opportunity(df, predictions, true_labels),
            "predictive_equality": self.predictive_equality(df, predictions, true_labels),
        }
        return metrics

if __name__ == '__main__':
    # Example Usage
    data = {
        'feature1': [10, 12, 15, 8, 11, 14, 9, 13],
        'sensitive_group': ['A', 'A', 'B', 'B', 'A', 'A', 'B', 'B'],
        'true_label': [1, 1, 0, 0, 1, 0, 0, 1],
        'prediction': [1, 0, 0, 1, 1, 0, 0, 1]
    }
    df = pd.DataFrame(data)
    predictions = df['prediction']
    true_labels = df['true_label']

    fairness_analyzer = FairnessMetrics(sensitive_attribute='sensitive_group', positive_label=1)

    print("\n--- Fairness Metrics Analysis ---")
    all_metrics = fairness_analyzer.calculate_all_metrics(df, predictions, true_labels)

    for metric_name, values in all_metrics.items():
        print(f"\n{metric_name.replace('_', ' ').title()}:")
        for group, value in values.items():
            print(f"  Group {group}: {value:.4f}")

    # Interpretation (example):
    # Demographic Parity: If values are close, positive prediction rate is similar across groups.
    # Equal Opportunity: If values are close, true positive rate (recall) is similar across groups.
    # Predictive Equality: If values are close, false positive rate is similar across groups.

    print("\n--- Example Complete ---")

# Commit timestamp: 2025-09-01 00:00:00 - 29
