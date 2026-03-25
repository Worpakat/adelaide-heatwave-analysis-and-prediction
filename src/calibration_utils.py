import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import TimeSeriesSplit

from src.modelling_utils import get_final_feature_matrices
from sklearn.calibration import CalibrationDisplay 


class XGBCalibrated:
    def __init__(self, params, n_splits=5, test_size=365*2, random_state=42):
        self.params = params
        self.n_splits = n_splits
        self.test_size = test_size
        self.random_state = random_state
        
        # Models and transformers to be saved
        self.xgb_base = None
        self.logreg_calib = None
        self.scaler = None
        self.pcas = None
        
        # Metadata for feature processing
        self.feature_groups_df = None
        self.non_standardized_features = None
        self.not_to_be_used = None

    def fit_and_calibrate(self, X_train, y_train, X_test, 
                          feature_groups_df, non_standardized_features, 
                          features_not_to_be_used):
        """
        Executes TimeSeriesSplit to generate calibration logits, 
        fits the final models, and stores all transformation parameters.
        """
        # Save metadata for future inference
        self.feature_groups_df = feature_groups_df
        self.non_standardized_features = non_standardized_features
        self.not_to_be_used = features_not_to_be_used

        tscv = TimeSeriesSplit(n_splits=self.n_splits, test_size=self.test_size)
        raw_logits_calib = np.array([])
        calib_start_idx = None

        print(f"Starting Cross-Validation for Calibration ({self.n_splits} splits)...")

        # 1. GENERATE OUT-OF-SAMPLE LOGITS
        for fold, (train_idx, test_idx) in enumerate(tscv.split(X_train)):
            X_train_cv, X_test_cv = X_train.iloc[train_idx], X_train.iloc[test_idx]
            y_train_cv, _ = y_train.iloc[train_idx], y_train.iloc[test_idx]

            if calib_start_idx is None:
                calib_start_idx = test_idx[0]

            # Process features for this fold (Temporary scaler/pcas)
            X_train_final, X_test_final, _, _ = get_final_feature_matrices(
                X_train=X_train_cv, 
                X_test=X_test_cv, 
                feature_groups_df=self.feature_groups_df,
                non_standardized_features=self.non_standardized_features,
                features_not_to_be_used=self.not_to_be_used
            )

            # Train base model
            xgb_cv = XGBClassifier(**self.params, random_state=self.random_state)
            xgb_cv.fit(X_train_final, y_train_cv)

            # Get raw logits (output_margin=True)
            test_logits = xgb_cv.predict(X_test_final, output_margin=True)
            raw_logits_calib = np.concatenate((raw_logits_calib, test_logits), axis=0)

        # 2. CALIBRATION TRAINING
        print("Training Logistic Calibrator...")
        self.logreg_calib = LogisticRegression(random_state=self.random_state)
        y_calib = y_train.iloc[calib_start_idx:]
        self.logreg_calib.fit(raw_logits_calib.reshape(-1, 1), y_calib)

        # 3. FINAL BASE MODEL TRAINING
        print("Training final XGBoost model on full training set...")
        X_train_final, X_test_final, self.scaler, self.pcas = get_final_feature_matrices(
            X_train=X_train, 
            X_test=X_test, 
            feature_groups_df=self.feature_groups_df,
            non_standardized_features=self.non_standardized_features,
            features_not_to_be_used=self.not_to_be_used
        )

        self.xgb_base = XGBClassifier(**self.params, random_state=self.random_state)
        self.xgb_base.fit(X_train_final, y_train)

        print("Fit and Calibration successful.")


    def _preprocess_inference(self, X_raw):
        """Internal helper to transform raw data for prediction."""
        _, X_processed, _, _ = get_final_feature_matrices(
            X_train=None,
            X_test=X_raw,
            feature_groups_df=self.feature_groups_df,
            non_standardized_features=self.non_standardized_features,
            features_not_to_be_used=self.not_to_be_used,
            scaler=self.scaler,
            pcas=self.pcas
        )
        return X_processed

    def predict_raw(self, X_raw, threshold=0.5):
        """Returns base XGB predictions (0 or 1). No calibration layer."""
        X_final = self._preprocess_inference(X_raw)
        raw_probs_positive = self.xgb_base.predict_proba(X_final)[:, 1]
        return (raw_probs_positive >= threshold).astype(int)

    def predict_proba_raw(self, X_raw):
        """Returns base XGB predicted probabilities. No calibration layer."""
        X_final = self._preprocess_inference(X_raw)
        return self.xgb_base.predict_proba(X_final)

    def predict_proba_calib(self, X_raw):
        """Returns calibrated probabilities (XGB Logits -> Logistic Regression)."""
        X_final = self._preprocess_inference(X_raw)
        
        # 1. Get raw logits (the 'margin') from XGBoost
        raw_logits = self.xgb_base.predict(X_final, output_margin=True).reshape(-1, 1)
        
        # 2. Pass through Logistic Calibrator to get probability of class 1
        calibrated_probs = self.logreg_calib.predict_proba(raw_logits)
        return calibrated_probs

    def predict_calib(self, X_raw, threshold=0.5):
        """Returns predictions (0 or 1) using calibrated probabilities and a threshold."""
        # Get the probability for class 1 (index 1)
        calib_probs_positive = self.predict_proba_calib(X_raw)[:, 1]
        return (calib_probs_positive >= threshold).astype(int)
    

def plot_calibration_curves(y_test, test_proba, y_train=None, train_proba=None, n_bins=10):
    cols = 1 if (y_train is None or train_proba is None) else 2
    fig, ax = plt.subplots(1, cols, figsize=((cols*5), 5))
    fig.set_tight_layout(True)
    disp_train = CalibrationDisplay.from_predictions(y_test, test_proba, n_bins=n_bins, ax=ax[0], name='Test', color='orange')
    
    if cols == 2: # Train is inclueded   
        disp_test = CalibrationDisplay.from_predictions(y_train, train_proba, n_bins=n_bins, ax=ax[1], name='Train', color='b')
    
    plt.show()


"""Source of function ``expected_calibration_error()``: 
'https://medium.com/data-science/expected-calibration-error-ece-a-step-by-step-visual-explanation-with-python-code-c3e9aa12937d'"""
def expected_calibration_error(samples, true_labels, n_bins=5):
    """ Returns the expected calibration error (ECE) between predicted and actual probabilities.
    """

    # uniform binning approach with M number of bins
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    # get max probability per sample i
    confidences = np.max(samples, axis=1)
    # get predictions from confidences (positional in this case)
    predicted_label = np.argmax(samples, axis=1)

    # get a boolean list of correct/false predictions
    accuracies = predicted_label==true_labels

    ece = np.zeros(1)
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        # determine if sample is in bin m (between bin lower & upper)
        in_bin = np.logical_and(confidences > bin_lower.item(), confidences <= bin_upper.item())
        # can calculate the empirical probability of a sample falling into bin m: (|Bm|/n)
        prob_in_bin = in_bin.mean()

        if prob_in_bin.item() > 0:
            # get the accuracy of bin m: acc(Bm)
            accuracy_in_bin = accuracies[in_bin].mean()
            # get the average confidence of bin m: conf(Bm)
            avg_confidence_in_bin = confidences[in_bin].mean()
            # calculate |acc(Bm) - conf(Bm)| * (|Bm|/n) for bin m and add to the total ECE
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prob_in_bin
    return ece

def ece_binary_with_bins(pos_probs, true_labels, n_bins=10):
    """
    pos_probs: np.array (n_samples,)
        predicted probability of positive class

    true_labels: np.array (n_samples,)
        binary labels {0,1}

    Returns
    -------
    ece : float
    bin_errors : np.array (n_bins,)
        |acc(Bm) - conf(Bm)| for each bin
    bin_weights : np.array (n_bins,)
        |Bm| / n for each bin
    bin_acc : np.array (n_bins,)
        empirical positive frequency
    bin_conf : np.array (n_bins,)
        mean predicted probability
    """

    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    n = len(pos_probs)

    bin_errors = np.zeros(n_bins)
    bin_weights = np.zeros(n_bins)
    bin_acc = np.zeros(n_bins)
    bin_conf = np.zeros(n_bins)

    for i, (lower, upper) in enumerate(zip(bin_lowers, bin_uppers)):

        in_bin = (pos_probs > lower) & (pos_probs <= upper)
        bin_count = np.sum(in_bin)

        if bin_count > 0:

            weight = bin_count / n
            acc = np.mean(true_labels[in_bin])
            conf = np.mean(pos_probs[in_bin])

            bin_errors[i] = np.abs(acc - conf)
            bin_weights[i] = weight
            bin_acc[i] = acc
            bin_conf[i] = conf

    ece = np.sum(bin_errors * bin_weights)

    return ece, bin_errors, bin_weights, bin_acc, bin_conf