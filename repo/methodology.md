# Methodology

Source: project report *Alzheimer's Disease Detection using Advanced Machine Learning Techniques* (Habib, Khan, Anwar; East Delta University). Settings that the report does not state, and that could not be recovered because the original training code was unavailable, are marked as *not recorded*.

**Research question.** Can two boosting classifiers, AdaBoost and Gradient Boosting, classify Alzheimer's disease (AD) from structured demographic, lifestyle, medical-history, clinical and cognitive variables, and how do they differ in accuracy and in the types of error they make?

## Data
- Alzheimer's Disease Dataset, Rabie El Kharoua, Kaggle (2024), CSV file.
- 2,149 records, 35 features covering demographics, lifestyle factors, medical history, clinical measurements, cognitive and functional assessments, and symptoms.
- Binary target `Diagnosis`: 0 = No Alzheimer (1,389; 64.6%), 1 = Alzheimer (760; 35.4%).
- **The dataset's publisher states it is synthetically generated for educational purposes and may not reflect real-world data.** The study therefore compares methods on a benchmark; it does not measure clinical performance.

## Preprocessing
1. Remove non-informative or redundant columns. *Not recorded: which columns were removed.*
2. Feature scaling: `StandardScaler` (zero mean, unit variance) for features with a Gaussian-like distribution; Min-Max scaling (0 to 1) for selected other features. *Not recorded: which features received which scaler.* The report describes scaling before the 80:20 split, so scalers may have been fitted on the full data.
3. 80:20 train/test split. Test set = 430 records (277 No Alzheimer, 153 Alzheimer; class proportions match the full data). *Not recorded: whether the split was stratified and which random seed was used.*

## Models
Both models are trained independently on the same training data (no combination of predictions).
- **AdaBoost** (Freund & Schapire, 1997): sequentially trained weak learners, each focusing on samples misclassified earlier. Final prediction F(x) = Σ α_t h_t(x), with α_t = ½ ln((1 − ε_t)/ε_t).
- **Gradient Boosting** (Friedman, 2001): additive model built stage-wise by fitting each weak learner to the residual errors of the previous model, F_m(x) = F_{m−1}(x) + γ_m h_m(x).
- *Not recorded: software library, version, and hyperparameters (number of estimators, learning rate, tree depth).*

## Evaluation
- Hold-out evaluation on the 430 test records only.
- Metrics: accuracy, per-class precision, recall, F1, macro and weighted averages, and confusion matrices. Alzheimer = positive class.
- Added in this paper from the reported confusion matrices: specificity, balanced accuracy, Matthews correlation coefficient, majority-class baseline accuracy, and 95% Wilson confidence intervals for accuracy, sensitivity and specificity.

## Reproducibility
- `make_figures.py` rebuilds every table and figure in the paper from the reported test-set counts (no raw data required) and checks that they agree with the report's Tables I and II.
- The original training notebook was not available, so the model-training step cannot be re-run from this repository; the reported test-set results are taken from the project report.
