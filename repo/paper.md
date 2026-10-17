# Alzheimer's Disease Detection Using Advanced Machine Learning Techniques

**Muhaimin Bin Habib, Ekram Khan, Md Sayeed Anwar**
Department of Computer Science and Engineering, East Delta University, Chattogram, Bangladesh

## Abstract

Early identification of Alzheimer's disease (AD) is difficult because conventional work-ups are costly and not widely accessible. We studied whether two boosting classifiers, AdaBoost and Gradient Boosting, can classify AD from structured demographic, lifestyle, medical-history, clinical and cognitive variables. We used the public Kaggle Alzheimer's Disease Dataset (2,149 records, 35 features, 35.4% positive), which its publisher describes as synthetically generated. After preprocessing, the data were split 80:20, both models were trained on the same training set, and each was evaluated once on 430 held-out records using accuracy, precision, recall and F1-score. Gradient Boosting outperformed AdaBoost (accuracy 95.8% vs. 91.2%; recall for Alzheimer cases 92.2% vs. 85.6%), making 18 errors instead of 38. Both exceeded the 64.4% majority-class baseline. Because the data are synthetic, evaluation used a single split, and no tuning is reported, the results demonstrate a method comparison rather than clinical performance.

## 1. Introduction

Alzheimer's disease is the most common cause of dementia and a growing public-health burden as populations age (Alzheimer's Association, 2023). Diagnosis is hard in the early stages, when symptoms are subtle and easily mistaken for normal ageing. Established work-ups rely on specialist assessment and on imaging or biomarker measurements, which are expensive and unevenly available. Large research programmes such as the Alzheimer's Disease Neuroimaging Initiative were created precisely to find imaging and biomarker measures that allow earlier and more reliable diagnosis (Mueller et al., 2005). At the same time, the pipeline of candidate treatments is large and active (Cummings et al., 2022), which raises the value of identifying people at risk as early as possible.

Machine learning offers a complementary route. Surveys of the field describe a wide range of approaches, including support vector machines, neural networks, deep learning and ensemble methods (Tanveer et al., 2020). A representative study built a learning framework on MRI-derived measures, together with age and cognitive scores, to predict which patients with mild cognitive impairment would convert to Alzheimer's disease (Moradi et al., 2015). Much less attention has gone to inexpensive tabular records that combine lifestyle, medical history, symptoms and brief cognitive tests, which could in principle support low-cost screening where scanners are unavailable.

Boosting is a natural family of methods for such mixed tabular data. AdaBoost combines weak learners by repeatedly re-weighting the training examples that earlier learners got wrong (Freund & Schapire, 1997). Gradient boosting instead builds an additive model stage by stage, with each new learner fitted to reduce the remaining loss of the current model (Friedman, 2001). Both are widely used, but they handle errors differently, and for a screening task the type of error matters: a missed Alzheimer case is usually costlier than a false alarm.

**Research question.** On a structured tabular Alzheimer's dataset, how do AdaBoost and Gradient Boosting compare in overall accuracy and in the balance of false negatives and false positives?

We hypothesized that Gradient Boosting would achieve higher accuracy and fewer missed Alzheimer cases than AdaBoost. We test this on the public Kaggle Alzheimer's Disease Dataset (El Kharoua, 2024). Importantly, the dataset's publisher describes it as synthetically generated for educational purposes, so our contribution is a controlled comparison of methods on a benchmark, not evidence of diagnostic accuracy in patients.

## 2. Methodology

The design is documented in `methodology.md`; we summarize it here.

**Data.** The dataset (El Kharoua, 2024) contains 2,149 records with 35 features spanning demographics, lifestyle factors, medical history, clinical measurements, cognitive and functional assessments, and symptoms. The binary target is diagnosis: 1,389 records (64.6%) are labelled No Alzheimer and 760 (35.4%) Alzheimer (Figure 1). We treat this as a binary classification task with Alzheimer as the positive class.

**Preprocessing.** Non-informative or redundant columns were removed to reduce dimensionality and noise (the specific columns removed were not recorded). Numerical features were scaled with `StandardScaler` (zero mean, unit variance) or, for selected features, Min-Max scaling to the range 0 to 1 (the assignment of scalers to features was not recorded). The data were then split 80:20 into training and test sets; the test set has 430 records (277 No Alzheimer, 153 Alzheimer; these class proportions closely match the full data), but whether the split was stratified and which random seed was used were not recorded.

**Models.** Two classifiers were trained independently on the same training data, without combining their predictions. *AdaBoost* forms a weighted vote of weak learners, F(x) = Σ α_t h_t(x), where the weight α_t = ½ ln((1 − ε_t)/ε_t) grows as the weak learner's weighted error ε_t falls. *Gradient Boosting* updates an additive model, F_m(x) = F_{m−1}(x) + γ_m h_m(x), by fitting each new learner h_m to the residual errors of the previous stage. The software library, its version and the models' hyperparameters were not recorded.

**Evaluation.** Each model was evaluated once on the held-out test set using accuracy, per-class precision, recall and F1-score, and confusion matrices. From the reported confusion matrices we additionally derived specificity, balanced accuracy, the Matthews correlation coefficient (MCC), the majority-class baseline, and 95% Wilson confidence intervals for accuracy, sensitivity and specificity. All tables and figures here are regenerated by `make_figures.py`, which also checks that they match the originally reported classification tables.

## 3. Results

### 3.1 Data and baseline

The classes are moderately imbalanced (Figure 1). A classifier that always predicts No Alzheimer would reach 64.4% accuracy on the test set (277 of 430) and would detect none of the 153 Alzheimer cases, so accuracy alone must be read against this baseline.

![Figure 1](figures/fig1_class_distribution.png)

*Figure 1. Class distribution of the 2,149 records in the Alzheimer's Disease Dataset (0 = No Alzheimer, 1 = Alzheimer).*

### 3.2 Classification performance

Table 1 gives the per-class classification reports. AdaBoost reached an accuracy of 0.91. Its precision and recall were 0.92 and 0.94 for No Alzheimer, and 0.89 and 0.86 for Alzheimer. Gradient Boosting reached an accuracy of 0.96, with precision and recall of 0.96 and 0.98 for No Alzheimer and 0.96 and 0.92 for Alzheimer. Both models were therefore stronger on the majority class, and both found it harder to recall Alzheimer cases than to recognize non-cases.

**Table 1. Classification reports on the test set (n = 430).**

| Model | Class | Precision | Recall | F1-score | Support |
|---|---|---|---|---|---|
| AdaBoost | No Alzheimer | 0.92 | 0.94 | 0.93 | 277 |
| AdaBoost | Alzheimer | 0.89 | 0.86 | 0.87 | 153 |
| AdaBoost | Accuracy | | | 0.91 | 430 |
| AdaBoost | Macro avg | 0.91 | 0.90 | 0.90 | 430 |
| Gradient Boosting | No Alzheimer | 0.96 | 0.98 | 0.97 | 277 |
| Gradient Boosting | Alzheimer | 0.96 | 0.92 | 0.94 | 153 |
| Gradient Boosting | Accuracy | | | 0.96 | 430 |
| Gradient Boosting | Macro avg | 0.96 | 0.95 | 0.95 | 430 |

![Figure 2](figures/fig2_confusion_matrices.png)

*Figure 2. Confusion matrices of AdaBoost (left) and Gradient Boosting (right) on the 430 test records. Cell percentages are of each true class.*

### 3.3 Error analysis and comparison

Figure 2 shows the error counts. AdaBoost produced 16 false positives and 22 false negatives (38 errors); Gradient Boosting produced 6 false positives and 12 false negatives (18 errors). Gradient Boosting therefore made 53% fewer errors overall and missed 12 of 153 Alzheimer cases (7.8%) compared with 22 of 153 (14.4%) for AdaBoost. Figure 3 and Table 2 summarize the comparison; Gradient Boosting scored higher on every metric, with an MCC of 0.908 versus 0.806.

![Figure 3](figures/fig3_metric_comparison.png)

*Figure 3. Test-set metrics for both models. The dashed line marks the majority-class accuracy (0.64); the vertical axis starts at 0.60.*

**Table 2. Derived test-set metrics with 95% Wilson confidence intervals (n = 430).**

| Model | Accuracy | Sensitivity (Alzheimer recall) | Specificity | FP | FN | Total errors | MCC |
|---|---|---|---|---|---|---|---|
| Majority baseline | 0.644 | 0.000 | 1.000 | 0 | 153 | 153 | 0.000 |
| AdaBoost | 0.912 (0.881–0.935) | 0.856 (0.792–0.903) | 0.942 (0.908–0.964) | 16 | 22 | 38 | 0.806 |
| Gradient Boosting | 0.958 (0.935–0.973) | 0.922 (0.868–0.955) | 0.978 (0.954–0.990) | 6 | 12 | 18 | 0.908 |

The accuracy intervals of the two models just touch (0.935 for AdaBoost's upper bound and for Gradient Boosting's lower bound), while the sensitivity intervals overlap more widely (0.868–0.903). Because both models were scored on the same test records, a paired test on their predictions would be more sensitive than this unpaired comparison, but the per-record predictions were not retained.

## 4. Discussion

**Gradient Boosting was the stronger model, supporting our hypothesis.** It improved accuracy by about 4.6 percentage points, cut false negatives from 22 to 12, and cut false positives from 16 to 6. Its advantage was largest on the clinically important measure: the fraction of Alzheimer cases it detected rose from 85.6% to 92.2%. A plausible explanation is that stage-wise fitting to residuals can model smoother and more complex interactions among the 35 features than AdaBoost's repeated re-weighting of examples, but this study did not test that explanation (for example with varied tree depths or feature-importance analysis), so it remains a hypothesis.

**Missed cases remain the weak point.** Even the better model failed to flag about 1 in 13 Alzheimer cases. The original report described AdaBoost's false-negative count as relatively low, but 22 of 153 (14.4%) missed cases would be substantial for a screening tool. Moving the decision threshold toward higher sensitivity would trade additional false positives for fewer missed cases, and the appropriate balance is a clinical decision; we did not explore it.

**Interpreting the numbers.** Accuracies above 90% are strong against a 64.4% baseline, but they should not be read as evidence that boosting can diagnose Alzheimer's disease from routine records. The dataset is described by its publisher as synthetic, and synthetic data can contain cleaner, more learnable relationships between features and labels than real clinical records do. Reported performance may therefore be optimistic compared with what would be seen in patients. What the results do support is a relative conclusion: on this benchmark, Gradient Boosting separated the classes better than AdaBoost.

**Limitations.**
1. *Synthetic data.* The dataset is not real patient data, so external clinical validity is unknown. Validation on real cohorts such as ADNI (Mueller et al., 2005) would be needed.
2. *Single split, no tuning reported.* Results come from one 80:20 hold-out split. Without repeated or cross-validated runs we cannot quantify variation due to the split, and hyperparameters were not tuned or reported, so each model's best performance may differ.
3. *Small test set and unpaired comparison.* With 430 test records and overlapping confidence intervals for sensitivity, the size of the gap between the models is uncertain.
4. *Unrecorded settings.* The original training code was not available when this paper was prepared, so the removed columns, scaler assignment, split seed and stratification, and model hyperparameters are unknown, and the reported results cannot be reproduced exactly from this repository. The project report also lists scaling before the split; if scalers were fitted on the full data this is a minor leakage risk, although tree-based boosting is insensitive to monotone feature scaling, so it is unlikely to matter here. Fitting scalers on training data only (inside a pipeline) is the safer practice.
5. *Limited metrics and models.* We report threshold-dependent metrics only (no ROC-AUC, calibration or decision-curve analysis), and only two models. The report's introduction also mentioned other models and the importance of cognitive scores such as MMSE; these were not evaluated here, so we make no claim about which features drive performance.

## 5. Conclusion

On the Kaggle Alzheimer's Disease Dataset, Gradient Boosting classified Alzheimer's disease more accurately than AdaBoost (95.8% vs. 91.2% test accuracy), halved the number of errors (18 vs. 38) and missed fewer Alzheimer cases (7.8% vs. 14.4%), with both models well above the 64.4% majority-class baseline. Because the data are synthetic, the evaluation used a single hold-out split, and no tuning or feature analysis was reported, these results establish a relative comparison of two methods rather than a validated diagnostic tool. The natural next steps are repeated cross-validation with tuned hyperparameters, threshold and ROC analysis focused on sensitivity, feature-importance analysis to test which cognitive and functional measures matter, and validation on real clinical data.

## References

Alzheimer's Association. (2023). 2023 Alzheimer's disease facts and figures. *Alzheimer's & Dementia, 19*(4), 1598–1695. https://doi.org/10.1002/alz.13016

Cummings, J., Lee, G., Nahed, P., Kambar, M. E. Z. N., Zhong, K., Fonseca, J., & Taghva, K. (2022). Alzheimer's disease drug development pipeline: 2022. *Alzheimer's & Dementia: Translational Research & Clinical Interventions, 8*(1), e12295. https://doi.org/10.1002/trc2.12295

El Kharoua, R. (2024). *Alzheimer's disease dataset* [Data set]. Kaggle. https://doi.org/10.34740/KAGGLE/DSV/8668279

Freund, Y., & Schapire, R. E. (1997). A decision-theoretic generalization of on-line learning and an application to boosting. *Journal of Computer and System Sciences, 55*(1), 119–139. https://doi.org/10.1006/jcss.1997.1504

Friedman, J. H. (2001). Greedy function approximation: A gradient boosting machine. *The Annals of Statistics, 29*(5), 1189–1232. https://doi.org/10.1214/aos/1013203451

Moradi, E., Pepe, A., Gaser, C., Huttunen, H., & Tohka, J. (2015). Machine learning framework for early MRI-based Alzheimer's conversion prediction in MCI subjects. *NeuroImage, 104*, 398–412. https://doi.org/10.1016/j.neuroimage.2014.09.059

Mueller, S. G., Weiner, M. W., Thal, L. J., Petersen, R. C., Jack, C. R., Jagust, W., Trojanowski, J. Q., Toga, A. W., & Beckett, L. (2005). Ways toward an early diagnosis in Alzheimer's disease: The Alzheimer's Disease Neuroimaging Initiative (ADNI). *Alzheimer's & Dementia, 1*(1), 55–66. https://doi.org/10.1016/j.jalz.2005.06.003

Tanveer, M., Richhariya, B., Khan, R. U., Rashid, A. H., Khanna, P., Prasad, M., & Lin, C. T. (2020). Machine learning techniques for the diagnosis of Alzheimer's disease: A review. *ACM Transactions on Multimedia Computing, Communications, and Applications, 16*(1s), 1–35. https://doi.org/10.1145/3344998
