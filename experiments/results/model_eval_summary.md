# Task 3: model evaluation rigor

## Class balance
| split | n | pos_rate |
|---|---|---|
| train | 30162 | 0.2489 |
| test | 15060 | 0.2457 |
Majority-class accuracy on test = 0.7543

## Bootstrap 95% CI (1000 resamples, test n=15060)
| model | metric | point | ci_lo | ci_hi | ci_width |
|---|---|---|---|---|---|
| gbm | accuracy | 0.8667 | 0.8610 | 0.8718 | 0.0109 |
| gbm | f1 | 0.6992 | 0.6865 | 0.7110 | 0.0245 |
| gbm | roc_auc | 0.9212 | 0.9164 | 0.9257 | 0.0093 |
| gbm | pr_auc | 0.8195 | 0.8094 | 0.8291 | 0.0197 |
| gbm | brier | 0.0930 | 0.0900 | 0.0960 | 0.0060 |
| lr_onehot | accuracy | 0.8418 | 0.8359 | 0.8475 | 0.0116 |
| lr_onehot | f1 | 0.6475 | 0.6345 | 0.6604 | 0.0259 |
| lr_onehot | roc_auc | 0.8968 | 0.8915 | 0.9018 | 0.0103 |
| lr_onehot | pr_auc | 0.7407 | 0.7274 | 0.7539 | 0.0265 |
| lr_onehot | brier | 0.1089 | 0.1060 | 0.1120 | 0.0060 |
| lr_ordinal | accuracy | 0.8135 | 0.8077 | 0.8193 | 0.0116 |
| lr_ordinal | f1 | 0.5442 | 0.5300 | 0.5592 | 0.0292 |
| lr_ordinal | roc_auc | 0.8416 | 0.8346 | 0.8479 | 0.0133 |
| lr_ordinal | pr_auc | 0.6508 | 0.6347 | 0.6664 | 0.0317 |
| lr_ordinal | brier | 0.1301 | 0.1270 | 0.1332 | 0.0062 |
| majority_prior | accuracy | 0.7543 | 0.7476 | 0.7609 | 0.0133 |
| majority_prior | f1 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| majority_prior | roc_auc | 0.5000 | 0.5000 | 0.5000 | 0.0000 |
| majority_prior | pr_auc | 0.2457 | 0.2391 | 0.2524 | 0.0133 |
| majority_prior | brier | 0.1853 | 0.1820 | 0.1887 | 0.0067 |
(CI = test-set sampling noise only. Training randomness is in section 5.)

## Paired bootstrap: GBM minus baseline
| gbm_minus | metric | diff | ci_lo | ci_hi | excludes_0 |
|---|---|---|---|---|---|
| lr_onehot | accuracy | 0.0248 | 0.0209 | 0.0288 | yes |
| lr_onehot | roc_auc | 0.0243 | 0.0219 | 0.0267 | yes |
| lr_onehot | pr_auc | 0.0788 | 0.0703 | 0.0866 | yes |
| lr_ordinal | accuracy | 0.0531 | 0.0475 | 0.0582 | yes |
| lr_ordinal | roc_auc | 0.0796 | 0.0745 | 0.0852 | yes |
| lr_ordinal | pr_auc | 0.1686 | 0.1571 | 0.1797 | yes |
| majority_prior | accuracy | 0.1122 | 0.1056 | 0.1189 | yes |
| majority_prior | roc_auc | 0.4211 | 0.4164 | 0.4257 | yes |
| majority_prior | pr_auc | 0.5737 | 0.5639 | 0.5836 | yes |

## Calibration (GBM)
Brier=0.0930  prior-only Brier=0.1853  Brier skill score=0.4983  weighted ECE=0.0080

## Per-group metrics (GBM, threshold 0.5)
sex and race are INPUT features of this model, so these gaps are not 'fairness through unawareness'. Group base rates also differ.
| attr | group | n | pos_rate | selection_rate | accuracy | acc_ci_lo | acc_ci_hi | tpr | fpr | roc_auc | small_n_flag |
|---|---|---|---|---|---|---|---|---|---|---|---|
| sex | Female | 4913 | 0.1134 | 0.0810 | 0.9347 | 0.9282 | 0.9414 | 0.5691 | 0.0186 | 0.9391 |  |
| sex | Male | 10147 | 0.3097 | 0.2541 | 0.8337 | 0.8271 | 0.8405 | 0.6417 | 0.0801 | 0.9023 |  |
| race | Amer-Indian-Eskimo | 149 | 0.1275 | 0.0805 | 0.9396 | 0.8993 | 0.9732 | 0.5789 | 0.0077 | 0.9644 |  |
| race | Asian-Pac-Islander | 408 | 0.2966 | 0.2696 | 0.8407 | 0.8039 | 0.8750 | 0.6860 | 0.0941 | 0.8982 |  |
| race | Black | 1411 | 0.1191 | 0.0843 | 0.9256 | 0.9117 | 0.9383 | 0.5417 | 0.0225 | 0.9438 |  |
| race | Other | 122 | 0.1967 | 0.1148 | 0.9016 | 0.8443 | 0.9508 | 0.5417 | 0.0102 | 0.9749 |  |
| race | White | 12970 | 0.2597 | 0.2098 | 0.8599 | 0.8534 | 0.8659 | 0.6342 | 0.0609 | 0.9173 |  |
sex (groups n>=100): TPR gap=0.073  selection-rate ratio min/max=0.319  accuracy gap=0.101
race (groups n>=100): TPR gap=0.144  selection-rate ratio min/max=0.299  accuracy gap=0.099

## Promotion gates vs noise
| gate | threshold | gbm_point | gbm_ci_lo | majority_baseline | majority_passes |
|---|---|---|---|---|---|
| accuracy>=0.75 | 0.7500 | 0.8667 | 0.8610 | 0.7543 | YES |
| f1>=0.70 | 0.7000 | 0.6992 | 0.6865 | 0.0000 | no |
| roc_auc>=0.75 | 0.7500 | 0.9212 | 0.9164 | 0.5000 | no |
Note: evaluate.py also has an F1>=0.70 gate (not in your task list). If majority_passes=YES, that gate cannot reject a model that always predicts <=50K.

Delta gate = 0.001 ROC-AUC. Models differ ONLY by random_state.
seed AUCs: [0.92116, 0.92124, 0.92152, 0.92198, 0.92201, 0.92112, 0.92173, 0.92154]
seed-to-seed AUC SD = 0.00035  range = 0.00088
SD of pairwise delta = 0.00050  paired test-set noise SD (bootstrap) = 0.00021
ordered pairs = 56
| gate | false_promotion_rate |
|---|---|
| current: delta >= 0.001 | 0.0000 |
| paired bootstrap: CI lower bound > 0 | 0.2143 |
| bootstrap CI > 0 AND delta >= 0.001 | 0.0000 |
False promotion = an equivalent model (different seed only) replacing the incumbent. The paired bootstrap removes TEST-set noise but not TRAINING noise, so it can still promote seed luck; compare the two numbers.

## Test size vs CI width
| n_test | acc_ci_width | auc_ci_width | acc_width_analytic |
|---|---|---|---|
| 500.0000 | 0.0591 | 0.0468 | 0.0596 |
| 1000.0000 | 0.0420 | 0.0363 | 0.0421 |
| 2500.0000 | 0.0256 | 0.0236 | 0.0267 |
| 5000.0000 | 0.0179 | 0.0153 | 0.0188 |
| 10000.0000 | 0.0131 | 0.0117 | 0.0133 |
| 15060.0000 | 0.0113 | 0.0089 | 0.0109 |
Analytic: n needed for accuracy CI half-width +/-0.5 pp = ~17756.

DATASET / TEST SIZE JUSTIFICATION (numbers measured by this script)
- UCI Adult Income after cleaning: 45222 rows (30162 train / 15060 test = 33% test).
- Positive rate train=0.2489, test=0.2457: imbalanced, so accuracy alone is misleading (majority baseline 0.7543); PR-AUC and ROC-AUC reported alongside.
- Test n=15060 gives accuracy 95% CI width 0.0113 and ROC-AUC CI width 0.0089.
- Smaller tests widen the CI: n=500 gives accuracy width 0.0591.
- Limits: one fixed split (no CV), tabular census extract, no timestamps (so no real temporal drift), test set reused for gate decisions and reporting, synthetic drift only.