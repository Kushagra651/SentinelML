# Task 2: detector evaluation
reference=30162 rows (train)  no-drift pool=15060 rows (test)  windows/cell=200  drifted reps/cell=50

## 1. FPR, independent no-drift windows (lower is better)
| window | raw_p05 | bonferroni | bh_fdr | ks_and_psi_crit | psi_warn | bonf_and_psi01 | bh_and_psi01 |
|---|---|---|---|---|---|---|---|
| 30.000 | 0.540 | 0.280 | 0.280 | 0.520 | 1.000 | 0.280 | 0.280 |
| 100.000 | 0.620 | 0.205 | 0.210 | 0.355 | 1.000 | 0.140 | 0.150 |
| 500.000 | 0.480 | 0.115 | 0.115 | 0.105 | 1.000 | 0.050 | 0.050 |
| 1000.000 | 0.440 | 0.065 | 0.070 | 0.010 | 0.870 | 0.015 | 0.015 |

Same raw rule when windows are sampled from the reference itself:
| window | raw_rule_FPR_sampled_from_ref |
|---|---|
| 30.000 | 0.520 |
| 100.000 | 0.515 |
| 500.000 | 0.415 |
| 1000.000 | 0.385 |
(Lower than the independent number = window overlaps the reference, which hides false alarms.)

## Features that fire most on no-drift windows (raw rule)
| source | window | feature | raw_fire_rate |
|---|---|---|---|
| independent | 30 | native_country | 0.215 |
| independent | 30 | education | 0.135 |
| independent | 30 | race | 0.090 |
| independent | 100 | native_country | 0.170 |
| independent | 100 | marital_status | 0.115 |
| independent | 100 | occupation | 0.105 |
| independent | 500 | native_country | 0.110 |
| independent | 500 | marital_status | 0.085 |
| independent | 500 | occupation | 0.085 |
| independent | 1000 | occupation | 0.105 |
| independent | 1000 | relationship | 0.065 |
| independent | 1000 | native_country | 0.065 |

## 2. Detection rate by intensity (rows) and rule (cols), per window

window = 30  (level 0.0 = no drift = FPR on test pool)
| level | raw_p05 | bonferroni | bh_fdr | ks_and_psi_crit | psi_warn | bonf_and_psi01 | bh_and_psi01 |
|---|---|---|---|---|---|---|---|
| 0.000 | 0.700 | 0.440 | 0.440 | 0.660 | 1.000 | 0.440 | 0.440 |
| 0.100 | 0.700 | 0.440 | 0.440 | 0.660 | 1.000 | 0.440 | 0.440 |
| 0.200 | 0.800 | 0.440 | 0.440 | 0.700 | 1.000 | 0.440 | 0.440 |
| 0.300 | 1.000 | 0.720 | 0.720 | 0.720 | 1.000 | 0.480 | 0.500 |
| 0.450 | 1.000 | 1.000 | 1.000 | 0.920 | 1.000 | 0.560 | 0.720 |
| 0.600 | 1.000 | 1.000 | 1.000 | 0.980 | 1.000 | 0.900 | 0.940 |

window = 100  (level 0.0 = no drift = FPR on test pool)
| level | raw_p05 | bonferroni | bh_fdr | ks_and_psi_crit | psi_warn | bonf_and_psi01 | bh_and_psi01 |
|---|---|---|---|---|---|---|---|
| 0.000 | 0.580 | 0.240 | 0.260 | 0.360 | 1.000 | 0.240 | 0.260 |
| 0.100 | 0.620 | 0.240 | 0.260 | 0.380 | 1.000 | 0.240 | 0.260 |
| 0.200 | 1.000 | 0.900 | 0.940 | 0.400 | 1.000 | 0.260 | 0.380 |
| 0.300 | 1.000 | 1.000 | 1.000 | 0.580 | 1.000 | 0.440 | 0.580 |
| 0.450 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| 0.600 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |

window = 500  (level 0.0 = no drift = FPR on test pool)
| level | raw_p05 | bonferroni | bh_fdr | ks_and_psi_crit | psi_warn | bonf_and_psi01 | bh_and_psi01 |
|---|---|---|---|---|---|---|---|
| 0.000 | 0.560 | 0.080 | 0.080 | 0.100 | 1.000 | 0.020 | 0.020 |
| 0.100 | 1.000 | 0.800 | 0.800 | 0.100 | 1.000 | 0.020 | 0.060 |
| 0.200 | 1.000 | 1.000 | 1.000 | 0.100 | 1.000 | 0.020 | 0.100 |
| 0.300 | 1.000 | 1.000 | 1.000 | 0.120 | 1.000 | 1.000 | 1.000 |
| 0.450 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| 0.600 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |

window = 1000  (level 0.0 = no drift = FPR on test pool)
| level | raw_p05 | bonferroni | bh_fdr | ks_and_psi_crit | psi_warn | bonf_and_psi01 | bh_and_psi01 |
|---|---|---|---|---|---|---|---|
| 0.000 | 0.420 | 0.060 | 0.080 | 0.020 | 0.920 | 0.000 | 0.000 |
| 0.100 | 1.000 | 1.000 | 1.000 | 0.020 | 0.920 | 0.000 | 0.000 |
| 0.200 | 1.000 | 1.000 | 1.000 | 0.020 | 0.920 | 0.000 | 0.000 |
| 0.300 | 1.000 | 1.000 | 1.000 | 0.020 | 1.000 | 1.000 | 1.000 |
| 0.450 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| 0.600 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |

## 3. Detection delay (samples of drifted stream until the rule fires)
| level | rule | detected_pct | median_delay_samples |
|---|---|---|---|
| 0.100 | raw_p05 | 100.000 | 40.000 |
| 0.100 | bonferroni | 100.000 | 300.000 |
| 0.100 | bh_fdr | 100.000 | 300.000 |
| 0.100 | ks_and_psi_crit | 85.000 | 30.000 |
| 0.100 | psi_warn | 100.000 | 30.000 |
| 0.100 | bonf_and_psi01 | 45.000 | 30.000 |
| 0.100 | bh_and_psi01 | 45.000 | 30.000 |
| 0.200 | raw_p05 | 100.000 | 30.000 |
| 0.200 | bonferroni | 100.000 | 50.000 |
| 0.200 | bh_fdr | 100.000 | 50.000 |
| 0.200 | ks_and_psi_crit | 95.000 | 30.000 |
| 0.200 | psi_warn | 100.000 | 30.000 |
| 0.200 | bonf_and_psi01 | 55.000 | 30.000 |
| 0.200 | bh_and_psi01 | 70.000 | 50.000 |
| 0.300 | raw_p05 | 100.000 | 30.000 |
| 0.300 | bonferroni | 100.000 | 30.000 |
| 0.300 | bh_fdr | 100.000 | 30.000 |
| 0.300 | ks_and_psi_crit | 100.000 | 40.000 |
| 0.300 | psi_warn | 100.000 | 30.000 |
| 0.300 | bonf_and_psi01 | 100.000 | 100.000 |
| 0.300 | bh_and_psi01 | 100.000 | 100.000 |
| 0.450 | raw_p05 | 100.000 | 30.000 |
| 0.450 | bonferroni | 100.000 | 30.000 |
| 0.450 | bh_fdr | 100.000 | 30.000 |
| 0.450 | ks_and_psi_crit | 100.000 | 30.000 |
| 0.450 | psi_warn | 100.000 | 30.000 |
| 0.450 | bonf_and_psi01 | 100.000 | 30.000 |
| 0.450 | bh_and_psi01 | 100.000 | 30.000 |
| 0.600 | raw_p05 | 100.000 | 30.000 |
| 0.600 | bonferroni | 100.000 | 30.000 |
| 0.600 | bh_fdr | 100.000 | 30.000 |
| 0.600 | ks_and_psi_crit | 100.000 | 30.000 |
| 0.600 | psi_warn | 100.000 | 30.000 |
| 0.600 | bonf_and_psi01 | 100.000 | 30.000 |
| 0.600 | bh_and_psi01 | 100.000 | 30.000 |
(median_delay_samples is over runs that fired within 2000 samples; detected_pct is the share that fired at all.)

## 4. Recommendation from the numbers
Criteria: FPR <= 0.05 AND detection at intensity 0.3 >= 0.80.
Intensity 0.1 was harmless in Task 1 (-0.18 pp accuracy, retrain gained nothing), so detection at 0.1 is a wasted-retrain cost; lower is better.
| window | rule | fpr | detect_0.3 | detect_0.1_(harmless) |
|---|---|---|---|---|
| 500 | bonf_and_psi01 | 0.050 | 1.000 | 0.020 |
| 500 | bh_and_psi01 | 0.050 | 1.000 | 0.060 |
| 1000 | bonf_and_psi01 | 0.015 | 1.000 | 0.000 |
| 1000 | bh_and_psi01 | 0.015 | 1.000 | 0.000 |

Pick: window=500, rule=Bonferroni AND PSI>=0.1.

## 5. Is a 6-hour interval justified?
From code: drift_check = '0 */6 * * *', window 6h, min rows 30; retrain_trigger = '30 */6 * * *' (30 min later), cooldown 12h, max 2/day.
Worst-case time from drift onset to retrain START = 6h (wait for next check) + 0.5h (trigger offset) = 6.5h, plus training time.
Requests per hour needed for a 6h window to reach each size:
| window | req_per_hour_needed |
|---|---|
| 30.0 | 5.0 |
| 100.0 | 16.7 |
| 500.0 | 83.3 |
| 1000.0 | 166.7 |
Traffic is UNKNOWN to me. Re-run with --traffic_per_hour <measured> (count of rows in ml.prediction_logs per hour) to finish this section.