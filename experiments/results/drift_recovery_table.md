Drift -> Degradation -> Recovery (mean +- std over 3 drift seeds)

| severity | stage | accuracy | ROC-AUC | PR-AUC |
|---|---|---|---|---|
| mild | A_old_clean | 0.8667 +- 0.0000 | 0.9212 +- 0.0000 | 0.8195 +- 0.0000 |
| mild | B_old_drifted | 0.8649 +- 0.0005 | 0.9193 +- 0.0001 | 0.8135 +- 0.0006 |
| mild | C_retrained_drifted | 0.8644 +- 0.0005 | 0.9190 +- 0.0001 | 0.8135 +- 0.0007 |
| mild | D_retrained_clean | 0.8659 +- 0.0004 | 0.9203 +- 0.0002 | 0.8177 +- 0.0006 |
| moderate | A_old_clean | 0.8667 +- 0.0000 | 0.9212 +- 0.0000 | 0.8195 +- 0.0000 |
| moderate | B_old_drifted | 0.8546 +- 0.0001 | 0.9096 +- 0.0002 | 0.7845 +- 0.0024 |
| moderate | C_retrained_drifted | 0.8584 +- 0.0001 | 0.9131 +- 0.0006 | 0.7976 +- 0.0006 |
| moderate | D_retrained_clean | 0.8610 +- 0.0008 | 0.9158 +- 0.0005 | 0.8076 +- 0.0010 |
| severe | A_old_clean | 0.8667 +- 0.0000 | 0.9212 +- 0.0000 | 0.8195 +- 0.0000 |
| severe | B_old_drifted | 0.7996 +- 0.0027 | 0.8625 +- 0.0018 | 0.6592 +- 0.0048 |
| severe | C_retrained_drifted | 0.8535 +- 0.0019 | 0.9077 +- 0.0000 | 0.7849 +- 0.0009 |
| severe | D_retrained_clean | 0.8493 +- 0.0013 | 0.9081 +- 0.0007 | 0.7843 +- 0.0008 |