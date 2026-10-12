
### jev3sym_6zs79  (n 1200, top-1 0.6783, dev-card 360)
shim parity max |d| 0.00e+00

**jev3sym_6zs79 / native / all** FIT 376 LTT 406 TEST 418; types {'FIT': {'choice': 138, 'noul': 222, 'score': 16}, 'LTT': {'choice': 141, 'noul': 250, 'score': 15}, 'TEST': {'choice': 137, 'noul': 269, 'score': 12}}
| alpha | var | TEST cov | flag | size | singleton | sing acc | lambda* (Bonf) | TEST act | acted err | FS-CRR c1 lam / act / err | FS-binom lam / act / err | per-type LTT act / err |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.1 | M | 0.880 | ok | 1.56 | 0.462 | 0.798 | - | 0.000 | - | - / 0.000 / - | - / 0.000 / - | 0.000 / - |
| 0.1 | T | 0.888 | ok | 1.62 | 0.476 | 0.789 | - | 0.000 | - | - / 0.000 / - | - / 0.000 / - | 0.000 / - |
| 0.1 | F | 0.880 | ok | 1.56 | 0.462 | 0.798 | - | 0.000 | - | - / 0.000 / - | - / 0.000 / - | 0.000 / - |
| 0.05 | M | 0.938 | ok | 1.85 | 0.323 | 0.844 | - | 0.000 | - | - / 0.000 / - | - / 0.000 / - | 0.000 / - |
| 0.05 | T | 0.935 | ok | 1.87 | 0.323 | 0.837 | - | 0.000 | - | - / 0.000 / - | - / 0.000 / - | 0.000 / - |
| 0.05 | F | 0.938 | ok | 1.85 | 0.323 | 0.844 | - | 0.000 | - | - / 0.000 / - | - / 0.000 / - | 0.000 / - |

per type / family, variant T, alpha 0.1 (groups {"choice": {"n_fit": 138, "qhat": 0.8151734338447949}, "noul": {"n_fit": 222, "qhat": 0.7231186920513306}, "pooled-small": {"n_fit": 16, "qhat": Infinity}}; pooled-small ['score'] full-set True)
| cell | n | cov | size | singleton | sing acc | top-1 | act | acted err | per-type-LTT act / err |
|---|---|---|---|---|---|---|---|---|---|
| choice | 137 | 0.912 | 1.74 | 0.350 | 0.854 | 0.650 | 0.000 | - | 0.000 / - |
| noul | 269 | 0.870 | 1.44 | 0.561 | 0.768 | 0.714 | 0.000 | - | 0.000 / - |
| score | 12 | 1.000 | 4.42 | 0.000 | - | 0.250 | 0.000 | - | 0.000 / - |
| adequacy | 63 | 0.937 | 1.84 | 0.159 | 0.600 | 0.619 | 0.000 | - | 0.000 / - |
| ambiguous | 62 | 0.952 | 1.95 | 0.081 | 0.400 | 0.516 | 0.000 | - | 0.000 / - |
| judge_hard | 80 | 0.800 | 1.35 | 0.650 | 0.692 | 0.675 | 0.000 | - | 0.000 / - |
| long_policy | 70 | 0.857 | 1.99 | 0.329 | 0.739 | 0.629 | 0.000 | - | 0.000 / - |
| policy | 74 | 0.892 | 1.14 | 0.865 | 0.875 | 0.824 | 0.000 | - | 0.000 / - |
| temporal_numeric | 69 | 0.913 | 1.61 | 0.652 | 0.889 | 0.783 | 0.000 | - | 0.000 / - |
LTT-half attribution: best lambda by p {"lambda": 0.95, "n_act": 71, "errors": 3, "p": 0.5228675724621862}; Bonferroni level 9.90e-04; zero-error acts needed 135 (single test 45); LTT singleton rate 0.510; per-type lambdas {'choice': None, 'noul': None}; FS-CRR c1 stop {"lambda": 0.95, "reason": "not rejected", "n_act": 71, "errors": 3, "p_value": 0.4520452780522165, "log_capital": 0.793972931484959, "weighted_error": 0.04225352112676056}; FS-CRR c8 lam None stop {"lambda": 0.95, "reason": "not rejected", "n_act": 71, "errors": 3, "p_value": 0.8992845384329501, "log_capital": 0.10615578910065979, "weighted_error": 0.04225352112676056}

per type / family, variant T, alpha 0.05 (groups {"choice": {"n_fit": 138, "qhat": 0.8787924500868126}, "noul": {"n_fit": 222, "qhat": 0.8934878279109186}, "pooled-small": {"n_fit": 16, "qhat": Infinity}}; pooled-small ['score'] full-set True)
| cell | n | cov | size | singleton | sing acc | top-1 | act | acted err | per-type-LTT act / err |
|---|---|---|---|---|---|---|---|---|---|
| choice | 137 | 0.934 | 2.13 | 0.248 | 0.882 | 0.650 | 0.000 | - | 0.000 / - |
| noul | 269 | 0.933 | 1.62 | 0.375 | 0.822 | 0.714 | 0.000 | - | 0.000 / - |
| score | 12 | 1.000 | 4.42 | 0.000 | - | 0.250 | 0.000 | - | 0.000 / - |
| adequacy | 63 | 0.984 | 1.94 | 0.063 | 0.750 | 0.619 | 0.000 | - | 0.000 / - |
| ambiguous | 62 | 0.984 | 2.02 | 0.032 | 0.500 | 0.516 | 0.000 | - | 0.000 / - |
| judge_hard | 80 | 0.938 | 1.71 | 0.287 | 0.783 | 0.675 | 0.000 | - | 0.000 / - |
| long_policy | 70 | 0.886 | 2.69 | 0.143 | 0.600 | 0.629 | 0.000 | - | 0.000 / - |
| policy | 74 | 0.905 | 1.18 | 0.824 | 0.885 | 0.824 | 0.000 | - | 0.000 / - |
| temporal_numeric | 69 | 0.928 | 1.78 | 0.507 | 0.886 | 0.783 | 0.000 | - | 0.000 / - |
LTT-half attribution: best lambda by p {"lambda": 0.95, "n_act": 71, "errors": 3, "p": 0.5228675724621862}; Bonferroni level 9.90e-04; zero-error acts needed 135 (single test 45); LTT singleton rate 0.296; per-type lambdas {'choice': None, 'noul': None}; FS-CRR c1 stop {"lambda": 0.95, "reason": "not rejected", "n_act": 71, "errors": 3, "p_value": 0.4520452780522165, "log_capital": 0.793972931484959, "weighted_error": 0.04225352112676056}; FS-CRR c8 lam None stop {"lambda": 0.95, "reason": "not rejected", "n_act": 71, "errors": 3, "p_value": 0.8992845384329501, "log_capital": 0.10615578910065979, "weighted_error": 0.04225352112676056}

**jev3sym_6zs79 / native / remainder** FIT 267 LTT 272 TEST 301; types {'FIT': {'choice': 104, 'noul': 151, 'score': 12}, 'LTT': {'choice': 97, 'noul': 162, 'score': 13}, 'TEST': {'choice': 101, 'noul': 190, 'score': 10}}
| alpha | var | TEST cov | flag | size | singleton | sing acc | lambda* (Bonf) | TEST act | acted err | FS-CRR c1 lam / act / err | FS-binom lam / act / err | per-type LTT act / err |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.1 | M | 0.874 | ok | 1.57 | 0.455 | 0.781 | - | 0.000 | - | - / 0.000 / - | - / 0.000 / - | 0.000 / - |
| 0.1 | T | 0.887 | ok | 1.66 | 0.465 | 0.779 | - | 0.000 | - | - / 0.000 / - | - / 0.000 / - | 0.000 / - |
| 0.1 | F | 0.874 | ok | 1.57 | 0.455 | 0.781 | - | 0.000 | - | - / 0.000 / - | - / 0.000 / - | 0.000 / - |
| 0.05 | M | 0.940 | ok | 1.87 | 0.322 | 0.845 | - | 0.000 | - | - / 0.000 / - | - / 0.000 / - | 0.000 / - |
| 0.05 | T | 0.950 | ok | 1.92 | 0.306 | 0.859 | - | 0.000 | - | - / 0.000 / - | - / 0.000 / - | 0.000 / - |
| 0.05 | F | 0.940 | ok | 1.87 | 0.322 | 0.845 | - | 0.000 | - | - / 0.000 / - | - / 0.000 / - | 0.000 / - |

per type / family, variant T, alpha 0.1 (groups {"choice": {"n_fit": 104, "qhat": 0.8241762571547657}, "noul": {"n_fit": 151, "qhat": 0.7240567832802329}, "pooled-small": {"n_fit": 12, "qhat": Infinity}}; pooled-small ['score'] full-set True)
| cell | n | cov | size | singleton | sing acc | top-1 | act | acted err | per-type-LTT act / err |
|---|---|---|---|---|---|---|---|---|---|
| choice | 101 | 0.921 | 1.85 | 0.297 | 0.833 | 0.624 | 0.000 | - | 0.000 / - |
| noul | 190 | 0.863 | 1.42 | 0.579 | 0.764 | 0.721 | 0.000 | - | 0.000 / - |
| score | 10 | 1.000 | 4.30 | 0.000 | - | 0.300 | 0.000 | - | 0.000 / - |
| adequacy | 35 | 0.914 | 1.83 | 0.171 | 0.500 | 0.571 | 0.000 | - | 0.000 / - |
| ambiguous | 51 | 0.941 | 1.94 | 0.098 | 0.400 | 0.529 | 0.000 | - | 0.000 / - |
| judge_hard | 62 | 0.806 | 1.37 | 0.629 | 0.692 | 0.710 | 0.000 | - | 0.000 / - |
| long_policy | 58 | 0.862 | 2.14 | 0.293 | 0.706 | 0.586 | 0.000 | - | 0.000 / - |
| policy | 57 | 0.912 | 1.12 | 0.877 | 0.900 | 0.860 | 0.000 | - | 0.000 / - |
| temporal_numeric | 38 | 0.921 | 1.68 | 0.605 | 0.870 | 0.763 | 0.000 | - | 0.000 / - |
LTT-half attribution: best lambda by p {"lambda": 0.95, "n_act": 51, "errors": 3, "p": 0.7494142190385397}; Bonferroni level 9.90e-04; zero-error acts needed 135 (single test 45); LTT singleton rate 0.507; per-type lambdas {'noul': None}; FS-CRR c1 stop {"lambda": 0.95, "reason": "not rejected", "n_act": 51, "errors": 3, "p_value": 0.5011376707437201, "log_capital": 0.6908744237420195, "weighted_error": 0.058823529411764705}; FS-CRR c8 lam None stop {"lambda": 0.95, "reason": "not rejected", "n_act": 51, "errors": 3, "p_value": 0.915790615178272, "log_capital": 0.08796752648770137, "weighted_error": 0.058823529411764705}

per type / family, variant T, alpha 0.05 (groups {"choice": {"n_fit": 104, "qhat": 0.89386211588959}, "noul": {"n_fit": 151, "qhat": 0.8993324322561739}, "pooled-small": {"n_fit": 12, "qhat": Infinity}}; pooled-small ['score'] full-set True)
| cell | n | cov | size | singleton | sing acc | top-1 | act | acted err | per-type-LTT act / err |
|---|---|---|---|---|---|---|---|---|---|
| choice | 101 | 0.970 | 2.29 | 0.168 | 0.941 | 0.624 | 0.000 | - | 0.000 / - |
| noul | 190 | 0.937 | 1.61 | 0.395 | 0.840 | 0.721 | 0.000 | - | 0.000 / - |
| score | 10 | 1.000 | 4.30 | 0.000 | - | 0.300 | 0.000 | - | 0.000 / - |
| adequacy | 35 | 0.971 | 1.94 | 0.057 | 0.500 | 0.571 | 0.000 | - | 0.000 / - |
| ambiguous | 51 | 1.000 | 2.06 | 0.000 | - | 0.529 | 0.000 | - | 0.000 / - |
| judge_hard | 62 | 0.935 | 1.73 | 0.274 | 0.765 | 0.710 | 0.000 | - | 0.000 / - |
| long_policy | 58 | 0.931 | 2.78 | 0.121 | 0.714 | 0.586 | 0.000 | - | 0.000 / - |
| policy | 57 | 0.930 | 1.16 | 0.842 | 0.917 | 0.860 | 0.000 | - | 0.000 / - |
| temporal_numeric | 38 | 0.947 | 1.89 | 0.474 | 0.889 | 0.763 | 0.000 | - | 0.000 / - |
LTT-half attribution: best lambda by p {"lambda": 0.95, "n_act": 51, "errors": 3, "p": 0.7494142190385397}; Bonferroni level 9.90e-04; zero-error acts needed 135 (single test 45); LTT singleton rate 0.290; per-type lambdas {'noul': None}; FS-CRR c1 stop {"lambda": 0.95, "reason": "not rejected", "n_act": 51, "errors": 3, "p_value": 0.5011376707437201, "log_capital": 0.6908744237420195, "weighted_error": 0.058823529411764705}; FS-CRR c8 lam None stop {"lambda": 0.95, "reason": "not rejected", "n_act": 51, "errors": 3, "p_value": 0.915790615178272, "log_capital": 0.08796752648770137, "weighted_error": 0.058823529411764705}

**jev3sym_6zs79 / native_shimcal / all** FIT 376 LTT 406 TEST 418; types {'FIT': {'choice': 138, 'noul': 222, 'score': 16}, 'LTT': {'choice': 141, 'noul': 250, 'score': 15}, 'TEST': {'choice': 137, 'noul': 269, 'score': 12}}
| alpha | var | TEST cov | flag | size | singleton | sing acc | lambda* (Bonf) | TEST act | acted err | FS-CRR c1 lam / act / err | FS-binom lam / act / err | per-type LTT act / err |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.1 | M | 0.873 | ok | 1.56 | 0.467 | 0.790 | - | 0.000 | - | - / 0.000 / - | - / 0.000 / - | 0.000 / - |
| 0.1 | T | 0.895 | ok | 1.65 | 0.462 | 0.798 | - | 0.000 | - | - / 0.000 / - | - / 0.000 / - | 0.000 / - |
| 0.1 | F | 0.873 | ok | 1.56 | 0.467 | 0.790 | - | 0.000 | - | - / 0.000 / - | - / 0.000 / - | 0.000 / - |
| 0.05 | M | 0.926 | FLAG | 1.82 | 0.352 | 0.816 | - | 0.000 | - | - / 0.000 / - | - / 0.000 / - | 0.000 / - |
| 0.05 | T | 0.933 | ok | 1.85 | 0.335 | 0.829 | - | 0.000 | - | - / 0.000 / - | - / 0.000 / - | 0.000 / - |
| 0.05 | F | 0.926 | FLAG | 1.82 | 0.352 | 0.816 | - | 0.000 | - | - / 0.000 / - | - / 0.000 / - | 0.000 / - |

per type / family, variant T, alpha 0.1 (groups {"choice": {"n_fit": 138, "qhat": 0.8373822546510632}, "noul": {"n_fit": 222, "qhat": 0.801}, "pooled-small": {"n_fit": 16, "qhat": Infinity}}; pooled-small ['score'] full-set True)
| cell | n | cov | size | singleton | sing acc | top-1 | act | acted err | per-type-LTT act / err |
|---|---|---|---|---|---|---|---|---|---|
| choice | 137 | 0.912 | 1.77 | 0.372 | 0.863 | 0.650 | 0.000 | - | 0.000 / - |
| noul | 269 | 0.881 | 1.47 | 0.528 | 0.775 | 0.714 | 0.000 | - | 0.000 / - |
| score | 12 | 1.000 | 4.42 | 0.000 | - | 0.250 | 0.000 | - | 0.000 / - |
| adequacy | 63 | 0.952 | 1.86 | 0.143 | 0.667 | 0.619 | 0.000 | - | 0.000 / - |
| ambiguous | 62 | 0.952 | 1.94 | 0.097 | 0.500 | 0.516 | 0.000 | - | 0.000 / - |
| judge_hard | 80 | 0.812 | 1.41 | 0.588 | 0.681 | 0.675 | 0.000 | - | 0.000 / - |
| long_policy | 70 | 0.857 | 2.07 | 0.314 | 0.727 | 0.629 | 0.000 | - | 0.000 / - |
| policy | 74 | 0.892 | 1.14 | 0.865 | 0.875 | 0.824 | 0.000 | - | 0.000 / - |
| temporal_numeric | 69 | 0.928 | 1.62 | 0.652 | 0.911 | 0.783 | 0.000 | - | 0.000 / - |
LTT-half attribution: best lambda by p {"lambda": 0.98, "n_act": 69, "errors": 3, "p": 0.5449641994872204}; Bonferroni level 9.90e-04; zero-error acts needed 135 (single test 45); LTT singleton rate 0.483; per-type lambdas {'choice': None, 'noul': None}; FS-CRR c1 stop {"lambda": 0.95, "reason": "not rejected", "n_act": 96, "errors": 7, "p_value": 0.5277593462669029, "log_capital": 0.6391148827621972, "weighted_error": 0.07291666666666667}; FS-CRR c8 lam None stop {"lambda": 0.95, "reason": "not rejected", "n_act": 96, "errors": 7, "p_value": 0.9216922173863737, "log_capital": 0.08154393177820446, "weighted_error": 0.07291666666666667}

per type / family, variant T, alpha 0.05 (groups {"choice": {"n_fit": 138, "qhat": 0.8987141256104405}, "noul": {"n_fit": 222, "qhat": 0.934538570984002}, "pooled-small": {"n_fit": 16, "qhat": Infinity}}; pooled-small ['score'] full-set True)
| cell | n | cov | size | singleton | sing acc | top-1 | act | acted err | per-type-LTT act / err |
|---|---|---|---|---|---|---|---|---|---|
| choice | 137 | 0.927 | 2.07 | 0.285 | 0.846 | 0.650 | 0.000 | - | 0.000 / - |
| noul | 269 | 0.933 | 1.62 | 0.375 | 0.822 | 0.714 | 0.000 | - | 0.000 / - |
| score | 12 | 1.000 | 4.42 | 0.000 | - | 0.250 | 0.000 | - | 0.000 / - |
| adequacy | 63 | 0.984 | 1.94 | 0.063 | 0.750 | 0.619 | 0.000 | - | 0.000 / - |
| ambiguous | 62 | 0.952 | 1.97 | 0.081 | 0.400 | 0.516 | 0.000 | - | 0.000 / - |
| judge_hard | 80 | 0.938 | 1.71 | 0.287 | 0.783 | 0.675 | 0.000 | - | 0.000 / - |
| long_policy | 70 | 0.900 | 2.61 | 0.157 | 0.636 | 0.629 | 0.000 | - | 0.000 / - |
| policy | 74 | 0.905 | 1.18 | 0.824 | 0.885 | 0.824 | 0.000 | - | 0.000 / - |
| temporal_numeric | 69 | 0.928 | 1.77 | 0.522 | 0.889 | 0.783 | 0.000 | - | 0.000 / - |
LTT-half attribution: best lambda by p {"lambda": 0.98, "n_act": 69, "errors": 3, "p": 0.5449641994872204}; Bonferroni level 9.90e-04; zero-error acts needed 135 (single test 45); LTT singleton rate 0.303; per-type lambdas {'choice': None, 'noul': None}; FS-CRR c1 stop {"lambda": 0.95, "reason": "not rejected", "n_act": 96, "errors": 7, "p_value": 0.5277593462669029, "log_capital": 0.6391148827621972, "weighted_error": 0.07291666666666667}; FS-CRR c8 lam None stop {"lambda": 0.95, "reason": "not rejected", "n_act": 96, "errors": 7, "p_value": 0.9216922173863737, "log_capital": 0.08154393177820446, "weighted_error": 0.07291666666666667}

**jev3sym_6zs79 / native_shimcal / remainder** FIT 267 LTT 272 TEST 301; types {'FIT': {'choice': 104, 'noul': 151, 'score': 12}, 'LTT': {'choice': 97, 'noul': 162, 'score': 13}, 'TEST': {'choice': 101, 'noul': 190, 'score': 10}}
| alpha | var | TEST cov | flag | size | singleton | sing acc | lambda* (Bonf) | TEST act | acted err | FS-CRR c1 lam / act / err | FS-binom lam / act / err | per-type LTT act / err |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.1 | M | 0.874 | ok | 1.59 | 0.452 | 0.779 | - | 0.000 | - | - / 0.000 / - | - / 0.000 / - | 0.000 / - |
| 0.1 | T | 0.890 | ok | 1.69 | 0.449 | 0.778 | - | 0.000 | - | - / 0.000 / - | - / 0.000 / - | 0.000 / - |
| 0.1 | F | 0.874 | ok | 1.59 | 0.452 | 0.779 | - | 0.000 | - | - / 0.000 / - | - / 0.000 / - | 0.000 / - |
| 0.05 | M | 0.940 | ok | 1.92 | 0.322 | 0.835 | - | 0.000 | - | - / 0.000 / - | - / 0.000 / - | 0.000 / - |
| 0.05 | T | 0.953 | ok | 1.97 | 0.302 | 0.857 | - | 0.000 | - | - / 0.000 / - | - / 0.000 / - | 0.000 / - |
| 0.05 | F | 0.940 | ok | 1.92 | 0.322 | 0.835 | - | 0.000 | - | - / 0.000 / - | - / 0.000 / - | 0.000 / - |

per type / family, variant T, alpha 0.1 (groups {"choice": {"n_fit": 104, "qhat": 0.84922670296685}, "noul": {"n_fit": 151, "qhat": 0.801}, "pooled-small": {"n_fit": 12, "qhat": Infinity}}; pooled-small ['score'] full-set True)
| cell | n | cov | size | singleton | sing acc | top-1 | act | acted err | per-type-LTT act / err |
|---|---|---|---|---|---|---|---|---|---|
| choice | 101 | 0.921 | 1.90 | 0.287 | 0.828 | 0.624 | 0.000 | - | 0.000 / - |
| noul | 190 | 0.868 | 1.44 | 0.558 | 0.764 | 0.721 | 0.000 | - | 0.000 / - |
| score | 10 | 1.000 | 4.30 | 0.000 | - | 0.300 | 0.000 | - | 0.000 / - |
| adequacy | 35 | 0.914 | 1.83 | 0.171 | 0.500 | 0.571 | 0.000 | - | 0.000 / - |
| ambiguous | 51 | 0.941 | 1.94 | 0.098 | 0.400 | 0.529 | 0.000 | - | 0.000 / - |
| judge_hard | 62 | 0.823 | 1.42 | 0.581 | 0.694 | 0.710 | 0.000 | - | 0.000 / - |
| long_policy | 58 | 0.862 | 2.22 | 0.259 | 0.667 | 0.586 | 0.000 | - | 0.000 / - |
| policy | 57 | 0.912 | 1.12 | 0.877 | 0.900 | 0.860 | 0.000 | - | 0.000 / - |
| temporal_numeric | 38 | 0.921 | 1.71 | 0.605 | 0.870 | 0.763 | 0.000 | - | 0.000 / - |
LTT-half attribution: best lambda by p {"lambda": 0.98, "n_act": 49, "errors": 3, "p": 0.77128597926257}; Bonferroni level 9.90e-04; zero-error acts needed 135 (single test 45); LTT singleton rate 0.482; per-type lambdas {'noul': None}; FS-CRR c1 stop {"lambda": 0.95, "reason": "not rejected", "n_act": 67, "errors": 5, "p_value": 0.541482029624577, "log_capital": 0.6134453993806386, "weighted_error": 0.07462686567164178}; FS-CRR c8 lam None stop {"lambda": 0.95, "reason": "not rejected", "n_act": 67, "errors": 5, "p_value": 0.9248503188002285, "log_capital": 0.07812337207684475, "weighted_error": 0.07462686567164178}

per type / family, variant T, alpha 0.05 (groups {"choice": {"n_fit": 104, "qhat": 0.9304309995746419}, "noul": {"n_fit": 151, "qhat": 0.9391913010284082}, "pooled-small": {"n_fit": 12, "qhat": Infinity}}; pooled-small ['score'] full-set True)
| cell | n | cov | size | singleton | sing acc | top-1 | act | acted err | per-type-LTT act / err |
|---|---|---|---|---|---|---|---|---|---|
| choice | 101 | 0.980 | 2.42 | 0.158 | 0.938 | 0.624 | 0.000 | - | 0.000 / - |
| noul | 190 | 0.937 | 1.61 | 0.395 | 0.840 | 0.721 | 0.000 | - | 0.000 / - |
| score | 10 | 1.000 | 4.30 | 0.000 | - | 0.300 | 0.000 | - | 0.000 / - |
| adequacy | 35 | 0.971 | 1.94 | 0.057 | 0.500 | 0.571 | 0.000 | - | 0.000 / - |
| ambiguous | 51 | 1.000 | 2.10 | 0.000 | - | 0.529 | 0.000 | - | 0.000 / - |
| judge_hard | 62 | 0.935 | 1.73 | 0.274 | 0.765 | 0.710 | 0.000 | - | 0.000 / - |
| long_policy | 58 | 0.948 | 2.93 | 0.103 | 0.667 | 0.586 | 0.000 | - | 0.000 / - |
| policy | 57 | 0.930 | 1.16 | 0.842 | 0.917 | 0.860 | 0.000 | - | 0.000 / - |
| temporal_numeric | 38 | 0.947 | 1.95 | 0.474 | 0.889 | 0.763 | 0.000 | - | 0.000 / - |
LTT-half attribution: best lambda by p {"lambda": 0.98, "n_act": 49, "errors": 3, "p": 0.77128597926257}; Bonferroni level 9.90e-04; zero-error acts needed 135 (single test 45); LTT singleton rate 0.279; per-type lambdas {'noul': None}; FS-CRR c1 stop {"lambda": 0.95, "reason": "not rejected", "n_act": 67, "errors": 5, "p_value": 0.541482029624577, "log_capital": 0.6134453993806386, "weighted_error": 0.07462686567164178}; FS-CRR c8 lam None stop {"lambda": 0.95, "reason": "not rejected", "n_act": 67, "errors": 5, "p_value": 0.9248503188002285, "log_capital": 0.07812337207684475, "weighted_error": 0.07462686567164178}

### h2o_x7c42  (n 3600, top-1 0.8461, dev-card 360)
shim parity max |d| 0.00e+00

**h2o_x7c42 / native / all** FIT 1203 LTT 1217 TEST 1180; types {'FIT': {'choice': 583, 'noul': 367, 'score': 253}, 'LTT': {'choice': 566, 'noul': 362, 'score': 289}, 'TEST': {'choice': 566, 'noul': 377, 'score': 237}}
| alpha | var | TEST cov | flag | size | singleton | sing acc | lambda* (Bonf) | TEST act | acted err | FS-CRR c1 lam / act / err | FS-binom lam / act / err | per-type LTT act / err |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.1 | M | 0.916 | ok | 1.16 | 0.845 | 0.915 | 0.90 | 0.608 | 0.024 | 0.90 / 0.608 / 0.024 | 0.88 / 0.642 / 0.029 | 0.236 / 0.000 |
| 0.1 | T | 0.919 | ok | 1.17 | 0.834 | 0.917 | 0.90 | 0.608 | 0.024 | 0.90 / 0.608 / 0.024 | 0.88 / 0.642 / 0.029 | 0.236 / 0.000 |
| 0.1 | F | 0.916 | ok | 1.16 | 0.845 | 0.915 | 0.90 | 0.608 | 0.024 | 0.90 / 0.608 / 0.024 | 0.88 / 0.642 / 0.029 | 0.236 / 0.000 |
| 0.05 | M | 0.950 | ok | 1.33 | 0.721 | 0.949 | 0.90 | 0.608 | 0.024 | 0.90 / 0.608 / 0.024 | 0.88 / 0.642 / 0.029 | 0.236 / 0.000 |
| 0.05 | T | 0.954 | ok | 1.37 | 0.706 | 0.954 | 0.90 | 0.608 | 0.024 | 0.90 / 0.608 / 0.024 | 0.88 / 0.642 / 0.029 | 0.236 / 0.000 |
| 0.05 | F | 0.950 | ok | 1.33 | 0.721 | 0.949 | 0.90 | 0.608 | 0.024 | 0.90 / 0.608 / 0.024 | 0.88 / 0.642 / 0.029 | 0.236 / 0.000 |

per type / family, variant T, alpha 0.1 (groups {"choice": {"n_fit": 583, "qhat": 0.7311405409509548}, "noul": {"n_fit": 367, "qhat": 0.7603291210094014}, "score": {"n_fit": 253, "qhat": 0.8250389049023151}}; pooled-small [] full-set False)
| cell | n | cov | size | singleton | sing acc | top-1 | act | acted err | per-type-LTT act / err |
|---|---|---|---|---|---|---|---|---|---|
| choice | 566 | 0.926 | 1.09 | 0.908 | 0.926 | 0.883 | 0.686 | 0.013 | 0.491 / 0.000 |
| noul | 377 | 0.926 | 1.18 | 0.825 | 0.910 | 0.833 | 0.634 | 0.038 | 0.000 / - |
| score | 237 | 0.895 | 1.38 | 0.671 | 0.899 | 0.789 | 0.384 | 0.033 | 0.000 / - |
| adequacy | 63 | 0.873 | 1.25 | 0.762 | 0.896 | 0.794 | 0.508 | 0.031 | 0.032 / 0.000 |
| adversarial | 55 | 0.927 | 1.36 | 0.673 | 0.919 | 0.855 | 0.382 | 0.000 | 0.145 / 0.000 |
| ambiguous | 75 | 0.987 | 1.00 | 1.000 | 0.987 | 0.987 | 0.933 | 0.014 | 0.587 / 0.000 |
| enum | 80 | 1.000 | 1.01 | 0.988 | 1.000 | 1.000 | 0.988 | 0.000 | 0.713 / 0.000 |
| fact | 71 | 0.986 | 1.03 | 0.972 | 0.986 | 0.972 | 0.930 | 0.015 | 0.000 / - |
| intent | 68 | 1.000 | 1.03 | 0.971 | 1.000 | 0.985 | 0.912 | 0.000 | 0.500 / 0.000 |
| judge_hard | 70 | 0.871 | 1.29 | 0.714 | 0.820 | 0.714 | 0.457 | 0.062 | 0.000 / - |
| long_policy | 70 | 0.914 | 1.14 | 0.857 | 0.917 | 0.871 | 0.543 | 0.026 | 0.086 / 0.000 |
| multi_hop | 57 | 0.895 | 1.14 | 0.860 | 0.878 | 0.842 | 0.561 | 0.062 | 0.263 / 0.000 |
| policy | 67 | 0.896 | 1.24 | 0.761 | 0.863 | 0.791 | 0.552 | 0.081 | 0.000 / - |
| probability | 67 | 0.881 | 1.37 | 0.657 | 0.864 | 0.731 | 0.194 | 0.000 | 0.119 / 0.000 |
| routing | 68 | 0.941 | 1.12 | 0.882 | 0.933 | 0.882 | 0.662 | 0.022 | 0.235 / 0.000 |
| severity | 63 | 0.889 | 1.29 | 0.714 | 0.889 | 0.778 | 0.540 | 0.059 | 0.000 / - |
| state_tracking | 54 | 0.889 | 1.09 | 0.870 | 0.894 | 0.815 | 0.685 | 0.000 | 0.370 / 0.000 |
| temporal_numeric | 66 | 0.818 | 1.30 | 0.697 | 0.761 | 0.576 | 0.182 | 0.000 | 0.136 / 0.000 |
| tool_selection | 53 | 1.000 | 1.00 | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | 0.943 / 0.000 |
| tradeoff | 70 | 0.786 | 1.43 | 0.657 | 0.717 | 0.686 | 0.129 | 0.222 | 0.000 / - |
| trap | 63 | 0.984 | 1.06 | 0.937 | 0.983 | 0.968 | 0.730 | 0.022 | 0.143 / 0.000 |
LTT-half attribution: best lambda by p {"lambda": 0.95, "n_act": 366, "errors": 3, "p": 9.747105917775468e-06}; Bonferroni level 9.90e-04; zero-error acts needed 135 (single test 45); LTT singleton rate 0.828; per-type lambdas {'choice': 0.94, 'noul': None, 'score': None}; FS-CRR c1 stop {"lambda": 0.89, "reason": "not rejected", "n_act": 751, "errors": 25, "p_value": 0.18266378176357842, "log_capital": 1.7001080740920278, "weighted_error": 0.033288948069241014}; FS-CRR c8 lam None stop {"lambda": 0.95, "reason": "not rejected", "n_act": 366, "errors": 3, "p_value": 0.22872551577460662, "log_capital": 1.475232615420254, "weighted_error": 0.00819672131147541}

per type / family, variant T, alpha 0.05 (groups {"choice": {"n_fit": 583, "qhat": 0.8901689138550315}, "noul": {"n_fit": 367, "qhat": 0.8599525097304201}, "score": {"n_fit": 253, "qhat": 0.9076263190134325}}; pooled-small [] full-set False)
| cell | n | cov | size | singleton | sing acc | top-1 | act | acted err | per-type-LTT act / err |
|---|---|---|---|---|---|---|---|---|---|
| choice | 566 | 0.954 | 1.27 | 0.806 | 0.961 | 0.883 | 0.686 | 0.013 | 0.491 / 0.000 |
| noul | 377 | 0.963 | 1.31 | 0.692 | 0.946 | 0.833 | 0.634 | 0.038 | 0.000 / - |
| score | 237 | 0.941 | 1.70 | 0.489 | 0.948 | 0.789 | 0.384 | 0.033 | 0.000 / - |
| adequacy | 63 | 0.968 | 1.49 | 0.571 | 0.972 | 0.794 | 0.508 | 0.031 | 0.032 / 0.000 |
| adversarial | 55 | 0.927 | 1.64 | 0.455 | 1.000 | 0.855 | 0.382 | 0.000 | 0.145 / 0.000 |
| ambiguous | 75 | 0.987 | 1.01 | 0.987 | 0.986 | 0.987 | 0.933 | 0.014 | 0.587 / 0.000 |
| enum | 80 | 1.000 | 1.01 | 0.988 | 1.000 | 1.000 | 0.988 | 0.000 | 0.713 / 0.000 |
| fact | 71 | 0.986 | 1.06 | 0.944 | 0.985 | 0.972 | 0.930 | 0.015 | 0.000 / - |
| intent | 68 | 1.000 | 1.07 | 0.941 | 1.000 | 0.985 | 0.912 | 0.000 | 0.500 / 0.000 |
| judge_hard | 70 | 0.957 | 1.50 | 0.500 | 0.914 | 0.714 | 0.457 | 0.062 | 0.000 / - |
| long_policy | 70 | 0.957 | 1.31 | 0.700 | 0.959 | 0.871 | 0.543 | 0.026 | 0.086 / 0.000 |
| multi_hop | 57 | 0.912 | 1.23 | 0.789 | 0.889 | 0.842 | 0.561 | 0.062 | 0.263 / 0.000 |
| policy | 67 | 0.925 | 1.34 | 0.657 | 0.886 | 0.791 | 0.552 | 0.081 | 0.000 / - |
| probability | 67 | 0.940 | 1.75 | 0.403 | 0.926 | 0.731 | 0.194 | 0.000 | 0.119 / 0.000 |
| routing | 68 | 0.971 | 1.21 | 0.824 | 0.964 | 0.882 | 0.662 | 0.022 | 0.235 / 0.000 |
| severity | 63 | 0.905 | 1.40 | 0.651 | 0.902 | 0.778 | 0.540 | 0.059 | 0.000 / - |
| state_tracking | 54 | 0.981 | 1.61 | 0.741 | 1.000 | 0.815 | 0.685 | 0.000 | 0.370 / 0.000 |
| temporal_numeric | 66 | 0.939 | 1.89 | 0.333 | 0.909 | 0.576 | 0.182 | 0.000 | 0.136 / 0.000 |
| tool_selection | 53 | 1.000 | 1.00 | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | 0.943 / 0.000 |
| tradeoff | 70 | 0.829 | 2.03 | 0.329 | 0.609 | 0.686 | 0.129 | 0.222 | 0.000 / - |
| trap | 63 | 0.984 | 1.16 | 0.841 | 0.981 | 0.968 | 0.730 | 0.022 | 0.143 / 0.000 |
LTT-half attribution: best lambda by p {"lambda": 0.95, "n_act": 366, "errors": 3, "p": 9.747105917775468e-06}; Bonferroni level 9.90e-04; zero-error acts needed 135 (single test 45); LTT singleton rate 0.690; per-type lambdas {'choice': 0.94, 'noul': None, 'score': None}; FS-CRR c1 stop {"lambda": 0.89, "reason": "not rejected", "n_act": 751, "errors": 25, "p_value": 0.18266378176357842, "log_capital": 1.7001080740920278, "weighted_error": 0.033288948069241014}; FS-CRR c8 lam None stop {"lambda": 0.95, "reason": "not rejected", "n_act": 366, "errors": 3, "p_value": 0.22872551577460662, "log_capital": 1.475232615420254, "weighted_error": 0.00819672131147541}

**h2o_x7c42 / native / remainder** FIT 1072 LTT 1092 TEST 1076; types {'FIT': {'choice': 507, 'noul': 332, 'score': 233}, 'LTT': {'choice': 500, 'noul': 321, 'score': 271}, 'TEST': {'choice': 511, 'noul': 343, 'score': 222}}
| alpha | var | TEST cov | flag | size | singleton | sing acc | lambda* (Bonf) | TEST act | acted err | FS-CRR c1 lam / act / err | FS-binom lam / act / err | per-type LTT act / err |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.1 | M | 0.914 | ok | 1.14 | 0.856 | 0.912 | 0.91 | 0.577 | 0.019 | 0.90 / 0.601 / 0.022 | 0.88 / 0.638 / 0.028 | 0.235 / 0.000 |
| 0.1 | T | 0.915 | ok | 1.15 | 0.849 | 0.911 | 0.91 | 0.577 | 0.019 | 0.90 / 0.601 / 0.022 | 0.88 / 0.638 / 0.028 | 0.235 / 0.000 |
| 0.1 | F | 0.914 | ok | 1.14 | 0.856 | 0.912 | 0.91 | 0.577 | 0.019 | 0.90 / 0.601 / 0.022 | 0.88 / 0.638 / 0.028 | 0.235 / 0.000 |
| 0.05 | M | 0.947 | ok | 1.31 | 0.733 | 0.946 | 0.91 | 0.577 | 0.019 | 0.90 / 0.601 / 0.022 | 0.88 / 0.638 / 0.028 | 0.235 / 0.000 |
| 0.05 | T | 0.950 | ok | 1.32 | 0.725 | 0.949 | 0.91 | 0.577 | 0.019 | 0.90 / 0.601 / 0.022 | 0.88 / 0.638 / 0.028 | 0.235 / 0.000 |
| 0.05 | F | 0.947 | ok | 1.31 | 0.733 | 0.946 | 0.91 | 0.577 | 0.019 | 0.90 / 0.601 / 0.022 | 0.88 / 0.638 / 0.028 | 0.235 / 0.000 |

per type / family, variant T, alpha 0.1 (groups {"choice": {"n_fit": 507, "qhat": 0.7293894048590858}, "noul": {"n_fit": 332, "qhat": 0.768278491053773}, "score": {"n_fit": 233, "qhat": 0.7860610350931938}}; pooled-small [] full-set False)
| cell | n | cov | size | singleton | sing acc | top-1 | act | acted err | per-type-LTT act / err |
|---|---|---|---|---|---|---|---|---|---|
| choice | 511 | 0.922 | 1.08 | 0.914 | 0.923 | 0.883 | 0.652 | 0.006 | 0.495 / 0.000 |
| noul | 343 | 0.927 | 1.20 | 0.802 | 0.909 | 0.825 | 0.595 | 0.034 | 0.000 / - |
| score | 222 | 0.883 | 1.24 | 0.770 | 0.883 | 0.815 | 0.378 | 0.036 | 0.000 / - |
| adequacy | 48 | 0.896 | 1.15 | 0.854 | 0.902 | 0.833 | 0.500 | 0.000 | 0.021 / 0.000 |
| adversarial | 55 | 0.927 | 1.33 | 0.691 | 0.921 | 0.855 | 0.327 | 0.000 | 0.145 / 0.000 |
| ambiguous | 73 | 0.986 | 1.00 | 1.000 | 0.986 | 0.986 | 0.932 | 0.015 | 0.603 / 0.000 |
| enum | 78 | 1.000 | 1.01 | 0.987 | 1.000 | 1.000 | 0.987 | 0.000 | 0.705 / 0.000 |
| fact | 63 | 0.984 | 1.03 | 0.968 | 0.984 | 0.968 | 0.937 | 0.017 | 0.000 / - |
| intent | 46 | 1.000 | 1.02 | 0.978 | 1.000 | 1.000 | 0.891 | 0.000 | 0.543 / 0.000 |
| judge_hard | 62 | 0.871 | 1.34 | 0.661 | 0.805 | 0.694 | 0.387 | 0.042 | 0.000 / - |
| long_policy | 69 | 0.928 | 1.14 | 0.855 | 0.932 | 0.884 | 0.507 | 0.029 | 0.087 / 0.000 |
| multi_hop | 53 | 0.906 | 1.15 | 0.849 | 0.889 | 0.849 | 0.491 | 0.038 | 0.245 / 0.000 |
| policy | 61 | 0.885 | 1.23 | 0.770 | 0.851 | 0.770 | 0.492 | 0.100 | 0.000 / - |
| probability | 64 | 0.844 | 1.28 | 0.734 | 0.830 | 0.734 | 0.172 | 0.000 | 0.109 / 0.000 |
| routing | 62 | 0.935 | 1.13 | 0.871 | 0.926 | 0.871 | 0.597 | 0.000 | 0.226 / 0.000 |
| severity | 54 | 0.870 | 1.15 | 0.852 | 0.870 | 0.852 | 0.611 | 0.061 | 0.000 / - |
| state_tracking | 50 | 0.880 | 1.08 | 0.880 | 0.886 | 0.800 | 0.680 | 0.000 | 0.400 / 0.000 |
| temporal_numeric | 62 | 0.806 | 1.31 | 0.694 | 0.744 | 0.565 | 0.145 | 0.000 | 0.129 / 0.000 |
| tool_selection | 45 | 1.000 | 1.00 | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | 0.956 / 0.000 |
| tradeoff | 69 | 0.783 | 1.29 | 0.710 | 0.735 | 0.696 | 0.116 | 0.125 | 0.000 / - |
| trap | 62 | 0.984 | 1.06 | 0.935 | 0.983 | 0.968 | 0.677 | 0.024 | 0.145 / 0.000 |
LTT-half attribution: best lambda by p {"lambda": 0.94, "n_act": 441, "errors": 6, "p": 4.185660871186509e-05}; Bonferroni level 9.90e-04; zero-error acts needed 135 (single test 45); LTT singleton rate 0.843; per-type lambdas {'choice': 0.94, 'noul': None, 'score': None}; FS-CRR c1 stop {"lambda": 0.89, "reason": "not rejected", "n_act": 672, "errors": 23, "p_value": 0.2828891667688979, "log_capital": 1.262700094907661, "weighted_error": 0.03422619047619048}; FS-CRR c8 lam None stop {"lambda": 0.95, "reason": "not rejected", "n_act": 329, "errors": 3, "p_value": 0.27309341916644, "log_capital": 1.2979413475835828, "weighted_error": 0.00911854103343465}

per type / family, variant T, alpha 0.05 (groups {"choice": {"n_fit": 507, "qhat": 0.8743346563880845}, "noul": {"n_fit": 332, "qhat": 0.8684988506375501}, "score": {"n_fit": 233, "qhat": 0.880264638476694}}; pooled-small [] full-set False)
| cell | n | cov | size | singleton | sing acc | top-1 | act | acted err | per-type-LTT act / err |
|---|---|---|---|---|---|---|---|---|---|
| choice | 511 | 0.953 | 1.23 | 0.824 | 0.960 | 0.883 | 0.652 | 0.006 | 0.495 / 0.000 |
| noul | 343 | 0.962 | 1.33 | 0.668 | 0.943 | 0.825 | 0.595 | 0.034 | 0.000 / - |
| score | 222 | 0.923 | 1.51 | 0.586 | 0.923 | 0.815 | 0.378 | 0.036 | 0.000 / - |
| adequacy | 48 | 0.938 | 1.29 | 0.708 | 0.941 | 0.833 | 0.500 | 0.000 | 0.021 / 0.000 |
| adversarial | 55 | 0.927 | 1.47 | 0.564 | 0.968 | 0.855 | 0.327 | 0.000 | 0.145 / 0.000 |
| ambiguous | 73 | 0.986 | 1.01 | 0.986 | 0.986 | 0.986 | 0.932 | 0.015 | 0.603 / 0.000 |
| enum | 78 | 1.000 | 1.01 | 0.987 | 1.000 | 1.000 | 0.987 | 0.000 | 0.705 / 0.000 |
| fact | 63 | 0.984 | 1.05 | 0.952 | 0.983 | 0.968 | 0.937 | 0.017 | 0.000 / - |
| intent | 46 | 1.000 | 1.09 | 0.935 | 1.000 | 1.000 | 0.891 | 0.000 | 0.543 / 0.000 |
| judge_hard | 62 | 0.968 | 1.58 | 0.419 | 0.923 | 0.694 | 0.387 | 0.042 | 0.000 / - |
| long_policy | 69 | 0.957 | 1.32 | 0.681 | 0.957 | 0.884 | 0.507 | 0.029 | 0.087 / 0.000 |
| multi_hop | 53 | 0.925 | 1.23 | 0.792 | 0.905 | 0.849 | 0.491 | 0.038 | 0.245 / 0.000 |
| policy | 61 | 0.918 | 1.33 | 0.672 | 0.878 | 0.770 | 0.492 | 0.100 | 0.000 / - |
| probability | 64 | 0.906 | 1.64 | 0.469 | 0.900 | 0.734 | 0.172 | 0.000 | 0.109 / 0.000 |
| routing | 62 | 0.968 | 1.21 | 0.806 | 0.960 | 0.871 | 0.597 | 0.000 | 0.226 / 0.000 |
| severity | 54 | 0.889 | 1.24 | 0.796 | 0.884 | 0.852 | 0.611 | 0.061 | 0.000 / - |
| state_tracking | 50 | 0.980 | 1.52 | 0.740 | 1.000 | 0.800 | 0.680 | 0.000 | 0.400 / 0.000 |
| temporal_numeric | 62 | 0.935 | 1.84 | 0.355 | 0.864 | 0.565 | 0.145 | 0.000 | 0.129 / 0.000 |
| tool_selection | 45 | 1.000 | 1.00 | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | 0.956 / 0.000 |
| tradeoff | 69 | 0.841 | 1.83 | 0.362 | 0.680 | 0.696 | 0.116 | 0.125 | 0.000 / - |
| trap | 62 | 0.984 | 1.11 | 0.887 | 0.982 | 0.968 | 0.677 | 0.024 | 0.145 / 0.000 |
LTT-half attribution: best lambda by p {"lambda": 0.94, "n_act": 441, "errors": 6, "p": 4.185660871186509e-05}; Bonferroni level 9.90e-04; zero-error acts needed 135 (single test 45); LTT singleton rate 0.708; per-type lambdas {'choice': 0.94, 'noul': None, 'score': None}; FS-CRR c1 stop {"lambda": 0.89, "reason": "not rejected", "n_act": 672, "errors": 23, "p_value": 0.2828891667688979, "log_capital": 1.262700094907661, "weighted_error": 0.03422619047619048}; FS-CRR c8 lam None stop {"lambda": 0.95, "reason": "not rejected", "n_act": 329, "errors": 3, "p_value": 0.27309341916644, "log_capital": 1.2979413475835828, "weighted_error": 0.00911854103343465}

**h2o_x7c42 / native_shimcal / all** FIT 1203 LTT 1217 TEST 1180; types {'FIT': {'choice': 583, 'noul': 367, 'score': 253}, 'LTT': {'choice': 566, 'noul': 362, 'score': 289}, 'TEST': {'choice': 566, 'noul': 377, 'score': 237}}
| alpha | var | TEST cov | flag | size | singleton | sing acc | lambda* (Bonf) | TEST act | acted err | FS-CRR c1 lam / act / err | FS-binom lam / act / err | per-type LTT act / err |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.1 | M | 0.916 | ok | 1.17 | 0.840 | 0.918 | 0.95 | 0.601 | 0.021 | 0.94 / 0.632 / 0.025 | 0.93 / 0.658 / 0.032 | 0.288 / 0.003 |
| 0.1 | T | 0.920 | ok | 1.18 | 0.832 | 0.915 | 0.95 | 0.601 | 0.021 | 0.94 / 0.632 / 0.025 | 0.93 / 0.658 / 0.032 | 0.288 / 0.003 |
| 0.1 | F | 0.916 | ok | 1.17 | 0.840 | 0.918 | 0.95 | 0.601 | 0.021 | 0.94 / 0.632 / 0.025 | 0.93 / 0.658 / 0.032 | 0.288 / 0.003 |
| 0.05 | M | 0.950 | ok | 1.32 | 0.733 | 0.948 | 0.95 | 0.601 | 0.021 | 0.94 / 0.632 / 0.025 | 0.93 / 0.658 / 0.032 | 0.288 / 0.003 |
| 0.05 | T | 0.954 | ok | 1.37 | 0.711 | 0.955 | 0.95 | 0.601 | 0.021 | 0.94 / 0.632 / 0.025 | 0.93 / 0.658 / 0.032 | 0.288 / 0.003 |
| 0.05 | F | 0.950 | ok | 1.32 | 0.733 | 0.948 | 0.95 | 0.601 | 0.021 | 0.94 / 0.632 / 0.025 | 0.93 / 0.658 / 0.032 | 0.288 / 0.003 |

per type / family, variant T, alpha 0.1 (groups {"choice": {"n_fit": 583, "qhat": 0.7513267971075602}, "noul": {"n_fit": 367, "qhat": 0.8089351626912717}, "score": {"n_fit": 253, "qhat": 0.8713237385542908}}; pooled-small [] full-set False)
| cell | n | cov | size | singleton | sing acc | top-1 | act | acted err | per-type-LTT act / err |
|---|---|---|---|---|---|---|---|---|---|
| choice | 566 | 0.926 | 1.09 | 0.913 | 0.925 | 0.883 | 0.689 | 0.013 | 0.601 / 0.003 |
| noul | 377 | 0.926 | 1.18 | 0.825 | 0.910 | 0.833 | 0.597 | 0.027 | 0.000 / - |
| score | 237 | 0.899 | 1.42 | 0.650 | 0.896 | 0.789 | 0.397 | 0.043 | 0.000 / - |
| adequacy | 63 | 0.873 | 1.25 | 0.762 | 0.896 | 0.794 | 0.476 | 0.033 | 0.063 / 0.000 |
| adversarial | 55 | 0.927 | 1.40 | 0.636 | 0.914 | 0.855 | 0.364 | 0.000 | 0.164 / 0.000 |
| ambiguous | 75 | 0.987 | 1.00 | 1.000 | 0.987 | 0.987 | 0.933 | 0.014 | 0.613 / 0.000 |
| enum | 80 | 1.000 | 1.01 | 0.988 | 1.000 | 1.000 | 0.988 | 0.000 | 0.762 / 0.000 |
| fact | 71 | 0.986 | 1.03 | 0.972 | 0.986 | 0.972 | 0.901 | 0.016 | 0.000 / - |
| intent | 68 | 1.000 | 1.03 | 0.971 | 1.000 | 0.985 | 0.897 | 0.000 | 0.750 / 0.000 |
| judge_hard | 70 | 0.871 | 1.29 | 0.714 | 0.820 | 0.714 | 0.429 | 0.033 | 0.000 / - |
| long_policy | 70 | 0.914 | 1.13 | 0.871 | 0.918 | 0.871 | 0.514 | 0.028 | 0.129 / 0.000 |
| multi_hop | 57 | 0.895 | 1.14 | 0.860 | 0.878 | 0.842 | 0.561 | 0.062 | 0.404 / 0.043 |
| policy | 67 | 0.896 | 1.24 | 0.761 | 0.863 | 0.791 | 0.507 | 0.059 | 0.030 / 0.000 |
| probability | 67 | 0.896 | 1.42 | 0.657 | 0.864 | 0.731 | 0.209 | 0.000 | 0.119 / 0.000 |
| routing | 68 | 0.941 | 1.12 | 0.882 | 0.933 | 0.882 | 0.662 | 0.000 | 0.441 / 0.000 |
| severity | 63 | 0.889 | 1.29 | 0.714 | 0.889 | 0.778 | 0.571 | 0.083 | 0.000 / - |
| state_tracking | 54 | 0.889 | 1.11 | 0.889 | 0.875 | 0.815 | 0.685 | 0.000 | 0.389 / 0.000 |
| temporal_numeric | 66 | 0.818 | 1.30 | 0.697 | 0.761 | 0.576 | 0.182 | 0.000 | 0.167 / 0.000 |
| tool_selection | 53 | 1.000 | 1.00 | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | 1.000 / 0.000 |
| tradeoff | 70 | 0.786 | 1.50 | 0.629 | 0.705 | 0.686 | 0.129 | 0.222 | 0.043 / 0.000 |
| trap | 63 | 0.984 | 1.06 | 0.937 | 0.983 | 0.968 | 0.746 | 0.021 | 0.143 / 0.000 |
LTT-half attribution: best lambda by p {"lambda": 0.98, "n_act": 373, "errors": 1, "p": 1.0125984046223173e-07}; Bonferroni level 9.90e-04; zero-error acts needed 135 (single test 45); LTT singleton rate 0.827; per-type lambdas {'choice': 0.97, 'noul': None, 'score': None}; FS-CRR c1 stop {"lambda": 0.93, "reason": "not rejected", "n_act": 785, "errors": 31, "p_value": 0.4107454486671429, "log_capital": 0.8897816026687106, "weighted_error": 0.03949044585987261}; FS-CRR c8 lam None stop {"lambda": 0.95, "reason": "not rejected", "n_act": 704, "errors": 17, "p_value": 0.1793726420099728, "log_capital": 1.7182898381436573, "weighted_error": 0.024147727272727272}

per type / family, variant T, alpha 0.05 (groups {"choice": {"n_fit": 583, "qhat": 0.9188442748047934}, "noul": {"n_fit": 367, "qhat": 0.9062445572739392}, "score": {"n_fit": 253, "qhat": 0.9431769651828973}}; pooled-small [] full-set False)
| cell | n | cov | size | singleton | sing acc | top-1 | act | acted err | per-type-LTT act / err |
|---|---|---|---|---|---|---|---|---|---|
| choice | 566 | 0.954 | 1.26 | 0.816 | 0.961 | 0.883 | 0.689 | 0.013 | 0.601 / 0.003 |
| noul | 377 | 0.963 | 1.31 | 0.692 | 0.946 | 0.833 | 0.597 | 0.027 | 0.000 / - |
| score | 237 | 0.941 | 1.71 | 0.489 | 0.948 | 0.789 | 0.397 | 0.043 | 0.000 / - |
| adequacy | 63 | 0.968 | 1.49 | 0.587 | 0.973 | 0.794 | 0.476 | 0.033 | 0.063 / 0.000 |
| adversarial | 55 | 0.927 | 1.65 | 0.455 | 1.000 | 0.855 | 0.364 | 0.000 | 0.164 / 0.000 |
| ambiguous | 75 | 0.987 | 1.01 | 0.987 | 0.986 | 0.987 | 0.933 | 0.014 | 0.613 / 0.000 |
| enum | 80 | 1.000 | 1.01 | 0.988 | 1.000 | 1.000 | 0.988 | 0.000 | 0.762 / 0.000 |
| fact | 71 | 0.986 | 1.06 | 0.944 | 0.985 | 0.972 | 0.901 | 0.016 | 0.000 / - |
| intent | 68 | 1.000 | 1.07 | 0.941 | 1.000 | 0.985 | 0.897 | 0.000 | 0.750 / 0.000 |
| judge_hard | 70 | 0.957 | 1.50 | 0.500 | 0.914 | 0.714 | 0.429 | 0.033 | 0.000 / - |
| long_policy | 70 | 0.957 | 1.31 | 0.700 | 0.959 | 0.871 | 0.514 | 0.028 | 0.129 / 0.000 |
| multi_hop | 57 | 0.912 | 1.21 | 0.807 | 0.891 | 0.842 | 0.561 | 0.062 | 0.404 / 0.043 |
| policy | 67 | 0.925 | 1.33 | 0.672 | 0.889 | 0.791 | 0.507 | 0.059 | 0.030 / 0.000 |
| probability | 67 | 0.940 | 1.75 | 0.418 | 0.929 | 0.731 | 0.209 | 0.000 | 0.119 / 0.000 |
| routing | 68 | 0.971 | 1.21 | 0.824 | 0.964 | 0.882 | 0.662 | 0.000 | 0.441 / 0.000 |
| severity | 63 | 0.905 | 1.40 | 0.651 | 0.902 | 0.778 | 0.571 | 0.083 | 0.000 / - |
| state_tracking | 54 | 0.981 | 1.61 | 0.741 | 1.000 | 0.815 | 0.685 | 0.000 | 0.389 / 0.000 |
| temporal_numeric | 66 | 0.939 | 1.88 | 0.348 | 0.913 | 0.576 | 0.182 | 0.000 | 0.167 / 0.000 |
| tool_selection | 53 | 1.000 | 1.00 | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | 1.000 / 0.000 |
| tradeoff | 70 | 0.829 | 2.03 | 0.343 | 0.625 | 0.686 | 0.129 | 0.222 | 0.043 / 0.000 |
| trap | 63 | 0.984 | 1.16 | 0.841 | 0.981 | 0.968 | 0.746 | 0.021 | 0.143 / 0.000 |
LTT-half attribution: best lambda by p {"lambda": 0.98, "n_act": 373, "errors": 1, "p": 1.0125984046223173e-07}; Bonferroni level 9.90e-04; zero-error acts needed 135 (single test 45); LTT singleton rate 0.696; per-type lambdas {'choice': 0.97, 'noul': None, 'score': None}; FS-CRR c1 stop {"lambda": 0.93, "reason": "not rejected", "n_act": 785, "errors": 31, "p_value": 0.4107454486671429, "log_capital": 0.8897816026687106, "weighted_error": 0.03949044585987261}; FS-CRR c8 lam None stop {"lambda": 0.95, "reason": "not rejected", "n_act": 704, "errors": 17, "p_value": 0.1793726420099728, "log_capital": 1.7182898381436573, "weighted_error": 0.024147727272727272}

**h2o_x7c42 / native_shimcal / remainder** FIT 1072 LTT 1092 TEST 1076; types {'FIT': {'choice': 507, 'noul': 332, 'score': 233}, 'LTT': {'choice': 500, 'noul': 321, 'score': 271}, 'TEST': {'choice': 511, 'noul': 343, 'score': 222}}
| alpha | var | TEST cov | flag | size | singleton | sing acc | lambda* (Bonf) | TEST act | acted err | FS-CRR c1 lam / act / err | FS-binom lam / act / err | per-type LTT act / err |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.1 | M | 0.915 | ok | 1.16 | 0.847 | 0.917 | 0.95 | 0.597 | 0.019 | 0.95 / 0.597 / 0.019 | 0.94 / 0.626 / 0.024 | 0.282 / 0.003 |
| 0.1 | T | 0.916 | ok | 1.15 | 0.852 | 0.912 | 0.95 | 0.597 | 0.019 | 0.95 / 0.597 / 0.019 | 0.94 / 0.626 / 0.024 | 0.282 / 0.003 |
| 0.1 | F | 0.915 | ok | 1.16 | 0.847 | 0.917 | 0.95 | 0.597 | 0.019 | 0.95 / 0.597 / 0.019 | 0.94 / 0.626 / 0.024 | 0.282 / 0.003 |
| 0.05 | M | 0.948 | ok | 1.32 | 0.735 | 0.946 | 0.95 | 0.597 | 0.019 | 0.95 / 0.597 / 0.019 | 0.94 / 0.626 / 0.024 | 0.282 / 0.003 |
| 0.05 | T | 0.948 | ok | 1.32 | 0.733 | 0.946 | 0.95 | 0.597 | 0.019 | 0.95 / 0.597 / 0.019 | 0.94 / 0.626 / 0.024 | 0.282 / 0.003 |
| 0.05 | F | 0.948 | ok | 1.32 | 0.735 | 0.946 | 0.95 | 0.597 | 0.019 | 0.95 / 0.597 / 0.019 | 0.94 / 0.626 / 0.024 | 0.282 / 0.003 |

per type / family, variant T, alpha 0.1 (groups {"choice": {"n_fit": 507, "qhat": 0.7442082004477284}, "noul": {"n_fit": 332, "qhat": 0.8173163162535464}, "score": {"n_fit": 233, "qhat": 0.8257684192193421}}; pooled-small [] full-set False)
| cell | n | cov | size | singleton | sing acc | top-1 | act | acted err | per-type-LTT act / err |
|---|---|---|---|---|---|---|---|---|---|
| choice | 511 | 0.922 | 1.08 | 0.924 | 0.922 | 0.883 | 0.681 | 0.009 | 0.593 / 0.003 |
| noul | 343 | 0.927 | 1.20 | 0.802 | 0.909 | 0.825 | 0.586 | 0.025 | 0.000 / - |
| score | 222 | 0.887 | 1.25 | 0.766 | 0.888 | 0.815 | 0.419 | 0.043 | 0.000 / - |
| adequacy | 48 | 0.896 | 1.12 | 0.875 | 0.905 | 0.833 | 0.583 | 0.036 | 0.062 / 0.000 |
| adversarial | 55 | 0.927 | 1.33 | 0.691 | 0.921 | 0.855 | 0.364 | 0.000 | 0.164 / 0.000 |
| ambiguous | 73 | 0.986 | 1.00 | 1.000 | 0.986 | 0.986 | 0.932 | 0.015 | 0.630 / 0.000 |
| enum | 78 | 1.000 | 1.01 | 0.987 | 1.000 | 1.000 | 0.987 | 0.000 | 0.756 / 0.000 |
| fact | 63 | 0.984 | 1.03 | 0.968 | 0.984 | 0.968 | 0.937 | 0.017 | 0.000 / - |
| intent | 46 | 1.000 | 1.02 | 0.978 | 1.000 | 1.000 | 0.870 | 0.000 | 0.761 / 0.000 |
| judge_hard | 62 | 0.871 | 1.34 | 0.661 | 0.805 | 0.694 | 0.371 | 0.000 | 0.000 / - |
| long_policy | 69 | 0.928 | 1.13 | 0.870 | 0.933 | 0.884 | 0.522 | 0.028 | 0.130 / 0.000 |
| multi_hop | 53 | 0.906 | 1.15 | 0.849 | 0.889 | 0.849 | 0.528 | 0.036 | 0.377 / 0.050 |
| policy | 61 | 0.885 | 1.23 | 0.770 | 0.851 | 0.770 | 0.492 | 0.067 | 0.033 / 0.000 |
| probability | 64 | 0.859 | 1.30 | 0.734 | 0.851 | 0.734 | 0.203 | 0.000 | 0.109 / 0.000 |
| routing | 62 | 0.935 | 1.13 | 0.871 | 0.926 | 0.871 | 0.645 | 0.000 | 0.419 / 0.000 |
| severity | 54 | 0.870 | 1.15 | 0.852 | 0.870 | 0.852 | 0.667 | 0.083 | 0.000 / - |
| state_tracking | 50 | 0.880 | 1.10 | 0.900 | 0.867 | 0.800 | 0.700 | 0.000 | 0.420 / 0.000 |
| temporal_numeric | 62 | 0.806 | 1.29 | 0.710 | 0.750 | 0.565 | 0.161 | 0.000 | 0.145 / 0.000 |
| tool_selection | 45 | 1.000 | 1.00 | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | 1.000 / 0.000 |
| tradeoff | 69 | 0.783 | 1.30 | 0.710 | 0.735 | 0.696 | 0.116 | 0.125 | 0.043 / 0.000 |
| trap | 62 | 0.984 | 1.06 | 0.935 | 0.983 | 0.968 | 0.742 | 0.022 | 0.145 / 0.000 |
LTT-half attribution: best lambda by p {"lambda": 0.98, "n_act": 332, "errors": 1, "p": 7.426607492709305e-07}; Bonferroni level 9.90e-04; zero-error acts needed 135 (single test 45); LTT singleton rate 0.844; per-type lambdas {'choice': 0.97, 'noul': None, 'score': None}; FS-CRR c1 stop {"lambda": 0.94, "reason": "not rejected", "n_act": 671, "errors": 21, "p_value": 0.10034011309407308, "log_capital": 2.2991897328181037, "weighted_error": 0.03129657228017884}; FS-CRR c8 lam None stop {"lambda": 0.95, "reason": "not rejected", "n_act": 626, "errors": 14, "p_value": 0.19223282077029605, "log_capital": 1.6490480333912205, "weighted_error": 0.022364217252396165}

per type / family, variant T, alpha 0.05 (groups {"choice": {"n_fit": 507, "qhat": 0.9043797540237527}, "noul": {"n_fit": 332, "qhat": 0.9137012337134511}, "score": {"n_fit": 233, "qhat": 0.9100603282102937}}; pooled-small [] full-set False)
| cell | n | cov | size | singleton | sing acc | top-1 | act | acted err | per-type-LTT act / err |
|---|---|---|---|---|---|---|---|---|---|
| choice | 511 | 0.951 | 1.24 | 0.828 | 0.957 | 0.883 | 0.681 | 0.009 | 0.593 / 0.003 |
| noul | 343 | 0.962 | 1.33 | 0.668 | 0.943 | 0.825 | 0.586 | 0.025 | 0.000 / - |
| score | 222 | 0.919 | 1.49 | 0.617 | 0.912 | 0.815 | 0.419 | 0.043 | 0.000 / - |
| adequacy | 48 | 0.938 | 1.29 | 0.708 | 0.941 | 0.833 | 0.583 | 0.036 | 0.062 / 0.000 |
| adversarial | 55 | 0.927 | 1.45 | 0.582 | 0.938 | 0.855 | 0.364 | 0.000 | 0.164 / 0.000 |
| ambiguous | 73 | 0.986 | 1.00 | 1.000 | 0.986 | 0.986 | 0.932 | 0.015 | 0.630 / 0.000 |
| enum | 78 | 1.000 | 1.01 | 0.987 | 1.000 | 1.000 | 0.987 | 0.000 | 0.756 / 0.000 |
| fact | 63 | 0.984 | 1.05 | 0.952 | 0.983 | 0.968 | 0.937 | 0.017 | 0.000 / - |
| intent | 46 | 1.000 | 1.09 | 0.935 | 1.000 | 1.000 | 0.870 | 0.000 | 0.761 / 0.000 |
| judge_hard | 62 | 0.968 | 1.58 | 0.419 | 0.923 | 0.694 | 0.371 | 0.000 | 0.000 / - |
| long_policy | 69 | 0.957 | 1.32 | 0.681 | 0.957 | 0.884 | 0.522 | 0.028 | 0.130 / 0.000 |
| multi_hop | 53 | 0.925 | 1.23 | 0.792 | 0.905 | 0.849 | 0.528 | 0.036 | 0.377 / 0.050 |
| policy | 61 | 0.918 | 1.33 | 0.672 | 0.878 | 0.770 | 0.492 | 0.067 | 0.033 / 0.000 |
| probability | 64 | 0.906 | 1.59 | 0.516 | 0.909 | 0.734 | 0.203 | 0.000 | 0.109 / 0.000 |
| routing | 62 | 0.968 | 1.21 | 0.806 | 0.960 | 0.871 | 0.645 | 0.000 | 0.419 / 0.000 |
| severity | 54 | 0.889 | 1.22 | 0.815 | 0.886 | 0.852 | 0.667 | 0.083 | 0.000 / - |
| state_tracking | 50 | 0.980 | 1.58 | 0.740 | 1.000 | 0.800 | 0.700 | 0.000 | 0.420 / 0.000 |
| temporal_numeric | 62 | 0.935 | 1.85 | 0.355 | 0.864 | 0.565 | 0.161 | 0.000 | 0.145 / 0.000 |
| tool_selection | 45 | 1.000 | 1.00 | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | 1.000 / 0.000 |
| tradeoff | 69 | 0.812 | 1.83 | 0.406 | 0.643 | 0.696 | 0.116 | 0.125 | 0.043 / 0.000 |
| trap | 62 | 0.984 | 1.11 | 0.887 | 0.982 | 0.968 | 0.742 | 0.022 | 0.145 / 0.000 |
LTT-half attribution: best lambda by p {"lambda": 0.98, "n_act": 332, "errors": 1, "p": 7.426607492709305e-07}; Bonferroni level 9.90e-04; zero-error acts needed 135 (single test 45); LTT singleton rate 0.720; per-type lambdas {'choice': 0.97, 'noul': None, 'score': None}; FS-CRR c1 stop {"lambda": 0.94, "reason": "not rejected", "n_act": 671, "errors": 21, "p_value": 0.10034011309407308, "log_capital": 2.2991897328181037, "weighted_error": 0.03129657228017884}; FS-CRR c8 lam None stop {"lambda": 0.95, "reason": "not rejected", "n_act": 626, "errors": 14, "p_value": 0.19223282077029605, "log_capital": 1.6490480333912205, "weighted_error": 0.022364217252396165}

### h2o_q668q  (n 3600, top-1 0.8922, dev-card 360)
shim parity max |d| 0.00e+00

**h2o_q668q / native / all** FIT 1203 LTT 1217 TEST 1180; types {'FIT': {'choice': 583, 'noul': 367, 'score': 253}, 'LTT': {'choice': 566, 'noul': 362, 'score': 289}, 'TEST': {'choice': 566, 'noul': 377, 'score': 237}}
| alpha | var | TEST cov | flag | size | singleton | sing acc | lambda* (Bonf) | TEST act | acted err | FS-CRR c1 lam / act / err | FS-binom lam / act / err | per-type LTT act / err |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.1 | M | 0.922 | ok | 1.04 | 0.947 | 0.922 | 0.85 | 0.805 | 0.036 | 0.84 / 0.815 / 0.037 | 0.79 / 0.837 / 0.041 | 0.375 / 0.011 |
| 0.1 | T | 0.918 | ok | 1.04 | 0.948 | 0.921 | 0.85 | 0.805 | 0.036 | 0.84 / 0.815 / 0.037 | 0.79 / 0.837 / 0.041 | 0.375 / 0.011 |
| 0.1 | F | 0.922 | ok | 1.04 | 0.947 | 0.922 | 0.85 | 0.805 | 0.036 | 0.84 / 0.815 / 0.037 | 0.79 / 0.837 / 0.041 | 0.375 / 0.011 |
| 0.05 | M | 0.957 | ok | 1.18 | 0.839 | 0.957 | 0.85 | 0.805 | 0.036 | 0.83 / 0.816 / 0.038 | 0.77 / 0.828 / 0.040 | 0.375 / 0.011 |
| 0.05 | T | 0.956 | ok | 1.18 | 0.844 | 0.955 | 0.85 | 0.805 | 0.036 | 0.84 / 0.814 / 0.037 | 0.78 / 0.837 / 0.044 | 0.375 / 0.011 |
| 0.05 | F | 0.957 | ok | 1.18 | 0.839 | 0.957 | 0.85 | 0.805 | 0.036 | 0.83 / 0.816 / 0.038 | 0.77 / 0.828 / 0.040 | 0.375 / 0.011 |

per type / family, variant T, alpha 0.1 (groups {"choice": {"n_fit": 583, "qhat": 0.5918274704681019}, "noul": {"n_fit": 367, "qhat": 0.6042874133596858}, "score": {"n_fit": 253, "qhat": 0.6757628614727532}}; pooled-small [] full-set False)
| cell | n | cov | size | singleton | sing acc | top-1 | act | acted err | per-type-LTT act / err |
|---|---|---|---|---|---|---|---|---|---|
| choice | 566 | 0.920 | 1.00 | 0.975 | 0.933 | 0.917 | 0.829 | 0.017 | 0.783 / 0.011 |
| noul | 377 | 0.915 | 1.10 | 0.899 | 0.906 | 0.862 | 0.780 | 0.061 | 0.000 / - |
| score | 237 | 0.916 | 1.04 | 0.962 | 0.917 | 0.903 | 0.789 | 0.043 | 0.000 / - |
| adequacy | 63 | 0.825 | 1.03 | 0.968 | 0.820 | 0.810 | 0.746 | 0.106 | 0.206 / 0.077 |
| adversarial | 55 | 0.982 | 1.04 | 0.964 | 0.981 | 0.964 | 0.873 | 0.000 | 0.218 / 0.000 |
| ambiguous | 75 | 1.000 | 1.00 | 1.000 | 1.000 | 1.000 | 0.987 | 0.000 | 0.773 / 0.000 |
| enum | 80 | 1.000 | 1.01 | 0.988 | 1.000 | 1.000 | 0.988 | 0.000 | 0.775 / 0.000 |
| fact | 71 | 0.972 | 1.00 | 1.000 | 0.972 | 0.972 | 0.944 | 0.015 | 0.000 / - |
| intent | 68 | 0.971 | 1.00 | 1.000 | 0.971 | 0.971 | 0.941 | 0.000 | 0.765 / 0.000 |
| judge_hard | 70 | 0.943 | 1.36 | 0.643 | 0.911 | 0.757 | 0.486 | 0.029 | 0.000 / - |
| long_policy | 70 | 0.900 | 1.01 | 0.986 | 0.899 | 0.900 | 0.843 | 0.068 | 0.257 / 0.000 |
| multi_hop | 57 | 0.895 | 1.02 | 0.982 | 0.893 | 0.877 | 0.895 | 0.098 | 0.772 / 0.091 |
| policy | 67 | 0.881 | 1.07 | 0.925 | 0.871 | 0.836 | 0.776 | 0.096 | 0.104 / 0.000 |
| probability | 67 | 0.925 | 1.04 | 0.955 | 0.922 | 0.910 | 0.597 | 0.050 | 0.358 / 0.000 |
| routing | 68 | 0.926 | 1.00 | 1.000 | 0.926 | 0.926 | 0.882 | 0.017 | 0.632 / 0.000 |
| severity | 63 | 0.921 | 1.03 | 0.968 | 0.934 | 0.921 | 0.825 | 0.000 | 0.000 / - |
| state_tracking | 54 | 0.889 | 0.94 | 0.907 | 0.959 | 0.889 | 0.796 | 0.000 | 0.481 / 0.000 |
| temporal_numeric | 66 | 0.742 | 1.05 | 0.833 | 0.764 | 0.682 | 0.485 | 0.125 | 0.303 / 0.000 |
| tool_selection | 53 | 1.000 | 1.00 | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | 1.000 / 0.000 |
| tradeoff | 70 | 0.771 | 1.04 | 0.957 | 0.761 | 0.757 | 0.457 | 0.125 | 0.029 / 0.000 |
| trap | 63 | 0.968 | 1.00 | 1.000 | 0.968 | 0.968 | 1.000 | 0.032 | 0.143 / 0.000 |
LTT-half attribution: best lambda by p {"lambda": 0.94, "n_act": 808, "errors": 10, "p": 6.564342554959548e-09}; Bonferroni level 9.90e-04; zero-error acts needed 135 (single test 45); LTT singleton rate 0.961; per-type lambdas {'choice': 0.9, 'noul': None, 'score': None}; FS-CRR c1 stop {"lambda": 0.83, "reason": "not rejected", "n_act": 1021, "errors": 36, "p_value": 0.16810241396870707, "log_capital": 1.7831818783571296, "weighted_error": 0.03525954946131244}; FS-CRR c8 lam 0.88 stop {"lambda": 0.87, "reason": "not rejected", "n_act": 975, "errors": 26, "p_value": 0.12703775719547075, "log_capital": 2.063270935956919, "weighted_error": 0.02666666666666667}

per type / family, variant T, alpha 0.05 (groups {"choice": {"n_fit": 583, "qhat": 0.8692410296092177}, "noul": {"n_fit": 367, "qhat": 0.7836277600549304}, "score": {"n_fit": 253, "qhat": 0.860744943242046}}; pooled-small [] full-set False)
| cell | n | cov | size | singleton | sing acc | top-1 | act | acted err | per-type-LTT act / err |
|---|---|---|---|---|---|---|---|---|---|
| choice | 566 | 0.970 | 1.19 | 0.860 | 0.971 | 0.917 | 0.829 | 0.017 | 0.783 / 0.011 |
| noul | 377 | 0.939 | 1.17 | 0.828 | 0.926 | 0.862 | 0.780 | 0.061 | 0.000 / - |
| score | 237 | 0.949 | 1.18 | 0.831 | 0.959 | 0.903 | 0.789 | 0.043 | 0.000 / - |
| adequacy | 63 | 0.905 | 1.25 | 0.778 | 0.878 | 0.810 | 0.746 | 0.106 | 0.206 / 0.077 |
| adversarial | 55 | 1.000 | 1.09 | 0.909 | 1.000 | 0.964 | 0.873 | 0.000 | 0.218 / 0.000 |
| ambiguous | 75 | 1.000 | 1.01 | 0.987 | 1.000 | 1.000 | 0.987 | 0.000 | 0.773 / 0.000 |
| enum | 80 | 1.000 | 1.01 | 0.988 | 1.000 | 1.000 | 0.988 | 0.000 | 0.775 / 0.000 |
| fact | 71 | 0.986 | 1.03 | 0.972 | 0.986 | 0.972 | 0.944 | 0.015 | 0.000 / - |
| intent | 68 | 0.985 | 1.04 | 0.956 | 1.000 | 0.971 | 0.941 | 0.000 | 0.765 / 0.000 |
| judge_hard | 70 | 0.986 | 1.49 | 0.514 | 0.972 | 0.757 | 0.486 | 0.029 | 0.000 / - |
| long_policy | 70 | 0.929 | 1.07 | 0.929 | 0.923 | 0.900 | 0.843 | 0.068 | 0.257 / 0.000 |
| multi_hop | 57 | 0.912 | 1.11 | 0.912 | 0.904 | 0.877 | 0.895 | 0.098 | 0.772 / 0.091 |
| policy | 67 | 0.910 | 1.18 | 0.821 | 0.891 | 0.836 | 0.776 | 0.096 | 0.104 / 0.000 |
| probability | 67 | 0.970 | 1.34 | 0.687 | 0.957 | 0.910 | 0.597 | 0.050 | 0.358 / 0.000 |
| routing | 68 | 0.941 | 1.07 | 0.926 | 0.937 | 0.926 | 0.882 | 0.017 | 0.632 / 0.000 |
| severity | 63 | 0.952 | 1.16 | 0.841 | 1.000 | 0.921 | 0.825 | 0.000 | 0.000 / - |
| state_tracking | 54 | 0.981 | 1.37 | 0.833 | 1.000 | 0.889 | 0.796 | 0.000 | 0.481 / 0.000 |
| temporal_numeric | 66 | 0.894 | 1.61 | 0.561 | 0.811 | 0.682 | 0.485 | 0.125 | 0.303 / 0.000 |
| tool_selection | 53 | 1.000 | 1.00 | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | 1.000 / 0.000 |
| tradeoff | 70 | 0.886 | 1.44 | 0.600 | 0.857 | 0.757 | 0.457 | 0.125 | 0.029 / 0.000 |
| trap | 63 | 0.968 | 1.00 | 1.000 | 0.968 | 0.968 | 1.000 | 0.032 | 0.143 / 0.000 |
LTT-half attribution: best lambda by p {"lambda": 0.94, "n_act": 808, "errors": 10, "p": 6.564342554959548e-09}; Bonferroni level 9.90e-04; zero-error acts needed 135 (single test 45); LTT singleton rate 0.855; per-type lambdas {'choice': 0.9, 'noul': None, 'score': None}; FS-CRR c1 stop {"lambda": 0.83, "reason": "not rejected", "n_act": 1018, "errors": 36, "p_value": 0.17694709985653154, "log_capital": 1.7319044619461597, "weighted_error": 0.03536345776031434}; FS-CRR c8 lam 0.88 stop {"lambda": 0.87, "reason": "not rejected", "n_act": 975, "errors": 26, "p_value": 0.12703775719547075, "log_capital": 2.063270935956919, "weighted_error": 0.02666666666666667}

**h2o_q668q / native / remainder** FIT 1072 LTT 1092 TEST 1076; types {'FIT': {'choice': 507, 'noul': 332, 'score': 233}, 'LTT': {'choice': 500, 'noul': 321, 'score': 271}, 'TEST': {'choice': 511, 'noul': 343, 'score': 222}}
| alpha | var | TEST cov | flag | size | singleton | sing acc | lambda* (Bonf) | TEST act | acted err | FS-CRR c1 lam / act / err | FS-binom lam / act / err | per-type LTT act / err |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.1 | M | 0.923 | ok | 1.04 | 0.951 | 0.925 | 0.85 | 0.813 | 0.037 | 0.84 / 0.823 / 0.038 | 0.79 / 0.845 / 0.042 | 0.380 / 0.012 |
| 0.1 | T | 0.920 | ok | 1.03 | 0.947 | 0.927 | 0.85 | 0.813 | 0.037 | 0.84 / 0.823 / 0.038 | 0.79 / 0.845 / 0.042 | 0.380 / 0.012 |
| 0.1 | F | 0.923 | ok | 1.04 | 0.951 | 0.925 | 0.85 | 0.813 | 0.037 | 0.84 / 0.823 / 0.038 | 0.79 / 0.845 / 0.042 | 0.380 / 0.012 |
| 0.05 | M | 0.949 | ok | 1.15 | 0.861 | 0.948 | 0.85 | 0.813 | 0.037 | 0.84 / 0.823 / 0.038 | 0.78 / 0.840 / 0.042 | 0.380 / 0.012 |
| 0.05 | T | 0.956 | ok | 1.17 | 0.856 | 0.953 | 0.85 | 0.813 | 0.037 | 0.84 / 0.823 / 0.038 | 0.76 / 0.848 / 0.044 | 0.380 / 0.012 |
| 0.05 | F | 0.949 | ok | 1.15 | 0.861 | 0.948 | 0.85 | 0.813 | 0.037 | 0.84 / 0.823 / 0.038 | 0.78 / 0.840 / 0.042 | 0.380 / 0.012 |

per type / family, variant T, alpha 0.1 (groups {"choice": {"n_fit": 507, "qhat": 0.5327452039545777}, "noul": {"n_fit": 332, "qhat": 0.6042874133596858}, "score": {"n_fit": 233, "qhat": 0.6235865573451943}}; pooled-small [] full-set False)
| cell | n | cov | size | singleton | sing acc | top-1 | act | acted err | per-type-LTT act / err |
|---|---|---|---|---|---|---|---|---|---|
| choice | 511 | 0.916 | 0.98 | 0.977 | 0.938 | 0.922 | 0.834 | 0.016 | 0.800 / 0.012 |
| noul | 343 | 0.915 | 1.11 | 0.889 | 0.905 | 0.857 | 0.773 | 0.064 | 0.000 / - |
| score | 222 | 0.937 | 1.03 | 0.968 | 0.935 | 0.923 | 0.829 | 0.043 | 0.000 / - |
| adequacy | 48 | 0.875 | 1.00 | 1.000 | 0.875 | 0.875 | 0.917 | 0.114 | 0.229 / 0.091 |
| adversarial | 55 | 0.982 | 1.04 | 0.964 | 0.981 | 0.964 | 0.873 | 0.000 | 0.218 / 0.000 |
| ambiguous | 73 | 1.000 | 1.00 | 1.000 | 1.000 | 1.000 | 0.986 | 0.000 | 0.781 / 0.000 |
| enum | 78 | 1.000 | 1.00 | 1.000 | 1.000 | 1.000 | 0.987 | 0.000 | 0.769 / 0.000 |
| fact | 63 | 0.968 | 1.00 | 1.000 | 0.968 | 0.968 | 0.952 | 0.017 | 0.000 / - |
| intent | 46 | 0.978 | 1.00 | 1.000 | 0.978 | 0.978 | 0.935 | 0.000 | 0.783 / 0.000 |
| judge_hard | 62 | 0.952 | 1.40 | 0.597 | 0.919 | 0.742 | 0.419 | 0.000 | 0.000 / - |
| long_policy | 69 | 0.913 | 1.00 | 1.000 | 0.913 | 0.913 | 0.855 | 0.068 | 0.261 / 0.000 |
| multi_hop | 53 | 0.868 | 1.00 | 1.000 | 0.868 | 0.868 | 0.906 | 0.104 | 0.792 / 0.095 |
| policy | 61 | 0.869 | 1.08 | 0.918 | 0.857 | 0.820 | 0.787 | 0.104 | 0.115 / 0.000 |
| probability | 64 | 0.922 | 1.05 | 0.953 | 0.918 | 0.906 | 0.594 | 0.053 | 0.359 / 0.000 |
| routing | 62 | 0.919 | 1.00 | 1.000 | 0.919 | 0.919 | 0.871 | 0.019 | 0.661 / 0.000 |
| severity | 54 | 0.981 | 1.02 | 0.981 | 0.981 | 0.981 | 0.944 | 0.000 | 0.000 / - |
| state_tracking | 50 | 0.880 | 0.92 | 0.920 | 0.957 | 0.900 | 0.800 | 0.000 | 0.540 / 0.000 |
| temporal_numeric | 62 | 0.726 | 1.02 | 0.790 | 0.776 | 0.677 | 0.468 | 0.138 | 0.290 / 0.000 |
| tool_selection | 45 | 1.000 | 1.00 | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | 1.000 / 0.000 |
| tradeoff | 69 | 0.768 | 1.00 | 0.942 | 0.785 | 0.768 | 0.449 | 0.097 | 0.043 / 0.000 |
| trap | 62 | 0.968 | 1.00 | 1.000 | 0.968 | 0.968 | 1.000 | 0.032 | 0.145 / 0.000 |
LTT-half attribution: best lambda by p {"lambda": 0.94, "n_act": 733, "errors": 9, "p": 3.01961188416502e-08}; Bonferroni level 9.90e-04; zero-error acts needed 135 (single test 45); LTT singleton rate 0.962; per-type lambdas {'choice': 0.89, 'noul': None, 'score': None}; FS-CRR c1 stop {"lambda": 0.83, "reason": "not rejected", "n_act": 923, "errors": 31, "p_value": 0.1185606867383909, "log_capital": 2.1323303251232772, "weighted_error": 0.03358613217768147}; FS-CRR c8 lam 0.89 stop {"lambda": 0.88, "reason": "not rejected", "n_act": 872, "errors": 20, "p_value": 0.11282117271790053, "log_capital": 2.181951256132055, "weighted_error": 0.022935779816513763}

per type / family, variant T, alpha 0.05 (groups {"choice": {"n_fit": 507, "qhat": 0.8692410296092177}, "noul": {"n_fit": 332, "qhat": 0.7978596740767063}, "score": {"n_fit": 233, "qhat": 0.8211847619229006}}; pooled-small [] full-set False)
| cell | n | cov | size | singleton | sing acc | top-1 | act | acted err | per-type-LTT act / err |
|---|---|---|---|---|---|---|---|---|---|
| choice | 511 | 0.969 | 1.18 | 0.869 | 0.971 | 0.922 | 0.834 | 0.016 | 0.800 / 0.012 |
| noul | 343 | 0.945 | 1.19 | 0.810 | 0.932 | 0.857 | 0.773 | 0.064 | 0.000 / - |
| score | 222 | 0.946 | 1.11 | 0.896 | 0.945 | 0.923 | 0.829 | 0.043 | 0.000 / - |
| adequacy | 48 | 0.875 | 1.02 | 0.979 | 0.872 | 0.875 | 0.917 | 0.114 | 0.229 / 0.091 |
| adversarial | 55 | 1.000 | 1.09 | 0.909 | 1.000 | 0.964 | 0.873 | 0.000 | 0.218 / 0.000 |
| ambiguous | 73 | 1.000 | 1.01 | 0.986 | 1.000 | 1.000 | 0.986 | 0.000 | 0.781 / 0.000 |
| enum | 78 | 1.000 | 1.01 | 0.987 | 1.000 | 1.000 | 0.987 | 0.000 | 0.769 / 0.000 |
| fact | 63 | 0.984 | 1.03 | 0.968 | 0.984 | 0.968 | 0.952 | 0.017 | 0.000 / - |
| intent | 46 | 0.978 | 1.04 | 0.957 | 1.000 | 0.978 | 0.935 | 0.000 | 0.783 / 0.000 |
| judge_hard | 62 | 1.000 | 1.55 | 0.452 | 1.000 | 0.742 | 0.419 | 0.000 | 0.000 / - |
| long_policy | 69 | 0.942 | 1.09 | 0.913 | 0.937 | 0.913 | 0.855 | 0.068 | 0.261 / 0.000 |
| multi_hop | 53 | 0.906 | 1.09 | 0.925 | 0.898 | 0.868 | 0.906 | 0.104 | 0.792 / 0.095 |
| policy | 61 | 0.902 | 1.16 | 0.836 | 0.882 | 0.820 | 0.787 | 0.104 | 0.115 / 0.000 |
| probability | 64 | 0.969 | 1.33 | 0.703 | 0.956 | 0.906 | 0.594 | 0.053 | 0.359 / 0.000 |
| routing | 62 | 0.952 | 1.10 | 0.903 | 0.946 | 0.919 | 0.871 | 0.019 | 0.661 / 0.000 |
| severity | 54 | 0.981 | 1.04 | 0.963 | 0.981 | 0.981 | 0.944 | 0.000 | 0.000 / - |
| state_tracking | 50 | 0.980 | 1.38 | 0.840 | 1.000 | 0.900 | 0.800 | 0.000 | 0.540 / 0.000 |
| temporal_numeric | 62 | 0.887 | 1.61 | 0.548 | 0.794 | 0.677 | 0.468 | 0.138 | 0.290 / 0.000 |
| tool_selection | 45 | 1.000 | 1.00 | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | 1.000 / 0.000 |
| tradeoff | 69 | 0.884 | 1.42 | 0.623 | 0.860 | 0.768 | 0.449 | 0.097 | 0.043 / 0.000 |
| trap | 62 | 0.968 | 1.00 | 1.000 | 0.968 | 0.968 | 1.000 | 0.032 | 0.145 / 0.000 |
LTT-half attribution: best lambda by p {"lambda": 0.94, "n_act": 733, "errors": 9, "p": 3.01961188416502e-08}; Bonferroni level 9.90e-04; zero-error acts needed 135 (single test 45); LTT singleton rate 0.869; per-type lambdas {'choice': 0.89, 'noul': None, 'score': None}; FS-CRR c1 stop {"lambda": 0.83, "reason": "not rejected", "n_act": 920, "errors": 31, "p_value": 0.12507953232885288, "log_capital": 2.0788054853757187, "weighted_error": 0.03369565217391304}; FS-CRR c8 lam 0.89 stop {"lambda": 0.88, "reason": "not rejected", "n_act": 872, "errors": 20, "p_value": 0.11282117271790053, "log_capital": 2.181951256132055, "weighted_error": 0.022935779816513763}

**h2o_q668q / native_shimcal / all** FIT 1203 LTT 1217 TEST 1180; types {'FIT': {'choice': 583, 'noul': 367, 'score': 253}, 'LTT': {'choice': 566, 'noul': 362, 'score': 289}, 'TEST': {'choice': 566, 'noul': 377, 'score': 237}}
| alpha | var | TEST cov | flag | size | singleton | sing acc | lambda* (Bonf) | TEST act | acted err | FS-CRR c1 lam / act / err | FS-binom lam / act / err | per-type LTT act / err |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.1 | M | 0.909 | ok | 1.03 | 0.964 | 0.910 | 0.92 | 0.797 | 0.032 | 0.90 / 0.813 / 0.038 | 0.87 / 0.831 / 0.041 | 0.381 / 0.011 |
| 0.1 | T | 0.925 | ok | 1.06 | 0.930 | 0.923 | 0.92 | 0.797 | 0.032 | 0.90 / 0.813 / 0.038 | 0.87 / 0.831 / 0.041 | 0.381 / 0.011 |
| 0.1 | F | 0.909 | ok | 1.03 | 0.964 | 0.910 | 0.92 | 0.797 | 0.032 | 0.90 / 0.813 / 0.038 | 0.87 / 0.831 / 0.041 | 0.381 / 0.011 |
| 0.05 | M | 0.958 | ok | 1.18 | 0.841 | 0.957 | 0.92 | 0.797 | 0.032 | 0.90 / 0.813 / 0.038 | 0.86 / 0.830 / 0.040 | 0.381 / 0.011 |
| 0.05 | T | 0.956 | ok | 1.19 | 0.843 | 0.956 | 0.92 | 0.797 | 0.032 | 0.90 / 0.813 / 0.038 | 0.85 / 0.836 / 0.042 | 0.381 / 0.011 |
| 0.05 | F | 0.958 | ok | 1.18 | 0.841 | 0.957 | 0.92 | 0.797 | 0.032 | 0.90 / 0.813 / 0.038 | 0.86 / 0.830 / 0.040 | 0.381 / 0.011 |

per type / family, variant T, alpha 0.1 (groups {"choice": {"n_fit": 583, "qhat": 0.5947006507669537}, "noul": {"n_fit": 367, "qhat": 0.801}, "score": {"n_fit": 253, "qhat": 0.6959238096255562}}; pooled-small [] full-set False)
| cell | n | cov | size | singleton | sing acc | top-1 | act | acted err | per-type-LTT act / err |
|---|---|---|---|---|---|---|---|---|---|
| choice | 566 | 0.920 | 1.00 | 0.982 | 0.926 | 0.917 | 0.829 | 0.017 | 0.793 / 0.011 |
| noul | 377 | 0.936 | 1.17 | 0.830 | 0.923 | 0.862 | 0.756 | 0.053 | 0.000 / - |
| score | 237 | 0.916 | 1.04 | 0.962 | 0.917 | 0.903 | 0.789 | 0.037 | 0.000 / - |
| adequacy | 63 | 0.825 | 1.03 | 0.968 | 0.820 | 0.810 | 0.730 | 0.109 | 0.206 / 0.077 |
| adversarial | 55 | 0.982 | 1.04 | 0.964 | 0.981 | 0.964 | 0.873 | 0.000 | 0.218 / 0.000 |
| ambiguous | 75 | 1.000 | 1.00 | 1.000 | 1.000 | 1.000 | 0.987 | 0.000 | 0.773 / 0.000 |
| enum | 80 | 1.000 | 1.01 | 0.988 | 1.000 | 1.000 | 0.988 | 0.000 | 0.775 / 0.000 |
| fact | 71 | 0.986 | 1.03 | 0.972 | 0.986 | 0.972 | 0.944 | 0.015 | 0.000 / - |
| intent | 68 | 0.971 | 1.00 | 1.000 | 0.971 | 0.971 | 0.926 | 0.000 | 0.779 / 0.000 |
| judge_hard | 70 | 0.986 | 1.49 | 0.514 | 0.972 | 0.757 | 0.471 | 0.030 | 0.000 / - |
| long_policy | 70 | 0.914 | 1.04 | 0.957 | 0.910 | 0.900 | 0.829 | 0.069 | 0.257 / 0.000 |
| multi_hop | 57 | 0.895 | 1.04 | 0.965 | 0.891 | 0.877 | 0.860 | 0.082 | 0.772 / 0.091 |
| policy | 67 | 0.910 | 1.18 | 0.821 | 0.891 | 0.836 | 0.776 | 0.096 | 0.104 / 0.000 |
| probability | 67 | 0.925 | 1.04 | 0.955 | 0.922 | 0.910 | 0.597 | 0.050 | 0.343 / 0.000 |
| routing | 68 | 0.926 | 1.00 | 1.000 | 0.926 | 0.926 | 0.882 | 0.017 | 0.676 / 0.000 |
| severity | 63 | 0.921 | 1.03 | 0.968 | 0.934 | 0.921 | 0.825 | 0.000 | 0.000 / - |
| state_tracking | 54 | 0.889 | 0.94 | 0.907 | 0.959 | 0.889 | 0.796 | 0.000 | 0.500 / 0.000 |
| temporal_numeric | 66 | 0.742 | 1.15 | 0.848 | 0.696 | 0.682 | 0.439 | 0.069 | 0.318 / 0.000 |
| tool_selection | 53 | 1.000 | 1.00 | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | 1.000 / 0.000 |
| tradeoff | 70 | 0.786 | 1.07 | 0.929 | 0.769 | 0.757 | 0.457 | 0.094 | 0.043 / 0.000 |
| trap | 63 | 0.968 | 1.00 | 1.000 | 0.968 | 0.968 | 1.000 | 0.032 | 0.143 / 0.000 |
LTT-half attribution: best lambda by p {"lambda": 0.98, "n_act": 731, "errors": 7, "p": 1.521422822353103e-09}; Bonferroni level 9.90e-04; zero-error acts needed 135 (single test 45); LTT singleton rate 0.947; per-type lambdas {'choice': 0.95, 'noul': None, 'score': None}; FS-CRR c1 stop {"lambda": 0.89, "reason": "not rejected", "n_act": 1021, "errors": 36, "p_value": 0.15806832178169034, "log_capital": 1.8447279230803606, "weighted_error": 0.03525954946131244}; FS-CRR c8 lam 0.94 stop {"lambda": 0.93, "reason": "not rejected", "n_act": 968, "errors": 25, "p_value": 0.11857868319601439, "log_capital": 2.1321785455376014, "weighted_error": 0.025826446280991736}

per type / family, variant T, alpha 0.05 (groups {"choice": {"n_fit": 583, "qhat": 0.9057368154885082}, "noul": {"n_fit": 367, "qhat": 0.8332262788044943}, "score": {"n_fit": 253, "qhat": 0.9043047029226199}}; pooled-small [] full-set False)
| cell | n | cov | size | singleton | sing acc | top-1 | act | acted err | per-type-LTT act / err |
|---|---|---|---|---|---|---|---|---|---|
| choice | 566 | 0.970 | 1.19 | 0.860 | 0.973 | 0.917 | 0.829 | 0.017 | 0.793 / 0.011 |
| noul | 377 | 0.939 | 1.17 | 0.828 | 0.926 | 0.862 | 0.756 | 0.053 | 0.000 / - |
| score | 237 | 0.949 | 1.19 | 0.827 | 0.959 | 0.903 | 0.789 | 0.037 | 0.000 / - |
| adequacy | 63 | 0.905 | 1.27 | 0.778 | 0.878 | 0.810 | 0.730 | 0.109 | 0.206 / 0.077 |
| adversarial | 55 | 1.000 | 1.09 | 0.909 | 1.000 | 0.964 | 0.873 | 0.000 | 0.218 / 0.000 |
| ambiguous | 75 | 1.000 | 1.01 | 0.987 | 1.000 | 1.000 | 0.987 | 0.000 | 0.773 / 0.000 |
| enum | 80 | 1.000 | 1.01 | 0.988 | 1.000 | 1.000 | 0.988 | 0.000 | 0.775 / 0.000 |
| fact | 71 | 0.986 | 1.03 | 0.972 | 0.986 | 0.972 | 0.944 | 0.015 | 0.000 / - |
| intent | 68 | 0.985 | 1.04 | 0.956 | 1.000 | 0.971 | 0.926 | 0.000 | 0.779 / 0.000 |
| judge_hard | 70 | 0.986 | 1.49 | 0.514 | 0.972 | 0.757 | 0.471 | 0.030 | 0.000 / - |
| long_policy | 70 | 0.929 | 1.07 | 0.929 | 0.923 | 0.900 | 0.829 | 0.069 | 0.257 / 0.000 |
| multi_hop | 57 | 0.912 | 1.11 | 0.912 | 0.904 | 0.877 | 0.860 | 0.082 | 0.772 / 0.091 |
| policy | 67 | 0.910 | 1.18 | 0.821 | 0.891 | 0.836 | 0.776 | 0.096 | 0.104 / 0.000 |
| probability | 67 | 0.970 | 1.34 | 0.687 | 0.957 | 0.910 | 0.597 | 0.050 | 0.343 / 0.000 |
| routing | 68 | 0.941 | 1.07 | 0.926 | 0.937 | 0.926 | 0.882 | 0.017 | 0.676 / 0.000 |
| severity | 63 | 0.952 | 1.16 | 0.841 | 1.000 | 0.921 | 0.825 | 0.000 | 0.000 / - |
| state_tracking | 54 | 0.981 | 1.39 | 0.833 | 1.000 | 0.889 | 0.796 | 0.000 | 0.500 / 0.000 |
| temporal_numeric | 66 | 0.894 | 1.64 | 0.576 | 0.816 | 0.682 | 0.439 | 0.069 | 0.318 / 0.000 |
| tool_selection | 53 | 1.000 | 1.00 | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | 1.000 / 0.000 |
| tradeoff | 70 | 0.886 | 1.47 | 0.571 | 0.875 | 0.757 | 0.457 | 0.094 | 0.043 / 0.000 |
| trap | 63 | 0.968 | 1.00 | 1.000 | 0.968 | 0.968 | 1.000 | 0.032 | 0.143 / 0.000 |
LTT-half attribution: best lambda by p {"lambda": 0.98, "n_act": 731, "errors": 7, "p": 1.521422822353103e-09}; Bonferroni level 9.90e-04; zero-error acts needed 135 (single test 45); LTT singleton rate 0.857; per-type lambdas {'choice': 0.95, 'noul': None, 'score': None}; FS-CRR c1 stop {"lambda": 0.89, "reason": "not rejected", "n_act": 1021, "errors": 36, "p_value": 0.15806832178169034, "log_capital": 1.8447279230803606, "weighted_error": 0.03525954946131244}; FS-CRR c8 lam 0.94 stop {"lambda": 0.93, "reason": "not rejected", "n_act": 968, "errors": 25, "p_value": 0.11857868319601439, "log_capital": 2.1321785455376014, "weighted_error": 0.025826446280991736}

**h2o_q668q / native_shimcal / remainder** FIT 1072 LTT 1092 TEST 1076; types {'FIT': {'choice': 507, 'noul': 332, 'score': 233}, 'LTT': {'choice': 500, 'noul': 321, 'score': 271}, 'TEST': {'choice': 511, 'noul': 343, 'score': 222}}
| alpha | var | TEST cov | flag | size | singleton | sing acc | lambda* (Bonf) | TEST act | acted err | FS-CRR c1 lam / act / err | FS-binom lam / act / err | per-type LTT act / err |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.1 | M | 0.912 | ok | 1.03 | 0.968 | 0.912 | 0.92 | 0.805 | 0.032 | 0.90 / 0.822 / 0.038 | 0.86 / 0.844 / 0.042 | 0.378 / 0.012 |
| 0.1 | T | 0.927 | ok | 1.05 | 0.922 | 0.935 | 0.92 | 0.805 | 0.032 | 0.90 / 0.822 / 0.038 | 0.86 / 0.844 / 0.042 | 0.378 / 0.012 |
| 0.1 | F | 0.912 | ok | 1.03 | 0.968 | 0.912 | 0.92 | 0.805 | 0.032 | 0.90 / 0.822 / 0.038 | 0.86 / 0.844 / 0.042 | 0.378 / 0.012 |
| 0.05 | M | 0.953 | ok | 1.16 | 0.861 | 0.949 | 0.92 | 0.805 | 0.032 | 0.90 / 0.822 / 0.038 | 0.86 / 0.842 / 0.042 | 0.378 / 0.012 |
| 0.05 | T | 0.956 | ok | 1.17 | 0.857 | 0.954 | 0.92 | 0.805 | 0.032 | 0.90 / 0.822 / 0.038 | 0.85 / 0.846 / 0.042 | 0.378 / 0.012 |
| 0.05 | F | 0.953 | ok | 1.16 | 0.861 | 0.949 | 0.92 | 0.805 | 0.032 | 0.90 / 0.822 / 0.038 | 0.86 / 0.842 / 0.042 | 0.378 / 0.012 |

per type / family, variant T, alpha 0.1 (groups {"choice": {"n_fit": 507, "qhat": 0.47632191248583444}, "noul": {"n_fit": 332, "qhat": 0.801}, "score": {"n_fit": 233, "qhat": 0.6422936373680126}}; pooled-small [] full-set False)
| cell | n | cov | size | singleton | sing acc | top-1 | act | acted err | per-type-LTT act / err |
|---|---|---|---|---|---|---|---|---|---|
| choice | 511 | 0.914 | 0.97 | 0.971 | 0.942 | 0.922 | 0.834 | 0.016 | 0.796 / 0.012 |
| noul | 343 | 0.939 | 1.18 | 0.819 | 0.925 | 0.857 | 0.746 | 0.055 | 0.000 / - |
| score | 222 | 0.937 | 1.03 | 0.968 | 0.935 | 0.923 | 0.829 | 0.038 | 0.000 / - |
| adequacy | 48 | 0.875 | 1.00 | 1.000 | 0.875 | 0.875 | 0.896 | 0.116 | 0.229 / 0.091 |
| adversarial | 55 | 0.982 | 1.04 | 0.964 | 0.981 | 0.964 | 0.873 | 0.000 | 0.218 / 0.000 |
| ambiguous | 73 | 1.000 | 1.00 | 1.000 | 1.000 | 1.000 | 0.986 | 0.000 | 0.781 / 0.000 |
| enum | 78 | 1.000 | 1.00 | 1.000 | 1.000 | 1.000 | 0.987 | 0.000 | 0.769 / 0.000 |
| fact | 63 | 0.984 | 1.03 | 0.968 | 0.984 | 0.968 | 0.952 | 0.017 | 0.000 / - |
| intent | 46 | 0.978 | 1.00 | 1.000 | 0.978 | 0.978 | 0.913 | 0.000 | 0.783 / 0.000 |
| judge_hard | 62 | 1.000 | 1.55 | 0.452 | 1.000 | 0.742 | 0.403 | 0.000 | 0.000 / - |
| long_policy | 69 | 0.913 | 1.01 | 0.957 | 0.924 | 0.913 | 0.841 | 0.069 | 0.261 / 0.000 |
| multi_hop | 53 | 0.868 | 1.02 | 0.981 | 0.865 | 0.868 | 0.868 | 0.087 | 0.774 / 0.098 |
| policy | 61 | 0.902 | 1.16 | 0.836 | 0.882 | 0.820 | 0.787 | 0.104 | 0.115 / 0.000 |
| probability | 64 | 0.922 | 1.03 | 0.938 | 0.933 | 0.906 | 0.594 | 0.053 | 0.344 / 0.000 |
| routing | 62 | 0.919 | 1.00 | 1.000 | 0.919 | 0.919 | 0.871 | 0.019 | 0.677 / 0.000 |
| severity | 54 | 0.981 | 1.02 | 0.981 | 0.981 | 0.981 | 0.944 | 0.000 | 0.000 / - |
| state_tracking | 50 | 0.880 | 0.92 | 0.920 | 0.957 | 0.900 | 0.800 | 0.000 | 0.520 / 0.000 |
| temporal_numeric | 62 | 0.726 | 1.06 | 0.742 | 0.761 | 0.677 | 0.419 | 0.077 | 0.290 / 0.000 |
| tool_selection | 45 | 1.000 | 1.00 | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | 1.000 / 0.000 |
| tradeoff | 69 | 0.783 | 1.01 | 0.899 | 0.806 | 0.768 | 0.449 | 0.065 | 0.043 / 0.000 |
| trap | 62 | 0.968 | 1.00 | 1.000 | 0.968 | 0.968 | 1.000 | 0.032 | 0.145 / 0.000 |
LTT-half attribution: best lambda by p {"lambda": 0.98, "n_act": 663, "errors": 6, "p": 5.009564510301394e-09}; Bonferroni level 9.90e-04; zero-error acts needed 135 (single test 45); LTT singleton rate 0.942; per-type lambdas {'choice': 0.95, 'noul': None, 'score': None}; FS-CRR c1 stop {"lambda": 0.89, "reason": "not rejected", "n_act": 923, "errors": 31, "p_value": 0.1111641800419021, "log_capital": 2.1967470709834074, "weighted_error": 0.03358613217768147}; FS-CRR c8 lam 0.94 stop {"lambda": 0.93, "reason": "not rejected", "n_act": 877, "errors": 22, "p_value": 0.13535729837704435, "log_capital": 1.9998373421218345, "weighted_error": 0.02508551881413911}

per type / family, variant T, alpha 0.05 (groups {"choice": {"n_fit": 507, "qhat": 0.9057368154885082}, "noul": {"n_fit": 332, "qhat": 0.8476404351200116}, "score": {"n_fit": 233, "qhat": 0.8631395551310793}}; pooled-small [] full-set False)
| cell | n | cov | size | singleton | sing acc | top-1 | act | acted err | per-type-LTT act / err |
|---|---|---|---|---|---|---|---|---|---|
| choice | 511 | 0.969 | 1.19 | 0.869 | 0.973 | 0.922 | 0.834 | 0.016 | 0.796 / 0.012 |
| noul | 343 | 0.945 | 1.19 | 0.810 | 0.932 | 0.857 | 0.746 | 0.055 | 0.000 / - |
| score | 222 | 0.946 | 1.10 | 0.901 | 0.945 | 0.923 | 0.829 | 0.038 | 0.000 / - |
| adequacy | 48 | 0.875 | 1.02 | 0.979 | 0.872 | 0.875 | 0.896 | 0.116 | 0.229 / 0.091 |
| adversarial | 55 | 1.000 | 1.07 | 0.927 | 1.000 | 0.964 | 0.873 | 0.000 | 0.218 / 0.000 |
| ambiguous | 73 | 1.000 | 1.01 | 0.986 | 1.000 | 1.000 | 0.986 | 0.000 | 0.781 / 0.000 |
| enum | 78 | 1.000 | 1.01 | 0.987 | 1.000 | 1.000 | 0.987 | 0.000 | 0.769 / 0.000 |
| fact | 63 | 0.984 | 1.03 | 0.968 | 0.984 | 0.968 | 0.952 | 0.017 | 0.000 / - |
| intent | 46 | 0.978 | 1.04 | 0.957 | 1.000 | 0.978 | 0.913 | 0.000 | 0.783 / 0.000 |
| judge_hard | 62 | 1.000 | 1.55 | 0.452 | 1.000 | 0.742 | 0.403 | 0.000 | 0.000 / - |
| long_policy | 69 | 0.942 | 1.09 | 0.913 | 0.937 | 0.913 | 0.841 | 0.069 | 0.261 / 0.000 |
| multi_hop | 53 | 0.906 | 1.09 | 0.925 | 0.898 | 0.868 | 0.868 | 0.087 | 0.774 / 0.098 |
| policy | 61 | 0.902 | 1.16 | 0.836 | 0.882 | 0.820 | 0.787 | 0.104 | 0.115 / 0.000 |
| probability | 64 | 0.969 | 1.33 | 0.703 | 0.956 | 0.906 | 0.594 | 0.053 | 0.344 / 0.000 |
| routing | 62 | 0.952 | 1.10 | 0.903 | 0.946 | 0.919 | 0.871 | 0.019 | 0.677 / 0.000 |
| severity | 54 | 0.981 | 1.04 | 0.963 | 0.981 | 0.981 | 0.944 | 0.000 | 0.000 / - |
| state_tracking | 50 | 0.980 | 1.40 | 0.840 | 1.000 | 0.900 | 0.800 | 0.000 | 0.520 / 0.000 |
| temporal_numeric | 62 | 0.887 | 1.65 | 0.565 | 0.800 | 0.677 | 0.419 | 0.077 | 0.290 / 0.000 |
| tool_selection | 45 | 1.000 | 1.00 | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | 1.000 / 0.000 |
| tradeoff | 69 | 0.884 | 1.43 | 0.609 | 0.881 | 0.768 | 0.449 | 0.065 | 0.043 / 0.000 |
| trap | 62 | 0.968 | 1.00 | 1.000 | 0.968 | 0.968 | 1.000 | 0.032 | 0.145 / 0.000 |
LTT-half attribution: best lambda by p {"lambda": 0.98, "n_act": 663, "errors": 6, "p": 5.009564510301394e-09}; Bonferroni level 9.90e-04; zero-error acts needed 135 (single test 45); LTT singleton rate 0.872; per-type lambdas {'choice': 0.95, 'noul': None, 'score': None}; FS-CRR c1 stop {"lambda": 0.89, "reason": "not rejected", "n_act": 923, "errors": 31, "p_value": 0.1111641800419021, "log_capital": 2.1967470709834074, "weighted_error": 0.03358613217768147}; FS-CRR c8 lam 0.94 stop {"lambda": 0.93, "reason": "not rejected", "n_act": 877, "errors": 22, "p_value": 0.13535729837704435, "log_capital": 1.9998373421218345, "weighted_error": 0.02508551881413911}
