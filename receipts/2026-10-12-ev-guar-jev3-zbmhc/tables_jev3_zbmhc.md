
### jev3lms_zbmhc  (n 1200, top-1 0.7117, dev-card 360)
shim parity max |d| 0.00e+00

**jev3lms_zbmhc / native_tfit / all** FIT 376 LTT 406 TEST 418; types {'FIT': {'choice': 138, 'noul': 222, 'score': 16}, 'LTT': {'choice': 141, 'noul': 250, 'score': 15}, 'TEST': {'choice': 137, 'noul': 269, 'score': 12}}
| alpha | var | TEST cov | flag | size | singleton | sing acc | lambda* (Bonf) | TEST act | acted err | FS-CRR c1 lam / act / err | FS-binom lam / act / err | per-type LTT act / err |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.1 | M | 0.904 | ok | 1.69 | 0.591 | 0.842 | - | 0.000 | - | - / 0.000 / - | - / 0.000 / - | 0.000 / - |
| 0.1 | T | 0.916 | ok | 1.61 | 0.526 | 0.868 | - | 0.000 | - | - / 0.000 / - | - / 0.000 / - | 0.000 / - |
| 0.1 | F | 0.904 | ok | 1.69 | 0.591 | 0.842 | - | 0.000 | - | - / 0.000 / - | - / 0.000 / - | 0.000 / - |
| 0.05 | M | 0.952 | ok | 2.19 | 0.383 | 0.875 | - | 0.000 | - | - / 0.000 / - | - / 0.000 / - | 0.000 / - |
| 0.05 | T | 0.955 | ok | 1.88 | 0.395 | 0.903 | - | 0.000 | - | - / 0.000 / - | - / 0.000 / - | 0.000 / - |
| 0.05 | F | 0.952 | ok | 2.19 | 0.383 | 0.875 | - | 0.000 | - | - / 0.000 / - | - / 0.000 / - | 0.000 / - |

per type / family, variant T, alpha 0.1 (groups {"choice": {"n_fit": 138, "qhat": 0.8564569624818352}, "noul": {"n_fit": 222, "qhat": 0.987212872236207}, "pooled-small": {"n_fit": 16, "qhat": Infinity}}; pooled-small ['score'] full-set True)
| cell | n | cov | size | singleton | sing acc | top-1 | act | acted err | per-type-LTT act / err |
|---|---|---|---|---|---|---|---|---|---|
| choice | 137 | 0.912 | 1.88 | 0.343 | 0.872 | 0.657 | 0.000 | - | 0.000 / - |
| noul | 269 | 0.914 | 1.36 | 0.643 | 0.867 | 0.770 | 0.000 | - | 0.000 / - |
| score | 12 | 1.000 | 4.42 | 0.000 | - | 0.667 | 0.000 | - | 0.000 / - |
| adequacy | 63 | 0.952 | 1.51 | 0.492 | 0.903 | 0.683 | 0.000 | - | 0.000 / - |
| ambiguous | 62 | 0.968 | 1.90 | 0.129 | 0.750 | 0.613 | 0.000 | - | 0.000 / - |
| judge_hard | 80 | 0.900 | 1.35 | 0.650 | 0.846 | 0.762 | 0.000 | - | 0.000 / - |
| long_policy | 70 | 0.829 | 2.23 | 0.329 | 0.739 | 0.671 | 0.000 | - | 0.000 / - |
| policy | 74 | 0.919 | 1.22 | 0.784 | 0.897 | 0.865 | 0.000 | - | 0.000 / - |
| temporal_numeric | 69 | 0.942 | 1.57 | 0.696 | 0.917 | 0.754 | 0.000 | - | 0.000 / - |
LTT-half attribution: best lambda by p {"lambda": 0.87, "n_act": 187, "errors": 19, "p": 0.9988027167269304}; Bonferroni level 9.90e-04; zero-error acts needed 135 (single test 45); LTT singleton rate 0.517; per-type lambdas {'choice': None, 'noul': None}; FS-CRR c1 stop {"lambda": 0.95, "reason": "not rejected", "n_act": 181, "errors": 19, "p_value": 0.9894619819676875, "log_capital": 0.010593936135257227, "weighted_error": 0.10497237569060773}; FS-CRR c8 lam None stop {"lambda": 0.95, "reason": "not rejected", "n_act": 181, "errors": 19, "p_value": 0.9986704886330467, "log_capital": 0.0013303959513209624, "weighted_error": 0.10497237569060773}

per type / family, variant T, alpha 0.05 (groups {"choice": {"n_fit": 138, "qhat": 0.9055823110212616}, "noul": {"n_fit": 222, "qhat": 0.9986791583806148}, "pooled-small": {"n_fit": 16, "qhat": Infinity}}; pooled-small ['score'] full-set True)
| cell | n | cov | size | singleton | sing acc | top-1 | act | acted err | per-type-LTT act / err |
|---|---|---|---|---|---|---|---|---|---|
| choice | 137 | 0.964 | 2.39 | 0.241 | 0.939 | 0.657 | 0.000 | - | 0.000 / - |
| noul | 269 | 0.948 | 1.51 | 0.491 | 0.894 | 0.770 | 0.000 | - | 0.000 / - |
| score | 12 | 1.000 | 4.42 | 0.000 | - | 0.667 | 0.000 | - | 0.000 / - |
| adequacy | 63 | 0.968 | 1.63 | 0.365 | 0.913 | 0.683 | 0.000 | - | 0.000 / - |
| ambiguous | 62 | 1.000 | 2.27 | 0.032 | 1.000 | 0.613 | 0.000 | - | 0.000 / - |
| judge_hard | 80 | 0.938 | 1.55 | 0.450 | 0.861 | 0.762 | 0.000 | - | 0.000 / - |
| long_policy | 70 | 0.929 | 2.87 | 0.214 | 0.867 | 0.671 | 0.000 | - | 0.000 / - |
| policy | 74 | 0.946 | 1.36 | 0.635 | 0.915 | 0.865 | 0.000 | - | 0.000 / - |
| temporal_numeric | 69 | 0.957 | 1.70 | 0.609 | 0.929 | 0.754 | 0.000 | - | 0.000 / - |
LTT-half attribution: best lambda by p {"lambda": 0.87, "n_act": 141, "errors": 11, "p": 0.9490179922243828}; Bonferroni level 9.90e-04; zero-error acts needed 135 (single test 45); LTT singleton rate 0.365; per-type lambdas {'choice': None, 'noul': None}; FS-CRR c1 stop {"lambda": 0.95, "reason": "not rejected", "n_act": 136, "errors": 11, "p_value": 0.9894619819676875, "log_capital": 0.010593936135257227, "weighted_error": 0.08088235294117647}; FS-CRR c8 lam None stop {"lambda": 0.95, "reason": "not rejected", "n_act": 136, "errors": 11, "p_value": 0.9986704886330467, "log_capital": 0.0013303959513209624, "weighted_error": 0.08088235294117647}

**jev3lms_zbmhc / native_tfit / remainder** FIT 267 LTT 272 TEST 301; types {'FIT': {'choice': 104, 'noul': 151, 'score': 12}, 'LTT': {'choice': 97, 'noul': 162, 'score': 13}, 'TEST': {'choice': 101, 'noul': 190, 'score': 10}}
| alpha | var | TEST cov | flag | size | singleton | sing acc | lambda* (Bonf) | TEST act | acted err | FS-CRR c1 lam / act / err | FS-binom lam / act / err | per-type LTT act / err |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.1 | M | 0.884 | ok | 1.64 | 0.605 | 0.824 | - | 0.000 | - | - / 0.000 / - | - / 0.000 / - | 0.000 / - |
| 0.1 | T | 0.897 | ok | 1.57 | 0.588 | 0.853 | - | 0.000 | - | - / 0.000 / - | - / 0.000 / - | 0.000 / - |
| 0.1 | F | 0.884 | ok | 1.64 | 0.605 | 0.824 | - | 0.000 | - | - / 0.000 / - | - / 0.000 / - | 0.000 / - |
| 0.05 | M | 0.940 | ok | 2.08 | 0.455 | 0.869 | - | 0.000 | - | - / 0.000 / - | - / 0.000 / - | 0.000 / - |
| 0.05 | T | 0.957 | ok | 1.95 | 0.372 | 0.893 | - | 0.000 | - | - / 0.000 / - | - / 0.000 / - | 0.000 / - |
| 0.05 | F | 0.940 | ok | 2.08 | 0.455 | 0.869 | - | 0.000 | - | - / 0.000 / - | - / 0.000 / - | 0.000 / - |

per type / family, variant T, alpha 0.1 (groups {"choice": {"n_fit": 104, "qhat": 0.8564569624818352}, "noul": {"n_fit": 151, "qhat": 0.9514718651382009}, "pooled-small": {"n_fit": 12, "qhat": Infinity}}; pooled-small ['score'] full-set True)
| cell | n | cov | size | singleton | sing acc | top-1 | act | acted err | per-type-LTT act / err |
|---|---|---|---|---|---|---|---|---|---|
| choice | 101 | 0.901 | 1.93 | 0.307 | 0.839 | 0.644 | 0.000 | - | 0.000 / - |
| noul | 190 | 0.889 | 1.23 | 0.768 | 0.856 | 0.779 | 0.000 | - | 0.000 / - |
| score | 10 | 1.000 | 4.30 | 0.000 | - | 0.800 | 0.000 | - | 0.000 / - |
| adequacy | 35 | 0.914 | 1.34 | 0.657 | 0.870 | 0.743 | 0.000 | - | 0.000 / - |
| ambiguous | 51 | 0.961 | 1.90 | 0.137 | 0.714 | 0.627 | 0.000 | - | 0.000 / - |
| judge_hard | 62 | 0.903 | 1.23 | 0.774 | 0.875 | 0.774 | 0.000 | - | 0.000 / - |
| long_policy | 58 | 0.828 | 2.26 | 0.345 | 0.750 | 0.638 | 0.000 | - | 0.000 / - |
| policy | 57 | 0.912 | 1.12 | 0.877 | 0.900 | 0.877 | 0.000 | - | 0.000 / - |
| temporal_numeric | 38 | 0.868 | 1.50 | 0.763 | 0.828 | 0.737 | 0.000 | - | 0.000 / - |
LTT-half attribution: best lambda by p {"lambda": 0.99, "n_act": 101, "errors": 10, "p": 0.9876917957277139}; Bonferroni level 9.90e-04; zero-error acts needed 135 (single test 45); LTT singleton rate 0.562; per-type lambdas {'noul': None}; FS-CRR c1 stop {"lambda": 0.95, "reason": "not rejected", "n_act": 130, "errors": 15, "p_value": 0.9871553027336019, "log_capital": 0.012927903665599923, "weighted_error": 0.11538461538461539}; FS-CRR c8 lam None stop {"lambda": 0.95, "reason": "not rejected", "n_act": 130, "errors": 15, "p_value": 0.9983761623515719, "log_capital": 0.0016251575017942645, "weighted_error": 0.11538461538461539}

per type / family, variant T, alpha 0.05 (groups {"choice": {"n_fit": 104, "qhat": 0.9164224904721608}, "noul": {"n_fit": 151, "qhat": 0.9985439751390554}, "pooled-small": {"n_fit": 12, "qhat": Infinity}}; pooled-small ['score'] full-set True)
| cell | n | cov | size | singleton | sing acc | top-1 | act | acted err | per-type-LTT act / err |
|---|---|---|---|---|---|---|---|---|---|
| choice | 101 | 0.970 | 2.55 | 0.178 | 0.889 | 0.644 | 0.000 | - | 0.000 / - |
| noul | 190 | 0.947 | 1.51 | 0.495 | 0.894 | 0.779 | 0.000 | - | 0.000 / - |
| score | 10 | 1.000 | 4.30 | 0.000 | - | 0.800 | 0.000 | - | 0.000 / - |
| adequacy | 35 | 0.971 | 1.66 | 0.343 | 0.917 | 0.743 | 0.000 | - | 0.000 / - |
| ambiguous | 51 | 1.000 | 2.33 | 0.000 | - | 0.627 | 0.000 | - | 0.000 / - |
| judge_hard | 62 | 0.952 | 1.55 | 0.452 | 0.893 | 0.774 | 0.000 | - | 0.000 / - |
| long_policy | 58 | 0.948 | 3.00 | 0.172 | 0.800 | 0.638 | 0.000 | - | 0.000 / - |
| policy | 57 | 0.947 | 1.33 | 0.667 | 0.921 | 0.877 | 0.000 | - | 0.000 / - |
| temporal_numeric | 38 | 0.921 | 1.68 | 0.632 | 0.875 | 0.737 | 0.000 | - | 0.000 / - |
LTT-half attribution: best lambda by p {"lambda": 0.9, "n_act": 88, "errors": 7, "p": 0.9266088944935377}; Bonferroni level 9.90e-04; zero-error acts needed 135 (single test 45); LTT singleton rate 0.346; per-type lambdas {'noul': None}; FS-CRR c1 stop {"lambda": 0.95, "reason": "not rejected", "n_act": 85, "errors": 7, "p_value": 0.9871553027336019, "log_capital": 0.012927903665599923, "weighted_error": 0.08235294117647059}; FS-CRR c8 lam None stop {"lambda": 0.95, "reason": "not rejected", "n_act": 85, "errors": 7, "p_value": 0.9387756802992969, "log_capital": 0.06317872043316652, "weighted_error": 0.08235294117647059}

**jev3lms_zbmhc / native_shimcal / all** FIT 376 LTT 406 TEST 418; types {'FIT': {'choice': 138, 'noul': 222, 'score': 16}, 'LTT': {'choice': 141, 'noul': 250, 'score': 15}, 'TEST': {'choice': 137, 'noul': 269, 'score': 12}}
| alpha | var | TEST cov | flag | size | singleton | sing acc | lambda* (Bonf) | TEST act | acted err | FS-CRR c1 lam / act / err | FS-binom lam / act / err | per-type LTT act / err |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.1 | M | 0.904 | ok | 1.54 | 0.524 | 0.858 | - | 0.000 | - | - / 0.000 / - | - / 0.000 / - | 0.000 / - |
| 0.1 | T | 0.914 | ok | 1.61 | 0.529 | 0.864 | - | 0.000 | - | - / 0.000 / - | - / 0.000 / - | 0.000 / - |
| 0.1 | F | 0.904 | ok | 1.54 | 0.524 | 0.858 | - | 0.000 | - | - / 0.000 / - | - / 0.000 / - | 0.000 / - |
| 0.05 | M | 0.950 | ok | 1.75 | 0.407 | 0.906 | - | 0.000 | - | - / 0.000 / - | - / 0.000 / - | 0.000 / - |
| 0.05 | T | 0.952 | ok | 1.85 | 0.397 | 0.904 | - | 0.000 | - | - / 0.000 / - | - / 0.000 / - | 0.000 / - |
| 0.05 | F | 0.950 | ok | 1.75 | 0.407 | 0.906 | - | 0.000 | - | - / 0.000 / - | - / 0.000 / - | 0.000 / - |

per type / family, variant T, alpha 0.1 (groups {"choice": {"n_fit": 138, "qhat": 0.879351119315873}, "noul": {"n_fit": 222, "qhat": 0.8047250214175409}, "pooled-small": {"n_fit": 16, "qhat": Infinity}}; pooled-small ['score'] full-set True)
| cell | n | cov | size | singleton | sing acc | top-1 | act | acted err | per-type-LTT act / err |
|---|---|---|---|---|---|---|---|---|---|
| choice | 137 | 0.905 | 1.85 | 0.350 | 0.854 | 0.657 | 0.000 | - | 0.000 / - |
| noul | 269 | 0.914 | 1.36 | 0.643 | 0.867 | 0.770 | 0.000 | - | 0.000 / - |
| score | 12 | 1.000 | 4.42 | 0.000 | - | 0.667 | 0.000 | - | 0.000 / - |
| adequacy | 63 | 0.952 | 1.51 | 0.492 | 0.903 | 0.683 | 0.000 | - | 0.000 / - |
| ambiguous | 62 | 0.952 | 1.89 | 0.145 | 0.667 | 0.613 | 0.000 | - | 0.000 / - |
| judge_hard | 80 | 0.900 | 1.35 | 0.650 | 0.846 | 0.762 | 0.000 | - | 0.000 / - |
| long_policy | 70 | 0.829 | 2.21 | 0.329 | 0.739 | 0.671 | 0.000 | - | 0.000 / - |
| policy | 74 | 0.919 | 1.22 | 0.784 | 0.897 | 0.865 | 0.000 | - | 0.000 / - |
| temporal_numeric | 69 | 0.942 | 1.55 | 0.696 | 0.917 | 0.754 | 0.000 | - | 0.000 / - |
LTT-half attribution: best lambda by p {"lambda": 0.97, "n_act": 75, "errors": 1, "p": 0.10559531481644245}; Bonferroni level 9.90e-04; zero-error acts needed 135 (single test 45); LTT singleton rate 0.542; per-type lambdas {'choice': None, 'noul': None}; FS-CRR c1 stop {"lambda": 0.95, "reason": "not rejected", "n_act": 98, "errors": 5, "p_value": 0.5514583722513069, "log_capital": 0.5951889240960565, "weighted_error": 0.05102040816326531}; FS-CRR c8 lam None stop {"lambda": 0.95, "reason": "not rejected", "n_act": 98, "errors": 5, "p_value": 0.9177922447041522, "log_capital": 0.08578422693272841, "weighted_error": 0.05102040816326531}

per type / family, variant T, alpha 0.05 (groups {"choice": {"n_fit": 138, "qhat": 0.9264534348475081}, "noul": {"n_fit": 222, "qhat": 0.8965518729828988}, "pooled-small": {"n_fit": 16, "qhat": Infinity}}; pooled-small ['score'] full-set True)
| cell | n | cov | size | singleton | sing acc | top-1 | act | acted err | per-type-LTT act / err |
|---|---|---|---|---|---|---|---|---|---|
| choice | 137 | 0.956 | 2.31 | 0.248 | 0.941 | 0.657 | 0.000 | - | 0.000 / - |
| noul | 269 | 0.948 | 1.51 | 0.491 | 0.894 | 0.770 | 0.000 | - | 0.000 / - |
| score | 12 | 1.000 | 4.42 | 0.000 | - | 0.667 | 0.000 | - | 0.000 / - |
| adequacy | 63 | 0.968 | 1.63 | 0.365 | 0.913 | 0.683 | 0.000 | - | 0.000 / - |
| ambiguous | 62 | 1.000 | 2.19 | 0.032 | 1.000 | 0.613 | 0.000 | - | 0.000 / - |
| judge_hard | 80 | 0.938 | 1.55 | 0.450 | 0.861 | 0.762 | 0.000 | - | 0.000 / - |
| long_policy | 70 | 0.914 | 2.79 | 0.229 | 0.875 | 0.671 | 0.000 | - | 0.000 / - |
| policy | 74 | 0.946 | 1.36 | 0.635 | 0.915 | 0.865 | 0.000 | - | 0.000 / - |
| temporal_numeric | 69 | 0.957 | 1.68 | 0.609 | 0.929 | 0.754 | 0.000 | - | 0.000 / - |
LTT-half attribution: best lambda by p {"lambda": 0.97, "n_act": 75, "errors": 1, "p": 0.10559531481644245}; Bonferroni level 9.90e-04; zero-error acts needed 135 (single test 45); LTT singleton rate 0.374; per-type lambdas {'choice': None, 'noul': None}; FS-CRR c1 stop {"lambda": 0.95, "reason": "not rejected", "n_act": 98, "errors": 5, "p_value": 0.5514583722513069, "log_capital": 0.5951889240960565, "weighted_error": 0.05102040816326531}; FS-CRR c8 lam None stop {"lambda": 0.95, "reason": "not rejected", "n_act": 98, "errors": 5, "p_value": 0.9177922447041522, "log_capital": 0.08578422693272841, "weighted_error": 0.05102040816326531}

**jev3lms_zbmhc / native_shimcal / remainder** FIT 267 LTT 272 TEST 301; types {'FIT': {'choice': 104, 'noul': 151, 'score': 12}, 'LTT': {'choice': 97, 'noul': 162, 'score': 13}, 'TEST': {'choice': 101, 'noul': 190, 'score': 10}}
| alpha | var | TEST cov | flag | size | singleton | sing acc | lambda* (Bonf) | TEST act | acted err | FS-CRR c1 lam / act / err | FS-binom lam / act / err | per-type LTT act / err |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.1 | M | 0.904 | ok | 1.52 | 0.538 | 0.852 | - | 0.000 | - | - / 0.000 / - | - / 0.000 / - | 0.000 / - |
| 0.1 | T | 0.907 | ok | 1.63 | 0.525 | 0.854 | - | 0.000 | - | - / 0.000 / - | - / 0.000 / - | 0.000 / - |
| 0.1 | F | 0.904 | ok | 1.52 | 0.538 | 0.852 | - | 0.000 | - | - / 0.000 / - | - / 0.000 / - | 0.000 / - |
| 0.05 | M | 0.950 | ok | 1.79 | 0.385 | 0.905 | - | 0.000 | - | - / 0.000 / - | - / 0.000 / - | 0.000 / - |
| 0.05 | T | 0.957 | ok | 1.98 | 0.372 | 0.893 | - | 0.000 | - | - / 0.000 / - | - / 0.000 / - | 0.000 / - |
| 0.05 | F | 0.950 | ok | 1.79 | 0.385 | 0.905 | - | 0.000 | - | - / 0.000 / - | - / 0.000 / - | 0.000 / - |

per type / family, variant T, alpha 0.1 (groups {"choice": {"n_fit": 104, "qhat": 0.879351119315873}, "noul": {"n_fit": 151, "qhat": 0.801}, "pooled-small": {"n_fit": 12, "qhat": Infinity}}; pooled-small ['score'] full-set True)
| cell | n | cov | size | singleton | sing acc | top-1 | act | acted err | per-type-LTT act / err |
|---|---|---|---|---|---|---|---|---|---|
| choice | 101 | 0.891 | 1.91 | 0.317 | 0.812 | 0.644 | 0.000 | - | 0.000 / - |
| noul | 190 | 0.911 | 1.34 | 0.663 | 0.865 | 0.779 | 0.000 | - | 0.000 / - |
| score | 10 | 1.000 | 4.30 | 0.000 | - | 0.800 | 0.000 | - | 0.000 / - |
| adequacy | 35 | 0.971 | 1.51 | 0.486 | 0.941 | 0.743 | 0.000 | - | 0.000 / - |
| ambiguous | 51 | 0.941 | 1.88 | 0.157 | 0.625 | 0.627 | 0.000 | - | 0.000 / - |
| judge_hard | 62 | 0.903 | 1.32 | 0.677 | 0.857 | 0.774 | 0.000 | - | 0.000 / - |
| long_policy | 58 | 0.828 | 2.31 | 0.276 | 0.688 | 0.638 | 0.000 | - | 0.000 / - |
| policy | 57 | 0.930 | 1.18 | 0.825 | 0.915 | 0.877 | 0.000 | - | 0.000 / - |
| temporal_numeric | 38 | 0.895 | 1.53 | 0.737 | 0.857 | 0.737 | 0.000 | - | 0.000 / - |
LTT-half attribution: best lambda by p {"lambda": 0.96, "n_act": 52, "errors": 1, "p": 0.2594969291207257}; Bonferroni level 9.90e-04; zero-error acts needed 135 (single test 45); LTT singleton rate 0.526; per-type lambdas {'noul': None}; FS-CRR c1 stop {"lambda": 0.95, "reason": "not rejected", "n_act": 60, "errors": 3, "p_value": 0.519264064302098, "log_capital": 0.6553427307918662, "weighted_error": 0.05}; FS-CRR c8 lam None stop {"lambda": 0.95, "reason": "not rejected", "n_act": 60, "errors": 3, "p_value": 0.9200024362594629, "log_capital": 0.08337896083444546, "weighted_error": 0.05}

per type / family, variant T, alpha 0.05 (groups {"choice": {"n_fit": 104, "qhat": 0.9502224858381388}, "noul": {"n_fit": 151, "qhat": 0.8935659943924871}, "pooled-small": {"n_fit": 12, "qhat": Infinity}}; pooled-small ['score'] full-set True)
| cell | n | cov | size | singleton | sing acc | top-1 | act | acted err | per-type-LTT act / err |
|---|---|---|---|---|---|---|---|---|---|
| choice | 101 | 0.970 | 2.65 | 0.178 | 0.889 | 0.644 | 0.000 | - | 0.000 / - |
| noul | 190 | 0.947 | 1.51 | 0.495 | 0.894 | 0.779 | 0.000 | - | 0.000 / - |
| score | 10 | 1.000 | 4.30 | 0.000 | - | 0.800 | 0.000 | - | 0.000 / - |
| adequacy | 35 | 0.971 | 1.66 | 0.343 | 0.917 | 0.743 | 0.000 | - | 0.000 / - |
| ambiguous | 51 | 1.000 | 2.45 | 0.000 | - | 0.627 | 0.000 | - | 0.000 / - |
| judge_hard | 62 | 0.952 | 1.55 | 0.452 | 0.893 | 0.774 | 0.000 | - | 0.000 / - |
| long_policy | 58 | 0.948 | 3.05 | 0.172 | 0.800 | 0.638 | 0.000 | - | 0.000 / - |
| policy | 57 | 0.947 | 1.33 | 0.667 | 0.921 | 0.877 | 0.000 | - | 0.000 / - |
| temporal_numeric | 38 | 0.921 | 1.71 | 0.632 | 0.875 | 0.737 | 0.000 | - | 0.000 / - |
LTT-half attribution: best lambda by p {"lambda": 0.96, "n_act": 52, "errors": 1, "p": 0.2594969291207257}; Bonferroni level 9.90e-04; zero-error acts needed 135 (single test 45); LTT singleton rate 0.342; per-type lambdas {'noul': None}; FS-CRR c1 stop {"lambda": 0.95, "reason": "not rejected", "n_act": 60, "errors": 3, "p_value": 0.519264064302098, "log_capital": 0.6553427307918662, "weighted_error": 0.05}; FS-CRR c8 lam None stop {"lambda": 0.95, "reason": "not rejected", "n_act": 60, "errors": 3, "p_value": 0.9200024362594629, "log_capital": 0.08337896083444546, "weighted_error": 0.05}

**jev3lms_zbmhc / native / all** FIT 376 LTT 406 TEST 418; types {'FIT': {'choice': 138, 'noul': 222, 'score': 16}, 'LTT': {'choice': 141, 'noul': 250, 'score': 15}, 'TEST': {'choice': 137, 'noul': 269, 'score': 12}}
| alpha | var | TEST cov | flag | size | singleton | sing acc | lambda* (Bonf) | TEST act | acted err | FS-CRR c1 lam / act / err | FS-binom lam / act / err | per-type LTT act / err |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.1 | M | 0.904 | ok | 1.52 | 0.519 | 0.853 | - | 0.000 | - | - / 0.000 / - | - / 0.000 / - | 0.000 / - |
| 0.1 | T | 0.916 | ok | 1.61 | 0.524 | 0.872 | - | 0.000 | - | - / 0.000 / - | - / 0.000 / - | 0.000 / - |
| 0.1 | F | 0.904 | ok | 1.52 | 0.519 | 0.853 | - | 0.000 | - | - / 0.000 / - | - / 0.000 / - | 0.000 / - |
| 0.05 | M | 0.950 | ok | 1.71 | 0.404 | 0.899 | - | 0.000 | - | - / 0.000 / - | - / 0.000 / - | 0.000 / - |
| 0.05 | T | 0.955 | ok | 1.89 | 0.392 | 0.902 | - | 0.000 | - | - / 0.000 / - | - / 0.000 / - | 0.000 / - |
| 0.05 | F | 0.950 | ok | 1.71 | 0.404 | 0.899 | - | 0.000 | - | - / 0.000 / - | - / 0.000 / - | 0.000 / - |

per type / family, variant T, alpha 0.1 (groups {"choice": {"n_fit": 138, "qhat": 0.855408100497548}, "noul": {"n_fit": 222, "qhat": 0.7563688098940358}, "pooled-small": {"n_fit": 16, "qhat": Infinity}}; pooled-small ['score'] full-set True)
| cell | n | cov | size | singleton | sing acc | top-1 | act | acted err | per-type-LTT act / err |
|---|---|---|---|---|---|---|---|---|---|
| choice | 137 | 0.912 | 1.87 | 0.336 | 0.891 | 0.657 | 0.000 | - | 0.000 / - |
| noul | 269 | 0.914 | 1.36 | 0.643 | 0.867 | 0.770 | 0.000 | - | 0.000 / - |
| score | 12 | 1.000 | 4.42 | 0.000 | - | 0.667 | 0.000 | - | 0.000 / - |
| adequacy | 63 | 0.952 | 1.51 | 0.492 | 0.903 | 0.683 | 0.000 | - | 0.000 / - |
| ambiguous | 62 | 0.984 | 1.92 | 0.113 | 0.857 | 0.613 | 0.000 | - | 0.000 / - |
| judge_hard | 80 | 0.900 | 1.35 | 0.650 | 0.846 | 0.762 | 0.000 | - | 0.000 / - |
| long_policy | 70 | 0.814 | 2.20 | 0.329 | 0.739 | 0.671 | 0.000 | - | 0.000 / - |
| policy | 74 | 0.919 | 1.22 | 0.784 | 0.897 | 0.865 | 0.000 | - | 0.000 / - |
| temporal_numeric | 69 | 0.942 | 1.57 | 0.696 | 0.917 | 0.754 | 0.000 | - | 0.000 / - |
LTT-half attribution: best lambda by p {"lambda": 0.94, "n_act": 76, "errors": 1, "p": 0.10138273576792182}; Bonferroni level 9.90e-04; zero-error acts needed 135 (single test 45); LTT singleton rate 0.522; per-type lambdas {'choice': None, 'noul': None}; FS-CRR c1 stop {"lambda": 0.95, "reason": "not rejected", "n_act": 70, "errors": 1, "p_value": 0.2944592725778634, "log_capital": 1.2226145789076914, "weighted_error": 0.014285714285714285}; FS-CRR c8 lam None stop {"lambda": 0.95, "reason": "not rejected", "n_act": 70, "errors": 1, "p_value": 0.7918010867540394, "log_capital": 0.2334450718002639, "weighted_error": 0.014285714285714285}

per type / family, variant T, alpha 0.05 (groups {"choice": {"n_fit": 138, "qhat": 0.9075120807444165}, "noul": {"n_fit": 222, "qhat": 0.8491037264888565}, "pooled-small": {"n_fit": 16, "qhat": Infinity}}; pooled-small ['score'] full-set True)
| cell | n | cov | size | singleton | sing acc | top-1 | act | acted err | per-type-LTT act / err |
|---|---|---|---|---|---|---|---|---|---|
| choice | 137 | 0.964 | 2.42 | 0.234 | 0.938 | 0.657 | 0.000 | - | 0.000 / - |
| noul | 269 | 0.948 | 1.51 | 0.491 | 0.894 | 0.770 | 0.000 | - | 0.000 / - |
| score | 12 | 1.000 | 4.42 | 0.000 | - | 0.667 | 0.000 | - | 0.000 / - |
| adequacy | 63 | 0.968 | 1.63 | 0.365 | 0.913 | 0.683 | 0.000 | - | 0.000 / - |
| ambiguous | 62 | 1.000 | 2.34 | 0.016 | 1.000 | 0.613 | 0.000 | - | 0.000 / - |
| judge_hard | 80 | 0.938 | 1.55 | 0.450 | 0.861 | 0.762 | 0.000 | - | 0.000 / - |
| long_policy | 70 | 0.929 | 2.87 | 0.214 | 0.867 | 0.671 | 0.000 | - | 0.000 / - |
| policy | 74 | 0.946 | 1.36 | 0.635 | 0.915 | 0.865 | 0.000 | - | 0.000 / - |
| temporal_numeric | 69 | 0.957 | 1.70 | 0.609 | 0.929 | 0.754 | 0.000 | - | 0.000 / - |
LTT-half attribution: best lambda by p {"lambda": 0.94, "n_act": 76, "errors": 1, "p": 0.10138273576792182}; Bonferroni level 9.90e-04; zero-error acts needed 135 (single test 45); LTT singleton rate 0.362; per-type lambdas {'choice': None, 'noul': None}; FS-CRR c1 stop {"lambda": 0.95, "reason": "not rejected", "n_act": 70, "errors": 1, "p_value": 0.2944592725778634, "log_capital": 1.2226145789076914, "weighted_error": 0.014285714285714285}; FS-CRR c8 lam None stop {"lambda": 0.95, "reason": "not rejected", "n_act": 70, "errors": 1, "p_value": 0.7918010867540394, "log_capital": 0.2334450718002639, "weighted_error": 0.014285714285714285}

**jev3lms_zbmhc / native / remainder** FIT 267 LTT 272 TEST 301; types {'FIT': {'choice': 104, 'noul': 151, 'score': 12}, 'LTT': {'choice': 97, 'noul': 162, 'score': 13}, 'TEST': {'choice': 101, 'noul': 190, 'score': 10}}
| alpha | var | TEST cov | flag | size | singleton | sing acc | lambda* (Bonf) | TEST act | acted err | FS-CRR c1 lam / act / err | FS-binom lam / act / err | per-type LTT act / err |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.1 | M | 0.904 | ok | 1.53 | 0.518 | 0.846 | - | 0.000 | - | - / 0.000 / - | - / 0.000 / - | 0.000 / - |
| 0.1 | T | 0.900 | ok | 1.57 | 0.581 | 0.857 | - | 0.000 | - | - / 0.000 / - | - / 0.000 / - | 0.000 / - |
| 0.1 | F | 0.904 | ok | 1.53 | 0.518 | 0.846 | - | 0.000 | - | - / 0.000 / - | - / 0.000 / - | 0.000 / - |
| 0.05 | M | 0.953 | ok | 1.77 | 0.369 | 0.910 | - | 0.000 | - | - / 0.000 / - | - / 0.000 / - | 0.000 / - |
| 0.05 | T | 0.957 | ok | 1.97 | 0.372 | 0.893 | - | 0.000 | - | - / 0.000 / - | - / 0.000 / - | 0.000 / - |
| 0.05 | F | 0.953 | ok | 1.77 | 0.369 | 0.910 | - | 0.000 | - | - / 0.000 / - | - / 0.000 / - | 0.000 / - |

per type / family, variant T, alpha 0.1 (groups {"choice": {"n_fit": 104, "qhat": 0.8583720813755027}, "noul": {"n_fit": 151, "qhat": 0.6847397923982959}, "pooled-small": {"n_fit": 12, "qhat": Infinity}}; pooled-small ['score'] full-set True)
| cell | n | cov | size | singleton | sing acc | top-1 | act | acted err | per-type-LTT act / err |
|---|---|---|---|---|---|---|---|---|---|
| choice | 101 | 0.911 | 1.95 | 0.287 | 0.862 | 0.644 | 0.000 | - | 0.000 / - |
| noul | 190 | 0.889 | 1.23 | 0.768 | 0.856 | 0.779 | 0.000 | - | 0.000 / - |
| score | 10 | 1.000 | 4.30 | 0.000 | - | 0.800 | 0.000 | - | 0.000 / - |
| adequacy | 35 | 0.914 | 1.34 | 0.657 | 0.870 | 0.743 | 0.000 | - | 0.000 / - |
| ambiguous | 51 | 0.980 | 1.94 | 0.098 | 0.800 | 0.627 | 0.000 | - | 0.000 / - |
| judge_hard | 62 | 0.903 | 1.23 | 0.774 | 0.875 | 0.774 | 0.000 | - | 0.000 / - |
| long_policy | 58 | 0.828 | 2.26 | 0.345 | 0.750 | 0.638 | 0.000 | - | 0.000 / - |
| policy | 57 | 0.912 | 1.12 | 0.877 | 0.900 | 0.877 | 0.000 | - | 0.000 / - |
| temporal_numeric | 38 | 0.868 | 1.50 | 0.763 | 0.828 | 0.737 | 0.000 | - | 0.000 / - |
LTT-half attribution: best lambda by p {"lambda": 0.93, "n_act": 49, "errors": 1, "p": 0.2898758071366523}; Bonferroni level 9.90e-04; zero-error acts needed 135 (single test 45); LTT singleton rate 0.559; per-type lambdas {'noul': None}; FS-CRR c1 stop {"lambda": 0.95, "reason": "not rejected", "n_act": 44, "errors": 1, "p_value": 0.606236756408093, "log_capital": 0.5004846820691045, "weighted_error": 0.022727272727272728}; FS-CRR c8 lam None stop {"lambda": 0.95, "reason": "not rejected", "n_act": 44, "errors": 1, "p_value": 0.8976358383440247, "log_capital": 0.10799081815690081, "weighted_error": 0.022727272727272728}

per type / family, variant T, alpha 0.05 (groups {"choice": {"n_fit": 104, "qhat": 0.9198289813053361}, "noul": {"n_fit": 151, "qhat": 0.8458161179607607}, "pooled-small": {"n_fit": 12, "qhat": Infinity}}; pooled-small ['score'] full-set True)
| cell | n | cov | size | singleton | sing acc | top-1 | act | acted err | per-type-LTT act / err |
|---|---|---|---|---|---|---|---|---|---|
| choice | 101 | 0.970 | 2.60 | 0.178 | 0.889 | 0.644 | 0.000 | - | 0.000 / - |
| noul | 190 | 0.947 | 1.51 | 0.495 | 0.894 | 0.779 | 0.000 | - | 0.000 / - |
| score | 10 | 1.000 | 4.30 | 0.000 | - | 0.800 | 0.000 | - | 0.000 / - |
| adequacy | 35 | 0.971 | 1.66 | 0.343 | 0.917 | 0.743 | 0.000 | - | 0.000 / - |
| ambiguous | 51 | 1.000 | 2.43 | 0.000 | - | 0.627 | 0.000 | - | 0.000 / - |
| judge_hard | 62 | 0.952 | 1.55 | 0.452 | 0.893 | 0.774 | 0.000 | - | 0.000 / - |
| long_policy | 58 | 0.948 | 3.00 | 0.172 | 0.800 | 0.638 | 0.000 | - | 0.000 / - |
| policy | 57 | 0.947 | 1.33 | 0.667 | 0.921 | 0.877 | 0.000 | - | 0.000 / - |
| temporal_numeric | 38 | 0.921 | 1.68 | 0.632 | 0.875 | 0.737 | 0.000 | - | 0.000 / - |
LTT-half attribution: best lambda by p {"lambda": 0.93, "n_act": 49, "errors": 1, "p": 0.2898758071366523}; Bonferroni level 9.90e-04; zero-error acts needed 135 (single test 45); LTT singleton rate 0.346; per-type lambdas {'noul': None}; FS-CRR c1 stop {"lambda": 0.95, "reason": "not rejected", "n_act": 44, "errors": 1, "p_value": 0.606236756408093, "log_capital": 0.5004846820691045, "weighted_error": 0.022727272727272728}; FS-CRR c8 lam None stop {"lambda": 0.95, "reason": "not rejected", "n_act": 44, "errors": 1, "p_value": 0.8976358383440247, "log_capital": 0.10799081815690081, "weighted_error": 0.022727272727272728}
