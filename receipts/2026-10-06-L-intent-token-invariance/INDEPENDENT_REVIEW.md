# Independent result review

Independent raw-data review completed by `/root/baseline_receipt_recompute` on 2026-10-06. It verified clean HEAD `c3c0d5aa21a08dd7d617c29ead1c18b614f85a9d`, measured source `c61b0930ca5e885b46fc46204ac9a892e6743607`, all receipt Git-blob hashes, and the immutable baseline data/source (`64f473bb143b367d655d4929217f16f3413cbdaf`). No retraining or inference rerun.

For each initialization 17, 29, and 43, the filtered variant scores **300/300** exact programs and the original baseline **200/300** on the same 300 iterative development tasks. The variant passes 100/100 comparison tasks; the baseline passes 0/100. Each seed has 100 improved, 200 unchanged and 0 regressed tasks. All variant outputs are kernel-accepted (300/300).

The paired task-cluster bootstrap reproduces **+33.333 percentage points**, 95% percentile interval **[28.333, 38.667]**, 10,000 resamples, RNG seed 1729. It averages across model seeds within each task before resampling; it does not treat 900 repeated task/seed rows as independent. The interval is descriptive and does not account for selecting the filter after inspecting this same cohort.

The measured source change removes fixed numeric, month-name and 34 common function-word tokens only from operation features. Independent source/feature review found no use of labels, target IDs, or option IDs in the filter. The fitting-record SHA is `eec306a03ba3e9791fb74b76ead1991e35f162085c33167a5ea6ace4404eb56c`; the paired task SHA is `d34df20c1448998791071833cc9154589b526e2ce0a68e7f98e307028d0a827f`; paired-analysis SHA is `74876e42d49e40bf54bcbdb6f57814f44aeae378960f5f675353af0bb817708c`. Receipt SHA is `1f69d058205bc0f458045f1cc06cf411323c746a63d9b56a8c4ca709cd5ecfe1`.

**Verdict:** verified within the single synthetic iterative-development cohort and stated scope. This is not untouched-domain transfer, an L0 gate, raw-language or arithmetic understanding, parent-retention evidence, or causal semantic mediation. That cohort is retired from confirmatory use.
