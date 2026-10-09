| # | family (area) | rows | max K | median tok | D13 | Clef-flash | Clef 27B | gap vs flash | gap vs 27B | class | evidence |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| 9 | Home-Appliance (tools) | 12 | 9 | 1915 | 0.0 | 91.7 | 75.0 | -3.10 | -2.54 | frozen | Home appliances, 18 q/case, case-exact; R4 joint decoding falsified (DEAD_ENDS) |
| 43 | CRUXEval-output-choice (knowledge) | 15 | 10 | 130 | 0.0 | 89.8 | 79.5 | -2.15 | -1.90 | data | CRUXEval: 0 rows in D1-24k (synthetic CRUXEval exists in D16/D17 draws); precedent: 877 GSM8K rows took D1 from 9.6 to 69.2 (subset) |
| 62 | When2Call MCQ (tools) | 21 | 4 | 687 | 23.8 | 74.6 | 74.6 | -1.72 | -1.72 | capacity | When2Call: in draw (895 rows); D13 regressed vs D1 55.6 -> 23.8 (tool_call AUROC 0.87 -> 0.59) |
| 3 | API-Bank-tool-selection (tools) | 25 | 53 | 6510 | 59.2 | 95.9 | 91.8 | -1.49 | -1.32 | format | API-Bank: eval K 53, 6.5k tok/row; D1-24k max K 16, T 2,048 |
| 56 | PhishNChips phishing decisions (retrieval) | 19 | 2 | 674 | 0.0 | 47.4 | 68.4 | -1.44 | -2.08 | data | PhishNChips: no training analog in any draw; D13 0.0 (answers 'legitimate' 17/19) |
| 4 | BANKING77 (retrieval) | 23 | 77 | 729 | 66.2 | 97.0 | 100.0 | -1.12 | -1.23 | format | BANKING77: eval K 77; train_max_k 16 dropped every K77 row |
| 48 | ForecastBench-binary (arts) | 18 | 2 | 1133 | 16.0 | 78.5 | 56.2 | -1.04 | -0.67 | data | ForecastBench: absent; global yes/no train prior 14-19% yes vs 28% gold (leaves Index 0.3) |
| 32 | MuSR (knowledge) | 15 | 5 | 1324 | 47.0 | 89.4 | 89.4 | -1.01 | -1.01 | frozen | MuSR |
| 44 | CLadder (knowledge) | 15 | 2 | 162 | 60.0 | 100.0 | 100.0 | -0.96 | -0.96 | frozen | CLadder |
| 61 | HoVer claim verification (retrieval) | 20 | 2 | 334 | 40.0 | 70.0 | 80.0 | -0.91 | -1.21 | data | HoVer: absent from D1-24k (in 100k-v3 / v2e) |
| 11 | ContractNLI (language) | 12 | 3 | 3908 | 52.3 | 75.1 | 65.1 | -0.56 | -0.31 | format | ContractNLI: 3.9k-token docs (100% > 2,048), NotMentioned 42%; anli short 3-way |
| 57 | MMLU-Pro (knowledge) | 18 | 10 | 177 | 56.3 | 75.0 | 87.5 | -0.54 | -0.90 | format | MMLU-Pro: eval K 10; mmlu_aux K 4 only |
| 37 | Amazon-ESCI (retrieval) | 19 | 4 | 268 | 7.5 | 25.2 | 34.2 | -0.54 | -0.81 | data | Amazon-ESCI: no 4-way E/S/C/I analog (nq binary only) |
| 36 | BRIGHT-retrieval (retrieval) | 16 | 2 | 9248 | 21.6 | 34.5 | 38.4 | -0.47 | -0.61 | format | BRIGHT: 9.2k tok, 32 docs/query; 5 queries refused by the 16k cap (split to context) |
| 41 | VAST (language) | 15 | 3 | 175 | 11.8 | 30.4 | 36.7 | -0.45 | -0.61 | data | VAST: absent from D1-24k |
| 59 | RAGTruth response-level hallucination (language) | 15 | 1 | 807 | 0.0 | 17.1 | 86.2 | -0.42 | -2.10 | data | RAGTruth: absent from D1-24k; D13 AUROC 0.18 = polarity misread (flip +1.18 diag, unclaimed) |
| 28 | WinoGrande (language) | 18 | 2 | 66 | 88.9 | 100.0 | 100.0 | -0.33 | -0.33 | capacity | WinoGrande: in draw, matched prior |
| 33 | SATA-Bench (knowledge) | 15 | 2 | 1361 | 32.5 | 46.0 | 46.0 | -0.32 | -0.32 | frozen | SATA (R2 top-m rule +0.16 split to readout) |
| 39 | FinEntity-entity-given (language) | 15 | 3 | 235 | 88.9 | 100.0 | 100.0 | -0.27 | -0.27 | format | FinEntity: sst5 score regression vs 3-way choice |
| 5 | CLINC150+OOS (retrieval) | 23 | 151 | 1086 | 54.2 | 61.0 | 88.9 | -0.25 | -1.26 | format | CLINC150: eval K 151 + 21% OOS; train K <= 16, OOS 0.8% |
| 2 | ToolRet-retrieval (tools) | 21 | 2 | 7467 | 62.7 | 69.4 | 68.5 | -0.23 | -0.20 | format | ToolRet: 7.5k tok, 32 docs/query; 1 query refused (split to context) |
| 29 | HellaSwag (language) | 18 | 4 | 258 | 92.6 | 100.0 | 100.0 | -0.22 | -0.22 | capacity | HellaSwag: in draw |
| 1 | BFCL-tool-selection (tools) | 25 | 2 | 975 | 89.7 | 94.8 | 100.0 | -0.21 | -0.42 | capacity | BFCL: in draw (glaive/toolace); R1 exactly-one rule +0.21 split to readout |
| 50 | Habermas-consensus (arts) | 10 | 4 | 1074 | 28.6 | 42.9 | 71.4 | -0.20 | -0.59 | data | Habermas: absent from D1-24k |
| 45 | HLE-text-MC (knowledge) | 18 | 12 | 193 | 13.6 | 20.2 | 0.0 | -0.19 | +0.39 | capacity | HLE |
| 38 | ACOS-category-sentiment (language) | 20 | 2 | 8046 | 14.9 | 21.7 | 26.7 | -0.16 | -0.29 | data | ACOS: absent; 58 q/row, 8k-token rows (also format) |
| 21 | Humicroedit (arts) | 9 | 2 | 78 | 0.0 | 11.1 | 11.1 | -0.15 | -0.15 | capacity | Humicroedit: in draw 50/50, neither arm discriminates |
| 20 | BPoMP-original-limerick (arts) | 24 | 2 | 167 | 91.7 | 94.4 | 100.0 | -0.04 | -0.12 | frozen | BPoMP |
| 22 | POP909-chord-pitch-class (arts) | 9 | 129 | 2582 | 0.0 | 0.0 | 37.0 | +0.00 | -0.51 | frozen | POP909 K 129 |
| 23 | CFColor-preference (arts) | 10 | 2 | 243 | 0.0 | 0.0 | 0.0 | +0.00 | +0.00 | capacity | CFColor: 0 for every arm |
| 31 | ChessBench-legal-move (knowledge) | 15 | 47 | 397 | 15.4 | 15.4 | 8.3 | +0.00 | +0.17 | frozen | ChessBench K 47 |
| 40 | iSarcasmEval-A-Ar (language) | 15 | 2 | 63 | 0.0 | 0.0 | 0.0 | +0.00 | +0.00 | capacity | iSarcasm: 0 for every arm |
| 64 | New Yorker caption matching (arts) | 8 | 5 | 157 | 68.8 | 68.8 | 68.8 | +0.00 | +0.00 | frozen | New Yorker |
| 42 | NLI4CT-2024 (language) | 27 | 2 | 485 | 49.5 | 41.6 | 62.1 | +0.19 | -0.31 | data | NLI4CT: clinical domain absent (train split quarantined for premise overlap) |
| 25 | GPQA-Diamond (knowledge) | 14 | 4 | 166 | 14.3 | 4.8 | 23.8 | +0.27 | -0.27 | capacity | GPQA |
| 58 | BBH fixed-option tasks (knowledge) | 18 | 18 | 184 | 59.7 | 43.6 | 59.7 | +0.46 | +0.00 | frozen | BBH |
| 12 | ANLI (language) | 18 | 3 | 160 | 65.2 | 41.8 | 49.4 | +0.69 | +0.46 | capacity | ANLI (D13 > Clef) |
| 30 | GSM8K-10 (knowledge) | 16 | 10 | 169 | 93.1 | 55.6 | 62.5 | +0.90 | +0.73 | capacity | GSM8K (D13 > Clef) |
