# L1 whole-world trainer consumer — CPU contract receipt

## Scope

Deterministic data-loader and fit-record conversion proof only. No checkpoint/model load, inference, optimizer update, calibration fit, GPU, Index, held-out data or paid compute occurred. This receipt does not establish an L1 capability or calibrated-performance result.

## Source and input identities

- Base: PR #134 exact head `d7de4ff7d099f63b0d8a8a00550b8c2f490c7452`, stacked on the #133/#132 cohort chain.
- Train source JSONL SHA-256: `175cf7cc2d7c39a8c931a3a452149be7c141db712bde81a0ce7f606b0a27e9f4`.
- Train source manifest SHA-256: `eaa38a852dab840a51175a12a1a58a74539338f3ee119088a3442f054d0586e7`.
- Validation JSONL SHA-256: `a25c938eba725611a74ba9ff9dac4b83ed4e28bb00a2bc50f9e76c1202c85339`.
- Validation manifest SHA-256: `140b0f7f577f23fd98090c583c1fdb3d5adc2cf7d1a13f626700724f55657870`.
- Fit partition SHA-256: `34909156035594729d7891d64c840efb8ed853dfe4fcb10a56f77b0fa8e91be0`.
- Calibration partition SHA-256: `f58d51a51cdb6621d9b7ae86d348d67406488e13bb0c6839c174aaa5c47dce28`.
- Whole-world split manifest SHA-256: `3ad9f962cc6896c4540a13633ea371a00d8537c46d061813b66188e43920b0ec`.

## Verification

- Command: `C:/Users/apumu/research/CHLabs/.venv-gpu/Scripts/python.exe -m pytest -q -p no:cacheprovider language_growth/tests/test_l1_world_training.py`.
- Result: **6 focused tests passed** (including the injected reserved-calibration mutation); with the repository lint suite, **73 passed**. The Item-conversion test uses only the existing trainer's Item converter with a one-token fake encoder; it does not construct a model or load a checkpoint.
- `git diff --check`: passed.
- Cohorts: 24 fit worlds / 3,072 fit Items; 8 reserved calibration worlds / 1,024 records; 32 separate free-running validation worlds / 1,280 tasks. World sets are disjoint. Item-to-world IDs remain in an external metric sidecar, never in `Item.row` or model requests.
- Tampered train bytes and replacing the 32-world train input with the one-world smoke cohort both fail the pinned-source SHA check before Item conversion.
- Appending reserved calibration records to the in-memory fit partition after loading fails the fit hash check before Item conversion.
- Hardware: local Windows CPU, CUDA not used. Cost: $0. Raw model predictions: none.

The focused command above is reproducible from the PR branch with the repository's `.venv-gpu` environment. No model result, benchmark score or promotion claim is made.
