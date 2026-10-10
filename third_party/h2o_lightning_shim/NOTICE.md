`h2o_lightning_shim.py` and `serve_config.json` are copied unmodified from
https://huggingface.co/h2oai/h2o-lightning-4b at tag `v1.1` (commit 193ad740925b176a3b70a5a13a7cff2f2fadd01e),
Apache-2.0 (`LICENSE`, copied from the same commit). Copyright H2O.ai. sha256 pins are in `__init__.py`.

Used only by jobs/jev_host_scan to replicate the published board row's readout (prompt, labels, temperature,
yes/no floor, answer format). The vLLM HTTP client in the shim is not used.
