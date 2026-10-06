# Vendored: bundled RWKV-7 Transformers code

- Source: `RWKV/RWKV7-G1k-13.3B-20260930` @ `ce3630b854cedbd673569f4600c6cf744bca9dba` (Hugging Face), files
  `modeling_rwkv7.py`, `configuration_rwkv7.py`, `tokenization_rwkv7.py`, `LICENSE`, `NOTICE`, copied unmodified.
- The repo NOTICE states the code comes from `huggingface/transformers@4ad9ed0747ed6ba75c787e8f9040dcd64b166ee2`
  ("Add RWKV-7 (Goose)", not in a transformers release as of v5.17.0), Apache-2.0.
- Pinned sha256 values live in `__init__.py` (`CODE_SHA256`, `CONFIG_SHA256`); `tests/test_m1_next_jobs.py` checks the
  vendored bytes, and every paid job checks the downloaded snapshot against them before loading weights.
- Why vendored: the repo's `auto_map` points at remote code (`custom_code`). It is not the `fla` `RWKV7ForCausalLM`
  that `decide.backends.hf_readout.load_rwkv7` loads (no `rwkv_vocab_v20230424.txt`, different module names), so that
  loader cannot read this checkpoint. Loading the vendored classes directly avoids `trust_remote_code`.
- Needs transformers >= 5.14 on CPU (tested with 5.14.1); the repo README asks for 5.15+, and the job images pin 5.17.0.
