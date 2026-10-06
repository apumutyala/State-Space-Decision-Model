# Vendored: Cloudflare Clef `joint_schema_model.py`

`joint_schema_model.py` in this directory is an **unmodified, byte-identical** copy of the file
Cloudflare publishes with its Clef open-weight decision models. `LICENSE` is the Apache-2.0 text
shipped in the same repositories. Redistribution is permitted by Apache-2.0 §4 (license copy kept,
no modifications made, so no change notices are needed).

| Item | Value |
|---|---|
| Source (Clef-flash) | `https://huggingface.co/Cloudflare/clef-flash/raw/main/joint_schema_model.py` |
| Source (Clef) | `https://huggingface.co/Cloudflare/clef/raw/main/joint_schema_model.py` |
| Clef-flash revision | `17f0b0ad64efb65d273590632833508766b2aae6` (lastModified 2026-10-01T15:23:49Z) |
| Clef revision | `2f3de3dd85f379784083b0814d997ab627200f0c` (lastModified 2026-10-01T15:23:46Z) |
| sha256 `joint_schema_model.py` | `0e304cf7c6500e8bb59bef7e2afd2c6373f82596dfb3b57d1aa93c175e2dc3a3` (identical in both repos) |
| sha256 `LICENSE` | `bbedc3fda3305820b977265f01b8619d87570a6739de3a5582c3464840f1e57a` (identical in both repos) |
| Fetched | 2026-10-06, as untrusted text, reviewed before first import |

`.gitattributes` marks `third_party/**` as `-text` so a Windows checkout with `core.autocrlf=true`
keeps the bytes (and the sha256) intact. `baselines/clef_backend.py::load_jsm` refuses to import
the file if its sha256 differs from the pin above.

## Review (2026-10-06)

Read in full before use. Findings:

- No network access except `huggingface_hub.snapshot_download` inside `load_release_model`, which
  only runs when `model_path` is not a local directory. Our loader always passes a local directory
  downloaded at the pinned revision.
- No `exec`/`eval`, no `trust_remote_code`, no pickle loading: weights load through
  `Qwen3_5ForConditionalGeneration.from_pretrained` and `safetensors.torch.load_file` (strict).
- Behaviour that matters for the baseline (each one is documented in `baselines/clef_backend.py`):
  1. **Silent truncation.** `encode_record` cuts the *state* to `max_length - fixed_length`
     tokens (default 16,384) and optionally to `max_state_tokens`, with no status. `systemone`
     reports `usage.input_tokens` after the cut.
  2. **Control tokens in untrusted text.** All request text (state, question ids, instructions,
     option ids and descriptions) is tokenized with `tokenizer(text, add_special_tokens=False)`,
     i.e. without `split_special_tokens`. A state containing `<|im_start|>` or `<|im_end|>` becomes
     real control tokens inside Clef's chat frame.
  3. **Question order is part of the input.** Fields are numbered in request insertion order;
     choice options are sorted by id (so option order in the request does not matter).
  4. **Ties** in `systemone_answer` go to the first option in request order.
  5. **Validation** is minimal (`ValueError` on missing model/state/questions or empty criteria);
     a 1-level `score` is accepted.
  6. Images/videos are encoded through the processor; our backends are text-only.
