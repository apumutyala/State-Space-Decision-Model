`modeling_d1.py`, `lfm2_vl.py`, `hybrid.py`, `runner.py`, `prompt.py` and `api.py` are copied unmodified from
https://huggingface.co/LiquidAI/d1-3B at commit 051bcc464b01b9f92942b364d9586b0ef5912432, Copyright Liquid AI, Inc.,
under the LFM Open License v1.0 (`LICENSE`, copied from the same commit). sha256 pins are in `__init__.py`.

Used only by jobs/jev_host_scan to evaluate d1-3B (research / evaluation use) through its documented
`system_one` readout. Reviewed 2026-10-09: torch modules, prompt rendering and option-token readout; no network,
file or process access beyond `from_pretrained` on the local snapshot.
